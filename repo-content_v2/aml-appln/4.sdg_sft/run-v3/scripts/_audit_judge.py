"""Full-corpus LLM-judge audit (per-record verdicts persisted).

Companion to `audit_llm_sample.py`, but:
  * defaults to **100%** sample rate (every record judged)
  * **persists per-record verdicts as JSONL** keyed on `record_id`
    so downstream tools (filter, re-roll planner, RL phase) can join.
  * supports per-bucket **resume** so a partial run can be picked up.
  * writes DD intermediate parquet to a dedicated subdir
    (`audit_llm_full/`) so the 20%-sample artifacts in `audit_llm/`
    are not clobbered.

Output layout:
    sft_data/data/manifests/audit_judge_{bucket}.jsonl
    sft_data/data/manifests/audit_judge_full_summary.json

Each output line:
    {
      "record_id": "record_1_0000001",
      "bucket":    "sar_pos",
      "task_type": "sar_judgment",
      "source":    "enterprise_fc",
      "variant":   "augmented",
      "verdict":   "PASS" | "ISSUES_FOUND",
      "issues":    [tag1, tag2, ...],
      "explain":   "..."
    }

The reviewer prompts (see `common/reviewer_prompts.py`) embed every
strategy check the doc requires:

    * sar_pos  -> back_rationalization, fabricated_facts,
                  off_topic_citation, label_evidence_mismatch,
                  weak_reasoning, internal_contradiction,
                  factual_error, missing_finding_citation,
                  accusatory_phrasing, vacuous_template,
                  wrong_threshold_applied, non_english_text
    * sar_neg  -> empty_narrative, missing_surface_flag,
                  missing_disambiguator, missing_disposition_phrase,
                  sycophantic_framing, fabricated_facts,
                  evidence_actually_suspicious, schema_violation
                  (v3: negatives REQUIRE non-empty grounded narrative;
                  `non_empty_narrative` as an issue was a v2 contract
                  and has been REMOVED.)
    * sar_adv  -> missing_inconsistency_flag, sycophantic_to_finding,
                  fabricated_facts, off_topic_citation, weak_reasoning
    * aux_num  -> back_rationalization, calculation_uses_invented_values,
                  evidence_too_generic, factual_error,
                  missing_calculation_steps
    * aux_cit  -> paraphrase_not_verbatim, evidence_too_generic,
                  answer_unsupported_by_evidence, off_topic_question
    * aux_stat -> label_disagreement, generic_reasoning, factual_error,
                  missing_statute_citation, internal_contradiction

Invoke via the audit dispatcher:
    python -m scripts.audit judge --sample-rate 1.0 \
        --buckets aux_stat,aux_beh,aux_num,aux_cit,sar_adv,sar_neg,sar_pos

(this file is a private helper; do not invoke directly).
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("audit_judge_full")

import pandas as pd

from scripts.common.dd_helpers import run_dd_pass, safe_json_loads
from scripts.common.reviewer_prompts import (
    REVIEWER_SYSTEM_BY_BUCKET,
    REVIEWER_USER_TEMPLATE,
)
from scripts.config import CONCURRENCY, FINAL_AUX, FINAL_NONAUX, INTERIM_NONAUX, MANIFESTS_DIR


SUMMARY_OUT = MANIFESTS_DIR / "audit_judge_full_summary.json"
DD_ARTIFACT_ROOT = INTERIM_NONAUX / "_dd_artifacts" / "audit_llm_full"
LOCKFILE = MANIFESTS_DIR / "_audit_judge_full.lock"


# ============================================================================
# Lockfile — prevents concurrent runs from double-writing JSONLs
# ============================================================================
def _acquire_lock(force: bool = False) -> None:
    """Refuse to start if another instance is already running.

    A concurrent run silently double-writes per-bucket JSONLs because each
    process sees an empty 'already_judged' set at startup. This crashed
    earlier; we now fail fast.
    """
    if LOCKFILE.exists():
        try:
            other_pid = int(LOCKFILE.read_text().strip())
            # Cheap liveness check — does /proc/<pid> exist (linux only)?
            alive = Path(f"/proc/{other_pid}").exists()
        except Exception:  # noqa: BLE001
            other_pid, alive = -1, False
        if alive and not force:
            raise SystemExit(
                f"Another audit_judge_full run appears to be active "
                f"(pid={other_pid}, lockfile={LOCKFILE}). "
                "Wait for it to finish, or pass --force to override."
            )
        if not alive:
            logger.warning("Stale lockfile (pid=%s not alive) — clearing.", other_pid)
            LOCKFILE.unlink(missing_ok=True)
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text(str(os.getpid()))
    atexit.register(lambda: LOCKFILE.unlink(missing_ok=True))

# Strategy doc: SDG_STRATEGY_SFT.md "LLM Reviewer Pass" section, table at
# lines 922-929 — pass-rate targets per bucket. A pass-rate below target
# triggers a corpus-level alert (NOT a loosening of the reviewer; per the
# strategy: "signals that the generator prompt or source data needs work").
# `aux_beh` is added per training_strategy.md §1 / §5.1 / §7.1; we tier it
# at the same 85% target as aux_num (closest analog: structured aux output).
PASS_RATE_TARGETS: dict[str, float] = {
    "sar_pos":  0.85,
    "sar_neg":  0.90,
    "sar_adv":  0.85,
    "aux_num":  0.85,
    "aux_cit":  0.95,
    "aux_stat": 0.90,
    "aux_beh":  0.85,
}

# Order matters: smaller buckets first so failures surface fast.
ALL_BUCKETS = ["aux_stat", "aux_beh", "aux_cit", "aux_num", "sar_adv", "sar_neg", "sar_pos"]


# ============================================================================
# Loading + bucketing — same logic as audit_llm_sample.py
# ============================================================================
def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def _bucket_for(rec: dict) -> str | None:
    md = rec.get("metadata", {}) or {}
    tt = md.get("task_type")
    if tt == "auxiliary_numeric":
        return "aux_num"
    if tt == "auxiliary_citation":
        return "aux_cit"
    if tt == "auxiliary_statutory":
        return "aux_stat"
    if tt == "auxiliary_behavioral":
        # Dedicated bucket per training_strategy.md §1 / §5.1 / §7.1 —
        # behavioral output schema {answer:{summary, metrics, evidence}}
        # is judged with a behavioral-specific reviewer, NOT aux_num
        # which expects {answer, calculation, evidence}.
        return "aux_beh"
    if tt == "sar_judgment":
        try:
            asst = json.loads(rec["messages"][2]["content"])
            is_susp = bool(asst.get("is_suspicious"))
        except Exception:  # noqa: BLE001
            return None
        if md.get("sar_variant") == "adversarial_aux":
            return "sar_adv"
        return "sar_pos" if is_susp else "sar_neg"
    return None


def _strat_cell(rec: dict, bucket: str) -> tuple:
    """Strategy-mandated stratification cell.

    SDG_STRATEGY_SFT.md line 1654:  "stratified by (typology × sar_variant ×
                                     label)" for sar_judgment audits
    SDG_STRATEGY_SFT.md line 1975:  "Stratified by (task_type × source)" for
                                     aux audits
    """
    md = rec.get("metadata", {}) or {}
    if bucket.startswith("sar_"):
        try:
            asst = json.loads(rec["messages"][2]["content"])
            label = bool(asst.get("is_suspicious"))
        except Exception:  # noqa: BLE001
            label = False
        return (bucket,
                md.get("typology")    or "?",
                md.get("sar_variant") or "?",
                label)
    # aux_*
    return (bucket, md.get("task_type") or "?", md.get("source") or "?")


def _stratify_sample(records: list[dict], rate: float, rng: random.Random) -> list[dict]:
    """Stratified sample by strategy-mandated cells. Rate=1.0 keeps everything."""
    if rate >= 1.0:
        return records
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        b = _bucket_for(r)
        if b is None:
            continue
        cells[_strat_cell(r, b)].append(r)
    out: list[dict] = []
    for _cell, lst in cells.items():
        n_take = max(1, int(round(rate * len(lst))))
        rng.shuffle(lst)
        out.extend(lst[:n_take])
    return out


# ============================================================================
# DD reviewer pass — like common/reviewer.py:review_records but accepts an
# explicit `artifact_path` so we can shard parquet into a dedicated subdir.
# ============================================================================
def _review_pass(
    records: list[dict],
    *,
    bucket: str,
    artifact_subdir: Path,
    dataset_name: str,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
) -> list[dict]:
    """Run the LLM judge over a list of records and return aligned verdicts."""
    if not records:
        return []
    if dry_run:
        return [
            {"verdict": "PASS", "issues": [], "explain": "dry-run"}
            for _ in records
        ]
    if bucket not in REVIEWER_SYSTEM_BY_BUCKET:
        raise ValueError(f"Unknown bucket {bucket!r}")

    # Build seed dataframe — same truncation as common/reviewer.py
    rows = []
    for r in records:
        msgs = r.get("messages", [])
        rows.append(
            {
                "user_content": str(msgs[1].get("content", ""))[:8000] if len(msgs) > 1 else "",
                "assistant_content": str(msgs[2].get("content", ""))[:4000] if len(msgs) > 2 else "",
            }
        )
    seed_df = pd.DataFrame(rows)

    try:
        gen = run_dd_pass(
            seed_df=seed_df,
            system_prompt=REVIEWER_SYSTEM_BY_BUCKET[bucket],
            user_template=REVIEWER_USER_TEMPLATE,
            output_column="judge_verdict",
            dataset_name=dataset_name,
            artifact_path=artifact_subdir,
            max_parallel=CONCURRENCY.per_pipeline_llm,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Reviewer pass failed (bucket=%s): %s", bucket, exc)
        return [
            {"verdict": "PASS", "issues": [], "explain": f"reviewer-infra-error: {exc!s}"[:200]}
            for _ in records
        ]

    out: list[dict] = []
    rows_back = gen.to_dict(orient="records")
    for j, gen_row in enumerate(rows_back):
        if j >= len(records):
            break
        verdict = safe_json_loads(gen_row.get("judge_verdict", ""))
        if not isinstance(verdict, dict):
            verdict = {
                "verdict": "PASS",  # conservative: don't kill records on judge bug
                "issues": [],
                "explain": f"reviewer-parse-error: {str(gen_row.get('judge_verdict',''))[:150]}",
            }
        verdict.setdefault("verdict", "PASS")
        verdict.setdefault("issues", [])
        verdict.setdefault("explain", "")
        if verdict["verdict"] not in ("PASS", "ISSUES_FOUND"):
            verdict["verdict"] = "ISSUES_FOUND"
        out.append(verdict)
    while len(out) < len(records):
        out.append({"verdict": "PASS", "issues": [], "explain": "reviewer-no-output"})
    return out


# ============================================================================
# Per-bucket processing with resume
# ============================================================================
def _bucket_jsonl(bucket: str) -> Path:
    return MANIFESTS_DIR / f"audit_judge_{bucket}.jsonl"


def _existing_record_ids(path: Path) -> set[str]:
    """Return the set of record_ids already judged in `path` (resume support)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                rid = json.loads(line).get("record_id")
                if rid:
                    out.add(rid)
            except Exception:  # noqa: BLE001
                continue
    return out


def _process_bucket(
    bucket: str,
    records: list[dict],
    *,
    resume: bool,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    chunk_size: int,
) -> dict:
    """Judge every record in `records` (already filtered to this bucket) and
    persist verdicts as JSONL. Resumable per-record by `record_id`."""
    out_path = _bucket_jsonl(bucket)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = _existing_record_ids(out_path) if resume else set()
    todo = [
        r for r in records if (r.get("metadata") or {}).get("record_id") not in already
    ]

    print(f"\n--- bucket={bucket}: total={len(records):,}  "
          f"already_judged={len(already):,}  todo={len(todo):,} ---")
    if not todo:
        print("  ✓ already complete — skipping.")
        return _summarise(out_path, n_records=len(records))

    issue_counter: Counter = Counter()
    n_pass = 0
    n_fail = 0
    n_written = 0

    # Process in chunks so the JSONL is appended incrementally — survives Ctrl-C.
    artifact_subdir = DD_ARTIFACT_ROOT / bucket
    artifact_subdir.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as fout:
        for chunk_idx, start in enumerate(range(0, len(todo), chunk_size)):
            chunk = todo[start:start + chunk_size]
            verdicts = _review_pass(
                chunk,
                bucket=bucket,
                artifact_subdir=artifact_subdir / f"chunk_{chunk_idx:04d}",
                dataset_name=f"audit_judge_full_{bucket}_{chunk_idx:04d}",
                max_tokens=max_tokens,
                temperature=temperature,
                dry_run=dry_run,
            )
            for r, v in zip(chunk, verdicts):
                md = r.get("metadata") or {}
                line = {
                    "record_id":  md.get("record_id"),
                    "bucket":     bucket,
                    "task_type":  md.get("task_type"),
                    "source":     md.get("source"),
                    "variant":    md.get("sar_variant"),
                    "typology":   md.get("typology"),
                    "verdict":    v.get("verdict", "PASS"),
                    "issues":     list(v.get("issues") or []),
                    "explain":    str(v.get("explain") or "")[:400],
                }
                fout.write(json.dumps(line) + "\n")
                n_written += 1
                if line["verdict"] == "PASS":
                    n_pass += 1
                else:
                    n_fail += 1
                for tag in line["issues"]:
                    issue_counter[str(tag)] += 1
            fout.flush()
            print(f"  chunk {chunk_idx + 1}/{(len(todo) + chunk_size - 1) // chunk_size} "
                  f"done — wrote {len(chunk)}; "
                  f"pass={n_pass} fail={n_fail}; "
                  f"top issues: {dict(issue_counter.most_common(5))}")

    return _summarise(out_path, n_records=len(records))


def _summarise(jsonl_path: Path, *, n_records: int) -> dict:
    """Re-scan the on-disk JSONL and return aggregate stats."""
    if not jsonl_path.exists():
        return {"n": 0, "pass": 0, "pass_rate": 1.0, "issues_top": {}}
    n = pass_n = 0
    issue_counter: Counter = Counter()
    with jsonl_path.open() as f:
        for line in f:
            try:
                v = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            n += 1
            if v.get("verdict") == "PASS":
                pass_n += 1
            for tag in (v.get("issues") or []):
                issue_counter[str(tag)] += 1
    return {
        "n":               n,
        "pass":            pass_n,
        "pass_rate":       round(pass_n / max(n, 1), 4),
        "issues_top":      dict(issue_counter.most_common(15)),
        "expected_records": n_records,
        "complete":        n == n_records,
    }


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-rate", type=float, default=1.0,
                   help="fraction of corpus to judge (1.0 = all)")
    p.add_argument("--buckets", default=",".join(ALL_BUCKETS),
                   help=f"comma-separated subset of: {ALL_BUCKETS}. "
                        "Order matters — earlier buckets run first; "
                        "default is small-to-large for fast feedback.")
    p.add_argument("--no-resume", action="store_true",
                   help="re-judge records even if they already appear in the JSONL")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--chunk-size", type=int, default=2000,
                   help="DD batch size per chunk; smaller = more frequent flushes "
                        "but more startup overhead")
    p.add_argument("--dry-run", action="store_true",
                   help="emit all-PASS verdicts without calling the LLM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="ignore stale lockfile and start anyway "
                        "(use only when sure no other instance is running)")
    args = p.parse_args()

    _acquire_lock(force=args.force)

    requested_buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    for b in requested_buckets:
        if b not in ALL_BUCKETS:
            raise SystemExit(f"unknown bucket {b!r} — valid: {ALL_BUCKETS}")

    rng = random.Random(args.seed)
    nonaux = _load_jsonl(FINAL_NONAUX)
    aux = _load_jsonl(FINAL_AUX)
    print(f"\n{'=' * 78}")
    print(f"FULL LLM-JUDGE AUDIT  (sample_rate={args.sample_rate:.0%})")
    print(f"{'=' * 78}")
    print(f"Total corpus: {len(nonaux):,} non-aux + {len(aux):,} aux = "
          f"{len(nonaux) + len(aux):,} records")

    # Sample (or pass-through if rate=1.0) then bucket
    sample = _stratify_sample(nonaux + aux, args.sample_rate, rng)
    print(f"Sample after stratification: {len(sample):,} records")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in sample:
        b = _bucket_for(r)
        if b is not None:
            by_bucket[b].append(r)

    print("\nDistribution by bucket:")
    for b in ALL_BUCKETS:
        print(f"  {b:10s}: {len(by_bucket.get(b, [])):>7,d}")

    # Run each bucket. If a previous run produced a summary file, merge with it
    # so partial / incremental runs build a complete picture.
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    full_summary: dict = {
        "sample_rate":       args.sample_rate,
        "total_corpus":      len(nonaux) + len(aux),
        "sampled":           len(sample),
        "buckets":           {},
    }
    if SUMMARY_OUT.exists():
        try:
            with SUMMARY_OUT.open() as f:
                prev = json.load(f) or {}
            full_summary["buckets"].update(prev.get("buckets") or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not merge previous summary %s: %s",
                           SUMMARY_OUT, exc)

    for b in requested_buckets:
        recs = by_bucket.get(b, [])
        if not recs:
            print(f"\n--- bucket={b}: 0 records, skipping ---")
            continue
        full_summary["buckets"][b] = _process_bucket(
            b,
            recs,
            resume=not args.no_resume,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            dry_run=args.dry_run,
            chunk_size=args.chunk_size,
        )
        # Flush summary after each bucket so an interrupt still leaves a useful file
        with SUMMARY_OUT.open("w") as f:
            json.dump(full_summary, f, indent=2)

    # Aggregate
    total_n = sum(b["n"] for b in full_summary["buckets"].values())
    total_pass = sum(b["pass"] for b in full_summary["buckets"].values())
    full_summary["overall"] = {
        "n_reviewed":       total_n,
        "n_pass":           total_pass,
        "overall_pass_rate": round(total_pass / max(total_n, 1), 4),
    }

    # Compare each bucket against its strategy-mandated pass-rate target
    # (SDG_STRATEGY_SFT.md lines 922-929). Per the strategy: a bucket below
    # target signals that the *generator* (not the reviewer) needs work.
    full_summary["pass_rate_targets"] = PASS_RATE_TARGETS
    full_summary["target_alerts"] = []
    print(f"\n{'=' * 78}")
    print(f"PER-BUCKET PASS-RATE vs STRATEGY TARGET")
    print(f"{'=' * 78}")
    print(f"{'bucket':<10s} {'n':>9s}  {'pass':>9s}  "
          f"{'rate':>7s}  {'target':>7s}  status")
    for b in ALL_BUCKETS:
        info = full_summary["buckets"].get(b)
        if not info:
            continue
        rate = info["pass_rate"]
        target = PASS_RATE_TARGETS.get(b, 0.0)
        ok = rate >= target
        status = "✓ OK" if ok else f"✗ ALERT (-{100*(target-rate):.1f}pp)"
        print(f"{b:<10s} {info['n']:>9,d}  {info['pass']:>9,d}  "
              f"{100*rate:>6.1f}%  {100*target:>6.1f}%  {status}")
        if not ok:
            full_summary["target_alerts"].append({
                "bucket":       b,
                "pass_rate":    rate,
                "target":       target,
                "n":            info["n"],
                "deficit_pp":   round(100 * (target - rate), 2),
                "top_issues":   info.get("issues_top", {}),
            })

    with SUMMARY_OUT.open("w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\n{'=' * 78}")
    print(f"OVERALL: {100 * full_summary['overall']['overall_pass_rate']:.1f}%  "
          f"({total_pass:,}/{total_n:,})")
    if full_summary["target_alerts"]:
        print(f"⚠ {len(full_summary['target_alerts'])} bucket(s) below strategy target — "
              "generator/source review needed (NOT reviewer loosening).")
    else:
        print("✓ All buckets meet strategy pass-rate targets.")
    print(f"Per-bucket JSONLs: sft_data/data/manifests/audit_judge_<bucket>.jsonl")
    print(f"Summary: {SUMMARY_OUT}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
