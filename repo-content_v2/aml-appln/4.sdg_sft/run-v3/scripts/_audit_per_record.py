"""Per-record audit dumper.

Runs every per-record validator (RULE-3-* through RULE-8-*) declared by the
strategy doc on 100% of the produced corpora and persists ONE LINE PER RECORD
to disk so downstream tools (filter, top-up planner, RL phase) can join on
`record_id` and decide what to keep.

Outputs (JSONL):
    sft_data/data/manifests/audit_per_record_nonaux.jsonl
    sft_data/data/manifests/audit_per_record_aux.jsonl

Each line has the shape:

    {
      "record_id": "record_1_0000001",
      "task_type": "sar_judgment",
      "source":    "enterprise_fc",
      "typology":  "layering",
      "sar_variant": "augmented",
      "is_suspicious": true,
      "hard_fails":     ["RULE-X-...", ...],   # any non-soft rule
      "soft_warnings":  ["RULE-7-LENGTH", ...],
      "fail_reasons":   {"RULE-X-...": "<short reason>"}
    }

This is the per-record companion to `audit.py` (which only keeps aggregate
counts in `audit_report.json`). Both `training_strategy.md` (Appendix G) and
`SDG_STRATEGY_SFT.md` (validation philosophy) require these per-record
verdicts to be available for the filter / top-up loop.

Invoke via the audit dispatcher:
    python -m scripts.audit per-record

(this file is a private helper; do not invoke directly).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("audit_per_record_dump")

from scripts.common.io import iter_jsonl, write_json
from scripts.config import FINAL_AUX, FINAL_NONAUX, MANIFESTS_DIR
from scripts.validators.rules_per_record import (
    rule_3_compat,
    rule_3_near_miss_benign,
    rule_3_none_benign,
    rule_4_counterparty_overlap,
    rule_5_none_empty,
    rule_6_cit_verbatim,
    rule_6_num_sum,
    rule_6_stat_label,
    rule_7_aug_cites,
    rule_7_bare_noleak,
    rule_7_length,
    # v3 narrative rules — both classes must carry grounded text
    rule_7_narrative_nonempty,
    rule_7_neg_disposition_marker,
    rule_7_no_leaky_hints,
    rule_7_objectivity,
    rule_8_adv_detect,
)

# Same rule list as audit.py — keep in sync.
PER_RECORD_RULES = [
    ("RULE-3-COMPAT",               rule_3_compat),
    ("RULE-3-NEAR-MISS-BENIGN",     rule_3_near_miss_benign),
    ("RULE-3-NONE-BENIGN",          rule_3_none_benign),
    ("RULE-4-COUNTERPARTY-OVERLAP", rule_4_counterparty_overlap),
    ("RULE-5-NONE-EMPTY",           rule_5_none_empty),
    ("RULE-6-NUM-SUM",              rule_6_num_sum),
    ("RULE-6-CIT-VERBATIM",         rule_6_cit_verbatim),
    ("RULE-6-STAT-LABEL",           rule_6_stat_label),
    # v3 per-record narrative rules (both classes carry grounded narratives)
    ("RULE-7-NARR-NONEMPTY",        rule_7_narrative_nonempty),
    ("RULE-7-NEG-DISPOSITION-MARKER", rule_7_neg_disposition_marker),
    ("RULE-7-NO-LEAKY-HINTS",       rule_7_no_leaky_hints),
    ("RULE-7-OBJECTIVITY",          rule_7_objectivity),
    ("RULE-7-BARE-NOLEAK",          rule_7_bare_noleak),
    ("RULE-7-AUG-CITES",            rule_7_aug_cites),
    ("RULE-7-LENGTH",               rule_7_length),
    ("RULE-8-ADV-DETECT",           rule_8_adv_detect),
]

# Soft rules: a failure becomes a warning, not a hard drop. Same convention as
# audit.py / `manifest_nonaux.json::soft_warnings`.
SOFT_RULES = {"RULE-7-LENGTH", "RULE-7-AUG-CITES"}

# Output paths
OUT_NONAUX = MANIFESTS_DIR / "audit_per_record_nonaux.jsonl"
OUT_AUX    = MANIFESTS_DIR / "audit_per_record_aux.jsonl"
OUT_SUMMARY = MANIFESTS_DIR / "audit_per_record_summary.json"


def _is_susp(rec: dict) -> bool | None:
    """Parse `is_suspicious` from the assistant message of a sar_judgment record."""
    md = rec.get("metadata") or {}
    if md.get("task_type") != "sar_judgment":
        return None
    try:
        ass = json.loads(rec["messages"][2]["content"])
        return bool(ass.get("is_suspicious"))
    except Exception:
        return None


def _eval_one(rec: dict) -> dict:
    """Run all per-record rules on a single record and return the verdict dict."""
    md = rec.get("metadata") or {}
    hard: list[str] = []
    soft: list[str] = []
    reasons: dict[str, str] = {}

    for rule_id, fn in PER_RECORD_RULES:
        try:
            ok, reason = fn(rec)
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, f"crash: {exc!r}"

        if ok:
            continue
        if rule_id in SOFT_RULES:
            soft.append(rule_id)
        else:
            hard.append(rule_id)
        # Keep one short reason per rule (truncated)
        reasons[rule_id] = (reason or "")[:240]

    return {
        "record_id":     md.get("record_id"),
        "task_type":     md.get("task_type"),
        "source":        md.get("source"),
        "typology":      md.get("typology"),
        "sar_variant":   md.get("sar_variant"),
        "is_suspicious": _is_susp(rec),
        "hard_fails":    hard,
        "soft_warnings": soft,
        "fail_reasons":  reasons,
    }


def _process(input_path: Path, output_path: Path, *, label: str) -> dict:
    """Stream a corpus through the rule set, writing one verdict line per record."""
    if not input_path.exists():
        logger.warning("%s not found — skipping %s", input_path, label)
        return {"label": label, "records": 0, "skipped": True}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_hard_any = 0
    n_soft_any = 0
    n_clean = 0
    by_rule_hard: Counter = Counter()
    by_rule_soft: Counter = Counter()
    rule_x_variant_hard: Counter = Counter()
    rule_x_source_hard: Counter = Counter()

    print(f"\n=== Per-record audit dump — {label} ===")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")

    with output_path.open("w") as out:
        for rec in iter_jsonl(input_path):
            v = _eval_one(rec)
            out.write(json.dumps(v) + "\n")
            n_total += 1
            if v["hard_fails"]:
                n_hard_any += 1
                for r in v["hard_fails"]:
                    by_rule_hard[r] += 1
                    rule_x_variant_hard[(r, v.get("sar_variant"))] += 1
                    rule_x_source_hard[(r, v.get("source"))] += 1
            if v["soft_warnings"]:
                n_soft_any += 1
                for r in v["soft_warnings"]:
                    by_rule_soft[r] += 1
            if not v["hard_fails"] and not v["soft_warnings"]:
                n_clean += 1
            if n_total % 20000 == 0:
                logger.info("  %s: processed %d records (clean=%d, hard=%d, soft=%d)",
                            label, n_total, n_clean, n_hard_any, n_soft_any)

    pct = lambda x: f"{100*x/max(n_total,1):.2f}%"
    print(f"  records:        {n_total}")
    print(f"  fully clean:    {n_clean} ({pct(n_clean)})")
    print(f"  any hard fail:  {n_hard_any} ({pct(n_hard_any)})")
    print(f"  any soft warn:  {n_soft_any} ({pct(n_soft_any)})")
    print(f"  hard fails by rule:")
    for rule_id, n in by_rule_hard.most_common():
        print(f"    {rule_id:<32s} {n:>7d}  ({pct(n)})")
    if by_rule_soft:
        print(f"  soft warnings by rule:")
        for rule_id, n in by_rule_soft.most_common():
            print(f"    {rule_id:<32s} {n:>7d}  ({pct(n)})")

    summary: dict = {
        "label":                label,
        "input":                str(input_path),
        "output":               str(output_path),
        "records":              n_total,
        "fully_clean":          n_clean,
        "any_hard_fail":        n_hard_any,
        "any_soft_warn":        n_soft_any,
        "hard_fails_by_rule":   dict(by_rule_hard),
        "soft_warnings_by_rule": dict(by_rule_soft),
        "hard_fails_by_rule_variant": {
            f"{r}|{v}": n for (r, v), n in rule_x_variant_hard.most_common(50)
        },
        "hard_fails_by_rule_source":  {
            f"{r}|{s}": n for (r, s), n in rule_x_source_hard.most_common(50)
        },
        "soft_rules": sorted(SOFT_RULES),
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nonaux", default=str(FINAL_NONAUX),
                   help="non-aux corpus JSONL")
    p.add_argument("--aux",    default=str(FINAL_AUX),
                   help="aux corpus JSONL")
    p.add_argument("--out-nonaux", default=str(OUT_NONAUX))
    p.add_argument("--out-aux",    default=str(OUT_AUX))
    p.add_argument("--out-summary", default=str(OUT_SUMMARY))
    p.add_argument("--skip-nonaux", action="store_true")
    p.add_argument("--skip-aux",    action="store_true")
    args = p.parse_args()

    summary: dict = {"by_corpus": {}}

    if not args.skip_nonaux:
        summary["by_corpus"]["nonaux"] = _process(
            Path(args.nonaux), Path(args.out_nonaux), label="nonaux")

    if not args.skip_aux:
        summary["by_corpus"]["aux"] = _process(
            Path(args.aux), Path(args.out_aux), label="aux")

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.out_summary)
    print(f"\nPer-corpus summary written to: {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
