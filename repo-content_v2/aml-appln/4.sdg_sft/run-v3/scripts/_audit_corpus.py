"""End-to-end QA audit against the strategy doc rules and constraints.

Runs every per-record validator across the produced corpora plus every
corpus-level audit (RULE-1-* and RULE-9-*). Also performs a Pydantic
schema sweep and the auxiliary↔Stage-6 output-shape contract test.

Designed to be run after `main.py` finishes, on the final JSONL outputs:

    sft_data/data/final/sar_judgment_non_auxillary_corpus.jsonl
    sft_data/data/final/auxiliary_corpus.jsonl

Invoke via the audit dispatcher:
    python -m scripts.audit corpus
    python -m scripts.audit corpus --report data/manifests/audit_report.json

(this file is a private helper; do not invoke directly).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.common.io import iter_jsonl, write_json
from scripts.config import FINAL_AUX, FINAL_NONAUX, MANIFESTS_DIR
from scripts.schemas import (
    ChatSFTRecord, CitationFinding, NumericFinding, StatutoryFinding,
)
from scripts.validators.rules_corpus import (
    audit_floor_near_miss, audit_floor_typology, audit_mix_label, audit_mix_variant,
    audit_source_cap, dedup_minhash, output_shape_contract_test,
)
from scripts.validators.rules_per_record import (
    rule_3_compat, rule_3_near_miss_benign, rule_3_none_benign,
    rule_4_counterparty_overlap, rule_5_none_empty,
    rule_6_cit_verbatim, rule_6_num_sum, rule_6_stat_label,
    rule_7_aug_cites, rule_7_bare_noleak, rule_7_length,
    # v3 narrative rules — both classes carry grounded text
    rule_7_narrative_nonempty,
    rule_7_neg_disposition_marker,
    rule_7_no_leaky_hints,
    rule_7_objectivity,
    rule_8_adv_detect,
)
from scripts.audit_semantic import (
    audit_amlgentex_cluster_uniqueness,
    audit_aux_field_distinctness,
    audit_narrative_diversity,
    audit_novel_citation_density,
    audit_per_source_field_coverage,
    audit_pool_projector_sanity,
    audit_question_well_formed,
    audit_source_task_semantic_match,
    audit_stage_manifests,
    audit_strategy_fixture,
)


# Per-record rules — every rule the strategy doc declares
PER_RECORD_RULES = [
    ("RULE-3-COMPAT", rule_3_compat),
    ("RULE-3-NEAR-MISS-BENIGN", rule_3_near_miss_benign),
    ("RULE-3-NONE-BENIGN", rule_3_none_benign),
    ("RULE-4-COUNTERPARTY-OVERLAP", rule_4_counterparty_overlap),
    ("RULE-5-NONE-EMPTY", rule_5_none_empty),
    ("RULE-6-NUM-SUM", rule_6_num_sum),
    ("RULE-6-CIT-VERBATIM", rule_6_cit_verbatim),
    ("RULE-6-STAT-LABEL", rule_6_stat_label),
    # v3 narrative rules (replace v2 RULE-7-NEG-EMPTY-NARRATIVE)
    ("RULE-7-NARR-NONEMPTY",        rule_7_narrative_nonempty),
    ("RULE-7-NEG-DISPOSITION-MARKER", rule_7_neg_disposition_marker),
    ("RULE-7-NO-LEAKY-HINTS",       rule_7_no_leaky_hints),
    ("RULE-7-OBJECTIVITY",          rule_7_objectivity),
    ("RULE-7-BARE-NOLEAK",          rule_7_bare_noleak),
    ("RULE-7-AUG-CITES",            rule_7_aug_cites),
    ("RULE-7-LENGTH",               rule_7_length),
    ("RULE-8-ADV-DETECT", rule_8_adv_detect),
]

SOFT_RULES = {"RULE-7-LENGTH"}


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_records(path: Path) -> list[dict]:
    return list(iter_jsonl(path)) if path.exists() else []


# ============================================================================
# Per-record sweep
# ============================================================================
def per_record_audit(records: list[dict], *, label: str) -> dict:
    banner(f"Per-record validators — {label} ({len(records)} records)")
    if not records:
        print("(no records)")
        return {}

    failures: Counter = Counter()
    samples: dict[str, tuple] = {}
    for rec in records:
        for rule_id, fn in PER_RECORD_RULES:
            try:
                ok, reason = fn(rec)
            except Exception as exc:  # noqa: BLE001
                ok, reason = False, f"crash: {exc}"
            if ok:
                continue
            failures[rule_id] += 1
            if rule_id not in samples:
                samples[rule_id] = (
                    rec.get("metadata", {}).get("record_id", "<no-id>"),
                    reason[:200],
                )

    n_total = len(records)
    print(f"{'Rule':<32s} {'Fails':>6s}  {'Pass-rate':>10s}  Sample")
    print("-" * 78)
    overall = {}
    for rule_id, _ in PER_RECORD_RULES:
        n_fails = failures.get(rule_id, 0)
        pass_rate = 100 * (1 - n_fails / n_total)
        soft = " (soft)" if rule_id in SOFT_RULES else ""
        flag = "✓" if n_fails == 0 else ("⚠" if rule_id in SOFT_RULES else "✗")
        rid, _reason = samples.get(rule_id, ("", ""))
        print(f"{flag} {rule_id:<30s} {n_fails:>6d}  {pass_rate:>9.2f}%  {rid[:30]}{soft}")
        overall[rule_id] = {"failures": n_fails, "pass_rate_pct": round(pass_rate, 2)}
    return overall


# ============================================================================
# Corpus-level audits — non-aux
# ============================================================================
def _print_audit(audit: dict) -> None:
    """Pretty-print a single corpus audit dict from rules_corpus."""
    rule = audit.get("rule", "?")
    passed = audit.get("passed", False)
    flag = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{rule}  {flag}")
    for k, v in audit.items():
        if k in ("rule", "passed"):
            continue
        print(f"  {k}: {v}")


def corpus_audit_nonaux(records: list[dict]) -> dict:
    banner(f"Corpus audits — non-aux ({len(records)} records)")
    if not records:
        print("(no records)")
        return {}
    audits = {
        "RULE-1-MIX-LABEL":       audit_mix_label(records),
        "RULE-1-MIX-VARIANT":     audit_mix_variant(records),
        "RULE-1-FLOOR-TYPOLOGY":  audit_floor_typology(records),
        "RULE-1-FLOOR-NEAR-MISS": audit_floor_near_miss(records),
        "RULE-9-SOURCE-CAP":      audit_source_cap(records),
    }
    for a in audits.values():
        _print_audit(a)
    return audits


# ============================================================================
# Corpus-level audits — auxiliary (per-task share + per-source cap)
# ============================================================================
def corpus_audit_aux(records: list[dict]) -> dict:
    banner(f"Corpus audits — aux ({len(records)} records)")
    if not records:
        print("(no records)")
        return {}

    target = {"auxiliary_numeric": 0.32, "auxiliary_citation": 0.32, "auxiliary_statutory": 0.36}
    counter: Counter = Counter(r["metadata"].get("task_type") for r in records)
    total = sum(counter.values()) or 1
    obs = {t: counter.get(t, 0) / total for t in target}
    diffs_pp = {t: abs(obs[t] - target[t]) * 100 for t in target}
    task_audit = {
        "rule": "AUX-TASK-MIX",
        "target": target,
        "observed": obs,
        "diffs_pp": diffs_pp,
        "tolerance_pp": 2.0,
        "passed": max(diffs_pp.values()) <= 2.0,
    }
    _print_audit(task_audit)

    # Per-task source cap (no source > 60% within its task type)
    by_task: dict[str, Counter] = {}
    for r in records:
        t = r["metadata"].get("task_type")
        s = r["metadata"].get("source")
        if t and s:
            by_task.setdefault(t, Counter())[s] += 1

    cap_audit = {
        "rule": "AUX-PER-TASK-SOURCE-CAP",
        "cap": 0.60,
        "by_task": {},
        "above_cap": {},
        "passed": True,
    }
    for task, srcs in by_task.items():
        ttotal = sum(srcs.values()) or 1
        cap_audit["by_task"][task] = {s: round(c / ttotal, 4) for s, c in srcs.items()}
        for s, c in srcs.items():
            if c / ttotal > 0.60:
                cap_audit["above_cap"][f"{task}/{s}"] = round(c / ttotal, 4)
                cap_audit["passed"] = False
    _print_audit(cap_audit)

    return {"AUX-TASK-MIX": task_audit, "AUX-PER-TASK-SOURCE-CAP": cap_audit}


# ============================================================================
# Schema sweep
# ============================================================================
def schema_audit(records: list[dict], *, label: str) -> dict:
    banner(f"Pydantic schema sweep — {label}")
    if not records:
        print("(no records)")
        return {"checked": 0, "passed": 0, "failed": 0}
    n_pass = n_fail = 0
    sample_fails: list[tuple] = []
    for r in records:
        try:
            ChatSFTRecord(**r)
            n_pass += 1
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            if len(sample_fails) < 3:
                sample_fails.append((r.get("metadata", {}).get("record_id", "?"), str(exc)[:200]))
    print(f"  Pass: {n_pass}/{len(records)} ({100 * n_pass / len(records):.2f}%)")
    if sample_fails:
        print("  Sample failures:")
        for rid, exc in sample_fails:
            print(f"    {rid}: {exc}")
    return {"checked": len(records), "passed": n_pass, "failed": n_fail,
            "samples": [{"record_id": r, "error": e} for r, e in sample_fails]}


# ============================================================================
# Stage-6 ↔ Aux-record shape contract
# ============================================================================
def shape_contract_audit(aux_records: list[dict], nonaux_records: list[dict]) -> dict:
    banner("Output-shape contract: aux records ≡ Stage-6 inline findings")
    inline = []
    for r in nonaux_records:
        try:
            user = json.loads(r["messages"][1]["content"])
        except Exception:  # noqa: BLE001
            continue
        aux = user.get("auxiliary_findings")
        if not isinstance(aux, dict):
            continue
        for kind in ("numeric", "citation", "statutory"):
            for entry in (aux.get(kind) or []):
                inline.append({"kind": kind, "payload": entry})
                if len(inline) >= 200:
                    break
            if len(inline) >= 200:
                break
        if len(inline) >= 200:
            break

    res = output_shape_contract_test(aux_records, inline, sample_n=100)
    flag = "✓ PASS" if res.get("passed") else "✗ FAIL"
    print(f"  {flag}  samples_checked={res.get('samples_checked')}  failures={res.get('failures')}")
    for f in res.get("sample_failures", [])[:3]:
        print(f"    {f[:200]}")
    return res


# ============================================================================
# Random sample inspection (qualitative)
# ============================================================================
def quality_samples(records: list[dict], *, n_per_cell: int = 1) -> None:
    banner("Quality samples — one per (variant × is_suspicious) cell")
    seen: set = set()
    for r in records:
        md = r["metadata"]
        try:
            out = json.loads(r["messages"][2]["content"])
        except Exception:  # noqa: BLE001
            continue
        is_susp = bool(out.get("is_suspicious"))
        key = (md.get("sar_variant"), is_susp, md.get("typology"))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 6:
            break
        narr = out.get("suspicious_activity_report") or ""
        print(f"\n--- {md.get('record_id', '?')}  variant={md.get('sar_variant')}  "
              f"typology={md.get('typology')}  is_suspicious={is_susp}  "
              f"narr_len={len(narr)} ---")
        if narr:
            print(narr[:700] + ("..." if len(narr) > 700 else ""))
        else:
            print("(empty narrative)")


# ============================================================================
# Top-level summary
# ============================================================================
def summary_table(report: dict) -> None:
    banner("SUMMARY")
    nonaux_total = report.get("nonaux", {}).get("schema", {}).get("checked", 0)
    aux_total = report.get("aux", {}).get("schema", {}).get("checked", 0)
    print(f"  Non-aux records:           {nonaux_total}")
    print(f"  Auxiliary records:         {aux_total}")
    print(f"  Combined deliverable:      {nonaux_total + aux_total}")

    # Per-record hard fails (excluding soft rules)
    nonaux_hard = sum(v["failures"] for k, v in report.get("nonaux", {}).get("per_record", {}).items()
                      if k not in SOFT_RULES)
    aux_hard = sum(v["failures"] for k, v in report.get("aux", {}).get("per_record", {}).items()
                   if k not in SOFT_RULES)
    print(f"  Non-aux hard rule fails:   {nonaux_hard}")
    print(f"  Aux hard rule fails:       {aux_hard}")

    # Corpus audits
    corpus_pass = sum(1 for a in report.get("nonaux", {}).get("corpus", {}).values() if a.get("passed"))
    corpus_total = len(report.get("nonaux", {}).get("corpus", {}))
    print(f"  Non-aux corpus audits:     {corpus_pass}/{corpus_total} passed")

    aux_corpus_pass = sum(1 for a in report.get("aux", {}).get("corpus", {}).values() if a.get("passed"))
    aux_corpus_total = len(report.get("aux", {}).get("corpus", {}))
    print(f"  Aux corpus audits:         {aux_corpus_pass}/{aux_corpus_total} passed")

    contract = report.get("contract", {})
    flag = "✓" if contract.get("passed") else "✗"
    print(f"  Shape contract test:       {flag}")


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nonaux", default=str(FINAL_NONAUX))
    p.add_argument("--aux", default=str(FINAL_AUX))
    p.add_argument("--report", default=str(MANIFESTS_DIR / "audit_report.json"))
    args = p.parse_args()

    nonaux = load_records(Path(args.nonaux))
    aux = load_records(Path(args.aux))

    report: dict = {"nonaux": {}, "aux": {}, "contract": {}}

    # Non-aux
    report["nonaux"]["schema"] = schema_audit(nonaux, label="non-aux")
    report["nonaux"]["per_record"] = per_record_audit(nonaux, label="non-aux")
    report["nonaux"]["corpus"] = corpus_audit_nonaux(nonaux)

    # Auxiliary
    report["aux"]["schema"] = schema_audit(aux, label="aux")
    report["aux"]["per_record"] = per_record_audit(aux, label="aux")
    report["aux"]["corpus"] = corpus_audit_aux(aux)

    # Cross-corpus contract
    report["contract"] = shape_contract_audit(aux, nonaux)

    # ========================================================================
    # Semantic audits (A1-A12) — added to catch field-level fidelity bugs
    # that the structural audits above cannot detect.
    # ========================================================================
    banner("Semantic audits — strategy-fidelity, field-distinctness, diversity")
    semantic_results = {}

    semantic_results["A1-strategy-fixture"]       = audit_strategy_fixture(nonaux, aux)
    semantic_results["A2-aux-field-distinctness"] = audit_aux_field_distinctness(aux)
    semantic_results["A3-question-well-formed"]   = audit_question_well_formed(aux)
    semantic_results["A5-source-task-match"]      = audit_source_task_semantic_match(aux)
    semantic_results["A6-narrative-diversity"]    = audit_narrative_diversity(nonaux)
    semantic_results["A7-novel-citation-density"] = audit_novel_citation_density(nonaux)
    semantic_results["A9-stage-manifests"]        = audit_stage_manifests(MANIFESTS_DIR)
    semantic_results["A10-amlgentex-uniqueness"]  = audit_amlgentex_cluster_uniqueness(nonaux)
    semantic_results["A11-source-field-coverage"] = audit_per_source_field_coverage(aux)

    for name, res in semantic_results.items():
        flag = "✓ PASS" if res.get("passed") else "✗ FAIL"
        print(f"\n{name}  {flag}")
        for k, v in res.items():
            if k in ("rule", "passed", "samples", "all_cells", "by_source",
                     "issues", "failing", "by_source_task", "by_reason",
                     "failing_cells", "summary"):
                continue
            print(f"  {k}: {v}")
        # Always show first 2 samples / issues / failing
        if res.get("samples"):
            print(f"  samples: {res['samples'][:2]}")
        if res.get("issues"):
            print(f"  issues (first 3): {res['issues'][:3]}")
        if res.get("failing"):
            print(f"  failing (first 3): {res['failing'][:3]}")
        if res.get("failing_cells"):
            print(f"  failing_cells (first 3): {res['failing_cells'][:3]}")

    report["semantic"] = semantic_results

    # A12: pool-projector sanity (run separately at the start of next run too)
    banner("A12 — Pool projector schema sanity (smoke samples per pool)")
    a12 = audit_pool_projector_sanity(sample_n=100)
    flag = "✓ PASS" if a12["passed"] else "✗ FAIL"
    print(f"  {flag}")
    for pool, info in a12.get("summary", {}).items():
        print(f"  {pool}: {info}")
    for issue in a12.get("issues", []):
        print(f"  ISSUE: {issue}")
    report["a12_pool_projector_sanity"] = a12

    # Qualitative samples
    quality_samples(nonaux)

    # Summary
    summary_table(report)

    # Print failing semantic audits in the summary
    failing_semantic = [k for k, v in semantic_results.items() if not v.get("passed")]
    if failing_semantic:
        print(f"  Semantic audits failing:   {failing_semantic}")
    else:
        print(f"  Semantic audits:           all {len(semantic_results)} passed")
    if not a12["passed"]:
        print(f"  Pool projector sanity:     ✗ ({len(a12.get('issues', []))} issues)")
    else:
        print(f"  Pool projector sanity:     ✓")

    write_json(report, args.report)
    print(f"\nFull JSON report written to: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
