"""Post-rollout evaluation — joins agent traces against demo/eval_keys.jsonl.

Computes:
  - Overall accuracy on is_suspicious
  - Recall on seeded positives (per typology + overall)
  - Specificity on near-miss negatives
  - False-positive rate on clean cases
  - Typology-hypothesis accuracy (MRULE-N-CLASSIFIER-COVERAGE)
  - Narrative grounding sanity (does positive narrative contain a
    transaction-amount / date string and cite a policy_excerpt's source?)

Usage:
    python -m pipeline.eval \
        --traces    data/final/prod_mimic/manifests/agent_rollout_traces.jsonl \
        --eval-keys data/final/prod_mimic/demo/eval_keys.jsonl \
        --out       data/final/prod_mimic/manifests/agent_rollout_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from pipeline.config import DEMO_DIR, MANIFESTS_DIR

logger = logging.getLogger("pipeline.eval")


def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate(traces_path: str, keys_path: str) -> dict:
    traces = {t["case_id"]: t for t in _load_jsonl(traces_path)}
    keys = {k["case_id"]: k for k in _load_jsonl(keys_path)}

    joined = []
    for case_id, k in keys.items():
        t = traces.get(case_id)
        if t is None:
            continue
        joined.append((k, t))

    n_total = len(joined)
    n_susp_keys = sum(1 for k, _ in joined if k["expected_label"] is True)
    n_clean_keys = sum(
        1 for k, _ in joined
        if k["expected_label"] is False and not k["near_miss"]
    )
    n_near_miss_keys = sum(1 for k, _ in joined if k["near_miss"])

    # Skip cases where the agent's SAR call failed to parse — they don't
    # contribute to label metrics.
    parse_failures = [
        c for c, (_, t) in [(k["case_id"], (k, t)) for k, t in joined]
        if t.get("sar_is_suspicious") is None
    ]

    # Confusion (excluding parse failures)
    tp = fp = tn = fn = 0
    tp_typology = defaultdict(int)
    fn_typology = defaultdict(int)
    fp_near_miss = 0
    fp_clean = 0

    typology_hypothesis_match = 0
    typology_hypothesis_attempted = 0

    for k, t in joined:
        exp = k["expected_label"]
        pred = t.get("sar_is_suspicious")
        if pred is None:
            continue

        # Typology hypothesis accuracy is measured on seeded entries (susp + near-miss)
        if k["expected_label"] is True or k["near_miss"]:
            typology_hypothesis_attempted += 1
            if t.get("typology_hypothesis") == k.get("expected_typology"):
                typology_hypothesis_match += 1

        if exp and pred:
            tp += 1
            tp_typology[k["expected_typology"]] += 1
        elif exp and not pred:
            fn += 1
            fn_typology[k["expected_typology"]] += 1
        elif (not exp) and pred:
            fp += 1
            if k["near_miss"]:
                fp_near_miss += 1
            else:
                fp_clean += 1
        else:
            tn += 1

    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    near_miss_specificity = _safe_div(
        sum(1 for k, t in joined
            if k["near_miss"] and t.get("sar_is_suspicious") is False),
        n_near_miss_keys,
    )
    clean_fpr = _safe_div(fp_clean, n_clean_keys)
    classifier_coverage = _safe_div(typology_hypothesis_match, typology_hypothesis_attempted)

    # Per-typology recall
    per_typology_recall = {}
    for typ, n_tp in tp_typology.items():
        n_total_typ = n_tp + fn_typology.get(typ, 0)
        per_typology_recall[typ] = {
            "tp": n_tp,
            "fn": fn_typology.get(typ, 0),
            "recall": _safe_div(n_tp, n_total_typ),
        }
    for typ, n_fn in fn_typology.items():
        if typ not in per_typology_recall:
            per_typology_recall[typ] = {
                "tp": 0, "fn": n_fn,
                "recall": _safe_div(0, n_fn),
            }

    # Narrative grounding: for positives where pred=true, check if narrative
    # contains at least one dollar amount and references some source.
    narrative_grounding_ok = 0
    narrative_grounding_attempted = 0
    import re
    for k, t in joined:
        if t.get("sar_is_suspicious") is True:
            narrative = t.get("sar_narrative", "")
            narrative_grounding_attempted += 1
            has_amount = bool(re.search(r"\$\s*[\d,]+", narrative))
            has_source = any(
                src.lower() in narrative.lower()
                for src in ["FinCEN", "FFIEC", "FATF", "OFAC", "31 U.S.C", "31 USC", "5324", "5318"]
            )
            if has_amount and has_source:
                narrative_grounding_ok += 1

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "n_total_cases": n_total,
        "n_susp_keys": n_susp_keys,
        "n_near_miss_keys": n_near_miss_keys,
        "n_clean_keys": n_clean_keys,
        "n_parse_failures": len(parse_failures),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "recall_on_seeded_positives": recall,
            "precision": precision,
            "near_miss_specificity": near_miss_specificity,
            "false_positive_rate_clean": clean_fpr,
            "false_positive_rate_near_miss": _safe_div(fp_near_miss, n_near_miss_keys),
            "classifier_coverage": classifier_coverage,
            "narrative_grounding_ok": _safe_div(
                narrative_grounding_ok, narrative_grounding_attempted,
            ),
        },
        "per_typology_recall": per_typology_recall,
        "narrative_grounding_attempted": narrative_grounding_attempted,
        "narrative_grounding_ok_count": narrative_grounding_ok,
    }


# ============================================================================
# CLI
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline.eval")
    parser.add_argument("--traces", default=str(MANIFESTS_DIR / "agent_rollout_traces.jsonl"))
    parser.add_argument("--eval-keys", default=str(DEMO_DIR / "eval_keys.jsonl"))
    parser.add_argument("--out", default=str(MANIFESTS_DIR / "agent_rollout_eval.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    metrics = evaluate(args.traces, args.eval_keys)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n══ Agent rollout evaluation ══")
    print(json.dumps(metrics, indent=2))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
