"""Score a trace directory against eval_keys.jsonl.

Pure stdlib + pandas (no NAT runtime). Joins per-case traces with the
held-out ground-truth keys and computes the workshop's headline metrics:

  * Confusion matrix (tp / fp / tn / fn)
  * F1 / precision / recall on the SAR-positive class
  * Near-miss specificity (fraction of near-miss-NEG correctly classified)
  * Clean false-positive rate
  * Per-typology recall (positives only)
  * Per-typology near-miss-specificity (negatives only) — v3.1 cohort
  * Narrative-length stats + JSON parse-failure rate

Usage:
    python -m scripts.score_traces \\
        --traces  ./data/traces_base_nemotron_v3p1 \\
        --eval-keys ./data/demo/eval_keys.jsonl \\
        --label "nemotron-3-nano (base)" \\
        --out ./data/traces_base_nemotron_v3p1/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_traces(trace_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fp in sorted(trace_dir.glob("*.json")):
        try:
            t = json.loads(fp.read_text())
        except Exception:
            continue
        case_id = t.get("case_id") or fp.stem
        out[case_id] = t
    return out


def load_eval_keys(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out[obj["case_id"]] = obj
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def _f1(p, r):
    if p == 0 and r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
def score(trace_dir: Path, eval_keys_path: Path, label: str) -> dict:
    traces = load_traces(trace_dir)
    keys   = load_eval_keys(eval_keys_path)

    tp = fp = tn = fn = 0
    fp_clean = tn_clean = 0
    parse_failures = 0
    errors = 0
    narrative_chars: list[int] = []
    n_non_empty = 0
    latencies: list[float] = []

    # per-typology buckets
    tpy_recall = defaultdict(lambda: {"tp": 0, "fn": 0})
    tpy_nmspec = defaultdict(lambda: {"correct": 0, "wrong": 0})

    n_traces_present = 0
    n_scored = 0

    for case_id, gt in keys.items():
        tr = traces.get(case_id)
        if tr is None:
            continue
        n_traces_present += 1

        if tr.get("error"):
            errors += 1
            continue
        if tr.get("sar_parse_error"):
            parse_failures += 1
            continue

        pred = tr.get("sar_is_suspicious")
        if pred is None:
            parse_failures += 1
            continue

        n_scored += 1

        gt_label = bool(gt.get("expected_label"))
        is_nm    = bool(gt.get("near_miss", False))
        typ      = gt.get("expected_typology", "none")

        # Confusion
        if gt_label:
            if pred:
                tp += 1
                tpy_recall[typ]["tp"] += 1
            else:
                fn += 1
                tpy_recall[typ]["fn"] += 1
        else:
            if pred:
                fp += 1
                if not is_nm:
                    fp_clean += 1
            else:
                tn += 1
                if not is_nm:
                    tn_clean += 1
            if is_nm:
                if pred:
                    tpy_nmspec[typ]["wrong"] += 1
                else:
                    tpy_nmspec[typ]["correct"] += 1

        # narrative
        narr = tr.get("sar_narrative", "") or ""
        if narr:
            n_non_empty += 1
            narrative_chars.append(len(narr))

        # latency
        if tr.get("wall_clock_ms") is not None:
            try:
                latencies.append(float(tr["wall_clock_ms"]))
            except Exception:
                pass

    # Headline metrics
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _f1(precision, recall)

    # NM specificity (total)
    nm_correct = sum(v["correct"] for v in tpy_nmspec.values())
    nm_wrong   = sum(v["wrong"]   for v in tpy_nmspec.values())
    near_miss_specificity = _safe_div(nm_correct, nm_correct + nm_wrong)

    # Clean FPR
    clean_fpr = _safe_div(fp_clean, fp_clean + tn_clean)

    # Per-typology summaries
    per_typology_recall = {}
    for typ, v in tpy_recall.items():
        n = v["tp"] + v["fn"]
        per_typology_recall[typ] = {
            "tp": v["tp"], "fn": v["fn"],
            "recall": round(_safe_div(v["tp"], n), 3),
        }
    per_typology_nm = {}
    for typ, v in tpy_nmspec.items():
        n = v["correct"] + v["wrong"]
        per_typology_nm[typ] = {
            "correct": v["correct"], "wrong": v["wrong"],
            "specificity": round(_safe_div(v["correct"], n), 3),
        }

    return {
        "label": label,
        "traces_dir": str(trace_dir),
        "n_total_keys": len(keys),
        "n_traces_present": n_traces_present,
        "n_scored": n_scored,
        "n_errors": errors,
        "n_parse_failures": parse_failures,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "f1": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "near_miss_specificity": round(near_miss_specificity, 3),
            "clean_fpr": round(clean_fpr, 3),
        },
        "per_typology_recall": per_typology_recall,
        "per_typology_nm_specificity": per_typology_nm,
        "narrative_stats": {
            "n_non_empty": n_non_empty,
            "mean_chars":   round(mean(narrative_chars), 1) if narrative_chars else 0,
            "median_chars": median(narrative_chars) if narrative_chars else 0,
            "min_chars": min(narrative_chars) if narrative_chars else 0,
            "max_chars": max(narrative_chars) if narrative_chars else 0,
        },
        "wall_clock_ms": {
            "mean":   round(mean(latencies), 1) if latencies else 0,
            "median": round(median(latencies), 1) if latencies else 0,
        },
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--eval-keys", required=True, type=Path)
    p.add_argument("--label", default="(unnamed)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    result = score(args.traces, args.eval_keys, args.label)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote: {args.out}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
