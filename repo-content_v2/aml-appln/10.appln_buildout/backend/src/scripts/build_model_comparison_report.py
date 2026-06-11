"""Fuse N per-endpoint `eval.json` files into one comparative report.

Output JSON shape is frontend-ready: a list of endpoints, a list of
metric rows (with per-endpoint values + winner annotation), and per-
typology breakdown tables. Pre-computed once after a benchmark run;
served as-is by `POST /api/demo/eval/model_comparison`.

Usage:
    python -m scripts.build_model_comparison_report \\
        --eval ./data/eval_base.json:"nemotron-3-nano (base)" \\
        --eval ./data/eval_gemma.json:"gemma-4-31b-it (frontier)" \\
        --eval ./data/eval_custom.json:"aml-custom-task-nim (SFT)" \\
        --out ./data/benchmarks/model_comparison.json

The script also writes a pointer file at
`./data/benchmarks/latest.json` containing the relative path of the
most recently built report; the API endpoint reads from there.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Static config (which metrics are "higher is better", display order, etc.)
# ---------------------------------------------------------------------------
_HIGHER_IS_BETTER: dict[str, bool] = {
    "f1": True,
    "precision": True,
    "recall": True,
    "near_miss_specificity": True,
    "clean_fpr": False,
}
_HEADLINE_ORDER  = ("f1", "precision", "recall", "near_miss_specificity", "clean_fpr")
_COVERAGE_FIELDS = ("n_scored", "n_parse_failures", "n_errors")


def _load_eval(path: Path) -> dict:
    return json.loads(path.read_text())


def _argmax(values: dict[str, float | int], higher_is_better: bool) -> Optional[str]:
    """Return the endpoint label whose metric value is best (None on ties / missing)."""
    if not values:
        return None
    # Skip None entries
    clean = {k: v for k, v in values.items() if v is not None}
    if not clean:
        return None
    target = max(clean.values()) if higher_is_better else min(clean.values())
    winners = [k for k, v in clean.items() if v == target]
    if len(winners) != 1:
        return None  # tie → no winner
    return winners[0]


def build_report(inputs: list[tuple[str, Path]], *,
                  demo_size: int, demo_version: str, notes: str) -> dict:
    """Combine N per-endpoint eval files into one comparative report.

    Args:
        inputs: list of (label, eval_json_path) tuples — order is preserved
                in the output (so the frontend can render columns in a
                deterministic order; convention is base → frontier → custom).
        demo_size: expected number of cases in the eval set.
        demo_version: human-readable demo dataset version.
        notes: optional free-text caveat (rendered as a banner in the UI).
    """
    endpoints: list[dict] = []
    raw_by_label: dict[str, dict] = {}
    for label, path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"eval json missing: {path}")
        ev = _load_eval(path)
        raw_by_label[label] = ev
        endpoints.append({
            "label": label,
            "eval_path": str(path),
            "traces_dir": ev.get("traces_dir"),
            "n_total_keys": ev.get("n_total_keys"),
        })

    # Coverage rows (n_scored / parse_failures / errors per endpoint)
    coverage_rows = []
    for field in _COVERAGE_FIELDS:
        coverage_rows.append({
            "field": field,
            "values": {lbl: raw_by_label[lbl].get(field) for lbl, _ in inputs},
        })

    # Headline metric rows
    headline_metrics = []
    for metric in _HEADLINE_ORDER:
        hib = _HIGHER_IS_BETTER[metric]
        vals = {lbl: raw_by_label[lbl].get("metrics", {}).get(metric)
                for lbl, _ in inputs}
        headline_metrics.append({
            "metric": metric,
            "higher_is_better": hib,
            "values": vals,
            "winner_label": _argmax(vals, hib),
        })

    # Confusion matrices per endpoint
    confusion = []
    for lbl, _ in inputs:
        c = raw_by_label[lbl].get("confusion", {})
        confusion.append({"label": lbl,
                           "tp": c.get("tp"), "fp": c.get("fp"),
                           "tn": c.get("tn"), "fn": c.get("fn")})

    # Per-typology recall + near-miss specificity
    all_recall_typs = set()
    all_nmspec_typs = set()
    for lbl, _ in inputs:
        all_recall_typs.update(raw_by_label[lbl].get("per_typology_recall", {}).keys())
        all_nmspec_typs.update(raw_by_label[lbl].get("per_typology_nm_specificity", {}).keys())

    per_typology_recall = []
    for typ in sorted(all_recall_typs):
        vals = {}
        ns = {}
        for lbl, _ in inputs:
            row = raw_by_label[lbl].get("per_typology_recall", {}).get(typ, {})
            vals[lbl] = row.get("recall")
            ns[lbl]   = row.get("tp", 0) + row.get("fn", 0)
        per_typology_recall.append({
            "typology": typ,
            "values": vals,
            "n_per_endpoint": ns,
            "winner_label": _argmax(vals, True),
        })

    per_typology_nm = []
    for typ in sorted(all_nmspec_typs):
        vals = {}
        ns = {}
        for lbl, _ in inputs:
            row = raw_by_label[lbl].get("per_typology_nm_specificity", {}).get(typ, {})
            vals[lbl] = row.get("specificity")
            ns[lbl]   = row.get("correct", 0) + row.get("wrong", 0)
        per_typology_nm.append({
            "typology": typ,
            "values": vals,
            "n_per_endpoint": ns,
            "winner_label": _argmax(vals, True),
        })

    # Narrative stats (n_non_empty / mean_chars / median_chars / min / max)
    narrative_stats = []
    NARR_FIELDS = ("n_non_empty", "mean_chars", "median_chars", "min_chars", "max_chars")
    for f in NARR_FIELDS:
        narrative_stats.append({
            "field": f,
            "values": {lbl: raw_by_label[lbl].get("narrative_stats", {}).get(f)
                       for lbl, _ in inputs},
        })

    # Wall-clock stats
    wc = []
    for f in ("mean", "median"):
        wc.append({
            "field": f,
            "values": {lbl: raw_by_label[lbl].get("wall_clock_ms", {}).get(f)
                       for lbl, _ in inputs},
        })

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "demo_size": demo_size,
        "demo_version": demo_version,
        "notes": notes,
        "endpoints": endpoints,
        "coverage": coverage_rows,
        "headline_metrics": headline_metrics,
        "confusion": confusion,
        "per_typology_recall": per_typology_recall,
        "per_typology_nm_specificity": per_typology_nm,
        "narrative_stats": narrative_stats,
        "wall_clock_ms": wc,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Build a multi-endpoint model comparison report.")
    p.add_argument("--eval", action="append", required=True, metavar="PATH:LABEL",
                    help="eval.json path + label, separated by ':'. May be given "
                          "multiple times; order is preserved in the output. "
                          "Convention: base → frontier → custom.")
    p.add_argument("--out", type=Path, default=None,
                    help="Output report path. Default: data/benchmarks/model_comparison_<ts>.json")
    p.add_argument("--demo-size", type=int, default=200)
    p.add_argument("--demo-version", default="v3.1 (prod-mimic v2)")
    p.add_argument("--notes", default="")
    p.add_argument("--update-latest", action="store_true",
                    help="Write data/benchmarks/latest.json pointing at the newly-built "
                          "report (consumed by the API endpoint).")
    args = p.parse_args(argv)

    inputs: list[tuple[str, Path]] = []
    for spec in args.eval:
        if ":" not in spec:
            print(f"bad --eval spec (need PATH:LABEL): {spec}", file=sys.stderr)
            return 2
        path_str, label = spec.split(":", 1)
        inputs.append((label, Path(path_str)))

    report = build_report(inputs, demo_size=args.demo_size,
                           demo_version=args.demo_version, notes=args.notes)

    # Resolve output path. Default is data/benchmarks/model_comparison_<ts>.json
    here = Path(__file__).resolve().parent
    backend_root = here.parent.parent
    if args.out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = backend_root / "data" / "benchmarks" / f"model_comparison_{ts}.json"
    else:
        out = args.out

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote report: {out}")

    if args.update_latest:
        # The pointer file stores (report_path, ran_at) — `eval_comparison.py`
        # follows this pointer to serve the active report.
        latest = out.parent / "latest.json"
        # The report_path is stored RELATIVE to the data dir so the
        # endpoint can resolve it independent of where the codebase lives.
        try:
            rel = out.relative_to(backend_root / "data")
            report_path_str = str(rel)
        except ValueError:
            report_path_str = str(out)
        latest.write_text(json.dumps({
            "report_path": report_path_str,
            "ran_at": report["ran_at"],
        }, indent=2))
        print(f"Updated pointer: {latest}")

    # Echo the headline section so a CLI run prints something useful
    print(json.dumps({
        "demo_size": report["demo_size"],
        "endpoints": [e["label"] for e in report["endpoints"]],
        "headline_metrics": report["headline_metrics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
