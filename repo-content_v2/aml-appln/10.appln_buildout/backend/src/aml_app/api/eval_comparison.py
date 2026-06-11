"""Pre-compiled side-by-side benchmark report API.

Serves the N-way comparison report produced offline by
`scripts/build_model_comparison_report.py`. The benchmark JSON files
live under `data/benchmarks/` and are regenerated whenever
`compare_endpoints.py` finishes a new sweep.

The endpoint is read-only — it does not invoke any model. It exists so
the workshop UI can render a side-by-side scorecard of:

  * `aml-custom-task-nim` — our SFT-trained Nemotron-3-Nano (3.2 B active)
  * `nemotron-3-nano (base)` — the same base checkpoint, untuned
  * `gemma-4-31b-it (frontier)` — a frontier teacher model
  * `gpt-5.2 (frontier)` — a frontier general-purpose model

against the 200-case prod-mimic demo eval set on the headline metrics
(F1, precision, recall, near-miss specificity, clean-cohort FPR) plus
per-typology breakdowns, confusion matrices, narrative statistics, and
median wall-clock latency.

Route: POST /api/demo/eval/model_comparison
Body:  {"report": "<filename or 'latest'>"}  — optional; defaults to latest.
"""
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


# ---------------------------------------------------------------------------
# I/O shapes
# ---------------------------------------------------------------------------
class ModelComparisonInput(BaseModel):
    """Selector for which pre-compiled benchmark report to serve.

    `report` may be:
      * the literal string ``"latest"`` (default) — serves whatever
        ``data/benchmarks/latest.json`` currently points to.
      * a bare filename like ``"four_way_pathA_20260528T054416Z.json"`` —
        served directly from ``data/benchmarks/``.
      * a full path beginning with ``/`` — served as-is (must still live
        under ``data/benchmarks/`` for safety).

    A bearer token gate via ``NAT_AML_EVAL_TOKEN`` mirrors the other
    ``/api/demo/eval/*`` routes.
    """

    report: str = Field(default="latest")
    token: Optional[str] = None


class ModelComparisonConfig(FunctionBaseConfig, name="demo_eval_model_comparison_report"):
    """No tunable knobs — just exposes the pre-compiled JSON files
    under ``data/benchmarks/`` via NAT's HTTP front-end.

    Note: the type-name carries the ``_report`` suffix because an older
    ``demo_eval_model_comparison`` stub still occupies that name in the
    registry (pending source-recovery cleanup). Route binding is by
    function_name in ``workflow.yaml``, so the public URL stays
    ``/api/demo/eval/model_comparison`` regardless.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _benchmarks_dir() -> Path:
    """Resolve the benchmarks directory from ``NAT_AML_DATA_DIR``.

    Falls back to ``./data/benchmarks`` (the repo-relative default) so
    the endpoint also works when invoked from a dev shell rooted at
    ``src/``.
    """
    data_dir = os.environ.get("NAT_AML_DATA_DIR", "./data")
    return Path(data_dir).resolve() / "benchmarks"


def _resolve_report_path(report: str, root: Path) -> Path:
    """Map the ``report`` selector to a file path under ``root``.

    ``latest.json`` is a tiny pointer file produced by
    ``build_model_comparison_report.py --update-latest``. It carries
    ``{report_path, ran_at}`` and points at the most recent full report.
    We follow the pointer here so the caller always gets the real
    benchmark document.
    """
    if report == "latest":
        pointer = root / "latest.json"
        if not pointer.exists():
            # Fall back to the lexicographically newest *.json under root
            candidates = sorted(
                p for p in root.glob("*.json") if p.name != "latest.json"
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No benchmark reports found under {root!s} "
                    "and no latest.json pointer exists."
                )
            return candidates[-1]
        with pointer.open() as f:
            ptr_doc = json.load(f)
        target = ptr_doc.get("report_path", "")
        if not target:
            raise ValueError(
                f"latest.json at {pointer!s} has no `report_path` field."
            )
        # Resolve relative paths against the data dir (one level up
        # from `benchmarks/`) so `"benchmarks/foo.json"` works.
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = root.parent / target
        return target_path.resolve()

    if report.startswith("/"):
        p = Path(report).resolve()
        if not str(p).startswith(str(root)):
            raise ValueError(
                f"Report path {p!s} is outside the benchmarks root {root!s}"
            )
        return p
    return root / report


def _check_token(supplied: Optional[str]) -> None:
    """Mirror the bearer-token gate used by other /api/demo/eval/* routes."""
    expected = os.environ.get("NAT_AML_EVAL_TOKEN", "").strip()
    if not expected:
        return
    if (supplied or "").strip() != expected:
        raise PermissionError(
            "Missing or invalid demo eval token. Set NAT_AML_EVAL_TOKEN "
            "or pass `token` in the request body."
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@register_function(config_type=ModelComparisonConfig)
async def demo_eval_model_comparison_report(config: ModelComparisonConfig, builder: Builder):
    """Return a pre-compiled N-way model-comparison report.

    The report is a JSON document with the following top-level fields:

        ran_at, demo_size, demo_version, notes
        endpoints[]            ← list of model labels + eval_path pointers
        coverage[]             ← n_scored, n_parse_failures, n_errors per endpoint
        headline_metrics[]     ← f1, precision, recall, near_miss_specificity,
                                 clean_fpr (with `winner_label` per metric)
        confusion[]            ← per-endpoint TP / FP / TN / FN
        per_typology_recall[]  ← per-typology recall across all endpoints
        per_typology_nm_specificity[]
                               ← per-typology near-miss specificity (precision battleground)
        narrative_stats[]      ← n_non_empty, mean_chars, median_chars, …
        wall_clock_ms[]        ← per-endpoint mean / median latency

    The shape is identical to what `scripts/build_model_comparison_report.py`
    writes — there's no transformation here, so the UI can rely on a stable
    schema.
    """

    async def _run(args: ModelComparisonInput) -> dict:
        _check_token(args.token)

        root = _benchmarks_dir()
        if not root.exists():
            raise FileNotFoundError(
                f"Benchmarks directory not found at {root!s}. "
                "Run scripts/compare_endpoints.py first to populate it."
            )

        path = _resolve_report_path(args.report, root)
        if not path.exists():
            available = sorted(p.name for p in root.glob("*.json"))
            raise FileNotFoundError(
                f"Report not found: {path.name}. "
                f"Available reports in {root!s}: {available}"
            )

        with path.open() as f:
            doc = json.load(f)

        # Add a tiny envelope so the UI can show provenance + freshness
        # without re-fetching the directory listing.
        return {
            "report_file": path.name,
            "report_path": str(path),
            "served_from": str(root),
            **doc,
        }

    yield FunctionInfo.from_fn(
        _run,
        description=(
            "Return a pre-compiled side-by-side comparison report of "
            "multiple LLM endpoints (custom-task NIM vs base Nemotron vs "
            "frontier models) on the 200-case prod-mimic demo eval set. "
            "Read-only — no LLM is invoked."
        ),
        input_schema=ModelComparisonInput,
    )
