"""Side-by-side end-to-end comparison of two LLM endpoints over the
prod-mimic demo eval set.

Runs the deterministic investigate_case workflow once per endpoint,
each writing to its own isolated trace dir. After both runs complete,
scores each trace dir against demo/eval_keys.jsonl and prints a
side-by-side metrics table.

Each per-case investigation makes exactly 4 LLM calls against the
configured NIM (3 aux skills + 1 SAR; behavioral is Python-deterministic
under Path A). The aux gate's LLM-as-Judge reviewer is bypassed by
default (NAT_AML_ENABLE_JUDGE=false) — that saves more calls per case
and still respects the v3.1 contract.

Usage:
    python -m scripts.compare_endpoints \\
        --limit 20 --concurrency 8

The two endpoints default to the workshop deployment (8089 = base
Nemotron, 8090 = Gemma teacher). Override with --endpoint-a-url etc.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Endpoint:
    label: str
    base_url: str
    model: str
    trace_dir: str          # subdir name under data/
    enable_judge: str = "false"


def _bool(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y")


# ---------------------------------------------------------------------------
# Single-endpoint run
# ---------------------------------------------------------------------------
async def run_one_endpoint(
    ep: Endpoint, *, limit: int | None, concurrency: int,
    backend_root: Path, skip_existing: bool = False,
) -> int:
    """Run run_batch.py against a single endpoint and rotate the traces
    directory afterwards into the endpoint's isolated dir."""
    data_dir = backend_root / "data"
    traces_default = data_dir / "traces"
    traces_target  = data_dir / ep.trace_dir

    traces_target.mkdir(parents=True, exist_ok=True)
    if traces_default.exists():
        if skip_existing and any(traces_target.glob("DEMO_*.json")):
            pass
        else:
            for f in traces_default.glob("DEMO_*.json"):
                try: f.unlink()
                except Exception: pass

    print(f"\n[{ep.label}] Launching run against {ep.base_url} (model={ep.model})", flush=True)
    print(f"[{ep.label}] Traces will land in: {traces_target}", flush=True)

    env = os.environ.copy()
    env["NAT_AML_NIM_BASE_URL"]     = ep.base_url
    env["NAT_AML_NIM_MODEL"]        = ep.model
    env["NAT_AML_REASONING_BASE_URL"]    = ep.base_url
    env["NAT_AML_REASONING_MODEL"]       = ep.model
    env["NAT_AML_ORCHESTRATOR_BASE_URL"] = ep.base_url
    env["NAT_AML_ORCHESTRATOR_MODEL"]    = ep.model
    env["NAT_AML_JUDGE_BASE_URL"]   = ep.base_url
    env["NAT_AML_JUDGE_MODEL"]      = ep.model
    env["NAT_AML_ENABLE_JUDGE"]     = ep.enable_judge
    env["NAT_AML_DATA_DIR"]         = str(data_dir)
    env["OMIT_DECISION_TARGET"]     = "1"
    # Disable Nemotron <think> blocks unless caller explicitly enabled them.
    env.setdefault("NAT_AML_NO_THINK", "1")
    env.setdefault("NAT_AML_THINKING_BUDGET", "0")
    env.setdefault("NAT_AML_SAR_MAX_TOKENS", "6000")
    env.setdefault("NAT_AML_AUX_MAX_TOKENS", "3000")
    env.setdefault("NAT_AML_BEHAVIORAL_MODE", "python_only")

    src_dir = backend_root / "src"
    log_path = data_dir / f"compare_run_{ep.trace_dir}.log"
    cmd = [
        sys.executable, "-m", "scripts.run_batch",
        "--concurrency", str(concurrency),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if skip_existing:
        cmd += ["--skip-existing"]

    with log_path.open("w") as logf:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(src_dir), env=env,
            stdout=logf, stderr=asyncio.subprocess.STDOUT,
        )
        rc = await proc.wait()

    # Move newly written traces from data/traces/ into the target dir.
    moved = 0
    for src in traces_default.glob("DEMO_*.json"):
        dst = traces_target / src.name
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            print(f"[{ep.label}] move {src.name} failed: {e}", file=sys.stderr)
    print(f"[{ep.label}] exit_code={rc}  moved={moved}  log={log_path}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# In-process scoring
# ---------------------------------------------------------------------------
def _score(label: str, traces_dir: Path, eval_keys: Path, backend_root: Path) -> dict:
    """Invoke scripts.score_traces in-process for crisp imports."""
    src = str(backend_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from scripts.score_traces import score
    return score(traces_dir, eval_keys, label)


# ---------------------------------------------------------------------------
# Side-by-side printer
# ---------------------------------------------------------------------------
def _print_side_by_side(a: dict, b: dict) -> None:
    """ASCII table comparing two scored results."""
    line = "─" * 92
    print(f"\n{line}")
    print(f"  SIDE-BY-SIDE — {a['label']}   vs   {b['label']}")
    print(line)
    print(f"  {'Metric':<38}  {a['label'][:24]:>24}   {b['label'][:24]:>24}")
    print(line)

    def _row(label, va, vb, fmt="{}"):
        sa = fmt.format(va) if va is not None else "—"
        sb = fmt.format(vb) if vb is not None else "—"
        print(f"  {label:<38}  {sa:>24}   {sb:>24}")

    _row("n_scored (cases joined)", a["n_scored"], b["n_scored"])
    _row("n_parse_failures", a["n_parse_failures"], b["n_parse_failures"])
    _row("n_errors", a["n_errors"], b["n_errors"])
    _row("wall_clock_ms (median per case)",
         a["wall_clock_ms"]["median"], b["wall_clock_ms"]["median"], "{:>14.0f}")
    print(line)
    print(f"  {'Confusion (positive class = SAR)':<38}")
    for k in ("tp", "fp", "tn", "fn"):
        _row(f"  {k.upper()}", a["confusion"][k], b["confusion"][k])
    print(line)
    print(f"  {'Headline metrics':<38}")
    for k in ("f1", "precision", "recall", "near_miss_specificity", "clean_fpr"):
        _row(f"  {k}", a["metrics"][k], b["metrics"][k], "{:>14.3f}")
    print(line)
    print(f"  {'Per-typology recall (POS class)':<38}")
    all_typs = sorted(set(a["per_typology_recall"].keys()) | set(b["per_typology_recall"].keys()))
    for t in all_typs:
        ra = a["per_typology_recall"].get(t); rb = b["per_typology_recall"].get(t)
        sa = (f"{ra['recall']:.2f}  ({ra['tp']}/{ra['tp']+ra['fn']})" if ra else "—")
        sb = (f"{rb['recall']:.2f}  ({rb['tp']}/{rb['tp']+rb['fn']})" if rb else "—")
        print(f"    {t:<36}  {sa:>24}   {sb:>24}")
    print(line)
    print(f"  {'Per-typology NM-NEG specificity':<38}")
    all_typs2 = sorted(set(a["per_typology_nm_specificity"].keys())
                       | set(b["per_typology_nm_specificity"].keys()))
    for t in all_typs2:
        ra = a["per_typology_nm_specificity"].get(t)
        rb = b["per_typology_nm_specificity"].get(t)
        sa = (f"{ra['specificity']:.2f}  ({ra['correct']}/{ra['correct']+ra['wrong']})" if ra else "—")
        sb = (f"{rb['specificity']:.2f}  ({rb['correct']}/{rb['correct']+rb['wrong']})" if rb else "—")
        print(f"    {t:<36}  {sa:>24}   {sb:>24}")
    print(line)
    print(f"  {'Narrative length (chars, non-empty)':<38}")
    for k in ("n_non_empty", "mean_chars", "median_chars", "min_chars", "max_chars"):
        _row(f"  {k}", a["narrative_stats"][k], b["narrative_stats"][k])
    print(line)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def main_async(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-a-url",   default="http://localhost:8089/v1")
    p.add_argument("--endpoint-a-model", default="nvidia/nemotron-3-nano")
    p.add_argument("--endpoint-a-label", default="nemotron-3-nano (base)")
    p.add_argument("--endpoint-a-dir",   default="traces_base_nemotron")
    p.add_argument("--endpoint-b-url",   default="http://localhost:8090/v1")
    p.add_argument("--endpoint-b-model", default="google/gemma-4-31b-it")
    p.add_argument("--endpoint-b-label", default="gemma-4-31b-it (frontier)")
    p.add_argument("--endpoint-b-dir",   default="traces_gemma")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--sequential", action="store_true",
                    help="Run endpoint-A then endpoint-B sequentially "
                          "(safer; default is to run them in parallel).")
    p.add_argument("--score-only", action="store_true",
                    help="Skip the runs, just score whatever's already in the trace dirs.")
    args = p.parse_args(argv)

    here = Path(__file__).resolve().parent
    backend_root = here.parent.parent
    eval_keys = backend_root / "data" / "demo" / "eval_keys.jsonl"

    ep_a = Endpoint(label=args.endpoint_a_label, base_url=args.endpoint_a_url,
                     model=args.endpoint_a_model, trace_dir=args.endpoint_a_dir)
    ep_b = Endpoint(label=args.endpoint_b_label, base_url=args.endpoint_b_url,
                     model=args.endpoint_b_model, trace_dir=args.endpoint_b_dir)

    if not args.score_only:
        t0 = time.time()
        if args.sequential:
            await run_one_endpoint(ep_a, limit=args.limit, concurrency=args.concurrency,
                                    backend_root=backend_root, skip_existing=args.skip_existing)
            await run_one_endpoint(ep_b, limit=args.limit, concurrency=args.concurrency,
                                    backend_root=backend_root, skip_existing=args.skip_existing)
        else:
            await asyncio.gather(
                run_one_endpoint(ep_a, limit=args.limit, concurrency=args.concurrency,
                                  backend_root=backend_root, skip_existing=args.skip_existing),
                run_one_endpoint(ep_b, limit=args.limit, concurrency=args.concurrency,
                                  backend_root=backend_root, skip_existing=args.skip_existing),
            )
        print(f"\nBoth runs complete in {(time.time()-t0)/60:.1f} min.")

    # Score both
    a_result = _score(ep_a.label, backend_root / "data" / ep_a.trace_dir,
                       eval_keys, backend_root)
    b_result = _score(ep_b.label, backend_root / "data" / ep_b.trace_dir,
                       eval_keys, backend_root)

    _print_side_by_side(a_result, b_result)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv or sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
