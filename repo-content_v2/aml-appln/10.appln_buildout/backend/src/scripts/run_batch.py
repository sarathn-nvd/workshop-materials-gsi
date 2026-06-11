"""Batch-run the deterministic investigate_case workflow over the demo
manifest. Persists one trace per case to ./data/traces/<case_id>.json and
prints progress + final eval summary."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


async def _run_one(workflow, case_id: str) -> dict:
    async with workflow.run({"case_id": case_id}) as runner:
        raw = await runner.result()
    return json.loads(raw) if isinstance(raw, str) else raw


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/workflow.yaml")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N cases (debug).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases whose trace already exists.")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    src_root = here.parent
    backend_root = src_root.parent
    os.chdir(src_root)
    os.environ.setdefault("NAT_AML_DATA_DIR", str(backend_root / "data"))

    import aml_app.register  # noqa: F401
    from nat.runtime.loader import (
        PluginTypes, discover_and_register_plugins, load_config,
    )
    from nat.builder.workflow_builder import WorkflowBuilder
    from aml_app.utils.data_loader import get_data_plane

    discover_and_register_plugins(PluginTypes.COMPONENT)
    cfg = load_config(args.config)
    dp = get_data_plane(os.environ["NAT_AML_DATA_DIR"])

    cases = dp.manifest()
    if args.limit:
        cases = cases[:args.limit]
    if args.skip_existing:
        done = {p.stem for p in dp.traces_dir.glob("*.json")}
        cases = [c for c in cases if c["case_id"] not in done]
    n_total = len(cases)
    print(f"\n========================================================")
    print(f"  AML investigation batch run")
    print(f"  Config           : {args.config}")
    print(f"  Cases to process : {n_total}")
    print(f"  Concurrency      : {args.concurrency}")
    print(f"========================================================\n",
          flush=True)
    if not cases:
        print("Nothing to do.")
        return 0

    async with WorkflowBuilder.from_config(cfg) as builder:
        workflow = await builder.build()
        sem = asyncio.Semaphore(args.concurrency)
        done_counter = {"n": 0, "pos": 0, "neg": 0, "err": 0, "t0": time.time()}

        async def _bounded(c: dict):
            async with sem:
                t = time.time()
                try:
                    trace = await _run_one(workflow, c["case_id"])
                except Exception as e:
                    trace = {"case_id": c["case_id"], "error": f"runner: {e}"}
                done_counter["n"] += 1
                if trace.get("error"):
                    done_counter["err"] += 1
                    verdict = "X"
                elif trace.get("sar_is_suspicious"):
                    done_counter["pos"] += 1
                    verdict = "+"
                else:
                    done_counter["neg"] += 1
                    verdict = "-"
                elapsed = time.time() - t
                cum = time.time() - done_counter["t0"]
                rate = done_counter["n"] / cum if cum else 0
                eta = (n_total - done_counter["n"]) / rate if rate else 0
                print(
                    f"  [{verdict}] {done_counter['n']:>3d}/{n_total}  "
                    f"{c['case_id']}  ({elapsed:>5.1f}s)  "
                    f"running +={done_counter['pos']} -={done_counter['neg']} "
                    f"err={done_counter['err']}  ETA {eta/60:.1f}min",
                    flush=True,
                )
                return trace

        await asyncio.gather(*[_bounded(c) for c in cases])

    total = time.time() - done_counter["t0"]
    print(f"\n========================================================")
    print(f"  BATCH COMPLETE: {n_total} cases in {total/60:.1f}min "
          f"(avg {total/n_total:.1f}s/case)")
    print(f"  Predictions: +={done_counter['pos']}  "
          f"-={done_counter['neg']}  err={done_counter['err']}")
    print(f"========================================================\n",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
