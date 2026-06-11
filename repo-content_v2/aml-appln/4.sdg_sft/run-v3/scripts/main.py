"""Top-level orchestrator.

Runs the non-auxiliary and auxiliary pipelines in parallel (two
multiprocessing.Process children, one per pipeline). Each pipeline runs its
stages in strict sequence. Both produce one final JSONL each. The combine
step is out of scope.

Usage:
    python -m scripts.main --total-records 75000
    python -m scripts.main --total-records 200            # smoke test
    python -m scripts.main --pipelines nonaux             # only non-aux
    python -m scripts.main --resume-from stage_6 --pipelines nonaux
    python -m scripts.main --dry-run                      # plan-only
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.common.io import read_manifest, write_json
from scripts.common.progress import configure_logging
from scripts.config import (
    CONCURRENCY, DEFAULT_SEED, DEFAULT_TOTAL_RECORDS,
    FINAL_AUX, FINAL_NONAUX, MANIFESTS_AUX, MANIFESTS_DIR, MANIFESTS_NONAUX,
    RUN_MANIFEST,
)

logger = configure_logging("main")


# ============================================================================
# Pipeline runners (each is the target of one Process)
# ============================================================================
def _run_nonaux(total_records: int, resume_from: str, skip_stages: list[str], dry_run: bool) -> None:
    from scripts.non_auxiliary.pipeline import run as run_nonaux
    run_nonaux(total_records=total_records, resume_from=resume_from,
               skip_stages=skip_stages, dry_run=dry_run)


def _run_aux(total_records: int, resume_from: str, skip_stages: list[str], dry_run: bool) -> None:
    from scripts.auxiliary.pipeline import run as run_aux
    run_aux(total_records=total_records, resume_from=resume_from,
            skip_stages=skip_stages, dry_run=dry_run)


# ============================================================================
# Argument parsing
# ============================================================================
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="scripts.main",
        description="SFT synthetic-data generation pipeline orchestrator.",
    )
    p.add_argument("--total-records", "-n", type=int, default=DEFAULT_TOTAL_RECORDS,
                   help="The pluggable N (default: 75000).")
    p.add_argument("--pipelines", default="nonaux,aux",
                   help="Comma-separated subset of {nonaux,aux} (default: both).")
    p.add_argument("--resume-from", default="all",
                   help="Stage id to start at (default: all).")
    p.add_argument("--skip-stages", default="",
                   help="Comma-separated list of stage ids to skip.")
    p.add_argument("--max-parallel-llm", type=int, default=CONCURRENCY.max_parallel_llm,
                   help="Total DataDesigner LLM concurrency budget.")
    p.add_argument("--max-cpu-workers", type=int, default=CONCURRENCY.max_cpu_workers,
                   help="Per-pipeline mp.Pool size.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global RNG seed.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan + verify configs; no LLM calls.")
    return p.parse_args(argv)


# ============================================================================
# Entry
# ============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Apply concurrency
    CONCURRENCY.max_parallel_llm = args.max_parallel_llm
    CONCURRENCY.max_cpu_workers = args.max_cpu_workers

    requested = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    valid = {"nonaux", "aux"}
    unknown = [p for p in requested if p not in valid]
    if unknown:
        logger.error("Unknown pipelines: %s. Valid: %s", unknown, sorted(valid))
        return 2

    skip_stages = [s for s in args.skip_stages.split(",") if s]

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    logger.info("Pipeline run start - N=%d pipelines=%s seed=%d resume_from=%s dry_run=%s",
                args.total_records, requested, args.seed, args.resume_from, args.dry_run)

    # Spawn parallel processes
    procs: dict[str, mp.Process] = {}
    if "nonaux" in requested:
        procs["nonaux"] = mp.Process(
            target=_run_nonaux, name="nonaux",
            kwargs={"total_records": args.total_records, "resume_from": args.resume_from,
                    "skip_stages": skip_stages, "dry_run": args.dry_run},
        )
    if "aux" in requested:
        procs["aux"] = mp.Process(
            target=_run_aux, name="aux",
            kwargs={"total_records": args.total_records, "resume_from": args.resume_from,
                    "skip_stages": skip_stages, "dry_run": args.dry_run},
        )

    if not procs:
        logger.error("No pipelines requested.")
        return 2

    for proc in procs.values():
        proc.start()

    exit_codes: dict[str, int] = {}
    for name, proc in procs.items():
        proc.join()
        exit_codes[name] = proc.exitcode if proc.exitcode is not None else -1
        logger.info("Pipeline %s exited with code %d", name, exit_codes[name])

    elapsed = time.time() - t0
    ended_at = datetime.now(timezone.utc).isoformat()

    # Build run summary
    run_summary = {
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": round(elapsed, 1),
        "args": vars(args),
        "exit_codes": exit_codes,
        "outputs": {
            "non_auxiliary": str(FINAL_NONAUX) if FINAL_NONAUX.exists() else None,
            "auxiliary":     str(FINAL_AUX) if FINAL_AUX.exists() else None,
        },
        "manifests": {
            "nonaux_dir": str(MANIFESTS_NONAUX),
            "aux_dir": str(MANIFESTS_AUX),
        },
    }
    write_json(run_summary, RUN_MANIFEST)
    logger.info("Run summary: %s", RUN_MANIFEST)

    return 0 if all(c == 0 for c in exit_codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
