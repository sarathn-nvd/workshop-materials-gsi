#!/usr/bin/env python3
"""Step 1 — Data Download.

Single entrypoint for acquiring all raw CPT / SFT / RL data sources
described in revised_strategy.md.

Usage:
    python download.py --list
    python download.py --tasks all
    python download.py --tasks cpt_l1,sft
    python download.py --tasks fincen,fatf --parallel-workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from sources import amlgentex, cfpb, courtlistener, fatf, ffiec, fincen, huggingface
from sources import icij, kaggle, ofac, uscode
from sources._common import (
    StatsTracker, setup_logging, write_report,
)

logger = logging.getLogger("download")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TaskFn = Callable[[StatsTracker], None]

# Individual tasks — every task is uniquely named and maps to a callable.
INDIVIDUAL_TASKS: dict[str, TaskFn] = {
    # CPT Layer 1 (HuggingFace)
    "pile_of_law": huggingface.download_pile_of_law,
    "edgar_corpus": huggingface.download_edgar_corpus,

    # CPT Layer 1 — USCode direct (Title 12, 18, 31; fills gaps in
    # pile-of-law's USCode packaging which is missing Title 12 entirely
    # and most of Title 18's main body)
    "uscode_house": uscode.download_uscode,

    # CPT Layer 2 — HuggingFace
    "caselaw_access_project": huggingface.download_caselaw_access,

    # CPT Layer 2 — CourtListener (alternative to gated Caselaw Access Project)
    "courtlistener": courtlistener.download_courtlistener,

    # CPT Layer 2 — FinCEN
    "fincen_advisories": fincen.download_fincen_advisories,
    "fincen_federal_register": fincen.download_fincen_federal_register,
    "fincen_sar_reviews": fincen.download_fincen_sar_reviews,
    "fincen_enforcement": fincen.download_fincen_enforcement,

    # CPT Layer 2 — FATF
    "fatf_publications": fatf.download_fatf_publications,

    # CPT Layer 2 — OFAC
    "ofac_enforcement": ofac.download_ofac_enforcement,
    "ofac_guidance": ofac.download_ofac_guidance,

    # CPT Layer 2 — ICIJ FinCEN Files
    "fincen_files": icij.download_fincen_files,

    # SFT — HuggingFace
    "enterprise_financial_crime": huggingface.download_enterprise_financial_crime,
    "finance_instruct_500k": huggingface.download_finance_instruct_500k,
    "finqa": huggingface.download_finqa,
    "tat_qa": huggingface.download_tat_qa,
    "financebench": huggingface.download_financebench,
    "legalbench": huggingface.download_legalbench,

    # SFT — Kaggle
    "sarsum": kaggle.download_sarsum,
    "ibm_aml_transactions": kaggle.download_ibm_aml_transactions,

    # SFT — scraped / bulk
    "ffiec_manual": ffiec.download_ffiec_manual,
    "cfpb_complaints": cfpb.download_cfpb_complaints,
    "amlgentex": amlgentex.download_amlgentex,
}

# Group aliases
GROUPS: dict[str, list[str]] = {
    "cpt_l1": [
        "pile_of_law", "edgar_corpus", "uscode_house",
    ],
    "cpt_l2": [
        "caselaw_access_project", "courtlistener",
        "fincen_advisories", "fincen_federal_register",
        "fincen_sar_reviews", "fincen_enforcement", "fincen_files",
        "fatf_publications",
        "ofac_enforcement", "ofac_guidance",
    ],
    "sft": [
        "enterprise_financial_crime", "finance_instruct_500k",
        "finqa", "tat_qa", "financebench", "legalbench",
        "sarsum", "ibm_aml_transactions",
        "ffiec_manual", "cfpb_complaints", "amlgentex",
    ],
    # Source-family groupings, useful for partial reruns
    "huggingface": [
        "pile_of_law", "edgar_corpus", "caselaw_access_project",
        "enterprise_financial_crime", "finance_instruct_500k",
        "finqa", "tat_qa", "financebench", "legalbench",
    ],
    "kaggle": ["sarsum", "ibm_aml_transactions"],
    "fincen": [
        "fincen_advisories", "fincen_federal_register",
        "fincen_sar_reviews", "fincen_enforcement", "fincen_files",
    ],
    "fatf": ["fatf_publications"],
    "ofac": ["ofac_enforcement", "ofac_guidance"],
    "ffiec": ["ffiec_manual"],
    "cfpb": ["cfpb_complaints"],
    "amlgentex": ["amlgentex"],
    "icij": ["fincen_files"],
}

GROUPS["cpt"] = GROUPS["cpt_l1"] + GROUPS["cpt_l2"]
GROUPS["all"] = sorted(INDIVIDUAL_TASKS.keys())


# ---------------------------------------------------------------------------
# Task resolution + dispatch
# ---------------------------------------------------------------------------

def resolve_tasks(requested: list[str]) -> list[str]:
    """Expand group aliases and validate task names. Returns ordered, deduped list."""
    resolved: list[str] = []
    seen: set[str] = set()
    for name in requested:
        name = name.strip()
        if not name:
            continue
        if name in GROUPS:
            for t in GROUPS[name]:
                if t not in seen:
                    resolved.append(t)
                    seen.add(t)
        elif name in INDIVIDUAL_TASKS:
            if name not in seen:
                resolved.append(name)
                seen.add(name)
        else:
            raise SystemExit(
                f"Unknown task or group: {name!r}. Run `python download.py --list`."
            )
    return resolved


def _run_one(task_name: str, tracker: StatsTracker) -> tuple[str, bool, str | None]:
    try:
        INDIVIDUAL_TASKS[task_name](tracker)
        return task_name, True, None
    except Exception as e:  # noqa: BLE001
        logger.exception("Task %s failed", task_name)
        return task_name, False, f"{type(e).__name__}: {e}"


def run_tasks(task_names: list[str], *, workers: int) -> int:
    """Dispatch tasks, optionally in parallel. Returns exit code (0/1)."""
    tracker = StatsTracker()
    results: list[tuple[str, bool, str | None]] = []

    if workers <= 1 or len(task_names) == 1:
        for name in task_names:
            logger.info("=== running task: %s ===", name)
            results.append(_run_one(name, tracker))
    else:
        logger.info("running %d tasks with %d workers", len(task_names), workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, n, tracker): n for n in task_names}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

    # Persist stats + render report
    tracker.save()
    write_report(tracker)

    ok = sum(1 for _, success, _ in results if success)
    failed = [(n, err) for n, success, err in results if not success]
    logger.info("=== run summary: %d succeeded, %d failed ===", ok, len(failed))
    for name, err in failed:
        logger.error("  FAILED  %s: %s", name, err)

    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="download.py",
        description="Step 1 — download raw data for AML agent training.",
    )
    p.add_argument(
        "--tasks", default="all",
        help=("Comma-separated list of tasks or group names. "
              "Groups: all, cpt, cpt_l1, cpt_l2, sft, huggingface, kaggle, "
              "fincen, fatf, ofac, icij, ffiec, cfpb, amlgentex"),
    )
    p.add_argument(
        "--parallel-workers", "-p", type=int, default=1,
        help="Number of tasks to run in parallel (default 1).",
    )
    p.add_argument("--list", action="store_true", help="List all tasks and groups and exit.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p.parse_args(argv)


def _print_registry() -> None:
    print("Individual tasks:")
    for name in sorted(INDIVIDUAL_TASKS):
        print(f"  - {name}")
    print()
    print("Group aliases:")
    for name in sorted(GROUPS):
        members = ", ".join(GROUPS[name])
        print(f"  - {name}: {members}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    setup_logging(verbose=args.verbose)

    if args.list:
        _print_registry()
        return 0

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = resolve_tasks(requested)
    if not tasks:
        logger.error("no tasks resolved from --tasks %r", args.tasks)
        return 2

    logger.info("resolved tasks (%d): %s", len(tasks), ", ".join(tasks))
    return run_tasks(tasks, workers=args.parallel_workers)


if __name__ == "__main__":
    raise SystemExit(main())
