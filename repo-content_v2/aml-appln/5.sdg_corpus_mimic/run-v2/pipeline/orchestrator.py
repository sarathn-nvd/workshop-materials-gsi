"""Top-level orchestrator for the production-mimic data pipeline.

Runs the 8 generation steps in order (Section 1.4 of the strategy doc):

    1  bootstrap                  one-time copy of SFT scripts into pipeline/
    2  inventory                  SFT entity inventory + clean-pool inventory
    3  build_tool_2_kyc           Tool 2 KYC store (Postgres-ready Parquet)
    4  build_tool_1_transactions  Tool 1 transactions store (Postgres-ready)
    5  wire_services              Tool 3/4/5 FastAPI service stubs + configs
    6  seed_subpopulations        ~50 suspicious + ~25 near-miss + counterparty injection
    7  sample_manifest            ~200-case demo manifest + hidden eval keys
    8  validate                   schema parity, drift, smoke test, classifier coverage

Each step is its own module under pipeline.steps.step_<n>_<name>; this
orchestrator dispatches by name.

Usage:
    python -m pipeline.orchestrator --step all --seed 42
    python -m pipeline.orchestrator --step 4
    python -m pipeline.orchestrator --step 4,5,6
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

from pipeline.config import DEFAULT_SEED

logger = logging.getLogger("pipeline.orchestrator")


# ============================================================================
# Step registry — each step's run() is imported lazily so that broken
# DataDesigner imports in unrelated steps don't block this orchestrator.
# ============================================================================
@dataclass(frozen=True)
class Step:
    n: int
    name: str
    description: str

    def run(self, *, seed: int) -> None:
        # Lazy import so transitive failures stay scoped.
        module_name = f"pipeline.steps.step_{self.n}_{self.name}"
        import importlib
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"step {self.n} ({self.name}) is not implemented yet — "
                f"could not import {module_name}: {e}"
            ) from e
        if not hasattr(mod, "run"):
            raise RuntimeError(
                f"step {self.n} ({self.name}) module {module_name} has no run() function"
            )
        run_fn: Callable[..., None] = mod.run
        run_fn(seed=seed)


STEPS: list[Step] = [
    Step(1, "bootstrap", "Copy SFT scripts into pipeline/ (one-time)"),
    Step(2, "inventory", "SFT entity inventory + clean-pool inventory"),
    Step(3, "build_tool_2_kyc", "Build Tool 2 KYC store (~2K entities)"),
    Step(4, "build_tool_1_transactions", "Build Tool 1 transactions store (~100K rows)"),
    Step(5, "wire_services", "Wire Tool 3/4/5 FastAPI services"),
    Step(6, "seed_subpopulations", "Seed suspicious + near-miss + counterparty injection"),
    Step(7, "sample_manifest", "Sample demo manifest + write hidden eval keys"),
    Step(8, "validate", "Schema parity, drift, smoke test, classifier coverage"),
]


# ============================================================================
# Step parsing
# ============================================================================
def _parse_step_spec(spec: str) -> list[Step]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(STEPS)
    out: list[Step] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_n, hi_n = int(lo), int(hi)
        else:
            lo_n = hi_n = int(part)
        for n in range(lo_n, hi_n + 1):
            match = next((s for s in STEPS if s.n == n), None)
            if match is None:
                raise ValueError(f"no such step: {n}")
            out.append(match)
    return out


# ============================================================================
# CLI
# ============================================================================
def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline.orchestrator",
        description="Run the production-mimic data pipeline.",
    )
    parser.add_argument(
        "--step", default="all",
        help="Step number(s) or 'all'. Examples: '4', '4,5,6', '2-7', 'all'.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    steps = _parse_step_spec(args.step)
    logger.info("Running %d step(s): %s", len(steps),
                ", ".join(f"{s.n}.{s.name}" for s in steps))

    for step in steps:
        t0 = time.time()
        logger.info("══ Step %d: %s — %s", step.n, step.name, step.description)
        try:
            step.run(seed=args.seed)
        except Exception:
            logger.exception("Step %d (%s) FAILED", step.n, step.name)
            return 1
        elapsed = time.time() - t0
        logger.info("Step %d done in %.1fs", step.n, elapsed)

    logger.info("All requested steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
