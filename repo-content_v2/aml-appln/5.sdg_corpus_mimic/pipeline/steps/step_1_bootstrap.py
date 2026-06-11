"""Step 1 — Bootstrap.

Delegates to pipeline.bootstrap.run(). Provided so the orchestrator's
step-number routing is uniform across all 8 steps.
"""
from __future__ import annotations

import logging

from pipeline import bootstrap

logger = logging.getLogger("pipeline.steps.step_1_bootstrap")


def run(*, seed: int, force: bool = False) -> None:  # noqa: ARG001 — seed unused
    summary = bootstrap.run(force=force)
    logger.info(
        "bootstrap: %d total | %d created | %d overwritten | %d unchanged | %d diverged",
        summary["n_total"], summary["n_created"], summary["n_overwritten"],
        summary["n_unchanged"], summary["n_diverged_kept_local"],
    )
