"""Per-source downloaders. Each module exposes a dict of task callables.

The top-level download.py composes these into a flat task registry that
groups tasks by phase (cpt_l1, cpt_l2, sft) and by group (huggingface,
kaggle, fincen, fatf, ofac, icij, ffiec, cfpb, amlgentex).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._common import StatsTracker

TaskFn = Callable[["StatsTracker"], None]
