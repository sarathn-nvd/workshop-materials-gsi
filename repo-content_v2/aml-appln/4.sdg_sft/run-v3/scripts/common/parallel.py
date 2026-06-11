"""Multiprocessing helpers — CPU parallelism for non-LLM stages.

The hardware (240 cores, 4 NUMA nodes, 2 TB RAM) lets us spin up large
process pools. Use NUMA-aware affinity hints when running both pipelines
in parallel to avoid cross-socket memory traffic.
"""
from __future__ import annotations

import logging
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


def safe_workers(requested: int, items: Optional[Sequence] = None) -> int:
    """Cap the worker count to a sane upper bound."""
    n = max(1, min(requested, cpu_count()))
    if items is not None:
        n = max(1, min(n, len(items)))
    return n


def parmap(
    fn: Callable[[T], U],
    items: Sequence[T],
    *,
    workers: int,
    chunksize: int = 1,
    desc: str = "",
) -> list[U]:
    """Parallel map with a process pool. Falls back to single-process for tiny inputs."""
    workers = safe_workers(workers, items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    if desc:
        logger.info("parmap(%s) workers=%d items=%d chunksize=%d",
                    desc, workers, len(items), chunksize)
    with Pool(processes=workers) as pool:
        return list(pool.imap(fn, items, chunksize=chunksize))


def parmap_iter(
    fn: Callable[[T], U],
    items: Iterable[T],
    *,
    workers: int,
    chunksize: int = 1,
) -> Iterable[U]:
    """Streaming variant — same as parmap but yields as workers complete."""
    workers = max(1, workers)
    if workers <= 1:
        for x in items:
            yield fn(x)
        return
    with Pool(processes=workers) as pool:
        yield from pool.imap_unordered(fn, items, chunksize=chunksize)


# ============================================================================
# NUMA-aware pinning (best effort; harmless on non-NUMA boxes)
# ============================================================================
NUMA_NODES: dict[int, list[int]] = {
    # Inferred from `lscpu` on the target box (240 cores, 2 sockets × 60 cores × 2 threads).
    # node0 = cores 0-29 + HT 120-149, etc.
    0: list(range(0, 30)) + list(range(120, 150)),
    1: list(range(30, 60)) + list(range(150, 180)),
    2: list(range(60, 90)) + list(range(180, 210)),
    3: list(range(90, 120)) + list(range(210, 240)),
}


def pin_to_numa_node(node: int) -> None:
    """Pin the calling process to a NUMA node (best-effort)."""
    cpus = NUMA_NODES.get(node, [])
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, cpus)
        logger.info("Pinned PID %d to NUMA node %d (cpus=%d)", os.getpid(), node, len(cpus))
    except Exception as exc:  # noqa: BLE001
        logger.debug("NUMA pin failed (non-fatal): %s", exc)


def numa_node_for_pipeline(pipeline: str) -> int:
    """Map pipeline name → NUMA node so non-aux + aux land on different sockets."""
    return {"nonaux": 0, "aux": 2}.get(pipeline, 0)
