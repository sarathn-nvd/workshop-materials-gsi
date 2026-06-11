"""Direct HuggingFace Hub downloads — bypasses ``datasets.load_dataset``.

Why this exists
---------------
``datasets`` 2.16+ has a known bug where ``dataset_module_factory`` reads
HTTP responses without honoring ``Content-Encoding: gzip``, raising
``UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b...`` for some
script-based or auto-converted datasets (FinQA, Caselaw, etc.). See
https://github.com/huggingface/datasets/issues/6851 and #6908.

Pinning ``datasets<3.0.0`` is required for our pile-of-law / edgar-corpus
streaming filters (the legacy ``trust_remote_code`` script loaders), so we
can't fix the bug by upgrading. Instead, for SFT datasets that are simple
file dumps on the Hub (CSV / JSON / JSONL / TSV / parquet), we go around
``datasets`` entirely and download the raw files via ``huggingface_hub``.

Two helpers are exposed:

* :func:`download_hf_files` — download every file matching a predicate from
  a dataset repo at a specific revision (``main`` by default; pass
  ``revision="refs/convert/parquet"`` to grab the auto-generated parquet
  view that the Hub maintains for any dataset).
* :func:`iter_parquet_records` — stream rows from local parquet files,
  optionally applying a ``predicate`` for caselaw-style filtering.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


def _hf_token() -> str | None:
    """Resolve the HF token from env. ``None`` if unset (public repos still work)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def list_repo_files(repo_id: str, *, revision: str = "main") -> list[str]:
    """List every file in an HF dataset repo at ``revision``."""
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token())
    return list(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))


def has_parquet_branch(repo_id: str) -> bool:
    """Return ``True`` if the Hub maintains a ``refs/convert/parquet`` branch.

    The Hub auto-converts most public datasets to parquet under this branch,
    even when the canonical ``main`` only carries a loader script.
    """
    try:
        files = list_repo_files(repo_id, revision="refs/convert/parquet")
        return any(f.endswith(".parquet") for f in files)
    except Exception as e:  # noqa: BLE001
        logger.debug("has_parquet_branch(%s): %s", repo_id, e)
        return False


def download_hf_files(
    repo_id: str,
    dest: Path,
    *,
    revision: str = "main",
    file_filter: Callable[[str], bool] | None = None,
    flatten: bool = False,
    max_workers: int = 4,
) -> tuple[list[Path], int]:
    """Download every matching file from a dataset repo to ``dest``.

    Parameters
    ----------
    repo_id:
        ``namespace/dataset`` slug.
    dest:
        Local directory. Created if missing.
    revision:
        Hub revision — ``main`` for canonical, ``refs/convert/parquet`` for
        the auto-converted parquet view.
    file_filter:
        Predicate that receives each remote path; only ``True`` paths are
        downloaded. Default keeps every file.
    flatten:
        If ``True``, drop subdirectories — useful for per-subtask repos
        where the directory layout doesn't matter downstream. Default
        preserves the source layout.
    max_workers:
        Parallel downloads.

    Returns
    -------
    (downloaded_paths, total_bytes)
        Paths actually written (skipped existing files are excluded), and
        the total bytes of every file in ``dest`` (so re-runs report the
        full corpus size, not just newly-downloaded bytes).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(repo_id, revision=revision)
    if file_filter is not None:
        files = [f for f in files if file_filter(f)]

    if not files:
        logger.warning("hf_download: no files matched in %s@%s", repo_id, revision)
        return [], 0

    token = _hf_token()
    written: list[Path] = []

    def _one(remote_path: str) -> Path | None:
        local_subpath = Path(remote_path).name if flatten else remote_path
        target = dest / local_subpath
        if target.exists() and target.stat().st_size > 0:
            logger.debug("hf_download skip existing: %s", target)
            return None
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
                local_dir=str(dest if not flatten else dest),
                token=token,
            )
            local_path = Path(local)
            if flatten and local_path.parent != dest:
                final = dest / local_path.name
                local_path.replace(final)
                local_path = final
            return local_path
        except Exception as e:  # noqa: BLE001
            logger.warning("hf_download failed %s/%s: %s", repo_id, remote_path, e)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, f) for f in files]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                written.append(res)

    total_bytes = 0
    for root, _dirs, fnames in os.walk(dest):
        for n in fnames:
            try:
                total_bytes += (Path(root) / n).stat().st_size
            except OSError:
                continue

    return written, total_bytes


def iter_parquet_records(
    paths: Iterable[Path],
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield records from a sequence of parquet files, optionally filtered."""
    import pyarrow.parquet as pq

    for p in paths:
        try:
            pf = pq.ParquetFile(p)
        except Exception as e:  # noqa: BLE001
            logger.warning("parquet read failed %s: %s", p, e)
            continue
        for batch in pf.iter_batches(batch_size=1024):
            for rec in batch.to_pylist():
                if predicate is None or predicate(rec):
                    yield rec
