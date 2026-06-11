"""Kaggle-hosted dataset downloads.

Sources:
  SFT: leonardovalves/sarsum
  SFT: ealtman2019/ibm-transactions-for-anti-money-laundering-aml (HI-Small only)

Requires ~/.kaggle/kaggle.json with 600 permissions or KAGGLE_USERNAME /
KAGGLE_KEY environment variables.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from ._common import StatsTracker, dir_size, source_dir

logger = logging.getLogger(__name__)


def _kaggle_api():
    """Return an authenticated Kaggle API client."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        raise RuntimeError(
            "Kaggle auth failed. Place kaggle.json at ~/.kaggle/kaggle.json "
            "(chmod 600) or set KAGGLE_USERNAME + KAGGLE_KEY."
        ) from e

    api = KaggleApi()
    api.authenticate()
    return api


def _download_kaggle_dataset(
    tracker: StatsTracker,
    *,
    source_name: str,
    slug: str,
    phase: str = "sft",
    file_filter: callable | None = None,
    unzip: bool = True,
) -> None:
    """Download a Kaggle dataset to the canonical SFT/<source>/ directory."""
    with tracker.track(source=source_name, phase=phase) as stats:
        out = source_dir(phase, source_name)
        api = _kaggle_api()

        if file_filter is None:
            logger.info("[%s] downloading all files from %s", source_name, slug)
            api.dataset_download_files(slug, path=str(out), unzip=unzip, quiet=False)
        else:
            files = api.dataset_list_files(slug).files
            targets = [f.name for f in files if file_filter(f.name)]
            if not targets:
                raise RuntimeError(
                    f"No files matched file_filter for kaggle dataset {slug!r}"
                )
            for name in targets:
                logger.info("[%s] downloading file %s", source_name, name)
                api.dataset_download_file(slug, name, path=str(out))
                # The Kaggle CLI writes <file>.zip unless the file is already an
                # uncompressed CSV / small payload; handle both cases.
                zipped = out / f"{name}.zip"
                if unzip and zipped.exists():
                    shutil.unpack_archive(zipped, out)
                    zipped.unlink()

        stats.files_written = sum(1 for _ in out.rglob("*") if _.is_file())
        stats.bytes_written = dir_size(out)


def download_sarsum(tracker: StatsTracker) -> None:
    _download_kaggle_dataset(
        tracker,
        source_name="sarsum",
        slug="leonardovalves/sarsum",
    )


def download_ibm_aml_transactions(tracker: StatsTracker) -> None:
    """Download only the HI-Small variant (per workshop scope)."""
    def _only_hi_small(name: str) -> bool:
        n = name.lower()
        return "hi-small" in n or "hi_small" in n

    _download_kaggle_dataset(
        tracker,
        source_name="ibm_aml_transactions",
        slug="ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        file_filter=_only_hi_small,
    )


TASKS_SFT = {
    "sarsum": download_sarsum,
    "ibm_aml_transactions": download_ibm_aml_transactions,
}

ALL_TASKS = dict(TASKS_SFT)
