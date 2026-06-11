"""OFAC sources — CPT Layer 2.

  1. Enforcement Actions — OpenSanctions bulk download (CSV + JSON)
  2. Guidance & Frameworks — PDFs scraped from treasury.gov file-finder
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ._common import StatsTracker, dir_size, source_dir
from ._http import build_session, download_file, polite_sleep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. OFAC Enforcement Actions via OpenSanctions
# ---------------------------------------------------------------------------

# OpenSanctions publishes bulk exports per dataset under a stable path.
OPEN_SANCTIONS_BASE = "https://data.opensanctions.org/datasets/latest/us_ofac_enforcement_actions"
# OpenSanctions changed the file set in late 2025; ``statements.csv`` no
# longer exists. We list every current resource (verified against the
# dataset's ``index.json``) so we don't waste a download attempt on a 404.
OPEN_SANCTIONS_FILES = [
    "entities.ftm.json",
    "names.txt",
    "senzing.json",
    "targets.nested.json",
    "targets.simple.csv",
    "index.json",
]


def download_ofac_enforcement(tracker: StatsTracker) -> None:
    with tracker.track(source="ofac_enforcement", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "ofac_enforcement", layer="level_2")
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()

        downloaded = 0
        for fname in OPEN_SANCTIONS_FILES:
            url = f"{OPEN_SANCTIONS_BASE}/{fname}"
            dest = out / fname
            try:
                nbytes = download_file(session, url, dest)
                if nbytes > 0 or dest.exists():
                    downloaded += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("ofac_enforcement: missing or failed %s: %s", url, e)

        if downloaded == 0:
            raise RuntimeError("ofac_enforcement: no files downloaded from OpenSanctions")

        stats.files_written = sum(1 for _ in out.rglob("*") if _.is_file())
        stats.bytes_written = dir_size(out)
        stats.notes["opensanctions_files_seen"] = downloaded


# ---------------------------------------------------------------------------
# 2. OFAC Guidance & Frameworks
# ---------------------------------------------------------------------------

OFAC_FILE_FINDER = "https://ofac.treasury.gov/file-finder"


def _collect_ofac_pdfs(session, index_url: str) -> set[str]:
    """Scrape ofac.treasury.gov/file-finder index pages for PDF links."""
    pdfs: set[str] = set()
    visited: set[str] = set()
    queue: list[str] = [index_url]

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger.warning("ofac_guidance: failed %s: %s", url, e)
            continue
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"].strip())
            if href.lower().endswith(".pdf"):
                pdfs.add(href)
            elif "file-finder" in href and "?page=" in href and href not in visited:
                queue.append(href)
        polite_sleep(0.3)
        # Guard against runaway crawls
        if len(visited) > 200:
            break
    return pdfs


def download_ofac_guidance(tracker: StatsTracker) -> None:
    with tracker.track(source="ofac_guidance", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "ofac_guidance", layer="level_2") / "pdfs"
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()
        pdfs = _collect_ofac_pdfs(session, OFAC_FILE_FINDER)
        logger.info("ofac_guidance: discovered %d PDFs", len(pdfs))

        def _task(u: str) -> int:
            name = u.rsplit("/", 1)[-1] or "doc.pdf"
            name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            dest = out / name
            try:
                return download_file(session, u, dest)
            except Exception as e:  # noqa: BLE001
                logger.warning("ofac_guidance: failed %s: %s", u, e)
                return 0

        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in as_completed([pool.submit(_task, u) for u in pdfs]):
                pass

        stats.files_written = sum(1 for p in out.glob("*.pdf") if p.stat().st_size > 0)
        stats.bytes_written = dir_size(out)
        stats.notes["urls_discovered"] = len(pdfs)


TASKS_CPT_L2 = {
    "ofac_enforcement": download_ofac_enforcement,
    "ofac_guidance": download_ofac_guidance,
}

ALL_TASKS = dict(TASKS_CPT_L2)
