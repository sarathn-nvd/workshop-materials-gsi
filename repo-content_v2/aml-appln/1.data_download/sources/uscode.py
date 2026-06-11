"""USCode Direct — CPT Layer 1 supplement.

Why this exists
---------------
``pile-of-law/pile-of-law``'s ``uscode`` subset is missing the two largest
and most AML-relevant Titles in the U.S. Code:

* **Title 12** (Banks and Banking) — entirely absent from the packaging
* **Title 18** (Crimes and Criminal Procedure) main body — only the small
  Appendix shipped, so Chapters 95 (RICO) and 96 (§§1956 / 1957 — the
  primary money-laundering statutes) are missing

The strategy doc explicitly targets these Titles for CPT Layer 1 register
training, so we pull them directly from the official publishing site at
``uscode.house.gov``. Title 31 (Money & Finance) is also fetched as a
duplicate of pile-of-law's coverage — Step 3 deduplication handles overlap.

URL structure
-------------
The latest release point is published on
``https://uscode.house.gov/download/download.shtml`` and exposes per-Title
HTML zips at::

    https://uscode.house.gov/download/releasepoints/us/pl/<C>/<N>/htm_usc<NN>@<C>-<N>.zip

where ``C-N`` is the congress-session number (e.g. ``119-84`` as of
Apr-2026). Each zip extracts to a single ``PRELIMusc<NN>.htm`` file
containing the entire Title in xhtml form (10–60 MB depending on Title).

This source resolves the *latest* release point from the download page
each run, then downloads + extracts the target-Title zips into
``data/raw/cpt/level_1/uscode_house/title_<NN>/<Title>.htm``. Step 2
(NV-Ingest) extracts clean text from these HTMLs the same way it does the
FFIEC manual.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ._common import StatsTracker, dir_size, source_dir
from ._http import build_session

logger = logging.getLogger(__name__)

DOWNLOAD_INDEX = "https://uscode.house.gov/download/download.shtml"
RELEASE_BASE = "https://uscode.house.gov/download"

# Per ``revised_strategy.md`` §1.1: Title 12 (Banks), Title 18 (Crimes,
# specifically chapters 95 RICO + 96 AML §§1956–1957), Title 31 (Money &
# Finance — already in pile-of-law but inexpensive to mirror).
TARGET_TITLES: tuple[str, ...] = ("12", "18", "31")


def _resolve_release_point(session) -> str:
    """Scrape ``download.shtml`` to find the latest release-point slug.

    Returns a string like ``"119-84"``. Falls back to a known-good value if
    the page can't be parsed (so a transient outage doesn't fail the task).
    """
    try:
        r = session.get(DOWNLOAD_INDEX, timeout=30)
        r.raise_for_status()
        # The page hard-codes filenames like ``htm_usc12@119-84.zip`` for
        # every title. Any of them lets us extract the release-point slug.
        m = re.search(r"htm_usc\d+[a-z]?@(\d+-\d+)\.zip", r.text)
        if m:
            slug = m.group(1)
            logger.info("uscode: resolved latest release point = %s", slug)
            return slug
    except Exception as e:  # noqa: BLE001
        logger.warning("uscode: release-point probe failed (%s); using fallback", e)
    fallback = "119-84"
    logger.info("uscode: using fallback release point = %s", fallback)
    return fallback


def _download_title(
    session,
    *,
    title: str,
    release: str,
    out_dir: Path,
) -> tuple[str, int]:
    """Download and extract one Title's HTML zip. Returns (title, bytes_written)."""
    congress, session_num = release.split("-", 1)
    zip_url = (
        f"{RELEASE_BASE}/releasepoints/us/pl/{congress}/{session_num}"
        f"/htm_usc{title}@{release}.zip"
    )
    title_dir = out_dir / f"title_{title}"
    title_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent skip: if any non-empty .htm already exists for this title,
    # we treat the title as downloaded.
    existing = [p for p in title_dir.glob("*.htm") if p.stat().st_size > 0]
    if existing:
        bytes_existing = sum(p.stat().st_size for p in existing)
        logger.info(
            "uscode: title %s already present (%d files, %d bytes) — skip",
            title, len(existing), bytes_existing,
        )
        return title, bytes_existing

    try:
        r = session.get(zip_url, timeout=180)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("uscode: title %s download failed (%s)", title, e)
        return title, 0

    if len(r.content) < 5_000:
        # The site returns an HTML "doc not found" stub at the wrong URL.
        # A real Title is at minimum ~1 MB compressed.
        logger.warning(
            "uscode: title %s response only %d bytes — likely a not-found stub",
            title, len(r.content),
        )
        return title, 0

    written = 0
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            members = zf.namelist()
            for name in members:
                if not name.lower().endswith(".htm"):
                    continue
                target = title_dir / Path(name).name  # flatten
                with zf.open(name) as src, target.open("wb") as dst:
                    chunk = src.read()
                    dst.write(chunk)
                    written += len(chunk)
            logger.info(
                "uscode: title %s extracted %d HTML files (%d bytes)",
                title, len(members), written,
            )
    except zipfile.BadZipFile as e:
        logger.warning("uscode: title %s bad zip (%s)", title, e)
        return title, 0

    return title, written


def download_uscode(tracker: StatsTracker) -> None:
    """Download USCode Title 12, 18, 31 directly from uscode.house.gov."""
    with tracker.track(source="uscode_house", phase="cpt", layer="level_1") as stats:
        out = source_dir("cpt", "uscode_house", layer="level_1")
        session = build_session()

        release = _resolve_release_point(session)
        stats.notes["release_point"] = release
        stats.notes["titles_requested"] = list(TARGET_TITLES)

        # Per-Title fetches in parallel (3 small downloads).
        successes: list[str] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_download_title, session, title=t, release=release, out_dir=out): t
                for t in TARGET_TITLES
            }
            for fut in as_completed(futures):
                title, n_bytes = fut.result()
                if n_bytes > 0:
                    successes.append(title)

        stats.files_written = sum(1 for p in out.rglob("*.htm") if p.stat().st_size > 0)
        stats.bytes_written = dir_size(out)
        stats.notes["titles_downloaded"] = sorted(successes)


TASKS_CPT_L1 = {"uscode_house": download_uscode}
ALL_TASKS = dict(TASKS_CPT_L1)
