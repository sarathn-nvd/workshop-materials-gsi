"""FinCEN sources — CPT Layer 2.

  1. Advisories & Guidance       (PDFs scraped from fincen.gov)
  2. Federal Register Notices    (PDFs scraped from fincen.gov)
  3. SAR Activity Reviews        (23 issues; direct PDF download)
  4. Enforcement Actions         (data.gov CSV + linked case docs)

All PDFs are saved as-is; extraction is deferred to Step 2 (NV-Ingest).
"""
from __future__ import annotations

import csv
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ._common import StatsTracker, dir_size, source_dir
from ._http import build_session, download_file, polite_sleep

logger = logging.getLogger(__name__)

BASE = "https://www.fincen.gov"


# ---------------------------------------------------------------------------
# Generic FinCEN scraper — walk an index page, collect .pdf hrefs, download
# ---------------------------------------------------------------------------

FINCEN_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Regex to also catch /sites/default/files/*.pdf relative links
_PDF_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)


def _discover_pdf_links(
    session,
    index_urls: list[str],
    *,
    follow_pagination: bool = True,
    follow_landing_pages: bool = True,
    max_pages: int = 60,
) -> set[str]:
    """Walk index pages and collect PDF URLs, optionally following landing pages.

    Behavior:
      1. Crawl every index URL (404s are warned but tolerated; we keep a list
         of candidate URLs because fincen.gov renames slugs periodically).
      2. Direct ``.pdf`` hrefs are collected immediately.
      3. ``follow_pagination`` follows ``?page=N`` Drupal links.
      4. ``follow_landing_pages`` opens internal ``/news/...`` and
         ``/resources/...`` pages that don't end in ``.pdf`` and looks for
         linked PDFs there. This is needed because FinCEN advisories often
         live on a per-advisory landing page rather than on the index.
    """
    pdfs: set[str] = set()
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(u, 0) for u in index_urls]

    def _is_internal(href: str) -> bool:
        return "fincen.gov" in href or href.startswith("/")

    def _looks_like_landing(href: str) -> bool:
        if not _is_internal(href):
            return False
        if _PDF_RE.search(href):
            return False
        path = href.split("?", 1)[0].lower()
        # Drupal article pages we care about
        return any(s in path for s in (
            "/news/", "/resources/", "/advisor", "/alert",
            "/financial-trend-analyses/", "/sar-activity-review",
        ))

    while queue:
        url, depth = queue.pop(0)
        if url in visited or len(visited) >= max_pages:
            continue
        visited.add(url)
        try:
            r = session.get(url, timeout=60, headers=FINCEN_HEADERS)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger.warning("fincen: failed to fetch index %s: %s", url, e)
            continue

        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"].strip())
            if _PDF_RE.search(href):
                pdfs.add(href)
                continue
            if (
                follow_pagination
                and "page=" in href
                and "fincen.gov" in href
                and href not in visited
            ):
                queue.append((href, depth))
                continue
            if (
                follow_landing_pages
                and depth < 1                  # only one extra hop
                and _looks_like_landing(href)
                and href not in visited
            ):
                queue.append((href, depth + 1))

        polite_sleep(0.2)

    return pdfs


def _download_pdfs_parallel(
    session,
    pdf_urls: set[str],
    out_dir: Path,
    workers: int = 8,
) -> tuple[int, int]:
    """Download a set of PDFs in parallel. Returns (files, bytes)."""
    files = 0
    bytes_written = 0

    def _task(u: str) -> tuple[str, int]:
        name = u.rsplit("/", 1)[-1] or "index.pdf"
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        dest = out_dir / name
        try:
            return u, download_file(session, u, dest)
        except Exception as e:  # noqa: BLE001
            logger.warning("fincen: failed %s: %s", u, e)
            return u, 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_task, u) for u in pdf_urls]
        for fut in as_completed(futures):
            _, nbytes = fut.result()
            if nbytes > 0:
                files += 1
                bytes_written += nbytes

    # Also count pre-existing files (re-run case)
    total_files = sum(1 for p in out_dir.glob("*.pdf") if p.stat().st_size > 0)
    return total_files, dir_size(out_dir)


# ---------------------------------------------------------------------------
# 1. Advisories & Guidance
# ---------------------------------------------------------------------------

# FinCEN renames these paths periodically; try every known location. As of
# Apr-2026 the canonical advisories index lives at the no-hyphen-between-words
# slug ``advisoriesbulletinsfact-sheets``; the older hyphenated slugs return
# 404 but are kept as fall-throughs in case the path is reverted.
ADVISORIES_INDEX = [
    f"{BASE}/resources/advisoriesbulletinsfact-sheets",
    f"{BASE}/resources/financial-trend-analyses",
    f"{BASE}/resources/statutes-regulations/guidance",
    f"{BASE}/resources/advisories-notices-bulletins-fact-sheets",
    f"{BASE}/news-room/advisories",
    f"{BASE}/news/advisories",
    f"{BASE}/resources/advisories",
]


def download_fincen_advisories(tracker: StatsTracker) -> None:
    with tracker.track(source="fincen_advisories", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "fincen_advisories", layer="level_2") / "pdfs"
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()
        pdfs = _discover_pdf_links(session, ADVISORIES_INDEX)
        logger.info("fincen_advisories: discovered %d PDF URLs", len(pdfs))
        if not pdfs:
            raise RuntimeError(
                "fincen_advisories: no PDFs discovered across any candidate index URL"
            )
        files, bytes_written = _download_pdfs_parallel(session, pdfs, out)
        stats.files_written = files
        stats.bytes_written = bytes_written
        stats.notes["urls_discovered"] = len(pdfs)


# ---------------------------------------------------------------------------
# 2. Federal Register Notices
# ---------------------------------------------------------------------------

FR_NOTICES_INDEX = [
    f"{BASE}/resources/statutes-regulations/federal-register-notices",
    f"{BASE}/resources/federal-register-notices",
    f"{BASE}/news-room/federal-register-notices",
    f"{BASE}/news/federal-register-notices",
]


def download_fincen_federal_register(tracker: StatsTracker) -> None:
    with tracker.track(source="fincen_federal_register", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "fincen_federal_register", layer="level_2") / "pdfs"
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()
        pdfs = _discover_pdf_links(session, FR_NOTICES_INDEX)
        logger.info("fincen_federal_register: discovered %d PDF URLs", len(pdfs))
        if not pdfs:
            raise RuntimeError(
                "fincen_federal_register: no PDFs discovered across any candidate index URL"
            )
        files, bytes_written = _download_pdfs_parallel(session, pdfs, out)
        stats.files_written = files
        stats.bytes_written = bytes_written
        stats.notes["urls_discovered"] = len(pdfs)


# ---------------------------------------------------------------------------
# 3. SAR Activity Reviews — 23 issues, direct PDF downloads
# ---------------------------------------------------------------------------

SAR_REVIEWS_INDEX = [
    f"{BASE}/sar-activity-review-trends-tips-issues",
]


def download_fincen_sar_reviews(tracker: StatsTracker) -> None:
    with tracker.track(source="fincen_sar_reviews", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "fincen_sar_reviews", layer="level_2") / "pdfs"
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()
        pdfs = _discover_pdf_links(session, SAR_REVIEWS_INDEX)
        logger.info("fincen_sar_reviews: discovered %d PDF URLs", len(pdfs))
        files, bytes_written = _download_pdfs_parallel(session, pdfs, out)
        stats.files_written = files
        stats.bytes_written = bytes_written
        stats.notes["urls_discovered"] = len(pdfs)


# ---------------------------------------------------------------------------
# 4. Enforcement Actions — data.gov CSV metadata + linked docs
# ---------------------------------------------------------------------------

CKAN_API = "https://catalog.data.gov/api/3/action"
ENFORCEMENT_CKAN_IDS = [
    "fincen-enforcement-actions-for-violations-of-the-bank-secrecy-act",
    "fincen-enforcement-actions",
]
ENFORCEMENT_CKAN_QUERIES = [
    "fincen enforcement",
    "bank secrecy act enforcement",
]

# FinCEN's own enforcement-actions landing pages (primary source now that
# data.gov's CKAN entry appears to have been retired).
ENFORCEMENT_FINCEN_PAGES = [
    f"{BASE}/news-room/enforcement-actions",
    f"{BASE}/news/enforcement-actions",
    f"{BASE}/enforcement/enforcement-actions",
]


def _ckan_get_package(session, package_id: str) -> dict | None:
    url = f"{CKAN_API}/package_show?id={package_id}"
    try:
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            return r.json().get("result") or None
    except Exception as e:  # noqa: BLE001
        logger.debug("ckan package_show %s failed: %s", package_id, e)
    return None


def _ckan_search_first(session, query: str) -> dict | None:
    from urllib.parse import quote_plus
    url = f"{CKAN_API}/package_search?q={quote_plus(query)}"
    try:
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            results = r.json().get("result", {}).get("results") or []
            if results:
                return results[0]
    except Exception as e:  # noqa: BLE001
        logger.debug("ckan package_search %s failed: %s", query, e)
    return None


def download_fincen_enforcement(tracker: StatsTracker) -> None:
    """Download FinCEN enforcement actions.

    Strategy (in order):
      1. CKAN known package IDs
      2. CKAN search fallback
      3. Direct scrape of FinCEN enforcement-actions landing pages
    Any successful path populates the output; we succeed if any one works.
    """
    with tracker.track(source="fincen_enforcement", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "fincen_enforcement", layer="level_2")
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()

        got_anything = False

        # 1. CKAN package_show with known IDs
        payload: dict | None = None
        for pid in ENFORCEMENT_CKAN_IDS:
            payload = _ckan_get_package(session, pid)
            if payload:
                stats.notes["ckan_id"] = pid
                break

        # 2. CKAN package_search fallback
        if payload is None:
            for q in ENFORCEMENT_CKAN_QUERIES:
                payload = _ckan_search_first(session, q)
                if payload:
                    stats.notes["ckan_id"] = payload.get("name", f"via_search:{q}")
                    break

        if payload is not None:
            resources = payload.get("resources", []) or []
            doc_urls: list[str] = []
            for res in resources:
                u = res.get("url")
                fmt = (res.get("format") or "").lower()
                if not u:
                    continue
                name = u.rsplit("/", 1)[-1] or f"resource_{res.get('id', 'x')}"
                name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
                dest = out / name
                try:
                    download_file(session, u, dest)
                    got_anything = True
                    if fmt in {"csv", "json"}:
                        doc_urls.extend(_extract_links_from_csv(dest))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "fincen_enforcement: ckan resource %s failed: %s", u, e,
                    )

            pdf_urls = {u for u in doc_urls if u.lower().endswith(".pdf")}
            if pdf_urls:
                pdf_out = out / "case_documents"
                pdf_out.mkdir(exist_ok=True)
                _download_pdfs_parallel(session, pdf_urls, pdf_out)
                stats.notes["case_pdfs_from_csv"] = len(pdf_urls)
        else:
            logger.info(
                "fincen_enforcement: CKAN lookup exhausted; falling back to direct scrape",
            )

        # 3. Direct scrape of FinCEN enforcement-actions landing pages
        pdfs = _discover_pdf_links(session, ENFORCEMENT_FINCEN_PAGES)
        if pdfs:
            pdf_out = out / "scraped_pdfs"
            pdf_out.mkdir(exist_ok=True)
            files, _ = _download_pdfs_parallel(session, pdfs, pdf_out)
            if files > 0:
                got_anything = True
                stats.notes["scraped_pdfs"] = len(pdfs)

        if not got_anything:
            raise RuntimeError(
                "fincen_enforcement: neither CKAN nor direct scrape produced any files",
            )

        stats.files_written = sum(1 for _ in out.rglob("*") if _.is_file())
        stats.bytes_written = dir_size(out)


def _extract_links_from_csv(path: Path) -> list[str]:
    urls: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return urls
            link_idx = [i for i, h in enumerate(header) if "url" in h.lower() or "link" in h.lower()]
            for row in reader:
                for i in link_idx:
                    if i < len(row) and row[i].startswith("http"):
                        urls.append(row[i].strip())
    except Exception as e:  # noqa: BLE001
        logger.debug("fincen_enforcement: csv link scan failed on %s: %s", path.name, e)
    return urls


# ---------------------------------------------------------------------------
# Public task registry
# ---------------------------------------------------------------------------

TASKS_CPT_L2 = {
    "fincen_advisories": download_fincen_advisories,
    "fincen_federal_register": download_fincen_federal_register,
    "fincen_sar_reviews": download_fincen_sar_reviews,
    "fincen_enforcement": download_fincen_enforcement,
}

ALL_TASKS = dict(TASKS_CPT_L2)
