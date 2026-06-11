"""ICIJ FinCEN Files — CPT Layer 2.

What ICIJ actually publishes
----------------------------
The FinCEN Files investigation (Sept-2020) is a journalism release; ICIJ
**does not** redistribute the leaked SAR PDFs (those were BuzzFeed's source
material and cannot be republished without legal exposure). ICIJ publishes:

1. A small **transactions CSV** of derived metadata (~2,500 transactions).
2. ~30 long-form **investigative articles** in the FinCEN Files series at
   ``icij.org/investigations/fincen-files/`` — these contain the AML-domain
   narrative register the strategy doc was originally aiming for.

This source therefore writes two artifacts under
``data/raw/cpt/level_2/fincen_files/``:

* ``data/<csv>``        — the transactions CSV (and any other data files
                          published on the download landing page).
* ``articles/<slug>.html`` + ``articles/<slug>.txt``
                        — every reachable FinCEN Files article HTML, plus
                          a stripped-text companion for direct CPT use.

The articles are the AML-narrative half of this source; Step 2 (NV-Ingest)
won't touch them because they're already plaintext-friendly HTML.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ._common import StatsTracker, dir_size, source_dir
from ._http import build_session, download_file, polite_sleep

logger = logging.getLogger(__name__)

ICIJ_BASE = "https://www.icij.org"
ICIJ_DOWNLOAD_LANDING = (
    f"{ICIJ_BASE}/investigations/fincen-files/download-fincen-files-transaction-data/"
)
# The investigation hub lists every story tagged FinCEN Files. Walking it
# (with pagination) gives us the canonical set of article URLs.
ICIJ_INVESTIGATION_HUB = f"{ICIJ_BASE}/investigations/fincen-files/"


# ---------------------------------------------------------------------------
# Part 1 — transactions CSV (and any other data files on the landing page)
# ---------------------------------------------------------------------------

def _download_transaction_data(session, out: Path) -> tuple[int, int]:
    """Pull the data files (CSV / XLSX / ZIP / JSON) from ICIJ's download
    landing page. Returns (urls_discovered, bytes_written_to_data_dir)."""
    try:
        r = session.get(ICIJ_DOWNLOAD_LANDING, timeout=60)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("fincen_files: data landing fetch failed: %s", e)
        return 0, 0

    soup = BeautifulSoup(r.text, "lxml")
    targets: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(ICIJ_DOWNLOAD_LANDING, a["href"].strip())
        if any(href.lower().endswith(ext) for ext in (".csv", ".xlsx", ".zip", ".json")):
            targets.add(href)

    if not targets:
        return 0, 0

    data_dir = out / "data"
    for u in targets:
        name = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            u.rsplit("/", 1)[-1].split("?", 1)[0] or "fincen_files_data",
        )
        try:
            download_file(session, u, data_dir / name)
        except Exception as e:  # noqa: BLE001
            logger.warning("fincen_files: data download failed %s: %s", u, e)

    return len(targets), dir_size(data_dir)


# ---------------------------------------------------------------------------
# Part 2 — investigation article HTML + plain-text extraction
# ---------------------------------------------------------------------------

def _discover_article_urls(session, max_pages: int = 6) -> set[str]:
    """Walk the FinCEN Files investigation hub (with pagination) and collect
    every article landing-page URL.

    Article URLs look like ``/investigations/fincen-files/<slug>/`` with
    no file extension. The hub itself, the download page, and the press-
    release sub-pages are filtered out via :data:`_NON_ARTICLE_SUFFIXES`.
    """
    found: set[str] = set()
    for page_num in range(max_pages):
        url = (
            ICIJ_INVESTIGATION_HUB
            if page_num == 0
            else f"{ICIJ_INVESTIGATION_HUB}page/{page_num + 1}/"
        )
        try:
            r = session.get(url, timeout=45)
            if r.status_code == 404:
                break
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger.warning("fincen_files: hub page %s failed: %s", url, e)
            break

        soup = BeautifulSoup(r.text, "lxml")
        new = 0
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"].strip())
            parsed = urlparse(href)
            if parsed.netloc and "icij.org" not in parsed.netloc:
                continue
            path = parsed.path.rstrip("/")
            if not path.startswith("/investigations/fincen-files/"):
                continue
            slug = path.rsplit("/", 1)[-1]
            if not slug or slug in _NON_ARTICLE_SUFFIXES:
                continue
            if "." in slug:  # asset link, not a story page
                continue
            if href not in found:
                found.add(href)
                new += 1
        logger.info(
            "fincen_files: hub page %d -> %d article URLs (cumulative %d)",
            page_num + 1, new, len(found),
        )
        if new == 0:
            break
        polite_sleep(0.4)
    return found


_NON_ARTICLE_SUFFIXES = {
    "fincen-files",                        # the hub root
    "download-fincen-files-transaction-data",
    "press-releases",
    "share",
}


_BOILERPLATE_RE = re.compile(
    r"(?:share this story|sign up for our newsletter|stay informed|"
    r"keep reading|read more from this investigation)",
    re.IGNORECASE,
)


def _extract_article_text(html: str) -> str:
    """Strip ICIJ's standard chrome and emit plain text for an article page."""
    soup = BeautifulSoup(html, "lxml")
    # Drop scripts, styles, and obvious chrome
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    article = soup.find("article") or soup.find("main") or soup.body or soup
    text = article.get_text("\n", strip=True)
    # Trim runs of obvious newsletter / share footers
    text = _BOILERPLATE_RE.split(text, maxsplit=1)[0]
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _slugify(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1] or "article"
    return re.sub(r"[^A-Za-z0-9._-]", "_", slug)


def _download_articles(session, out: Path) -> int:
    """Download every FinCEN Files article and emit ``<slug>.html`` plus a
    stripped ``<slug>.txt`` companion. Returns the count of stored articles."""
    articles_dir = out / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    urls = _discover_article_urls(session)
    if not urls:
        logger.warning("fincen_files: 0 article URLs discovered on the hub")
        return 0

    def _one(url: str) -> bool:
        slug = _slugify(url)
        html_path = articles_dir / f"{slug}.html"
        txt_path = articles_dir / f"{slug}.txt"
        if html_path.exists() and txt_path.exists() and html_path.stat().st_size > 0:
            return True
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger.warning("fincen_files: article fetch failed %s: %s", url, e)
            return False
        html_path.write_text(r.text, encoding="utf-8")
        try:
            txt_path.write_text(_extract_article_text(r.text), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("fincen_files: article text-extract failed %s: %s", url, e)
        return True

    ok = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, u) for u in sorted(urls)]
        for fut in as_completed(futures):
            if fut.result():
                ok += 1
            polite_sleep(0.1)
    logger.info("fincen_files: stored %d/%d articles", ok, len(urls))
    return ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def download_fincen_files(tracker: StatsTracker) -> None:
    with tracker.track(source="fincen_files", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "fincen_files", layer="level_2")
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()

        # Part 1: transactions CSV + any other data files on the download
        # landing page. Historically this was the only thing the script
        # picked up; that misses the high-value narrative articles.
        data_urls, _ = _download_transaction_data(session, out)

        # Part 2: investigation articles (the AML narrative register half).
        articles_count = _download_articles(session, out)

        stats.files_written = sum(1 for _ in out.rglob("*") if _.is_file())
        stats.bytes_written = dir_size(out)
        stats.notes["data_urls_discovered"] = data_urls
        stats.notes["articles_stored"] = articles_count


TASKS_CPT_L2 = {"fincen_files": download_fincen_files}
ALL_TASKS = dict(TASKS_CPT_L2)
