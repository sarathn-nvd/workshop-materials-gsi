"""FFIEC BSA/AML Examination Manual — SFT source.

Authoritative reference used by US bank examiners. Published online as
HTML; we save each section page verbatim so Step 2 (NV-Ingest) can
extract structured text.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ._common import StatsTracker, dir_size, source_dir
from ._http import build_session, polite_sleep

logger = logging.getLogger(__name__)

FFIEC_START = "https://bsaaml.ffiec.gov/manual/Introduction/01"
FFIEC_ROOT_NETLOC = "bsaaml.ffiec.gov"


def _is_manual_section(href: str) -> bool:
    p = urlparse(href)
    if p.netloc and p.netloc != FFIEC_ROOT_NETLOC:
        return False
    return p.path.startswith("/manual/") and not p.path.endswith((".pdf", ".zip"))


def _slugify(path: str) -> str:
    p = path.strip("/").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]", "_", p) or "index"


def download_ffiec_manual(tracker: StatsTracker) -> None:
    """Crawl the FFIEC BSA/AML Manual sections and save each as HTML."""
    with tracker.track(source="ffiec_manual", phase="sft") as stats:
        out = source_dir("sft", "ffiec_manual") / "html"
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()

        seen: set[str] = set()
        queue: list[str] = [FFIEC_START]
        files = 0

        while queue:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                r = session.get(url, timeout=60)
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                logger.warning("ffiec: failed %s: %s", url, e)
                continue

            dest = out / f"{_slugify(urlparse(url).path)}.html"
            dest.write_text(r.text, encoding="utf-8")
            files += 1

            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"].strip())
                if _is_manual_section(href) and href not in seen:
                    queue.append(href)

            polite_sleep(0.25)
            if len(seen) > 500:
                logger.warning("ffiec: crawl cap reached at %d pages", len(seen))
                break

        stats.files_written = files
        stats.bytes_written = dir_size(out)
        stats.notes["pages_crawled"] = len(seen)


TASKS_SFT = {"ffiec_manual": download_ffiec_manual}
ALL_TASKS = dict(TASKS_SFT)
