"""HTTP helpers: retrying requests session, cloudscraper, Playwright fallback.

Sites like fatf-gafi.org gate static scrapers; we use cloudscraper first and
fall back to Playwright (headless Chromium) for JS-rendered index pages.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def build_session(timeout: int = 60, user_agent: str = DEFAULT_UA) -> requests.Session:
    """Return a Session with retry/backoff and a sane UA."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def build_cloudscraper() -> Any:
    """Return a cloudscraper client with retry behavior similar to our session."""
    import cloudscraper

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False},
    )
    scraper.headers.update({"User-Agent": DEFAULT_UA})
    return scraper


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 1 << 16,
    timeout: int = 120,
    headers: dict | None = None,
) -> int:
    """Stream a URL to a local file. Returns bytes written (0 if skipped)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        logger.debug("skip existing: %s", dest.name)
        return 0

    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout, headers=headers or {}) as r:
        r.raise_for_status()
        written = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
        tmp.rename(dest)
    return written


def playwright_get_html(url: str, *, wait_selector: str | None = None, timeout_ms: int = 30000) -> str:
    """Render a URL with headless Chromium and return the final HTML.

    Used as a fallback when cloudscraper cannot get past JS-only pages.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=DEFAULT_UA)
            page = context.new_page()
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=timeout_ms)
                except Exception:  # noqa: BLE001
                    logger.warning("playwright: selector %r not found on %s", wait_selector, url)
            # Let client-side renders settle
            page.wait_for_timeout(1500)
            return page.content()
        finally:
            browser.close()


def polite_sleep(seconds: float = 0.25) -> None:
    """Minimal inter-request delay for scraping index pages."""
    if seconds > 0:
        time.sleep(seconds)
