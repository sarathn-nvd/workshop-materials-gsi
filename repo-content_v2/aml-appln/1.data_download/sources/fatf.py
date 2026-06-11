"""FATF Publications — CPT Layer 2.

The FATF site (fatf-gafi.org) is a JS-rendered AEM CMS behind Cloudflare.
As of Apr-2026 the publication index pages render document tiles via
client-side fetch — a static scrape returns 0 PDFs because direct ``.pdf``
hrefs simply do not exist on the index pages anymore.

Two-pass strategy
-----------------
Pass 1 — *discovery*. For each FATF publications index page (e.g.
``/en/publications/mutualevaluations.html``), we render with Playwright,
let the card list settle, scroll to load every card, then collect the
``/en/publications/.../...html`` document landing-page URLs.

Pass 2 — *resolution*. For each landing page, fetch the rendered HTML
(cloudscraper or Playwright fallback) and regex-scrape ``/content/dam/...pdf``
URLs. The CMS sometimes appends ``.coredownload.inline.pdf`` or
``.coredownload.pdf`` suffixes; both are valid PDFs — we keep them all.

PDFs are then downloaded in parallel and stored under ``pdfs/`` with names
slugified from the URL basename.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests

from ._common import StatsTracker, dir_size, source_dir
from ._http import polite_sleep

logger = logging.getLogger(__name__)

FATF_BASE = "https://www.fatf-gafi.org"

FATF_INDEX_PAGES = [
    f"{FATF_BASE}/en/publications/mutualevaluations.html",
    f"{FATF_BASE}/en/publications/methodsandtrends.html",
    f"{FATF_BASE}/en/publications/fatfrecommendations.html",
    f"{FATF_BASE}/en/publications/fatfgeneral.html",
    f"{FATF_BASE}/en/publications/high-risk-and-other-monitored-jurisdictions.html",
]

# AEM faceted-search endpoints. ``methodsandtrends`` exposes the full set of
# ~90 typology / methods-and-trends reports through this JSON API; the other
# topic pages don't use the same component (they render server-side and we
# fall back to scraped links on the index pages themselves).
FATF_FACETED_API = [
    f"{FATF_BASE}/content/fatf-gafi/en/topics/methods-and-trends"
    f"/jcr:content/root/container/faceted_search/results.facets.json",
]

# Regex to capture both absolute and relative ``.pdf`` references that may
# appear in FATF page HTML (often inside JSON blobs embedded in <script>).
_PDF_REGEX = re.compile(
    r"(?:https?://[^\"'\s<>]+|/[^\"'\s<>]+)\.pdf(?:\?[^\"'\s<>]*)?",
    re.IGNORECASE,
)

# Match document landing-page URLs anchored from any of the index pages.
# FATF uses inconsistent capitalization (mixed "Mutualevaluations" /
# "mutualevaluations"); we handle that case-insensitively.
_LANDING_REGEX = re.compile(
    r"/en/publications/[^\"'\s<>]+\.html",
    re.IGNORECASE,
)

# Hrefs we should NOT treat as document landing pages even though they match
# the URL prefix (these are filter / category indices themselves).
_LANDING_BLOCKLIST = {
    "mutualevaluations.html",
    "methodsandtrends.html",
    "fatfrecommendations.html",
    "fatfgeneral.html",
    "high-risk-and-other-monitored-jurisdictions.html",
}


# ---------------------------------------------------------------------------
# Pass 1 — discover document landing pages
# ---------------------------------------------------------------------------

def _render_index_with_scroll(ctx, url: str) -> str:
    """Render an index page in ``ctx``, scrolling to load every card.

    The FATF index is lazy-loaded; we scroll to the bottom several times to
    force every card into the DOM before snapshotting the HTML. Uses the
    caller-supplied Playwright BrowserContext so the same Cloudflare cookie
    is shared with PDF downloads later in the run.
    """
    page = ctx.new_page()
    try:
        page.goto(url, timeout=90_000, wait_until="domcontentloaded")
        page.wait_for_timeout(6000)  # let JS settle
        for _ in range(10):
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(800)
        return page.content()
    finally:
        page.close()


def _extract_landing_pages(html: str, index_url: str) -> set[str]:
    """Return absolute URLs of document landing pages found on an index page."""
    landings: set[str] = set()
    for match in _LANDING_REGEX.findall(html):
        slug = match.rsplit("/", 1)[-1].lower()
        if slug in _LANDING_BLOCKLIST:
            continue
        if "/" in match.strip("/").lower():
            absolute = urljoin(index_url, match)
            landings.add(absolute)
    return landings


# ---------------------------------------------------------------------------
# Pass 2 — resolve PDF URLs from each landing page
# ---------------------------------------------------------------------------

def _extract_pdf_urls_from_landing(html: str, base_url: str) -> set[str]:
    pdfs: set[str] = set()
    for match in _PDF_REGEX.findall(html):
        # Strip stray HTML-encoded suffixes like ``\\&quot;>`` that occasionally
        # leak from inlined JSON. Keep only the URL portion.
        cleaned = re.split(r"[\\\"<>]", match, maxsplit=1)[0]
        cleaned = cleaned.rstrip("\\")
        if not cleaned.lower().endswith((".pdf",)) and ".pdf?" not in cleaned.lower():
            # Pattern allows query strings; only accept if this still ends in pdf-ish
            if not re.search(r"\.pdf(\?|$)", cleaned, re.IGNORECASE):
                continue
        absolute = urljoin(base_url, cleaned)
        # Restrict to fatf-gafi.org so we don't follow stray external URLs
        host = urlparse(absolute).netloc
        if host and "fatf-gafi.org" not in host.lower():
            continue
        pdfs.add(absolute)
    return pdfs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _slugify_filename(url: str) -> str:
    name = url.split("?", 1)[0].rsplit("/", 1)[-1] or "doc.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _playwright_pdf_session():
    """Return a Playwright request context with a warmed Cloudflare cookie.

    FATF's Cloudflare instance returns 403 to ``requests`` and ``cloudscraper``
    even with a perfect UA. The only thing it accepts is a real Chromium
    request context that carries the challenge cookie. We launch one browser
    per call site, visit a publications page to satisfy the challenge, then
    return ``(browser, context)`` so the caller can issue many
    ``context.request.get(...)`` calls before closing.
    """
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        ),
    )
    # Warm-up: visit the FATF homepage so the Cloudflare challenge cookie is
    # set on the context. We deliberately visit the root URL (not one of the
    # publication indexes) so the per-page render cache stays clean for the
    # 5 index visits that follow.
    page = ctx.new_page()
    try:
        page.goto(f"{FATF_BASE}/", timeout=90_000, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
    finally:
        page.close()
    return pw, browser, ctx


def _playwright_download_pdf(ctx, url: str, dest: Path) -> int:
    """Download a PDF via the warmed Playwright request context."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    resp = ctx.request.get(url, timeout=120_000)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    body = resp.body()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return len(body)


# ---------------------------------------------------------------------------
# Internet Archive Wayback Machine fallback
# ---------------------------------------------------------------------------
#
# When FATF's Cloudflare WAF rate-limits this IP (every PDF returns 403), we
# can route around the live origin via web.archive.org. The Wayback CDX API
# tells us if a snapshot exists and the most recent snapshot timestamp; we
# then fetch the PDF body from the special ``id_`` rendering URL which
# returns the original asset bytes (not a Wayback HTML frame).
#
# Hit rate is high in practice: FATF documents are well-archived (most
# `/content/dam/...pdf` URLs have one or more snapshots dating back years).

WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
WAYBACK_FETCH = "https://web.archive.org/web/{ts}id_/{url}"


def _wayback_lookup(url: str, *, session: requests.Session,
                    max_retries: int = 3) -> str | None:
    """Return the most recent successful Wayback snapshot timestamp, or None.

    Uses the lightweight Availability API rather than the full CDX search —
    CDX can take 30-60 s per query under load and is overkill for our
    "give me the closest snapshot to today" need. Honors ``Retry-After`` on
    429 responses so we don't get into a feedback loop with archive.org's
    rate limiter.
    """
    import time as _time

    for attempt in range(max_retries):
        try:
            r = session.get(
                WAYBACK_AVAILABLE,
                params={"url": url},
                timeout=20,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("wayback lookup HTTP failed for %s: %s", url, e)
            return None

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                return None
            snap = (data.get("archived_snapshots") or {}).get("closest") or {}
            if snap.get("available") and str(snap.get("status", "")).startswith("2"):
                return snap.get("timestamp")
            return None

        if r.status_code == 429:
            # Respect Retry-After if present, else exponential backoff.
            wait = int(r.headers.get("Retry-After", 0) or (5 * (2 ** attempt)))
            logger.warning("wayback 429; sleeping %ds (attempt %d)", wait, attempt + 1)
            _time.sleep(wait)
            continue

        # Non-200 / non-429 — give up on this URL.
        logger.debug("wayback lookup HTTP %d for %s", r.status_code, url)
        return None
    return None


def _wayback_download(url: str, dest: Path, *, session: requests.Session) -> int:
    """Try to download ``url`` via the Wayback Machine. Returns bytes or 0."""
    ts = _wayback_lookup(url, session=session)
    if not ts:
        return 0
    fetch_url = WAYBACK_FETCH.format(ts=ts, url=quote(url, safe=":/?&=%"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    try:
        with session.get(fetch_url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                logger.debug(
                    "wayback fetch HTTP %d for %s (snapshot %s)",
                    r.status_code, url, ts,
                )
                return 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            written = 0
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            if written < 1024:
                # Probably an error page or thin redirect; discard.
                tmp.unlink(missing_ok=True)
                return 0
            tmp.rename(dest)
            return written
    except Exception as e:  # noqa: BLE001
        logger.debug("wayback download failed for %s: %s", url, e)
        return 0


def _path_to_landing_url(jcr_path: str) -> str:
    """Convert an AEM ``/content/fatf-gafi/en/...`` path to a public URL."""
    if jcr_path.startswith("/content/fatf-gafi/"):
        suffix = jcr_path[len("/content/fatf-gafi/"):]
        return f"{FATF_BASE}/{suffix}.html"
    return urljoin(FATF_BASE, jcr_path)


def _fetch_faceted_search(ctx, api_url: str) -> list[str]:
    """Return public landing-page URLs from an AEM faceted search API.

    The AEM endpoint returns only the first page (~15 results) and rejects
    every pagination query string we tried; this gives us recent FATF docs
    that supplement (but don't fully replace) the index-page scrape.
    """
    out: list[str] = []
    try:
        r = ctx.request.get(api_url, timeout=30_000)
        if not r.ok:
            logger.warning("fatf facets: HTTP %d for %s", r.status, api_url)
            return out
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("fatf facets failed %s: %s", api_url, e)
        return out

    results = data.get("results", []) or []
    for item in results:
        path = item.get("path") or ""
        if path:
            out.append(_path_to_landing_url(path))
    total = data.get("totalMatches", len(out))
    label = api_url.rsplit("/topics/", 1)[-1].split("/", 1)[0]
    logger.info("fatf facets: %s -> %d (of %s) documents", label, len(out), total)
    return out


def download_fatf_publications(tracker: StatsTracker) -> None:
    with tracker.track(source="fatf_publications", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "fatf_publications", layer="level_2") / "pdfs"
        out.mkdir(parents=True, exist_ok=True)

        # We use one Playwright session for everything: index discovery,
        # API calls, landing-page resolution, and PDF downloads. This way the
        # Cloudflare challenge cookie is established once and reused.
        pw, browser, ctx = _playwright_pdf_session()
        try:
            # Pass 1a: scrape index pages (gives us mutualevaluations + high-risk
            # tile links that aren't exposed via the API).
            landing_pages: set[str] = set()
            for idx_url in FATF_INDEX_PAGES:
                try:
                    html = _render_index_with_scroll(ctx, idx_url)
                    found = _extract_landing_pages(html, idx_url)
                    logger.info("fatf: index %s -> %d landing pages", idx_url, len(found))
                    landing_pages.update(found)
                except Exception as e:  # noqa: BLE001
                    logger.warning("fatf: index %s failed: %s", idx_url, e)
                polite_sleep(0.5)

            # Pass 1b: AEM faceted-search API (~91 methods-and-trends docs)
            for api_url in FATF_FACETED_API:
                landing_pages.update(_fetch_faceted_search(ctx, api_url))

            if not landing_pages:
                raise RuntimeError("fatf: discovered 0 landing pages across every index")

            # Pass 2: resolve each landing page to its PDF URLs.
            #
            # We use the warmed Playwright request context for landing-page
            # HTML too — Cloudflare blocks plain ``requests``/cloudscraper for
            # most landing pages, just like for the PDFs themselves.
            pdf_urls: set[str] = set()
            for landing_url in sorted(landing_pages):
                try:
                    resp = ctx.request.get(landing_url, timeout=60_000)
                    if not resp.ok:
                        logger.debug("fatf landing %d: %s", resp.status, landing_url)
                        continue
                    pdf_urls.update(
                        _extract_pdf_urls_from_landing(resp.text(), landing_url),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("fatf: landing fetch failed %s: %s", landing_url, e)
            logger.info(
                "fatf: %d landing pages resolved to %d PDF URLs",
                len(landing_pages), len(pdf_urls),
            )

            if not pdf_urls:
                raise RuntimeError(
                    "fatf: 0 PDFs resolved from landing pages — site layout "
                    "may have changed; inspect _PDF_REGEX / _LANDING_REGEX",
                )

            # Pass 3: download. Two-stage strategy:
            #   (a) Try the live FATF origin via the warmed Playwright context.
            #       This is fastest and gives the most up-to-date PDF.
            #   (b) After 5 consecutive 403s — the FATF WAF clearly has
            #       us throttled — switch the rest of the batch to the
            #       Internet Archive Wayback Machine, which mirrors most
            #       FATF documents and has no per-IP rate limit on us.
            wayback_session = requests.Session()
            wayback_session.headers.update({
                "User-Agent": (
                    "gsi-training-data-downloader/1.0 "
                    "(+anti-money-laundering research)"
                ),
            })

            ok_live = 0
            ok_wayback = 0
            consecutive_403 = 0
            wayback_mode = False
            for u in sorted(pdf_urls):
                dest = out / _slugify_filename(u)

                if not wayback_mode:
                    try:
                        nbytes = _playwright_download_pdf(ctx, u, dest)
                        if nbytes > 0:
                            ok_live += 1
                        consecutive_403 = 0
                        polite_sleep(1.0)
                        continue
                    except Exception as e:  # noqa: BLE001
                        msg = str(e)
                        logger.warning("fatf: live download failed %s: %s", u, msg)
                        if "403" in msg:
                            consecutive_403 += 1
                            if consecutive_403 >= 5:
                                logger.warning(
                                    "fatf: 5 consecutive 403s from origin — "
                                    "switching remaining %d URLs to Wayback Machine",
                                    len(pdf_urls) - (ok_live + ok_wayback) - 1,
                                )
                                wayback_mode = True

                # Wayback fallback (either we just switched or already in Wayback mode).
                try:
                    nbytes = _wayback_download(u, dest, session=wayback_session)
                    if nbytes > 0:
                        ok_wayback += 1
                    else:
                        logger.warning("fatf: wayback miss for %s", u)
                except Exception as e:  # noqa: BLE001
                    logger.warning("fatf: wayback failed %s: %s", u, e)
                polite_sleep(0.5)

            logger.info(
                "fatf: downloaded %d PDFs total (live=%d, wayback=%d, total_urls=%d)",
                ok_live + ok_wayback, ok_live, ok_wayback, len(pdf_urls),
            )
        finally:
            browser.close()
            pw.stop()

        stats.files_written = sum(1 for p in out.glob("*.pdf") if p.stat().st_size > 0)
        stats.bytes_written = dir_size(out)
        stats.notes["index_pages"] = len(FATF_INDEX_PAGES)
        stats.notes["landing_pages"] = len(landing_pages)
        stats.notes["pdf_urls_resolved"] = len(pdf_urls)


TASKS_CPT_L2 = {"fatf_publications": download_fatf_publications}
ALL_TASKS = dict(TASKS_CPT_L2)
