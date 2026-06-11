"""CourtListener — CPT Layer 2 (alternative to gated Caselaw Access Project).

Why CourtListener
-----------------
The free-law/Caselaw_Access_Project HF dataset is access-gated and the
approval workflow is opaque. CourtListener is run by the same organization
(Free Law Project) and exposes the *same* underlying opinion corpus through
a public REST API. We use the API to materialize a focused AML-jurisprudence
sub-corpus without waiting on HF approval.

Coverage strategy
-----------------
We run several statute / keyword queries that map to the strategy doc's
"federal districts + 2nd/9th Circuit + NY/CA/FL/TX state" jurisdiction filter
and the AML keyword set::

    money laundering, Bank Secrecy Act, structuring, wire fraud,
    RICO, 5324, 1956, 1957, willful blindness

For each query we paginate the search API up to ``MAX_PAGES`` (20 results
per page), then for each result fetch the full opinion text from the
``opinions`` endpoint. Results are deduplicated by opinion ``id``. Output is
written as parquet shards under
``data/raw/cpt/level_2/courtlistener/`` mirroring the schema we'd have
gotten from Caselaw Access Project (text + jurisdiction + court + year).

Authentication
--------------
CourtListener allows anonymous calls but at a low rate limit. For a real
pull set ``COURTLISTENER_TOKEN`` in the environment to a personal API
token (free, sign up at https://www.courtlistener.com/sign-up/). The token
is sent as ``Authorization: Token <token>``.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from ._common import StatsTracker, dir_size, source_dir

logger = logging.getLogger(__name__)

API_BASE = "https://www.courtlistener.com/api/rest/v4"
SEARCH_URL = f"{API_BASE}/search/"
OPINION_URL_TMPL = f"{API_BASE}/opinions/{{id}}/"

# Statute / keyword queries — each is a separate call; we union the results.
# We use phrase queries (quoted) to keep precision high; legal text is dense
# enough that bare substring searches would surface a lot of noise.
QUERIES = (
    '"money laundering"',
    '"Bank Secrecy Act"',
    '"structuring"',
    '"willful blindness"',
    '"18 U.S.C. § 1956"',
    '"18 U.S.C. § 1957"',
    '"31 U.S.C. § 5324"',
    'RICO "money laundering"',
)

# Cap the per-query and total result counts so a runaway query doesn't burn
# the rate limit. With 8 queries and 20 pages of 20 results each, the upper
# bound is ~3,200 unique opinions — comfortably within a free-tier daily
# call budget.
MAX_PAGES_PER_QUERY = 20
PAGE_SIZE = 20
MAX_TOTAL_OPINIONS = 5_000


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    token = os.environ.get("COURTLISTENER_TOKEN")
    if token:
        s.headers.update({"Authorization": f"Token {token}"})
        logger.info("courtlistener: using authenticated API token")
    else:
        logger.warning(
            "courtlistener: COURTLISTENER_TOKEN not set — anonymous calls "
            "will be rate-limited; consider setting it for a full pull",
        )
    s.headers.update({
        "User-Agent": "gsi-training-data-downloader/1.0 (+anti-money-laundering research)",
        "Accept": "application/json",
    })
    return s


def _get_with_retry(session: requests.Session, url: str, params: dict | None = None,
                    *, max_retries: int = 4) -> dict | None:
    """GET with exponential backoff on 429 / 5xx."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=60)
        except Exception as e:  # noqa: BLE001
            logger.warning("courtlistener: %s failed (%s); attempt %d", url, e, attempt + 1)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                logger.warning("courtlistener: non-JSON 200 from %s", url)
                return None
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", 0) or (2 ** attempt))
            logger.warning(
                "courtlistener: HTTP %d on %s (retry %d, sleep %ds)",
                r.status_code, url, attempt + 1, wait,
            )
            time.sleep(wait)
            continue
        logger.warning(
            "courtlistener: HTTP %d on %s — body: %s",
            r.status_code, url, r.text[:160],
        )
        return None
    return None


# ---------------------------------------------------------------------------
# Search and opinion-detail fetch
# ---------------------------------------------------------------------------

def _search_opinion_ids(session: requests.Session) -> list[int]:
    """Run every QUERY, paginate, return the union of cluster IDs.

    The v4 search API returns ``cluster_id`` per hit (a "cluster" is a court
    decision; opinions are individual sub-documents inside a cluster). We
    track cluster IDs and resolve to opinion text per-cluster afterward.
    """
    seen: set[int] = set()
    for q in QUERIES:
        logger.info("courtlistener: search %r", q)
        for page in range(1, MAX_PAGES_PER_QUERY + 1):
            params = {
                "type": "o",  # opinions
                "q": q,
                "order_by": "score desc",
                "page_size": PAGE_SIZE,
                "page": page,
            }
            payload = _get_with_retry(session, SEARCH_URL, params=params)
            if not payload:
                break
            results = payload.get("results", [])
            if not results:
                break
            for hit in results:
                cid = hit.get("cluster_id") or hit.get("id")
                if isinstance(cid, int):
                    seen.add(cid)
            if len(seen) >= MAX_TOTAL_OPINIONS:
                logger.info(
                    "courtlistener: hit MAX_TOTAL_OPINIONS=%d; stopping search",
                    MAX_TOTAL_OPINIONS,
                )
                return sorted(seen)
            if not payload.get("next"):
                break
            time.sleep(0.3)
    return sorted(seen)


_OPINION_TEXT_FIELDS = ("plain_text", "html_with_citations", "html", "html_lawbox", "xml_harvard")


def _opinion_to_record(payload: dict) -> dict[str, Any] | None:
    """Pick the best available text representation and emit a flat record."""
    text = ""
    for f in _OPINION_TEXT_FIELDS:
        v = payload.get(f) or ""
        if v:
            text = v
            break
    if not text or len(text) < 200:
        return None
    # Strip HTML tags if we picked an HTML field (cheap; we're not parsing
    # citation structure here — Step 3 curation handles that).
    if "<" in text and ">" in text:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return {
        "id": payload.get("id"),
        "cluster_id": payload.get("cluster"),
        "absolute_url": payload.get("absolute_url"),
        "date_filed": payload.get("date_filed") or payload.get("date_created"),
        "type": payload.get("type"),
        "author_str": payload.get("author_str"),
        "text": text,
    }


def _resolve_clusters_to_records(
    session: requests.Session,
    cluster_ids: Iterable[int],
) -> Iterable[dict[str, Any]]:
    """Yield one flat record per opinion in the listed clusters.

    A cluster's ``sub_opinions`` field lists opinion-detail URLs we can
    fetch directly. We yield records lazily so the caller can shard them
    on the fly without holding everything in RAM.
    """
    cluster_url_tmpl = f"{API_BASE}/clusters/{{id}}/"
    seen_opinion_ids: set[int] = set()
    for cid in cluster_ids:
        cluster = _get_with_retry(session, cluster_url_tmpl.format(id=cid))
        if not cluster:
            continue
        court = cluster.get("court_id") or ""
        case_name = cluster.get("case_name") or cluster.get("case_name_short") or ""
        date_filed = cluster.get("date_filed") or ""
        sub_opinions = cluster.get("sub_opinions") or []
        for opinion_url in sub_opinions:
            payload = _get_with_retry(session, opinion_url)
            if not payload:
                continue
            rec = _opinion_to_record(payload)
            if rec is None:
                continue
            oid = rec.get("id")
            if isinstance(oid, int):
                if oid in seen_opinion_ids:
                    continue
                seen_opinion_ids.add(oid)
            rec["court"] = court
            rec["case_name"] = case_name
            rec["cluster_date_filed"] = date_filed
            yield rec
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Parquet shard writer (vendored to avoid an import cycle with huggingface.py)
# ---------------------------------------------------------------------------

def _write_parquet_shards(
    records: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    records_per_shard: int = 5_000,
    prefix: str = "courtlistener",
) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    bytes_w = 0
    shard_idx = 0
    buf: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal shard_idx, bytes_w, buf
        if not buf:
            return
        path = out_dir / f"{prefix}-{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf), path, compression="zstd")
        bytes_w += path.stat().st_size
        shard_idx += 1
        buf = []

    for rec in records:
        buf.append(rec)
        total += 1
        if len(buf) >= records_per_shard:
            _flush()
    _flush()
    return total, bytes_w


# ---------------------------------------------------------------------------
# Public task
# ---------------------------------------------------------------------------

def download_courtlistener(tracker: StatsTracker) -> None:
    """Download AML-relevant opinions from CourtListener and shard to parquet."""
    with tracker.track(source="courtlistener", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "courtlistener", layer="level_2")
        session = _session()

        cluster_ids = _search_opinion_ids(session)
        logger.info("courtlistener: %d unique clusters to resolve", len(cluster_ids))
        if not cluster_ids:
            stats.notes["clusters_discovered"] = 0
            return

        records, bytes_w = _write_parquet_shards(
            _resolve_clusters_to_records(session, cluster_ids), out,
        )
        stats.records_kept = records
        stats.files_written = len(list(out.glob("*.parquet")))
        stats.bytes_written = dir_size(out)
        stats.notes["clusters_discovered"] = len(cluster_ids)
        stats.notes["queries"] = list(QUERIES)


TASKS_CPT_L2 = {"courtlistener": download_courtlistener}
ALL_TASKS = dict(TASKS_CPT_L2)
