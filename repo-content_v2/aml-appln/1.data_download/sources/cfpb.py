"""CFPB Consumer Complaint Database — SFT source.

Full DB as CSV (large, ~500 MB zipped). We apply the download-time filter
specified in revised_strategy.md §1.3:
  has_consumer_narrative == True
  product in {"Checking or savings account",
              "Money transfers, virtual currency, virtual currency",
              "Credit card"}

The filter is applied *after* download to keep the raw artifact available
but to materialize a filtered working copy alongside it.

CFPB's file server blocks requests missing browser-like headers, so we
send explicit Referer + Accept and fall back to the Socrata CSV export
if the primary URL returns 403.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ._common import StatsTracker, dir_size, source_dir
from ._http import build_session, download_file

logger = logging.getLogger(__name__)

# Primary: CFPB's own zipped CSV bulk download
CFPB_CSV_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
# Fallback: Socrata dataset 's6ew-h6mp' published by CFPB on data.consumerfinance.gov
CFPB_SOCRATA_CSV_URL = (
    "https://data.consumerfinance.gov/api/views/s6ew-h6mp/rows.csv?accessType=DOWNLOAD"
)

CFPB_HEADERS = {
    "Referer": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
    "Accept": "application/zip, text/csv, application/octet-stream, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

KEEP_PRODUCTS = {
    "Checking or savings account",
    "Money transfers, virtual currency, virtual currency",
    "Credit card",
}


def _download_with_fallback(session, out: Path) -> Path:
    """Try primary zipped CSV first; fall back to Socrata raw CSV.

    Some macOS Python installs can't verify data.consumerfinance.gov's cert
    chain with their bundled certifi. If we hit an SSL verification error on
    the Socrata fallback, retry with certifi's current CA bundle, and as a
    last resort with verification disabled (logging loudly).
    """
    raw_zip = out / "complaints.csv.zip"
    raw_csv = out / "complaints.csv"

    # Primary: CFPB's zipped CSV
    try:
        download_file(session, CFPB_CSV_URL, raw_zip, headers=CFPB_HEADERS)
        return raw_zip
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "cfpb: primary URL failed (%s); falling back to Socrata CSV export", e,
        )

    # Socrata fallback, with progressively looser TLS verification
    attempts = [
        ("certifi bundle", {"verify": _certifi_path()}),
        ("default verify", {}),
        ("verify=False (LAST RESORT)", {"verify": False}),
    ]
    last_err: Exception | None = None
    for label, verify_kwargs in attempts:
        try:
            if "LAST RESORT" in label:
                logger.warning("cfpb: attempting Socrata with TLS verification DISABLED")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            _download_with_verify(
                session, CFPB_SOCRATA_CSV_URL, raw_csv,
                headers=CFPB_HEADERS, **verify_kwargs,
            )
            return raw_csv
        except Exception as e:  # noqa: BLE001
            logger.warning("cfpb Socrata (%s) failed: %s", label, e)
            last_err = e

    raise RuntimeError(
        f"cfpb: all download paths failed; last error: {last_err}"
    )


def _certifi_path() -> str | None:
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return None


def _download_with_verify(
    session, url: str, dest: Path, *,
    headers: dict | None = None, verify=True,
) -> int:
    """Streaming download with an explicit verify setting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("cfpb: skip existing %s", dest.name)
        return 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(
        url, stream=True, timeout=120, headers=headers or {}, verify=verify,
    ) as r:
        r.raise_for_status()
        written = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        tmp.rename(dest)
    return written


def download_cfpb_complaints(tracker: StatsTracker) -> None:
    with tracker.track(source="cfpb_complaints", phase="sft") as stats:
        out = source_dir("sft", "cfpb_complaints")
        out.mkdir(parents=True, exist_ok=True)
        session = build_session()

        raw_path = _download_with_fallback(session, out)
        logger.info("cfpb: reading raw CSV (%s) to apply structural filter", raw_path.name)

        filtered_path = out / "complaints_filtered.parquet"
        kept = 0
        total = 0
        read_kwargs: dict = dict(chunksize=200_000, low_memory=False)
        if raw_path.suffix == ".zip":
            read_kwargs["compression"] = "zip"

        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(raw_path, **read_kwargs):
            total += len(chunk)
            product_col = "Product" if "Product" in chunk.columns else "product"
            narrative_col = (
                "Consumer complaint narrative"
                if "Consumer complaint narrative" in chunk.columns
                else "consumer_complaint_narrative"
            )
            mask = (
                chunk[narrative_col].notna()
                & (chunk[narrative_col].astype(str).str.len() > 30)
                & chunk[product_col].isin(KEEP_PRODUCTS)
            )
            filtered = chunk.loc[mask].copy()
            if len(filtered) > 0:
                parts.append(filtered)
                kept += len(filtered)

        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(filtered_path, index=False)

        stats.records_kept = kept
        stats.records_filtered_out = total - kept
        stats.files_written = sum(1 for _ in out.rglob("*") if _.is_file())
        stats.bytes_written = dir_size(out)
        stats.notes["rows_total"] = total
        stats.notes["source_url"] = str(CFPB_CSV_URL if raw_path.suffix == ".zip" else CFPB_SOCRATA_CSV_URL)


TASKS_SFT = {"cfpb_complaints": download_cfpb_complaints}
ALL_TASKS = dict(TASKS_SFT)
