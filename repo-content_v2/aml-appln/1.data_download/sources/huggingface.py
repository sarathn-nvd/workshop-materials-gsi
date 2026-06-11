"""HuggingFace dataset downloads for CPT and SFT.

Two strategies are used, picked per source:

* **Streaming via ``datasets.load_dataset``** — needed for pile-of-law and
  edgar-corpus, which are massive (100 GB+) and ship as legacy script
  loaders that we filter on the fly. We pin ``datasets<3.0.0`` so the
  ``trust_remote_code`` path keeps working.
* **Direct file download via ``huggingface_hub``** (in :mod:`._hf`) — used
  for every other source. ``datasets`` 2.16+ has a known bug where
  ``dataset_module_factory`` doesn't honor ``Content-Encoding: gzip`` and
  raises ``UnicodeDecodeError: 0x8b`` for some auto-converted repos
  (notably FinQA). Bypassing ``datasets`` entirely also future-proofs us
  against the ``datasets`` 3.x removal of script loaders.

Sources covered:
  CPT Layer 1: pile-of-law/pile-of-law (6 subsets), eloukas/edgar-corpus
  CPT Layer 2: free-law/Caselaw_Access_Project (per-jurisdiction parquet)
  SFT:         Webopen2026/enterprise-financial-crime-ai-dataset,
               Josephgflowers/Finance-Instruct-500k,
               dreamerdeo/finqa, next-tat/TAT-QA,
               PatronusAI/financebench, nguha/legalbench
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from ._common import StatsTracker, dir_size, source_dir
from ._hf import (
    download_hf_files,
    has_parquet_branch,
    iter_parquet_records,
    list_repo_files,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming helpers (used only for pile-of-law and edgar-corpus)
# ---------------------------------------------------------------------------

def _load_dataset(
    repo_id: str,
    config: str | None = None,
    split: str = "train",
    streaming: bool = True,
    **kwargs: Any,
):
    """Thin wrapper around datasets.load_dataset with consistent defaults.

    Used only for pile-of-law and edgar-corpus, which need ``trust_remote_code``
    to execute their builder scripts. SFT datasets bypass this path entirely.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    return load_dataset(
        repo_id,
        config,
        split=split,
        streaming=streaming,
        token=token,
        trust_remote_code=True,
        **kwargs,
    )


def _write_parquet_shards(
    records: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    records_per_shard: int = 50_000,
    prefix: str = "data",
) -> tuple[int, int]:
    """Write an iterable of records to parquet shards. Returns (records, bytes)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    total_records = 0
    total_bytes = 0
    shard_idx = 0
    buffer: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal shard_idx, total_bytes, buffer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        shard_path = out_dir / f"{prefix}-{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        total_bytes += shard_path.stat().st_size
        shard_idx += 1
        buffer = []

    for rec in records:
        buffer.append(rec)
        total_records += 1
        if len(buffer) >= records_per_shard:
            _flush()
    _flush()
    return total_records, total_bytes


def _stream_filtered(
    repo_id: str,
    *,
    config: str | None,
    split: str,
    predicate: Callable[[dict[str, Any]], bool] | None,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    max_records: int | None = None,
) -> tuple[Iterable[dict[str, Any]], dict[str, int]]:
    """Stream an HF dataset, apply a predicate, yield records."""
    ds = _load_dataset(repo_id, config=config, split=split, streaming=True)
    counters = {"seen": 0, "kept": 0, "filtered_out": 0}

    def _iter() -> Iterable[dict[str, Any]]:
        for rec in ds:
            counters["seen"] += 1
            if predicate is None or predicate(rec):
                counters["kept"] += 1
                yield transform(rec) if transform else rec
            else:
                counters["filtered_out"] += 1
            if max_records is not None and counters["kept"] >= max_records:
                break

    return _iter(), counters


# ---------------------------------------------------------------------------
# CPT Layer 1 — Pile of Law subsets (script-based; streaming + filter)
# ---------------------------------------------------------------------------

# Pile of Law schema (verified Apr-2026):
#   {text, created_timestamp, downloaded_timestamp, url}
# The ``url`` field is a single bulk-download zip URL (e.g. govinfo.gov's
# CFR-2020.zip), so it does NOT encode title number per record. The Title
# number is inside the ``text`` body — each record corresponds to one Title:
#
#   CFR record text starts:    "Title 14\n      Aeronautics..."
#   USCode record text starts: "Title 51\nUSCTitle\n51\n..."
#
# We therefore extract the Title from the first ~1000 chars of text using
# a leading-anchor regex. ``federal_register`` and ``oig`` continue to
# discriminate on text content (financial agency / OIG names).

_CFR_TITLES = {"12", "17", "31"}                # Banks, SEC/CFTC, Money & Finance
_USCODE_TITLES = {"12", "18", "31"}              # Banks, Crimes (incl. RICO/AML), Money & Finance
_USCODE_T18_CHAPTERS = {"95", "96"}              # documented for downstream chapter filtering

# Match ``Title 14`` / ``Title 51`` / ``TITLE 18`` etc. — but anchor to the
# *start* of the record. Pile-of-Law packs one Title per record and prints
# the title number on (effectively) the first non-blank line; without
# anchoring we'd misfire on cross-references like "see Title 17 §240" that
# can appear early in a different Title's record.
_TITLE_RE = re.compile(
    r"\A\s*(?:.*?\n)?\s*Title\s+(\d+)\b",
    re.IGNORECASE | re.DOTALL,
)

# Federal-Register / OIG content filters work on the body text rather than
# the URL. We scan a large window so agency mentions that appear after a long
# preamble or table-of-contents still register.
_FILTER_TEXT_WINDOW = 32_000

# Full agency names — matched case-insensitively with word boundaries.
# Kept separate from short acronyms so we can be strict about the acronyms.
_FINANCIAL_AGENCY_PHRASES = (
    "Financial Crimes Enforcement Network",
    "Department of the Treasury", "Treasury Department",
    "Securities and Exchange Commission",
    "Commodity Futures Trading Commission",
    "Federal Deposit Insurance Corporation",
    "Office of the Comptroller of the Currency",
    "Federal Reserve System", "Board of Governors of the Federal Reserve",
    "Federal Reserve Board",
    "Consumer Financial Protection Bureau",
    "National Credit Union Administration",
    "Office of Foreign Assets Control",
    "Bureau of Engraving and Printing",
    "Internal Revenue Service",
    "Bank Secrecy Act",
    "anti-money laundering",
    "suspicious activity report",
    "currency transaction report",
)

# Short acronyms — matched CASE-SENSITIVELY with word boundaries.
# Without case sensitivity "SEC" picks up "section" / "Sec."; without word
# boundaries "BSA" picks up "basal", etc. The combination is what makes the
# filter actually select financial-agency Federal-Register notices rather
# than any issue that happens to mention "sec." in prose.
_FINANCIAL_AGENCY_ACRONYMS = (
    "FinCEN", "SEC", "CFTC", "FDIC", "OCC", "FRB",
    "CFPB", "NCUA", "OFAC", "BSA", "IRS",
)

_FINANCIAL_AGENCY_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _FINANCIAL_AGENCY_PHRASES) + r")\b",
    flags=re.IGNORECASE,
)
_FINANCIAL_AGENCY_ACRONYM_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _FINANCIAL_AGENCY_ACRONYMS) + r")\b",
)

# OIG allow-list — strict. Each entry is an agency-specific phrase. We
# deliberately do NOT include a bare "Office of Inspector General" catch-all:
# every federal OIG uses that boilerplate phrase, so a catch-all defeats the
# filter and pulls in DHS / EPA / HUD / DOD / USDA / Interior / Energy
# inspector-general reports that are off-topic for CPT.
_FINANCIAL_OIG = (
    "Treasury OIG", "Treasury Office of Inspector General",
    "FDIC OIG", "FDIC Office of Inspector General",
    "OCC OIG", "OCC Office of Inspector General",
    "Federal Reserve OIG", "Federal Reserve Office of Inspector General",
    "TIGTA",  # Treasury Inspector General for Tax Administration
    "Treasury Inspector General for Tax Administration",
    "Treasury Inspector General",
    "SEC OIG", "Securities and Exchange Commission Office of Inspector General",
    "CFPB OIG", "Consumer Financial Protection Bureau Office of Inspector General",
    "FHFA OIG", "Federal Housing Finance Agency Office of Inspector General",
    "NCUA OIG", "National Credit Union Administration Office of Inspector General",
)

_FINANCIAL_OIG_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _FINANCIAL_OIG) + r")\b",
    flags=re.IGNORECASE,
)


def _title_from_text(text: str | None, *, head: int = 4000) -> str | None:
    """Extract the leading "Title N" from a Pile-of-Law CFR/USCode record body.

    The relevant Pile-of-Law subsets pack one Title per record; the Title
    number appears at (or very near) the start of the ``text`` field. Using
    a start-anchored regex avoids misfiring on legal cross-references like
    "see Title 17 §240..." that can appear early inside a different Title.
    """
    if not text:
        return None
    m = _TITLE_RE.match(text[:head])
    return m.group(1) if m else None


# revised_strategy.md uses short slugs (``sec``, ``doj_guidance``) but the
# real Pile-of-Law HF configs are named differently. The directory layout we
# write to disk keeps the strategy's slugs (so downstream steps see
# ``pile_of_law_sec/``, ``pile_of_law_doj_guidance/`` as documented), while
# we look up the actual HF builder config via this mapping.
_POL_HF_CONFIG = {
    "sec": "sec_administrative_proceedings",
    "cfr": "cfr",
    "federal_register": "federal_register",
    "uscode": "uscode",
    "oig": "oig",
    "doj_guidance": "doj_guidance_documents",
}


def _pile_of_law_filter_factory(subset: str) -> Callable[[dict[str, Any]], bool]:
    if subset == "sec":
        return lambda _r: True
    if subset == "doj_guidance":
        return lambda _r: True

    if subset == "cfr":
        # Pile-of-Law packs one Title per record; chapter-level filtering for
        # CFR Title 12/17/31 happens at Step 3 (curation).
        def _f(r: dict[str, Any]) -> bool:
            return _title_from_text(r.get("text")) in _CFR_TITLES
        return _f

    if subset == "uscode":
        # Same record granularity as CFR — one Title per record. We keep the
        # full text of Titles 12, 18, 31. The strategy's tighter "Title 18
        # Ch. 95–96 only" cut is enforced at Step 3 since chapters live
        # inside the text body, not as separate records here.
        def _f(r: dict[str, Any]) -> bool:
            return _title_from_text(r.get("text")) in _USCODE_TITLES
        return _f

    if subset == "federal_register":
        def _f(r: dict[str, Any]) -> bool:
            text = (r.get("text") or "")[:_FILTER_TEXT_WINDOW]
            if _FINANCIAL_AGENCY_PHRASE_RE.search(text):
                return True
            if _FINANCIAL_AGENCY_ACRONYM_RE.search(text):
                return True
            return False
        return _f

    if subset == "oig":
        def _f(r: dict[str, Any]) -> bool:
            text = (r.get("text") or "")[:_FILTER_TEXT_WINDOW]
            return bool(_FINANCIAL_OIG_RE.search(text))
        return _f

    raise ValueError(f"Unknown Pile of Law subset: {subset!r}")


_POL_SUBSETS = ("sec", "cfr", "federal_register", "uscode", "oig", "doj_guidance")


def _download_pile_of_law_subset(subset: str, tracker: StatsTracker) -> None:
    with tracker.track(source=f"pile_of_law_{subset}", phase="cpt", layer="level_1") as stats:
        out = source_dir("cpt", f"pile_of_law_{subset}", layer="level_1")
        predicate = _pile_of_law_filter_factory(subset)
        hf_config = _POL_HF_CONFIG.get(subset, subset)
        records, counters = _stream_filtered(
            "pile-of-law/pile-of-law",
            config=hf_config,
            split="train",
            predicate=predicate,
        )
        _write_parquet_shards(records, out, prefix=f"pol_{subset}")
        stats.records_kept = counters["kept"]
        stats.records_filtered_out = counters["filtered_out"]
        stats.files_written = len(list(out.glob("*.parquet")))
        stats.bytes_written = dir_size(out)
        stats.notes["streamed_records"] = counters["seen"]
        stats.notes["hf_config"] = hf_config


def download_pile_of_law(tracker: StatsTracker) -> None:
    """Download all 6 Pile of Law subsets, applying per-subset structural filters."""
    for subset in _POL_SUBSETS:
        _download_pile_of_law_subset(subset, tracker)


# ---------------------------------------------------------------------------
# CPT Layer 1 — EDGAR-CORPUS (script-based; streaming + SIC filter)
# ---------------------------------------------------------------------------

# EDGAR-CORPUS schema (verified Apr-2026):
#   {filename, cik, year, section_1, section_1A, ..., section_15}
# Note: there is **no** ``sic`` field on the records, so the strategy doc's
# "SIC 6000–6999" cut cannot be applied directly during streaming. We use a
# content-based proxy — record's section_1 (Business description) must
# contain at least one financial-sector keyword. This keeps the spirit of
# the strategy ("financial-sector 10-Ks") while removing the dependency on
# an external CIK→SIC lookup.

_EDGAR_FINANCIAL_TERMS = (
    "bank", "banking", "savings institution", "savings and loan", "thrift",
    "credit union", "trust company", "financial holding",
    "insurance", "insurer", "reinsurance",
    "broker-dealer", "broker dealer", "securities firm", "investment bank",
    "investment company", "investment adviser", "investment advisor",
    "asset management", "asset manager", "wealth management",
    "mutual fund", "exchange-traded fund", "hedge fund", "private equity",
    "real estate investment trust", "reit",
    "mortgage", "consumer finance", "commercial finance", "lending",
    "financial services", "financial institution",
)


def _edgar_financial_filter(rec: dict[str, Any]) -> bool:
    """Keep 10-Ks whose Business section mentions a financial-sector activity.

    Proxy for the strategy's "SIC 6000–6999" filter (the field doesn't exist
    in eloukas/edgar-corpus). We scan ``section_1`` (Business) which is the
    primary description of company activities; falls back to ``section_1A``
    (Risk Factors) when section_1 is empty.
    """
    body = rec.get("section_1") or rec.get("section_1A") or ""
    if not body:
        return False
    head = body[:6000].lower()
    return any(term in head for term in _EDGAR_FINANCIAL_TERMS)


def download_edgar_corpus(tracker: StatsTracker) -> None:
    """Download EDGAR-CORPUS filtered to financial-sector 10-Ks.

    Uses a content-based proxy filter (financial-sector keywords in the
    Business section). This replaces the original SIC 6000–6999 filter
    because eloukas/edgar-corpus does not expose a ``sic`` field. Step 3
    can apply a stricter SIC-coded filter via an SEC CIK→SIC lookup if
    needed.
    """
    with tracker.track(source="edgar_corpus", phase="cpt", layer="level_1") as stats:
        out = source_dir("cpt", "edgar_corpus", layer="level_1")
        records, counters = _stream_filtered(
            "eloukas/edgar-corpus",
            config="full",
            split="train",
            predicate=_edgar_financial_filter,
        )
        _write_parquet_shards(records, out, prefix="edgar")
        stats.records_kept = counters["kept"]
        stats.records_filtered_out = counters["filtered_out"]
        stats.files_written = len(list(out.glob("*.parquet")))
        stats.bytes_written = dir_size(out)
        stats.notes["streamed_records"] = counters["seen"]
        stats.notes["filter"] = "content-based financial keywords (no SIC field in source)"


# ---------------------------------------------------------------------------
# CPT Layer 2 — Caselaw Access Project
#
# Strategy: download per-jurisdiction parquet files directly from the Hub
# (the dataset ships as one parquet per jurisdiction on ``main``), then
# stream-filter to AML-relevant opinions and re-shard.
# ---------------------------------------------------------------------------

_CASELAW_KEYWORDS = re.compile(
    r"\b(money\s+laundering|bank\s+secrecy\s+act|structuring|wire\s+fraud|"
    r"RICO|§?\s*5324|§?\s*1956|§?\s*1957|willful\s+blindness)\b",
    re.IGNORECASE,
)

# Per ``revised_strategy.md`` §1.2: federal + NY/CA/FL/TX state. The dataset's
# parquet files are organized by jurisdiction slug at the *path* level
# (one parquet per jurisdiction at ``<slug>/<slug>.parquet``). The slugs we
# need are ``us`` (federal pool: includes 2nd & 9th Circuit federal cases),
# ``ny``, ``cal``, ``florida``, ``tex``.
_CASELAW_TARGET_JURISDICTIONS = ("us", "ny", "cal", "florida", "tex")


def _caselaw_text(rec: dict[str, Any]) -> str:
    body = rec.get("casebody") or rec.get("text") or rec.get("opinion_text") or ""
    if isinstance(body, dict):
        body = body.get("text") or body.get("data") or ""
    return body if isinstance(body, str) else ""


def _caselaw_filter(rec: dict[str, Any]) -> bool:
    text = _caselaw_text(rec)[:8000]
    return bool(_CASELAW_KEYWORDS.search(text))


def download_caselaw_access(tracker: StatsTracker) -> None:
    """Download Caselaw Access Project parquet shards for target jurisdictions.

    The dataset is access-gated: HF returns 403 unless the account behind
    ``HF_TOKEN`` has been granted access. Visit
    https://huggingface.co/datasets/free-law/Caselaw_Access_Project and click
    "Agree and access repository", then re-export ``HF_TOKEN`` and re-run.
    """
    repo_id = "free-law/Caselaw_Access_Project"
    with tracker.track(source="caselaw_access_project", phase="cpt", layer="level_2") as stats:
        out = source_dir("cpt", "caselaw_access_project", layer="level_2")
        raw_dir = out / "raw_parquet"

        try:
            all_files = list_repo_files(repo_id, revision="main")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "403" in msg or "GatedRepoError" in msg:
                logger.error(
                    "caselaw_access_project: HF returned 403. Request access at "
                    "https://huggingface.co/datasets/free-law/Caselaw_Access_Project "
                    "and ensure HF_TOKEN is exported.",
                )
            raise

        target_files = [
            f for f in all_files
            if f.endswith(".parquet")
            and any(f.startswith(f"{j}/") or f.startswith(f"{j}.") for j in _CASELAW_TARGET_JURISDICTIONS)
        ]
        if not target_files:
            target_files = [
                f for f in all_files
                if f.endswith(".parquet")
                and any(j in f.split("/")[0].lower() for j in _CASELAW_TARGET_JURISDICTIONS)
            ]
        logger.info(
            "caselaw: %d/%d parquet files match target jurisdictions",
            len(target_files), sum(1 for f in all_files if f.endswith(".parquet")),
        )

        downloaded, _ = download_hf_files(
            repo_id, raw_dir,
            file_filter=lambda f: f in set(target_files),
            max_workers=4,
        )
        local_parquets = sorted(p for p in raw_dir.rglob("*.parquet") if p.stat().st_size > 0)

        # Stream-filter to AML-relevant opinions and rewrite as our own shards.
        kept = 0
        seen = 0
        def _generate():
            nonlocal kept, seen
            for rec in iter_parquet_records(local_parquets):
                seen += 1
                if _caselaw_filter(rec):
                    kept += 1
                    yield rec

        shards_dir = out / "filtered"
        _write_parquet_shards(_generate(), shards_dir, prefix="caselaw")

        stats.records_kept = kept
        stats.records_filtered_out = max(0, seen - kept)
        stats.files_written = (
            len(local_parquets) + len(list(shards_dir.glob("*.parquet")))
        )
        stats.bytes_written = dir_size(out)
        stats.notes["jurisdictions"] = list(_CASELAW_TARGET_JURISDICTIONS)
        stats.notes["raw_parquet_files"] = len(local_parquets)
        stats.notes["streamed_records"] = seen


# ---------------------------------------------------------------------------
# SFT — HuggingFace-hosted sources (direct file downloads)
# ---------------------------------------------------------------------------

def _download_hf_repo_files(
    tracker: StatsTracker,
    *,
    source_name: str,
    repo_id: str,
    keep_extensions: tuple[str, ...],
    revision: str = "main",
    use_parquet_branch_if_empty: bool = True,
    extra_notes: dict[str, Any] | None = None,
) -> None:
    """Generic SFT downloader: pull every file matching ``keep_extensions``.

    Falls back to the auto-generated parquet branch when ``main`` carries
    only a loader script (e.g. ``dreamerdeo/finqa``).
    """
    with tracker.track(source=source_name, phase="sft") as stats:
        out = source_dir("sft", source_name)

        def _matches(path: str) -> bool:
            low = path.lower()
            return any(low.endswith(ext) for ext in keep_extensions)

        downloaded, _ = download_hf_files(
            repo_id, out, revision=revision, file_filter=_matches,
        )

        used_revision = revision
        # If main had no data files (loader-script-only repo), try parquet branch.
        if not downloaded and use_parquet_branch_if_empty:
            if has_parquet_branch(repo_id):
                logger.info(
                    "%s: no data files at @main; falling back to refs/convert/parquet",
                    source_name,
                )
                downloaded, _ = download_hf_files(
                    repo_id, out,
                    revision="refs/convert/parquet",
                    file_filter=lambda f: f.lower().endswith(".parquet"),
                )
                used_revision = "refs/convert/parquet"

        files = [p for p in out.rglob("*") if p.is_file()]
        stats.files_written = len(files)
        stats.bytes_written = dir_size(out)
        stats.records_kept = stats.files_written  # 1:1 — we don't filter here
        stats.notes["repo_id"] = repo_id
        stats.notes["revision"] = used_revision
        if extra_notes:
            stats.notes.update(extra_notes)


def download_enterprise_financial_crime(tracker: StatsTracker) -> None:
    """310K-record SAR drafting dataset; CSVs + supporting json/jsonl/xlsx/pdf."""
    _download_hf_repo_files(
        tracker,
        source_name="enterprise_financial_crime",
        repo_id="Webopen2026/enterprise-financial-crime-ai-dataset",
        keep_extensions=(".csv", ".json", ".jsonl", ".xlsx", ".pdf", ".txt"),
    )


def download_finance_instruct_500k(tracker: StatsTracker) -> None:
    _download_hf_repo_files(
        tracker,
        source_name="finance_instruct_500k",
        repo_id="Josephgflowers/Finance-Instruct-500k",
        keep_extensions=(".json", ".jsonl", ".parquet", ".csv"),
    )


def download_finqa(tracker: StatsTracker) -> None:
    """FinQA repo carries only a loader script; pulls from the parquet branch."""
    _download_hf_repo_files(
        tracker,
        source_name="finqa",
        repo_id="dreamerdeo/finqa",
        keep_extensions=(".json", ".jsonl", ".parquet"),
    )


def download_tat_qa(tracker: StatsTracker) -> None:
    _download_hf_repo_files(
        tracker,
        source_name="tat_qa",
        repo_id="next-tat/TAT-QA",
        keep_extensions=(".json", ".jsonl", ".parquet"),
    )


def download_financebench(tracker: StatsTracker) -> None:
    _download_hf_repo_files(
        tracker,
        source_name="financebench",
        repo_id="PatronusAI/financebench",
        keep_extensions=(".jsonl", ".json", ".parquet"),
    )


# LegalBench — the 6 logical families in revised_strategy.md §3.4.2 map to
# the following real HF configs on ``nguha/legalbench``. The HF repo ships
# per-subtask TSV files; we grab only the configs we use.
LEGALBENCH_SUBTASKS: tuple[str, ...] = (
    # rule_qa family
    "rule_qa",
    # statutory reasoning (SARA)
    "sara_entailment",
    "sara_numeric",
    # hearsay (single task)
    "hearsay",
    # consumer contracts QA (single task)
    "consumer_contracts_qa",
    # contract_nli family (representative subset)
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_return_of_confidential_information",
    # supply_chain_disclosure family (representative subset)
    "supply_chain_disclosure_disclosed_accountability",
    "supply_chain_disclosure_disclosed_training",
    "supply_chain_disclosure_disclosed_verification",
    "supply_chain_disclosure_best_practice_accountability",
    "supply_chain_disclosure_best_practice_training",
)


def download_legalbench(tracker: StatsTracker) -> None:
    """Download selected LegalBench sub-tasks via direct TSV download.

    LegalBench's HF repo stores per-subtask data under ``data/<subtask>/``
    with files like ``train.tsv`` and ``test.tsv``. We grab every TSV/CSV
    in that subdirectory for each sub-task in :data:`LEGALBENCH_SUBTASKS`.
    """
    repo_id = "nguha/legalbench"
    for subtask in LEGALBENCH_SUBTASKS:
        source_name = f"legalbench__{subtask}"
        with tracker.track(source=source_name, phase="sft") as stats:
            out = source_dir("sft", source_name)
            prefix = f"data/{subtask}/"
            downloaded, _ = download_hf_files(
                repo_id, out,
                file_filter=lambda f, p=prefix: f.startswith(p)
                and f.lower().endswith((".tsv", ".csv", ".json", ".jsonl", ".md")),
                flatten=True,  # write directly into <source>/, not <source>/data/<subtask>/
            )
            files = [p for p in out.rglob("*") if p.is_file()]
            stats.files_written = len(files)
            stats.records_kept = len(files)
            stats.bytes_written = dir_size(out)
            stats.notes["repo_id"] = repo_id
            stats.notes["subtask"] = subtask


# ---------------------------------------------------------------------------
# Public task groups — consumed by download.py
# ---------------------------------------------------------------------------

TASKS_CPT_L1 = {
    "pile_of_law": download_pile_of_law,
    "edgar_corpus": download_edgar_corpus,
}

TASKS_CPT_L2 = {
    "caselaw_access_project": download_caselaw_access,
}

TASKS_SFT = {
    "enterprise_financial_crime": download_enterprise_financial_crime,
    "finance_instruct_500k": download_finance_instruct_500k,
    "finqa": download_finqa,
    "tat_qa": download_tat_qa,
    "financebench": download_financebench,
    "legalbench": download_legalbench,
}

ALL_TASKS: dict[str, Callable[[StatsTracker], None]] = {
    **TASKS_CPT_L1, **TASKS_CPT_L2, **TASKS_SFT,
}
