"""Partition the curated CPT corpus into the two training-phase folders.

Reads curated JSONL produced by `../1.data_curation/` (mirrored into
`../data/raw/{level_1,level_2}/<source>.jsonl`) and emits two folders that
each training phase consumes directly:

    data/final/
    +-- level_1/<source>.jsonl     <-- Phase 1 trains on this (1 epoch)
    +-- level_2/<source>.jsonl     <-- Phase 2 trains on this (3-4 epochs)

What goes where (`../README.md` Section 3 design, simplified per the latest
scoping decision):

  * EDGAR splits 87% to level_1 / 13% to level_2 (override with --edgar_l2_pct).
    The level_2 share is the L1-replay slice that protects Phase 2 against
    catastrophic forgetting -- Phase 2 sees those EDGAR documents for the
    first time, mixed in with the natural L2 corpus, so the replay signal
    stays strong rather than being re-shown content the model already
    memorized in Phase 1. The 13% target sizes the slice so Phase 2's per-epoch
    token budget (replay + native L2 upsampled) lands at the README §3.3
    design point of ~540 M effective tokens / epoch.

  * The seven small L1 sources (pile_of_law_*, uscode_house) all go 100% to
    level_1. Their combined ~340 M tokens is too small to spare a replay
    share; losing any of it would hurt Phase 1's broad-register coverage more
    than it would help Phase 2's anti-forgetting signal.

  * All L2 sources (FinCEN / OFAC / FATF / courtlistener) go 100% to level_2.

  * `enterprise_financial_crime` is dropped (1 record / ~400 tokens after
    curation; below useful threshold).

Two contracts the partition enforces:

  1. DETERMINISTIC + DISJOINT. Routing is a deterministic hash on the
     document id salted by `--seed`, so re-running on the same input is
     bit-for-bit reproducible and an EDGAR document never lands in both
     level_1 and level_2.

  2. PER-DOC 5% CAP. If the largest document in a source exceeds 5% of that
     source's total characters, every long document is paragraph-aware
     chunked into pieces of at most 5% of the source. Each chunk gets a
     fresh hash and is routed independently. Without this cap, the two giant
     `uscode_house` HTML renders, the two `pile_of_law_uscode` records, and
     the few `fatf_publications` PDFs would each dominate their entire source.

Run:

    python split_pools.py \
        --input_dir  ../data/raw \
        --output_dir ../data/final \
        [--seed 42] [--workers 8] [--no_chunk_cap]

Re-running with the same `--seed` is idempotent: the output is byte-for-byte
identical and the existing files are overwritten.

What this script does NOT do (intentionally):
  * No tokenization or .bin/.idx packing -- handled downstream.
  * No document shuffling -- handled at the tokenizer / dataloader stage.
  * No train/val holdout split -- handled at training time via the NeMo
    dataloader's `split:` arg (the reference recipe used "0.95, 0.05, 0.0").
  * No L2 upsampling for Phase 2 -- handled at training time via dataloader
    blend weights, not by physically duplicating records here.
  * No length / quality / dedup / PII filters -- the curator already did all
    of those; we trust the input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bucket space [0, BUCKET_MOD). Larger -> finer-grained percentages; 1000
# is enough for the smallest slice we currently use (1%).
BUCKET_MOD = 1000

# Per-doc cap as a fraction of the source's total character count. A document
# (or paragraph-chunked sub-document) larger than this is split before bucket
# assignment. Matches `../README.md` Section 3.1 / planning todo.
PER_DOC_CAP_FRAC = 0.05

# Pool name -> on-disk subdirectory under output_dir. Pool names are arbitrary
# labels used internally by the routing table; the only thing downstream sees
# is the directory layout.
POOL_TO_DIR: dict[str, str] = {
    "level_1": "level_1",
    "level_2": "level_2",
}

# Per-source bucket ranges. Each range is half-open [lo, hi); a doc whose
# bucket falls in a range routes to that pool. Ranges within a source must be
# disjoint and cover [0, BUCKET_MOD); a missing range means "drop".
#
# EDGAR is the only source split across both pools; its cutoff is overridable
# via --edgar_l2_pct (default 13%). Concretely, with the default cutoff EDGAR
# contributes ~2.45 B tokens to level_1 and ~370 M tokens to level_2 -- the
# latter is the Phase-2 L1-replay slice that, mixed with the upsampled native
# L2, gives Phase 2 ~540 M effective tokens / epoch (README §3.3).
DEFAULT_EDGAR_L2_PCT = 13  # 13% of EDGAR docs go to level_2; rest go to level_1


def build_routing(edgar_l2_pct: float) -> dict[str, dict[str, tuple[int, int]]]:
    """Build the per-source bucket routing table.

    `edgar_l2_pct` is the share of EDGAR documents (by count, not by tokens)
    that should land in level_2 as the Phase-2 L1-replay slice. Default 13%
    sizes the slice so Phase 2's per-epoch budget hits the README §3.3 design
    point of ~540 M effective tokens / epoch (after dataloader-side L2
    upsampling).
    """
    if not (0.0 <= edgar_l2_pct <= 100.0):
        raise ValueError(f"edgar_l2_pct must be in [0, 100], got {edgar_l2_pct}")
    edgar_cut = int(round(BUCKET_MOD * (100.0 - edgar_l2_pct) / 100.0))
    return {
        # --- L1 ---
        "edgar_corpus": {
            "level_1": (0, edgar_cut),                 # (100 - edgar_l2_pct)% -> Phase 1
            "level_2": (edgar_cut, BUCKET_MOD),        # edgar_l2_pct%         -> Phase 2 L1-replay
        },
        "pile_of_law_oig":              {"level_1": (0, BUCKET_MOD)},
        "pile_of_law_federal_register": {"level_1": (0, BUCKET_MOD)},
        "pile_of_law_sec":              {"level_1": (0, BUCKET_MOD)},
        "pile_of_law_cfr":              {"level_1": (0, BUCKET_MOD)},
        "uscode_house":                 {"level_1": (0, BUCKET_MOD)},
        "pile_of_law_doj_guidance":     {"level_1": (0, BUCKET_MOD)},
        "pile_of_law_uscode":           {"level_1": (0, BUCKET_MOD)},
        # --- L2 ---
        "ofac_guidance":                {"level_2": (0, BUCKET_MOD)},
        "fincen_federal_register":      {"level_2": (0, BUCKET_MOD)},
        "fincen_sar_reviews":           {"level_2": (0, BUCKET_MOD)},
        "fincen_advisories":            {"level_2": (0, BUCKET_MOD)},
        "fincen_enforcement":           {"level_2": (0, BUCKET_MOD)},
        "courtlistener":                {"level_2": (0, BUCKET_MOD)},
        "fatf_publications":            {"level_2": (0, BUCKET_MOD)},
        "fincen_files":                 {"level_2": (0, BUCKET_MOD)},
        # --- Explicit drops ---
        "enterprise_financial_crime":   {},  # 1 record / ~400 tokens after curation
    }


# Default routing instance. Rebuilt with the CLI value in main() and passed
# explicitly to each worker so subprocess spawn semantics don't matter.
ROUTING: dict[str, dict[str, tuple[int, int]]] = build_routing(DEFAULT_EDGAR_L2_PCT)


# ---------------------------------------------------------------------------
# Per-source stats containers
# ---------------------------------------------------------------------------


@dataclass
class PoolCounts:
    docs: int = 0
    chars: int = 0


@dataclass
class SourceResult:
    source: str
    layer: str
    input_path: str
    in_records: int = 0
    in_chars: int = 0
    max_doc_chars: int = 0
    chunk_max_chars: int | None = None
    chunks_emitted: int = 0
    docs_dropped_no_route: int = 0
    pool_counts: dict[str, PoolCounts] = field(default_factory=dict)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Hashing + bucket assignment
# ---------------------------------------------------------------------------


def stable_bucket(seed: int, key: str, mod: int = BUCKET_MOD) -> int:
    """Deterministic bucket in [0, mod) for `key` salted by `seed`.

    Uses BLAKE2b (fast, fixed digest, stdlib) to avoid the per-interpreter
    randomization that affects Python's built-in hash().
    """
    h = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % mod


def route_bucket(bucket: int, source_routing: dict[str, tuple[int, int]]) -> str | None:
    for pool_name, (lo, hi) in source_routing.items():
        if lo <= bucket < hi:
            return pool_name
    return None


# ---------------------------------------------------------------------------
# Per-doc 5% cap chunking (paragraph-aware, char-fallback for runaway paragraphs)
# ---------------------------------------------------------------------------


def chunk_document(text: str, doc_id: str, max_chars: int) -> Iterator[tuple[str, str]]:
    """Yield (chunk_id, chunk_text) tuples each <= max_chars characters.

    Splits on the most semantically meaningful boundary that fits the budget:
    paragraphs first ('\\n\\n'), then a hard char split for any single
    paragraph that itself exceeds max_chars (rare; happens with the giant
    `uscode_house` HTML renders that contain unbroken statute blocks).

    chunk_id is `f"{doc_id}#chunk_{i}"`, ensuring each chunk routes
    independently while staying traceable to its parent.
    """
    if len(text) <= max_chars:
        yield (doc_id, text)
        return

    paragraphs = text.split("\n\n")
    buf: list[str] = []
    buf_len = 0
    chunk_idx = 0

    def flush() -> Iterator[tuple[str, str]]:
        nonlocal buf, buf_len, chunk_idx
        if buf:
            yield (f"{doc_id}#chunk_{chunk_idx}", "\n\n".join(buf))
            chunk_idx += 1
            buf, buf_len = [], 0

    for p in paragraphs:
        plen = len(p) + 2  # account for the rejoined '\n\n' delimiter

        if plen > max_chars:
            # Single paragraph blows the budget. Drain whatever we have buffered
            # first, then char-split the runaway paragraph into max_chars pieces.
            yield from flush()
            for i in range(0, len(p), max_chars):
                yield (f"{doc_id}#chunk_{chunk_idx}", p[i:i + max_chars])
                chunk_idx += 1
            continue

        if buf_len + plen > max_chars:
            yield from flush()

        buf.append(p)
        buf_len += plen

    yield from flush()


# ---------------------------------------------------------------------------
# JSONL streaming + record helpers
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterable[dict]:
    """Stream-parse a JSONL file. Skips blank lines; logs and skips bad JSON."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning("[%s] skipped malformed line %d: %s", path.name, lineno, e)


def get_doc_id(rec: dict, fallback_text: str) -> str:
    """Stable per-record key for routing.

    Prefer the curator's `id` field (always present post-curation), fall back
    to `doc_id`, and as a last resort hash the text itself so every record
    still gets a unique stable key.
    """
    for k in ("id", "doc_id"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    return "sha256:" + hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Per-source worker
# ---------------------------------------------------------------------------


def process_source(
    source: str,
    layer: str,
    input_path: Path,
    output_dir: Path,
    seed: int,
    apply_chunk_cap: bool,
    routing_table: dict[str, dict[str, tuple[int, int]]],
) -> SourceResult:
    """Two-pass partition for one source file.

    `routing_table` is the full per-source ROUTING dict produced by
    build_routing(); we pass it in explicitly so the main process's CLI-tuned
    edgar_l2_pct cutoff applies even under spawn-style multiprocessing where
    the worker re-imports the module and would otherwise see DEFAULT_EDGAR_L2_PCT.

    Pass 1 (cheap, streaming): compute total chars + max doc chars so we know
    if the per-doc cap will trigger and what the chunk budget should be.

    Pass 2: route each doc (chunking if needed), append to the appropriate
    pool's per-source output file. Output files are pre-opened and held open
    for the duration of the source pass so we don't repeatedly fopen/close on
    every record.
    """
    t0 = time.time()
    routing = routing_table.get(source, {})
    result = SourceResult(source=source, layer=layer, input_path=str(input_path))

    # ---- Pass 1: stats (streaming, no record buffering) ----
    for rec in iter_jsonl(input_path):
        text = rec.get("text", "") or ""
        n = len(text)
        result.in_chars += n
        if n > result.max_doc_chars:
            result.max_doc_chars = n
        result.in_records += 1

    if result.in_records == 0:
        result.duration_s = time.time() - t0
        return result

    if not routing:
        logging.info("[%s] DROPPED per ROUTING (in_records=%d)", source, result.in_records)
        result.duration_s = time.time() - t0
        return result

    # Decide chunking: only if some doc actually exceeds 5% of source total.
    if apply_chunk_cap:
        cap = int(PER_DOC_CAP_FRAC * result.in_chars)
        if cap > 0 and result.max_doc_chars > cap:
            result.chunk_max_chars = cap

    # ---- Pass 2: route + write ----
    writers: dict[str, "open"] = {}
    try:
        for pool_name in routing:
            out_path = output_dir / POOL_TO_DIR[pool_name] / f"{source}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writers[pool_name] = out_path.open("w", encoding="utf-8")
            result.pool_counts[pool_name] = PoolCounts()

        for rec in iter_jsonl(input_path):
            text = rec.get("text", "") or ""
            if not text:
                continue
            doc_id = get_doc_id(rec, text)

            chunks: Iterable[tuple[str, str]]
            if result.chunk_max_chars is None:
                chunks = ((doc_id, text),)
            else:
                chunks = chunk_document(text, doc_id, result.chunk_max_chars)

            for chunk_id, chunk_text in chunks:
                bucket = stable_bucket(seed, chunk_id)
                pool = route_bucket(bucket, routing)
                if pool is None:
                    result.docs_dropped_no_route += 1
                    continue

                # Minimal training-shape record. Strip everything the
                # tokenizer does not need (lang_id, dedup id, non_alpha_ratio,
                # element_type, page, metadata, ...). `source` and `id` are
                # kept for downstream auditing and shard-level traceability.
                out_rec = {"text": chunk_text, "source": source, "id": chunk_id}
                writers[pool].write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                pc = result.pool_counts[pool]
                pc.docs += 1
                pc.chars += len(chunk_text)
                result.chunks_emitted += 1
    finally:
        for w in writers.values():
            w.close()

    result.duration_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Discovery + orchestration
# ---------------------------------------------------------------------------


def discover_sources(
    input_dir: Path,
    routing_table: dict[str, dict[str, tuple[int, int]]],
) -> list[tuple[str, str, Path]]:
    """Return [(source, layer, input_path), ...] for everything under input_dir."""
    found: list[tuple[str, str, Path]] = []
    for layer in ("level_1", "level_2"):
        layer_dir = input_dir / layer
        if not layer_dir.is_dir():
            logging.warning("layer dir missing: %s", layer_dir)
            continue
        for p in sorted(layer_dir.glob("*.jsonl")):
            source = p.stem
            if source not in routing_table:
                logging.warning(
                    "[%s] no ROUTING entry for source -- skipping. Add it to build_routing() "
                    "(or to the explicit-drop set with `{}`) to silence this.",
                    source,
                )
                continue
            found.append((source, layer, p))
    return found


def write_summary(
    output_dir: Path,
    results: list[SourceResult],
    seed: int,
    apply_chunk_cap: bool,
    edgar_l2_pct: float,
) -> Path:
    """Write the human-readable + machine-readable partition summary."""
    pool_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"docs": 0, "chars": 0, "sources_with_records": 0}
    )
    per_source: list[dict] = []
    for r in results:
        per_source.append(
            {
                "source": r.source,
                "layer": r.layer,
                "input_path": r.input_path,
                "in_records": r.in_records,
                "in_chars": r.in_chars,
                "max_doc_chars": r.max_doc_chars,
                "chunk_max_chars": r.chunk_max_chars,
                "chunks_emitted": r.chunks_emitted,
                "docs_dropped_no_route": r.docs_dropped_no_route,
                "duration_s": round(r.duration_s, 2),
                "pools": {p: {"docs": pc.docs, "chars": pc.chars} for p, pc in r.pool_counts.items()},
            }
        )
        for pool_name, pc in r.pool_counts.items():
            pt = pool_totals[pool_name]
            pt["docs"] += pc.docs
            pt["chars"] += pc.chars
            if pc.docs > 0:
                pt["sources_with_records"] += 1

    pool_totals_out = {
        pool: {"subdir": POOL_TO_DIR[pool], **vals} for pool, vals in sorted(pool_totals.items())
    }

    summary = {
        "schema_version": "1.0",
        "seed": seed,
        "bucket_mod": BUCKET_MOD,
        "edgar_l2_pct": edgar_l2_pct,
        "per_doc_cap_frac": PER_DOC_CAP_FRAC if apply_chunk_cap else None,
        "pool_layout": POOL_TO_DIR,
        "pool_totals": pool_totals_out,
        "per_source": per_source,
    }
    out_path = output_dir / "split_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    return out_path


def fmt_n(n: int | None) -> str:
    return "n/a" if n is None else f"{n:,}"


def log_pool_totals(results: list[SourceResult]) -> None:
    pool_totals: dict[str, PoolCounts] = defaultdict(PoolCounts)
    for r in results:
        for pool, pc in r.pool_counts.items():
            pool_totals[pool].docs += pc.docs
            pool_totals[pool].chars += pc.chars

    logging.info("---- pool totals ----")
    for pool in ("level_1", "level_2"):
        pc = pool_totals.get(pool)
        if pc is None:
            continue
        logging.info(
            "  %-10s -> %-10s | docs=%10s | chars=%14s",
            pool, POOL_TO_DIR[pool], fmt_n(pc.docs), fmt_n(pc.chars),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "split_pools.log"
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    logging.info("log file: %s", log_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Curated raw root with level_1/ and level_2/ subdirs of <source>.jsonl files.")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Final partition output root (creates level_1/, level_2/ subdirs).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Hash salt for deterministic per-doc bucket assignment (default: 42).")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)),
                    help="Per-source parallel worker processes (default: min(8, cpu_count)).")
    ap.add_argument("--no_chunk_cap", action="store_true",
                    help="Disable the per-doc 5%% cap chunking. Off-by-default; only use if "
                         "you want raw documents passed through untouched.")
    ap.add_argument("--edgar_l2_pct", type=float, default=DEFAULT_EDGAR_L2_PCT,
                    help=f"Share of EDGAR docs (by count, deterministic-hash) to route to level_2 "
                         f"as the Phase-2 L1-replay slice. Default: {DEFAULT_EDGAR_L2_PCT}%%.")
    args = ap.parse_args()

    setup_logging(args.output_dir)
    apply_chunk_cap = not args.no_chunk_cap
    routing_table = build_routing(args.edgar_l2_pct)
    edgar_cut = routing_table["edgar_corpus"]["level_1"][1]
    logging.info(
        "seed=%d workers=%d chunk_cap=%s edgar_l2_pct=%.2f%% (bucket cut=%d/%d) "
        "input_dir=%s output_dir=%s",
        args.seed, args.workers, "on" if apply_chunk_cap else "off",
        args.edgar_l2_pct, edgar_cut, BUCKET_MOD,
        args.input_dir, args.output_dir,
    )

    sources = discover_sources(args.input_dir, routing_table)
    if not sources:
        logging.error("no sources discovered under %s", args.input_dir)
        return 1
    logging.info("discovered %d source files", len(sources))

    t_start = time.time()
    results: list[SourceResult] = []

    if args.workers <= 1 or len(sources) == 1:
        for source, layer, path in sources:
            logging.info("[start] %-32s (%s, %.1f MB)",
                         source, layer, path.stat().st_size / (1 << 20))
            r = process_source(source, layer, path, args.output_dir, args.seed,
                               apply_chunk_cap, routing_table)
            log_source_done(r)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_source, source, layer, path, args.output_dir,
                            args.seed, apply_chunk_cap, routing_table): (source, layer, path)
                for source, layer, path in sources
            }
            for source, layer, path in sources:
                logging.info("[queued] %-32s (%s, %.1f MB)",
                             source, layer, path.stat().st_size / (1 << 20))
            for fut in as_completed(futures):
                source, layer, path = futures[fut]
                try:
                    r = fut.result()
                except Exception:
                    logging.exception("[FAIL] %s (%s)", source, layer)
                    raise
                log_source_done(r)
                results.append(r)

    results.sort(key=lambda r: (0 if r.layer == "level_1" else 1, r.source))

    summary_path = write_summary(args.output_dir, results, args.seed, apply_chunk_cap,
                                 args.edgar_l2_pct)
    logging.info("wrote summary: %s", summary_path)
    log_pool_totals(results)
    logging.info("done in %.1fs", time.time() - t_start)
    return 0


def log_source_done(r: SourceResult) -> None:
    cap_note = (
        f"chunk_max={fmt_n(r.chunk_max_chars)}c (max_doc={fmt_n(r.max_doc_chars)}c)"
        if r.chunk_max_chars is not None
        else f"no chunking (max_doc={fmt_n(r.max_doc_chars)}c)"
    )
    if not r.pool_counts:
        logging.info(
            "[done ] %-32s in_records=%-8s in_chars=%-14s | DROPPED",
            r.source, fmt_n(r.in_records), fmt_n(r.in_chars),
        )
        return
    pool_summary = " ".join(
        f"{p}={fmt_n(pc.docs)}d/{fmt_n(pc.chars)}c"
        for p, pc in sorted(r.pool_counts.items())
        if pc.docs > 0
    )
    logging.info(
        "[done ] %-32s in_records=%-8s in_chars=%-14s | %s | %s | %.1fs",
        r.source, fmt_n(r.in_records), fmt_n(r.in_chars),
        cap_note, pool_summary, r.duration_s,
    )


if __name__ == "__main__":
    sys.exit(main())
