"""Staged, resumable CPT data curation pipeline.

Reads per-source JSONL produced by Step 2 (`2.data_processing/data/cpt`)
across **both CPT layers jointly**, runs the curation stages prescribed in
`approch.md` Section 6.2, and finally segregates the survivors by their
`layer` field into `<output_dir>/cpt/<layer>/<source>.jsonl`.

Joint dedup (one corpus, dedup once, segregate at the end) is the standard
pattern used by FineWeb / RedPajama / Dolma / NeMo CC -- it ensures the same
content doesn't appear in both L1 and L2, while keeping the curriculum L1->L2
distinction available at training time via the layer-segregated output.

Phases (each is checkpointed via `_work/checkpoint/meta.json`):

  0. INGEST          -- Map `content` -> `text`, preserve `layer`/`source`,
                        union both layers under one staging dir, split large
                        source files into N-record chunks for downstream
                        Ray pipeline parallelism.
  1. TEXT_CLEAN      -- One Curator Pipeline:
                          BYTES_REPR -> HTML_STRIP -> ENGLISH_CLEAN ->
                          LANG (en, score>=0.5) -> LENGTH ->
                          ADD_ID -> QUALITY -> BOILERPLATE -> PII
  2. EXACT_DEDUP     -- SHA-256 of text (Curator workflow)
  3. FUZZY_DEDUP     -- MinHash 128p, Jaccard ~0.8 (Curator workflow)
  4. XSOURCE_DEDUP   -- Targeted source-pair dedup, richness-scored
  5. WRITE_CURATED   -- Group survivors by (layer, source) ->
                        cpt/<layer>/<source>.jsonl

Run:

  python main.py \
      --input_dir  /path/to/2.data_processing/data/cpt \
      --output_dir /path/to/3.cpt/1.data_curation/data

Re-run the same command to resume from the last completed phase.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

from config import PathsConfig, PipelineConfig
from utils import (
    BoilerplateLineRemover,
    CheckpointManager,
    EnglishTextCleaner,
    FixBytesReprModifier,
    HTMLStripper,
    PiiTagRedactor,
    aggregate_pii_audit,
    clear_dir,
    collect_recurring_lines,
    compute_char_and_token_stats,
    consolidate_pii_audits,
    count_records_by_layer_source,
    ensure_dir,
    filter_jsonl_by_id,
    get_logger,
    is_target_lang,
    iter_jsonl,
    length_in_range,
    list_jsonl,
    load_denylists,
    maybe_download,
    write_jsonl,
    xsource_pair_dedup,
)


# ---------------------------------------------------------------------------- #
# Phase helpers                                                                 #
# ---------------------------------------------------------------------------- #


def _stage_dir(paths: PathsConfig, name: str) -> Path:
    return ensure_dir(paths.stages_dir / name)


#: Hard cap on records per chunk in 00_ingest. The downstream Curator
#  `FilePartitioningStage` cannot split a single JSONL file (it groups files
#  by total bytes via `blocksize`, but an individual file is atomic). Without
#  chunking, EDGAR's 13 GB single file becomes ONE serial Ray task that
#  bottlenecks the entire 14-stage pipeline -- 8 GPUs idle, 200 CPU cores
#  idle, ~1 task/6 min. Splitting at the INGEST boundary turns it into ~64
#  parallel tasks and unblocks the rest of the pipeline.
INGEST_RECORDS_PER_CHUNK = 1000


def phase_ingest(paths: PathsConfig, cfg: PipelineConfig, logger) -> None:
    """Read both `<input_dir>/<layer>/*.jsonl` subtrees, map `content` -> `text`,
    tag every record with its `layer`, write into one joint staging dir.

    Files larger than `INGEST_RECORDS_PER_CHUNK` records are split into shards
    named `<layer>__<source>__chunk0001.jsonl, chunk0002.jsonl, ...` so the
    downstream Ray pipeline can parallelise across many tasks. Files at or
    below the threshold keep their original `<layer>__<source>.jsonl` name.
    Per-record `source` is preserved across all shards, so downstream phases
    that key on `source` (boilerplate, xsource dedup, write_curated) see the
    same logical source regardless of how many shards it spans.
    """
    out_dir = _stage_dir(paths, "00_ingest")
    clear_dir(out_dir)
    keep_cols = ("source", "phase", "layer", "doc_id", "element_type", "page", "metadata")

    def _normalise(rec: dict, layer: str) -> dict | None:
        text = rec.get("content") or rec.get("text") or ""
        if not text:
            return None
        out: dict = {cfg.text_field: text}
        for c in keep_cols:
            if c in rec:
                out[c] = rec[c]
        out[cfg.layer_field] = layer  # force-set, overrides Step 2 value if absent
        return out

    grand_in = 0
    grand_out = 0
    files_seen = 0
    chunks_written = 0

    for layer_name in cfg.layer_dirs:
        layer_dir = paths.input_dir / layer_name
        if not layer_dir.exists():
            logger.info(f"INGEST: skipping missing layer dir {layer_dir}")
            continue
        for src_jsonl in list_jsonl(layer_dir):
            files_seen += 1
            local_in = 0
            local_out = 0
            local_chunks = 0
            base_stem = src_jsonl.stem  # e.g., "edgar_corpus"
            buf: list[dict] = []

            def _flush(force_split_naming: bool = False) -> None:
                nonlocal local_out, local_chunks, chunks_written
                if not buf:
                    return
                if force_split_naming or local_chunks > 0:
                    fname = f"{layer_name}__{base_stem}__chunk{local_chunks + 1:04d}.jsonl"
                else:
                    # Single-shard file -- keep the original name without a chunk suffix.
                    fname = f"{layer_name}__{src_jsonl.name}"
                n = write_jsonl(iter(buf), out_dir / fname)
                local_out += n
                local_chunks += 1
                chunks_written += 1
                buf.clear()

            for rec in iter_jsonl(src_jsonl):
                local_in += 1
                norm = _normalise(rec, layer_name)
                if norm is None:
                    continue
                buf.append(norm)
                if len(buf) >= INGEST_RECORDS_PER_CHUNK:
                    _flush(force_split_naming=True)
            _flush(force_split_naming=local_chunks > 0)

            grand_in += local_in
            grand_out += local_out
            chunk_note = f" (split into {local_chunks} chunks)" if local_chunks > 1 else ""
            logger.info(
                f"INGEST {layer_name}/{src_jsonl.name}: read={local_in} wrote={local_out}{chunk_note}"
            )

    if files_seen == 0:
        msg = (
            f"INGEST: no input files found under {paths.input_dir}/(level_1|level_2)/*.jsonl. "
            f"Did Step 2 finish writing the CPT outputs?"
        )
        raise FileNotFoundError(msg)
    logger.info(
        f"INGEST done: {grand_out} records out of {grand_in} input lines from "
        f"{files_seen} files into {chunks_written} shards"
    )


def phase_text_clean(paths: PathsConfig, cfg: PipelineConfig, logger) -> None:
    """One Curator Pipeline: BYTES_REPR -> HTML -> CLEAN -> LANG -> LENGTH ->
    ADD_ID -> QUALITY -> BOILERPLATE -> PII. Reads stages/00_ingest -> stages/01_clean.
    """
    # NOTE: in nemo_curator 1.1.0rc0 (the version inside nvcr.io/nvidia/nemo-curator:26.02),
    # `Filter`, `ScoreFilter`, `Modify`, and `AddId` all live under `stages.text.modules`,
    # and the concrete filter classes are flat under `stages.text.filters` (NOT nested
    # under `.heuristic` / `.fasttext` / `.repetition` subpackages as in the upstream
    # source clone). Don't trust the source clone -- only the container's runtime SDK.
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.filters import (
        CommonEnglishWordsFilter,
        FastTextLangId,
        NonAlphaNumericFilter,
        RepeatedLinesFilter,
    )
    from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.stages.text.modules import AddId, Filter, Modify, ScoreFilter

    in_dir = paths.stages_dir / "00_ingest"
    out_dir = _stage_dir(paths, "01_clean")
    clear_dir(out_dir)

    logger.info("TEXT_CLEAN: ensuring lid.176.bin is present")
    maybe_download(cfg.language_id.model_url, paths.lid_model_path)

    logger.info("TEXT_CLEAN: collecting per-source recurring lines (boilerplate pre-pass)")
    recurring_by_source: dict[str, set[str]] = {}
    # Filenames in 00_ingest are either `<layer>__<source>.jsonl` (unchunked)
    # or `<layer>__<source>__chunk<NNNN>.jsonl` (chunked). The BoilerplateLineRemover
    # keys by record-level `source`, so all chunks of the same source must map back
    # to the same key here. Parse: strip the optional `__chunk<NNNN>` suffix, then
    # the `<layer>__` prefix.
    for src_jsonl in list_jsonl(in_dir):
        stem = src_jsonl.stem
        if "__chunk" in stem:
            stem = stem.rsplit("__chunk", 1)[0]
        source = stem.split("__", 1)[1] if "__" in stem else stem
        rec_set = collect_recurring_lines(
            src_jsonl,
            text_field=cfg.text_field,
            doc_ratio_threshold=cfg.boilerplate.recurring_line_doc_ratio,
            min_chars=cfg.boilerplate.recurring_line_min_chars,
        )
        if rec_set:
            recurring_by_source.setdefault(source, set()).update(rec_set)
            logger.info(f"  {source} ({src_jsonl.name}): {len(rec_set)} recurring lines")

    denylist_path = Path(__file__).parent / cfg.boilerplate.denylist_path
    denylist_by_source = load_denylists(denylist_path)
    logger.info(f"TEXT_CLEAN: denylist sources loaded: {sorted(denylist_by_source)}")

    audit_dir = paths.pii_dir / "_shards"
    audit_dir.mkdir(parents=True, exist_ok=True)
    salt_path = paths.pii_dir / ".salt"
    if not salt_path.exists():
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(os.urandom(16))
        try:
            salt_path.chmod(0o600)
        except OSError:
            pass
    salt = salt_path.read_bytes()

    pipeline = Pipeline(
        name="cpt_text_clean",
        description="BYTES_REPR -> HTML -> CLEAN -> LANG -> LENGTH -> ADD_ID -> QUALITY -> BOILERPLATE -> PII",
        stages=[
            JsonlReader(file_paths=str(in_dir), blocksize=cfg.blocksize),

            # CLEAN -- order matters:
            # FixBytesRepr must run before everything else (some upstream sources,
            # notably pile_of_law_oig, carry text as Python `repr()` of bytes),
            # otherwise the `b'...\n...'` wrapper survives into HTML strip / lid /
            # quality filters and gets the doc dropped as gibberish.
            Modify(modifier_fn=FixBytesReprModifier(), input_fields=cfg.text_field),
            Modify(modifier_fn=HTMLStripper(), input_fields=cfg.text_field),
            Modify(modifier_fn=EnglishTextCleaner(), input_fields=cfg.text_field),

            # LANG -- run lid, then keep only target lang code at score>=cutoff
            ScoreFilter(
                filter_obj=FastTextLangId(
                    model_path=str(paths.lid_model_path),
                    min_langid_score=cfg.language_id.min_langid_score,
                ),
                text_field=cfg.text_field,
                score_field="lang_id",
            ),
            Filter(
                filter_fn=partial(is_target_lang, target_code=cfg.language_id.target_lang_code),
                filter_field="lang_id",
            ),

            # LENGTH
            Filter(
                filter_fn=partial(
                    length_in_range, lo=cfg.length.min_chars, hi=cfg.length.max_chars
                ),
                filter_field=cfg.text_field,
            ),

            # ADD_ID
            AddId(id_field=cfg.id_field, id_prefix=cfg.id_prefix.rstrip("_"), overwrite=True),

            # QUALITY
            ScoreFilter(
                filter_obj=NonAlphaNumericFilter(
                    max_non_alpha_numeric_to_text_ratio=cfg.quality.max_non_alphanumeric_ratio
                ),
                text_field=cfg.text_field,
                score_field="non_alpha_ratio",
            ),
            ScoreFilter(
                filter_obj=RepeatedLinesFilter(
                    max_repeated_line_fraction=cfg.quality.min_unique_line_ratio
                ),
                text_field=cfg.text_field,
                score_field="unique_line_ratio",
            ),
            ScoreFilter(
                filter_obj=CommonEnglishWordsFilter(min_num_common_words=cfg.quality.min_common_words),
                text_field=cfg.text_field,
                score_field="common_word_count",
            ),

            # BOILERPLATE -- multi-input modifier
            Modify(
                modifier_fn=BoilerplateLineRemover(
                    recurring_by_source=recurring_by_source,
                    denylist_by_source=denylist_by_source,
                ),
                input_fields=[[cfg.text_field, cfg.source_field]],
                output_fields=cfg.text_field,
            ),

            # PII -- multi-input modifier (text, source, id)
            Modify(
                modifier_fn=PiiTagRedactor(
                    audit_dir=audit_dir if cfg.pii.write_audit else None,
                    salt=salt,
                    enable_ein=cfg.pii.enable_ein_regex,
                    enable_account_id=cfg.pii.enable_account_id_regex,
                ),
                input_fields=[[cfg.text_field, cfg.source_field, cfg.id_field]],
                output_fields=cfg.text_field,
            ),

            JsonlWriter(path=str(out_dir)),
        ],
    )
    t0 = time.time()
    pipeline.run()
    logger.info(f"TEXT_CLEAN: pipeline finished in {time.time() - t0:.1f}s")

    if cfg.pii.write_audit:
        consolidate_pii_audits(audit_dir, paths.pii_dir)
        logger.info(f"TEXT_CLEAN: PII audits consolidated to {paths.pii_dir}")


def phase_exact_dedup(paths: PathsConfig, cfg: PipelineConfig, logger) -> None:
    """Identify SHA-256 duplicates and write a deduped copy."""
    from nemo_curator.stages.deduplication.exact.workflow import (
        ExactDeduplicationWorkflow,
    )
    from nemo_curator.stages.text.deduplication.removal_workflow import (
        TextDuplicatesRemovalWorkflow,
    )

    in_dir = paths.stages_dir / "01_clean"
    work_dir = ensure_dir(paths.cache_dir / "exact")
    out_dir = _stage_dir(paths, "02_exact")
    clear_dir(out_dir)
    clear_dir(work_dir)

    logger.info("EXACT_DEDUP: identifying duplicates")
    ident = ExactDeduplicationWorkflow(
        output_path=str(work_dir),
        input_path=str(in_dir),
        input_filetype="jsonl",
        text_field=cfg.text_field,
        assign_id=False,
        id_field=cfg.id_field,
    )
    ident.run()

    logger.info("EXACT_DEDUP: removing duplicates -> stages/02_exact")
    removal = TextDuplicatesRemovalWorkflow(
        input_path=str(in_dir),
        ids_to_remove_path=str(work_dir),
        output_path=str(out_dir),
        input_filetype="jsonl",
        output_filetype="jsonl",
        input_id_field=cfg.id_field,
        ids_to_remove_duplicate_id_field=cfg.id_field,
    )
    removal.run()


def phase_fuzzy_dedup(paths: PathsConfig, cfg: PipelineConfig, logger) -> None:
    """Identify near-duplicates via MinHash + LSH, write a deduped copy."""
    from nemo_curator.stages.deduplication.fuzzy.workflow import (
        FuzzyDeduplicationWorkflow,
    )
    from nemo_curator.stages.text.deduplication.removal_workflow import (
        TextDuplicatesRemovalWorkflow,
    )

    in_dir = paths.stages_dir / "02_exact"
    work_dir = ensure_dir(paths.cache_dir / "fuzzy")
    out_dir = _stage_dir(paths, "03_fuzzy")
    clear_dir(out_dir)
    clear_dir(work_dir)

    logger.info("FUZZY_DEDUP: identifying near-duplicates")
    ident = FuzzyDeduplicationWorkflow(
        cache_path=str(work_dir / "cache"),
        output_path=str(work_dir),
        input_path=str(in_dir),
        input_filetype="jsonl",
        input_blocksize=cfg.fuzzy_dedup.input_blocksize,
        text_field=cfg.text_field,
        char_ngrams=cfg.fuzzy_dedup.char_ngrams,
        num_bands=cfg.fuzzy_dedup.num_bands,
        minhashes_per_band=cfg.fuzzy_dedup.minhashes_per_band,
        use_64_bit_hash=cfg.fuzzy_dedup.use_64_bit_hash,
        bands_per_iteration=cfg.fuzzy_dedup.bands_per_iteration,
        seed=cfg.fuzzy_dedup.seed,
    )
    ident.run()

    logger.info("FUZZY_DEDUP: removing near-duplicates -> stages/03_fuzzy")
    # Canonical pattern from the official Curator tutorial
    # (backup/Curator/tutorials/math/5_deduplication.py).
    #
    # Critical contracts (each one was a bug we hit before):
    #   * `ids_to_remove_path` MUST point at the `FuzzyDuplicateIds` SUBFOLDER,
    #     NOT the parent `output_path`. If you give it the parent, pyarrow
    #     scans recursively and schema-unifies with the MinHash/LSH cache
    #     parquets, which have list<int64> columns. The schema merge then
    #     fails with `ArrowNotImplementedError: Unsupported cast from
    #     list<element: int64> to int64`.
    #   * `id_generator_path` lets the JsonlReader re-assign the same
    #     `_curator_dedup_id` to each input record at read time, matching the
    #     IDs the fuzzy workflow used during identification.
    #   * `input_blocksize` MUST match what fuzzy used during identification
    #     (default "1GiB"), otherwise the IdGenerator's batch registry won't
    #     have the key for the new partitioning -> KeyError.
    #   * `ids_to_remove_duplicate_id_field` overrides the SDK's default of
    #     "id" to match what fuzzy actually wrote (`_curator_dedup_id`).
    #
    # Our doc-level `id` column rides through unchanged; it just isn't the
    # match key for fuzzy -- `_curator_dedup_id` is.
    duplicate_ids_path = work_dir / "FuzzyDuplicateIds"
    id_generator_path = work_dir / "fuzzy_id_generator.json"
    removal = TextDuplicatesRemovalWorkflow(
        input_path=str(in_dir),
        ids_to_remove_path=str(duplicate_ids_path),
        output_path=str(out_dir),
        input_filetype="jsonl",
        output_filetype="jsonl",
        input_blocksize=cfg.fuzzy_dedup.input_blocksize,
        id_generator_path=str(id_generator_path),
        ids_to_remove_duplicate_id_field="_curator_dedup_id",
        # input_id_field defaults to "_curator_dedup_id" -- correct here.
    )
    removal.run()


def _resplit_by_source(in_dir: Path, out_dir: Path, source_field: str, logger) -> dict[str, int]:
    """Re-split mixed-source partition files into per-source files."""
    ensure_dir(out_dir)
    clear_dir(out_dir)
    fps: dict[str, "object"] = {}
    counts: dict[str, int] = {}
    try:
        for src_part in sorted(Path(in_dir).rglob("*.jsonl")):
            for rec in iter_jsonl(src_part):
                source = rec.get(source_field) or "_unknown_"
                fp = fps.get(source)
                if fp is None:
                    fp = open(out_dir / f"{source}.jsonl", "a", encoding="utf-8")
                    fps[source] = fp
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[source] = counts.get(source, 0) + 1
    finally:
        for fp in fps.values():
            fp.close()
    for source, n in sorted(counts.items()):
        logger.info(f"  {source}: {n} records")
    return counts


def phase_xsource_dedup(paths: PathsConfig, cfg: PipelineConfig, logger) -> None:
    """Targeted cross-source dedup over configured pairs, richness-scored.

    Operates on per-source files re-split from stages/03_fuzzy.
    """
    in_dir = paths.stages_dir / "03_fuzzy"
    by_source = _stage_dir(paths, "03_fuzzy_by_source")
    clear_dir(by_source)
    logger.info("XSOURCE_DEDUP: re-splitting fuzzy-dedup output by source")
    _resplit_by_source(in_dir, by_source, cfg.source_field, logger)

    out_dir = _stage_dir(paths, "04_xsource")
    clear_dir(out_dir)

    drop_ids: set[str] = set()
    pair_count = 0
    for pair in cfg.xsource_dedup.pairs:
        a = by_source / f"{pair.source_a}.jsonl"
        b = by_source / f"{pair.source_b}.jsonl"
        if not (a.exists() and b.exists()):
            logger.info(
                f"XSOURCE_DEDUP: skipping pair ({pair.source_a}, {pair.source_b}) -- missing"
            )
            continue
        pair_drops = xsource_pair_dedup(
            source_a_jsonl=a,
            source_b_jsonl=b,
            text_field=cfg.text_field,
            id_field=cfg.id_field,
            weights=pair.richness_weights,
        )
        logger.info(
            f"XSOURCE_DEDUP: pair ({pair.source_a}, {pair.source_b}) -> dropping {len(pair_drops)} losers"
        )
        drop_ids.update(pair_drops)
        pair_count += 1

    logger.info(f"XSOURCE_DEDUP: total dropped across {pair_count} pairs: {len(drop_ids)}")

    for src_jsonl in sorted(by_source.glob("*.jsonl")):
        kept, dropped = filter_jsonl_by_id(
            in_path=src_jsonl,
            out_path=out_dir / src_jsonl.name,
            id_field=cfg.id_field,
            drop_ids=drop_ids,
        )
        logger.info(f"  {src_jsonl.name}: kept={kept} dropped={dropped}")


def phase_write_curated(paths: PathsConfig, cfg: PipelineConfig, logger) -> dict[str, dict[str, int]]:
    """Group survivors of stages/04_xsource by (layer, source) and write to
    `cpt/<layer>/<source>.jsonl`. The layer field rides through every phase
    untouched, so each record self-routes to the correct output bucket.
    """
    in_dir = paths.stages_dir / "04_xsource"
    ensure_dir(paths.curated_dir)
    clear_dir(paths.curated_dir)

    counts: dict[tuple[str, str], int] = {}
    fps: dict[tuple[str, str], object] = {}
    try:
        for src_jsonl in sorted(in_dir.glob("*.jsonl")):
            for rec in iter_jsonl(src_jsonl):
                layer = str(rec.get(cfg.layer_field) or "_unknown_")
                source = str(rec.get(cfg.source_field) or "_unknown_")
                key = (layer, source)
                fp = fps.get(key)
                if fp is None:
                    out_path = paths.curated_dir / layer / f"{source}.jsonl"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fp = open(out_path, "w", encoding="utf-8")
                    fps[key] = fp
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[key] = counts.get(key, 0) + 1
    finally:
        for fp in fps.values():
            fp.close()  # type: ignore[attr-defined]

    summary: dict[str, dict[str, int]] = {}
    for (layer, source), n in sorted(counts.items()):
        logger.info(f"WRITE_CURATED  {layer}/{source}: {n} records")
        summary.setdefault(layer, {})[source] = n
    return summary


# ---------------------------------------------------------------------------- #
# Stats helpers (drive the rich summary.json)                                   #
# ---------------------------------------------------------------------------- #


def _phase_output_dirs(paths: PathsConfig) -> dict[str, Path]:
    return {
        "INGEST": paths.stages_dir / "00_ingest",
        "TEXT_CLEAN": paths.stages_dir / "01_clean",
        "EXACT_DEDUP": paths.stages_dir / "02_exact",
        "FUZZY_DEDUP": paths.stages_dir / "03_fuzzy",
        "XSOURCE_DEDUP": paths.stages_dir / "04_xsource",
    }


def _load_summary(paths: PathsConfig) -> dict:
    if paths.summary_path.exists():
        try:
            return json.loads(paths.summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_summary(paths: PathsConfig, summary: dict) -> None:
    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _record_phase_counts(
    phase_name: str,
    paths: PathsConfig,
    cfg: PipelineConfig,
    logger,
) -> None:
    """After a phase completes, count records by (layer, source) in its output
    directory and merge into `summary.json` under `by_layer.<layer>.<source>.phase_counts`.
    """
    out_dir = _phase_output_dirs(paths).get(phase_name)
    if out_dir is None or not out_dir.exists():
        return
    counts = count_records_by_layer_source(out_dir, cfg.layer_field, cfg.source_field)
    summary = _load_summary(paths)
    by_layer = summary.setdefault("by_layer", {})
    for layer, src_map in counts.items():
        for source, n in src_map.items():
            entry = by_layer.setdefault(layer, {}).setdefault(source, {})
            entry.setdefault("phase_counts", {})[phase_name] = n
    _save_summary(paths, summary)
    total = sum(n for layer in counts.values() for n in layer.values())
    logger.info(
        f"[stats] {phase_name}: {total} records across "
        f"{sum(len(v) for v in counts.values())} (layer,source) buckets"
    )


def build_summary(paths: PathsConfig, cfg: PipelineConfig, logger) -> dict:
    """Enrich `summary.json` with chars, tokens, PII tallies, and layer totals.

    Idempotent: re-running just refreshes the chars/tokens/PII sections plus the
    derived totals on top of whatever phase_counts have been recorded so far.
    """
    summary = _load_summary(paths)
    by_layer: dict[str, dict[str, dict]] = summary.setdefault("by_layer", {})

    logger.info("[summary] computing char + token counts on curated output")
    char_token = compute_char_and_token_stats(
        curated_dir=paths.curated_dir,
        text_field=cfg.text_field,
        layer_field=cfg.layer_field,
        source_field=cfg.source_field,
        tokenizer_id=cfg.tokenizer.model_id if cfg.tokenizer.enabled else None,
        hf_token_env=cfg.tokenizer.hf_token_env,
        logger=logger,
        workers=cfg.tokenizer.workers,
    )
    layers_seen: set[str] = set(by_layer)
    for layer, src_map in char_token.items():
        layers_seen.add(layer)
        for source, info in src_map.items():
            entry = by_layer.setdefault(layer, {}).setdefault(source, {})
            entry["chars"] = info["chars"]
            entry["tokens"] = info["tokens"]

    logger.info("[summary] aggregating PII redaction tallies")
    pii_by_source = aggregate_pii_audit(paths.pii_dir)
    for source, ent_counts in pii_by_source.items():
        for layer in layers_seen:
            entry = by_layer.get(layer, {}).get(source)
            if entry is not None:
                entry["pii_redactions"] = ent_counts

    layer_totals: dict[str, dict] = {}
    for layer, src_map in by_layer.items():
        chars = sum(int(e.get("chars", 0) or 0) for e in src_map.values())
        tokens_vals = [e.get("tokens") for e in src_map.values() if e.get("tokens") is not None]
        tokens = sum(tokens_vals) if tokens_vals else None
        final_recs = sum(
            int((e.get("phase_counts", {}) or {}).get("XSOURCE_DEDUP", 0) or 0)
            for e in src_map.values()
        )
        layer_totals[layer] = {
            "final_records": final_recs,
            "chars": chars,
            "tokens": tokens,
            "sources": len(src_map),
        }

    summary["tokenizer"] = cfg.tokenizer.model_id if cfg.tokenizer.enabled else None
    summary["layer_totals"] = layer_totals
    _save_summary(paths, summary)
    logger.info(f"[summary] -> {paths.summary_path}")

    for layer in sorted(layer_totals):
        t = layer_totals[layer]
        tok_str = f"{t['tokens']:,} tokens" if t["tokens"] is not None else "tokens=N/A"
        logger.info(
            f"[summary] {layer}: {t['final_records']:,} records | "
            f"{t['chars']:,} chars | {tok_str} | sources={t['sources']}"
        )
    return summary


# ---------------------------------------------------------------------------- #
# Orchestrator                                                                  #
# ---------------------------------------------------------------------------- #

def run(paths: PathsConfig, cfg: PipelineConfig, logger) -> None:
    ensure_dir(paths.output_dir)
    ensure_dir(paths.work_dir)
    ensure_dir(paths.stages_dir)
    ensure_dir(paths.curated_dir)
    ensure_dir(paths.checkpoint_dir)
    ensure_dir(paths.cache_dir)
    ensure_dir(paths.pii_dir)

    ckpt = CheckpointManager(paths.checkpoint_dir)

    phases = [
        ("INGEST", lambda: phase_ingest(paths, cfg, logger)),
        ("TEXT_CLEAN", lambda: phase_text_clean(paths, cfg, logger)),
        ("EXACT_DEDUP", lambda: phase_exact_dedup(paths, cfg, logger)),
        ("FUZZY_DEDUP", lambda: phase_fuzzy_dedup(paths, cfg, logger)),
        ("XSOURCE_DEDUP", lambda: phase_xsource_dedup(paths, cfg, logger)),
    ]

    for name, fn in phases:
        if ckpt.is_done(name):
            logger.info(f"[skip] {name} already complete")
            continue
        logger.info(f"[start] {name}")
        t0 = time.time()
        fn()
        ckpt.mark_done(name)
        logger.info(f"[done]  {name} in {time.time() - t0:.1f}s")
        try:
            _record_phase_counts(name, paths, cfg, logger)
        except Exception as e:
            logger.warning(f"[stats] {name}: failed to record counts: {e}")

    if not ckpt.is_done("WRITE_CURATED"):
        logger.info("[start] WRITE_CURATED")
        t0 = time.time()
        phase_write_curated(paths, cfg, logger)
        ckpt.mark_done("WRITE_CURATED")
        logger.info(f"[done]  WRITE_CURATED in {time.time() - t0:.1f}s")
    else:
        logger.info("[skip] WRITE_CURATED already complete")

    # Always (re)build the final summary -- it's idempotent and cheap apart
    # from the optional tokenizer pass.
    build_summary(paths, cfg, logger)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPT Data Curation Pipeline (joint dedup, layer-segregated output)")
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Parent of `level_1/` and `level_2/` (e.g. .../data/cpt). Both subdirs are read jointly.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write `cpt/<layer>/<source>.jsonl` and `_work/` intermediates.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Wipe checkpoint and start over",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if "/data/sft" in str(args.input_dir).replace("\\", "/"):
        msg = "Refusing to run: this pipeline is CPT-only; pass an input_dir under data/cpt/."
        raise SystemExit(msg)

    paths = PathsConfig(input_dir=args.input_dir, output_dir=args.output_dir)
    cfg = PipelineConfig()
    logger = get_logger(paths.log_path)
    logger.info("CPT data curation: joint dedup over both layers")
    logger.info(f"  input_dir  = {paths.input_dir}")
    logger.info(f"  output_dir = {paths.output_dir}")
    logger.info(f"  layer dirs = {list(cfg.layer_dirs)}")

    if args.reset:
        CheckpointManager(paths.checkpoint_dir).reset()
        logger.info("Checkpoint reset.")

    from nemo_curator.core.client import RayClient

    ray_client = RayClient()
    ray_client.start()
    try:
        run(paths, cfg, logger)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
