"""Configuration for the CPT data curation pipeline.

All knobs live here as dataclasses. The pipeline runs **once over the union of
both CPT layers** (joint dedup is the industry-standard pattern --
FineWeb / RedPajama / Dolma / Llama / NeMo CC all do this); the final
`WRITE_CURATED` phase is the only stage that knows about the layer split,
which it reads from each record's `layer` field (preserved from Step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LanguageIdConfig:
    model_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_filename: str = "lid.176.bin"
    target_lang_code: str = "EN"
    min_langid_score: float = 0.50


@dataclass
class LengthConfig:
    min_chars: int = 200
    # `max_chars=None` disables the upper bound. Default-disabled because our
    # CPT corpus includes whole-statute documents (CFR Titles, U.S. Code Titles,
    # uscode_house HTML renders, FATF publications) that are 2-10 MB each --
    # exactly the foundational AML statutes the model needs to ground SAR
    # citations in. A 1 M-char cap dropped them 100%. Set to an int to re-enable.
    max_chars: int | None = None


@dataclass
class QualityConfig:
    """§6.2 step 4. Single set of thresholds across both layers (joint dedup).

    Tuned to the looser end of the §6.2 range so L1's ~24x larger corpus
    isn't over-pruned. Curriculum-driven L1-vs-L2 tightening, if needed
    later, applies as a per-layer post-filter on `cpt/<layer>/<source>.jsonl`
    (separate concern from corpus-level curation).
    """

    max_non_alphanumeric_ratio: float = 0.30
    min_unique_line_ratio: float = 0.70
    min_common_words: int = 2


@dataclass
class BoilerplateConfig:
    # Lines seen in >= recurring_line_doc_ratio of a source's documents are stripped.
    recurring_line_doc_ratio: float = 0.01
    # Minimum line character length to be eligible for the recurring-line denylist
    # (avoids stripping ubiquitous short tokens like "Page 1").
    recurring_line_min_chars: int = 20
    # YAML file with per-source hand-curated denylists.
    denylist_path: str = "boilerplate/denylists.yaml"


@dataclass
class PiiConfig:
    """Typed-tag PII redaction. Tags follow `approch.md` §6.2 step 6."""

    # Presidio entity types -> replacement tag. Order matters (longest match wins inside Presidio).
    entity_tags: dict[str, str] = field(
        default_factory=lambda: {
            "US_SSN": "[SSN]",
            "CREDIT_CARD": "[CREDIT_CARD]",
            "IBAN_CODE": "[IBAN]",
            "PHONE_NUMBER": "[PHONE]",
            "EMAIL_ADDRESS": "[EMAIL]",
            # PERSON is gated per-source via `person_recognizer_sources`. Tag is uniform.
            "PERSON": "[PERSON]",
        }
    )
    # Sources where Presidio's PERSON recogniser is enabled. Off by default on
    # legal/regulatory text (case citations, public officials, statutes acting
    # in official capacity are not PII and should not be redacted).
    person_recognizer_sources: list[str] = field(
        default_factory=lambda: [
            "fincen_files",
            "cfpb_complaints",
        ]
    )
    # Always-on regex passes (Presidio under-recalls these in financial text).
    enable_ein_regex: bool = True
    enable_account_id_regex: bool = True
    # Audit side-file with hashed-only spans, per source. Set to False to disable.
    write_audit: bool = True


@dataclass
class TokenizerConfig:
    """Tokenizer for the final char + token summary. Matches Step 2's tokenizer
    so token counts compare apples-to-apples across the two pipelines.
    Skipped silently with a warning if HF auth or `transformers` is unavailable.

    Tokenization is **file-level parallel** via `multiprocessing.Pool` -- each
    worker loads its own tokenizer instance and processes whole curated JSONL
    files independently. Single-threaded encoding takes 45-80 min on this
    corpus; with 32 workers it drops to 5-10 min (bottleneck = largest file).
    """

    model_id: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    hf_token_env: str = "HF_TOKEN"
    enabled: bool = True
    # Parallel workers for the tokenizer pass. None -> use os.cpu_count() // 2.
    workers: int | None = 32


@dataclass
class FuzzyDedupConfig:
    """MinHash + LSH per `approch.md` §6.2 step 7b: 128 perms, Jaccard ~0.80."""

    char_ngrams: int = 24
    # 16 bands * 8 minhashes = 128 permutations; LSH threshold ~ (1/16)^(1/8) ~= 0.79
    num_bands: int = 16
    minhashes_per_band: int = 8
    use_64_bit_hash: bool = False
    bands_per_iteration: int = 4
    seed: int = 42
    # Must match the input_blocksize used by the FuzzyDeduplicationWorkflow's
    # FilePartitioningStage (default "1GiB") so the IdGenerator's batch
    # registry stays consistent between identification and removal.
    input_blocksize: str = "1GiB"


@dataclass
class XSourceDedupPair:
    source_a: str
    source_b: str
    # Weights for richness scoring: (length, unique_section_headings, citation_density).
    richness_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


@dataclass
class XSourceDedupConfig:
    """Targeted cross-source dedup over configured CPT pairs.

    For each pair, exact-match on a normalized SHA-256 hash; on collision, keep
    the doc with the higher richness score and drop the rest. Operates on the
    per-source files re-split from the global FUZZY output -- both `source_a`
    and `source_b` may sit in either CPT layer; only the `source` name matters.
    """

    pairs: list[XSourceDedupPair] = field(
        default_factory=lambda: [
            XSourceDedupPair("pile_of_law_uscode", "uscode_house"),
            XSourceDedupPair("pile_of_law_cfr", "derived_cfr_31_X"),
        ]
    )


@dataclass
class PathsConfig:
    """Single-run layout. Final output mirrors Step 2's `cpt/<layer>/<source>.jsonl`
    shape so downstream consumers can swap the path prefix
    `2.data_processing/data/cpt/...` for
    `3.cpt/1.data_curation/data/cpt/...` with no other change.

    Layout under `output_dir`::

        cpt/<layer>/<source>.jsonl       <-- final curated output (= Step 2 shape)
        _work/stages/...                 <-- per-phase intermediate JSONL (joint, both layers)
        _work/checkpoint/meta.json       <-- resume marker
        _work/cache/...                  <-- exact + fuzzy dedup workflow caches
        _work/pii/...                    <-- PII audit shards + per-source audit
        _work/models/lid.176.bin         <-- FastText language-id model
        _work/log_data_curator.txt
        _work/summary.json               <-- single rich summary (per-phase counts +
                                              chars + tokens + PII + layer totals)
    """

    input_dir: Path
    output_dir: Path

    # Derived
    work_dir: Path = field(init=False)
    stages_dir: Path = field(init=False)
    curated_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    pii_dir: Path = field(init=False)
    log_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    lid_model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()

        self.work_dir = self.output_dir / "_work"
        self.stages_dir = self.work_dir / "stages"
        self.curated_dir = self.output_dir / "cpt"
        self.checkpoint_dir = self.work_dir / "checkpoint"
        self.cache_dir = self.work_dir / "cache"
        self.pii_dir = self.work_dir / "pii"
        self.log_path = self.work_dir / "log_data_curator.txt"
        self.summary_path = self.work_dir / "summary.json"
        self.lid_model_path = self.work_dir / "models" / "lid.176.bin"


@dataclass
class PipelineConfig:
    language_id: LanguageIdConfig = field(default_factory=LanguageIdConfig)
    length: LengthConfig = field(default_factory=LengthConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    boilerplate: BoilerplateConfig = field(default_factory=BoilerplateConfig)
    pii: PiiConfig = field(default_factory=PiiConfig)
    fuzzy_dedup: FuzzyDedupConfig = field(default_factory=FuzzyDedupConfig)
    xsource_dedup: XSourceDedupConfig = field(default_factory=XSourceDedupConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    id_prefix: str = "cpt_"
    text_field: str = "text"
    source_field: str = "source"
    layer_field: str = "layer"
    id_field: str = "id"

    # JSONL read blocksize for the linear text-clean pipeline (Ray FilePartitioning hint).
    blocksize: str = "256MiB"

    # Layer subdirectories under the input directory to read jointly.
    layer_dirs: tuple[str, ...] = ("level_1", "level_2")
