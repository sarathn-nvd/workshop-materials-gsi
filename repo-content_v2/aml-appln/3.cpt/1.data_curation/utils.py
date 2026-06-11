"""Helpers + custom modifiers for the CPT data curation pipeline.

Kept deliberately small. Curator SDK 26.02 (Ray-based) provides the heavy
machinery (filters, dedup workflows, IO). This module only adds:

- A `CheckpointManager` so the staged pipeline is resumable.
- Custom `DocumentModifier`s for bytes-repr decode, HTML strip, English text
  cleanup, boilerplate line removal, and typed-tag PII redaction.
- Top-level filter callables for LANG / LENGTH (named functions pickle cleanly
  for Ray workers).
- A pre-pass utility `collect_recurring_lines(...)` used to seed the boilerplate
  modifier with per-source recurring-line denylists.
- A small `xsource_pair_dedup(...)` helper that picks the richer doc when two
  configured CPT sources contain the same content.
- Stats helpers used by `main.build_summary` to populate the rich `summary.json`.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import unicodedata
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier

from pii_patterns import build_regex_replacements


# ---------------------------------------------------------------------------- #
# Logging                                                                       #
# ---------------------------------------------------------------------------- #

def get_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cpt_curator")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------- #
# Checkpointing                                                                 #
# ---------------------------------------------------------------------------- #

PHASES = [
    "INGEST",
    "TEXT_CLEAN",
    "EXACT_DEDUP",
    "FUZZY_DEDUP",
    "XSOURCE_DEDUP",
    "WRITE_CURATED",
]


@dataclass
class CheckpointManager:
    checkpoint_dir: Path

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.checkpoint_dir / "meta.json"

    def _load(self) -> dict:
        if not self.meta_path.exists():
            return {"completed": []}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"completed": []}

    def is_done(self, phase: str) -> bool:
        return phase in self._load().get("completed", [])

    def mark_done(self, phase: str) -> None:
        meta = self._load()
        completed = list(meta.get("completed", []))
        if phase not in completed:
            completed.append(phase)
        self.meta_path.write_text(json.dumps({"completed": completed}, indent=2), encoding="utf-8")

    def reset(self) -> None:
        if self.meta_path.exists():
            self.meta_path.unlink()


# ---------------------------------------------------------------------------- #
# Filesystem helpers                                                            #
# ---------------------------------------------------------------------------- #

def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def clear_dir(path: Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def maybe_download(url: str, out_path: Path) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(out_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def iter_jsonl(path: Path):
    """Yield records from a (possibly gzipped) JSONL file. Skips malformed lines."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(records: Iterable[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def list_jsonl(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).glob("*.jsonl") if p.is_file())


def count_records(path: Path) -> int:
    n = 0
    for _ in iter_jsonl(path):
        n += 1
    return n


# ---------------------------------------------------------------------------- #
# Lang + length filter helpers (top-level so they pickle for Ray workers)       #
# ---------------------------------------------------------------------------- #


def is_target_lang(lang_id_value: object, target_code: str = "EN") -> bool:
    """Parse `FastTextLangId.score_document` output and return True iff the
    detected language code matches `target_code`.

    `FastTextLangId.score_document` returns either a `[score, lang_code]` list
    or its string repr (e.g. "[0.99, 'EN']").
    """
    if isinstance(lang_id_value, (list, tuple)) and len(lang_id_value) == 2:
        return str(lang_id_value[1]).upper() == target_code.upper()
    if isinstance(lang_id_value, str):
        try:
            parsed = ast.literal_eval(lang_id_value)
        except Exception:
            return target_code.upper() in lang_id_value.upper()
        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
            return str(parsed[1]).upper() == target_code.upper()
    return False


def length_in_range(text: object, lo: int, hi: int | None) -> bool:
    """Length filter. `hi=None` disables the upper bound (useful for whole-statute
    corpora where individual documents can be 5-10 MB and shouldn't be dropped).
    """
    n = len(text or "")  # type: ignore[arg-type]
    if n < lo:
        return False
    if hi is not None and n > hi:
        return False
    return True


# ---------------------------------------------------------------------------- #
# Custom DocumentModifier: bytes-repr decoder                                   #
# ---------------------------------------------------------------------------- #

class FixBytesReprModifier(DocumentModifier):
    """Decode strings that are the Python ``repr()`` of a bytes object back to
    real UTF-8 text.

    Some upstream parquet files (notably ``pile_of_law_oig``) carry text columns
    that were serialized as ``str(some_bytes)`` instead of ``some_bytes.decode()``,
    so the JSONL we receive contains literal ``b'...\\n...'`` strings: every
    record arrives with a leading ``b'`` wrapper, two-character ``\\n``
    sequences instead of real newlines, and dense ``\\xNN`` escape soup.

    Detection is conservative: a string only qualifies if it starts with
    ``b'`` / ``b"`` and ends with the matching quote, AND parses cleanly via
    :func:`ast.literal_eval` to a ``bytes`` object. Anything else passes through
    untouched. Decoding uses ``errors='replace'`` so a partial mojibake doesn't
    crash the pipeline.

    Must run **before** :class:`HTMLStripper`, language ID, and any heuristic
    filter -- otherwise the wrapper / escape sequences confuse downstream
    stages (lid would classify these docs as gibberish and drop them).
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "fix_bytes_repr"

    @staticmethod
    def _looks_like_bytes_repr(s: str) -> bool:
        if len(s) < 3:
            return False
        if not (s.startswith("b'") or s.startswith('b"')):
            return False
        return s.endswith(s[1])

    def modify_document(self, text: str) -> str:
        if not text or not self._looks_like_bytes_repr(text):
            return text
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if not isinstance(parsed, bytes):
            return text
        return parsed.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------- #
# Custom DocumentModifier: HTML strip                                           #
# ---------------------------------------------------------------------------- #

class HTMLStripper(DocumentModifier):
    """Strips HTML tags and decodes a few common entities."""

    _tag_re = re.compile(r"<[^>]+>")
    _entity_re = re.compile(r"&(amp|lt|gt|quot|nbsp|#\d+);")

    def __init__(self) -> None:
        super().__init__()
        self._name = "html_stripper"

    def modify_document(self, text: str) -> str:
        if not text:
            return ""
        text = self._tag_re.sub(" ", text)
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&nbsp;", " ")
        )
        text = self._entity_re.sub(" ", text)
        return text


# ---------------------------------------------------------------------------- #
# Custom DocumentModifier: English text cleanup                                 #
# ---------------------------------------------------------------------------- #

class EnglishTextCleaner(DocumentModifier):
    """NFC normalize + collapse whitespace + drop control chars + CRLF -> LF.

    Conservative: leaves URLs, citations, numbers in place. Step-2 already gave
    us reasonably clean text; this is a light final pass.
    """

    _ws_re = re.compile(r"[ \t\f\v]+")
    _multi_newline_re = re.compile(r"\n{3,}")
    _control_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self) -> None:
        super().__init__()
        self._name = "english_text_cleaner"

    def modify_document(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        # Normalize CRLF / CR -> LF before any other whitespace handling, otherwise
        # \r survives (it's intentionally excluded from `_control_re`) and pollutes
        # downstream line-based matchers (boilerplate, recurring-line collector).
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._control_re.sub(" ", text)
        text = self._ws_re.sub(" ", text)
        text = self._multi_newline_re.sub("\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------- #
# Recurring-line pre-pass + boilerplate modifier                                #
# ---------------------------------------------------------------------------- #

def collect_recurring_lines(
    source_jsonl: Path,
    text_field: str,
    doc_ratio_threshold: float,
    min_chars: int,
) -> set[str]:
    """Return lines seen in >= `doc_ratio_threshold` of docs in this source.

    Streams the file once: for each doc, dedup lines within the doc, then
    increment a per-line doc-count. Lines crossing the threshold get returned.
    """
    total_docs = 0
    line_doc_count: Counter[str] = Counter()
    for rec in iter_jsonl(source_jsonl):
        total_docs += 1
        text = rec.get(text_field) or ""
        seen_in_doc: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if len(line) < min_chars:
                continue
            if line in seen_in_doc:
                continue
            seen_in_doc.add(line)
        for line in seen_in_doc:
            line_doc_count[line] += 1
    if total_docs == 0:
        return set()
    threshold = max(2, int(total_docs * doc_ratio_threshold))
    return {line for line, n in line_doc_count.items() if n >= threshold}


def load_denylists(yaml_path: Path) -> dict[str, list[str]]:
    p = Path(yaml_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, list[str]] = {}
    for source, lines in data.items():
        if not isinstance(lines, list):
            continue
        out[source] = [str(line).strip() for line in lines if str(line).strip()]
    return out


class BoilerplateLineRemover(DocumentModifier):
    """Strips both per-source recurring lines and YAML denylist lines.

    Multi-input modifier: needs `text` and `source` columns at call time.
    """

    def __init__(
        self,
        recurring_by_source: dict[str, set[str]],
        denylist_by_source: dict[str, list[str]],
    ) -> None:
        super().__init__()
        self._name = "boilerplate_line_remover"
        # Pre-compile fast lookup sets
        self._strip: dict[str, set[str]] = {}
        for source in set(recurring_by_source) | set(denylist_by_source):
            joined = set()
            joined.update(recurring_by_source.get(source, set()))
            joined.update(denylist_by_source.get(source, []))
            self._strip[source] = joined

    def modify_document(self, text: str, source: str) -> str:
        if not text:
            return ""
        strip_set = self._strip.get(source)
        if not strip_set:
            return text
        kept_lines = []
        for raw in text.splitlines():
            if raw.strip() in strip_set:
                continue
            kept_lines.append(raw)
        return "\n".join(kept_lines)


# ---------------------------------------------------------------------------- #
# PII typed-tag redaction                                                       #
# ---------------------------------------------------------------------------- #

# Stand-alone regexes. Presidio integration can be layered on top later for
# the few CPT sources where PERSON detection is desirable (currently `fincen_files`);
# see README for the extension point.

_REGEX_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Order matters: more specific (e.g. SSN, IBAN) before the broader ones.
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # CREDIT_CARD — restricted to CARD-FORMATTED patterns only (groups of 4
    # digits separated by spaces/dashes, or AmEx 4-6-5). An earlier version
    # used `\b(?:\d[ -]*?){13,19}\b` which matched any 13-19-digit run and
    # tagged ~21K EDGAR CUSIPs, ~18K OIG contract numbers, and thousands of
    # FinCEN case docket numbers as credit cards. The formatted-only
    # patterns below match real card numbers without false-positive on
    # regulatory identifiers / table data.
    (
        re.compile(r"\b(?:\d{4}[ \-]){3}\d{4}\b"),
        "[CREDIT_CARD]",
    ),
    (
        re.compile(r"\b3\d{3}[ \-]\d{6}[ \-]\d{5}\b"),  # AmEx 4-6-5
        "[CREDIT_CARD]",
    ),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "[EMAIL]",
    ),
    (
        re.compile(
            r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
        ),
        "[PHONE]",
    ),
]


class PiiTagRedactor(DocumentModifier):
    """Regex-based typed-tag PII redaction with audit side-file.

    Multi-input modifier: needs `text`, `source`, and `id` columns. Each worker
    appends to its own audit JSONL under `audit_dir/`; the orchestrator
    concatenates them per source at end of stage.

    The audit row is metadata-only -- the original PII span is replaced by a
    salted SHA-256 hash, never written in cleartext.
    """

    def __init__(
        self,
        audit_dir: Optional[Path],
        salt: bytes,
        enable_ein: bool,
        enable_account_id: bool,
    ) -> None:
        super().__init__()
        self._name = "pii_tag_redactor"
        self._audit_dir = Path(audit_dir) if audit_dir is not None else None
        self._salt = salt
        # Combine canonical regexes with the optional EIN / ACCOUNT_ID extras.
        self._patterns: list[tuple[re.Pattern[str], str]] = list(_REGEX_PII_PATTERNS)
        self._patterns.extend(
            build_regex_replacements(enable_ein=enable_ein, enable_account_id=enable_account_id)
        )
        self._audit_fp = None  # opened lazily per worker

    def _open_audit(self) -> None:
        if self._audit_dir is None or self._audit_fp is not None:
            return
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        fname = f"audit_{os.getpid()}_{uuid.uuid4().hex[:8]}.jsonl"
        self._audit_fp = open(self._audit_dir / fname, "a", encoding="utf-8")

    def _hash(self, span: str) -> str:
        h = hashlib.sha256()
        h.update(self._salt)
        h.update(span.encode("utf-8"))
        return h.hexdigest()[:32]

    def _audit(self, doc_id: str, source: str, entity: str, span: str, replacement: str) -> None:
        if self._audit_dir is None:
            return
        self._open_audit()
        if self._audit_fp is None:
            return
        rec = {
            "doc_id": doc_id,
            "source": source,
            "entity_type": entity,
            "span_sha256": self._hash(span),
            "span_len": len(span),
            "replacement": replacement,
        }
        self._audit_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def modify_document(self, text: str, source: str, id: str) -> str:
        if not text:
            return ""
        new_text = text
        for pattern, replacement in self._patterns:
            entity = replacement.strip("[]")

            def _sub(m: re.Match[str]) -> str:
                span = m.group(0)
                self._audit(id, source, entity, span, replacement)
                return replacement

            new_text = pattern.sub(_sub, new_text)
        return new_text


def consolidate_pii_audits(audit_dir: Path, by_source_dir: Path) -> None:
    """Concatenate per-worker audit shards into one file per source."""
    audit_dir = Path(audit_dir)
    by_source_dir = Path(by_source_dir)
    if not audit_dir.exists():
        return
    by_source_dir.mkdir(parents=True, exist_ok=True)
    fps: dict[str, "os.TextIOWrapper"] = {}
    try:
        for shard in sorted(audit_dir.glob("audit_*.jsonl")):
            for line in shard.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = rec.get("source", "_unknown_")
                fp = fps.get(source)
                if fp is None:
                    fp = open(by_source_dir / f"{source}.audit.jsonl", "a", encoding="utf-8")
                    fps[source] = fp
                fp.write(line + "\n")
            shard.unlink(missing_ok=True)
    finally:
        for fp in fps.values():
            fp.close()


# ---------------------------------------------------------------------------- #
# Cross-source dedup with richness scoring                                      #
# ---------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^[A-Z0-9§\-\.\s]{4,}$", re.MULTILINE)
_CITATION_RE = re.compile(
    r"(?:U\.?S\.?C\.?|C\.?F\.?R\.?|Pub(?:lic)?\.?\s*L(?:aw)?\.?|Reg\.?|§|Section\s+\d)",
    re.IGNORECASE,
)


def richness_score(text: str, weights: tuple[float, float, float]) -> float:
    if not text:
        return 0.0
    w_len, w_head, w_cite = weights
    nchars = len(text)
    headings = len(_HEADING_RE.findall(text))
    citations = len(_CITATION_RE.findall(text))
    return (
        w_len * nchars
        + w_head * (headings * 1000.0)
        + w_cite * (citations * 1000.0)
    )


def _normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def xsource_pair_dedup(
    source_a_jsonl: Path,
    source_b_jsonl: Path,
    text_field: str,
    id_field: str,
    weights: tuple[float, float, float],
) -> set[str]:
    """Find normalized-text exact matches across the two source files.

    For each cluster (set of docs sharing the same normalized hash) that spans
    both sources, keep the doc with the highest richness score and return the
    IDs of the rest (the losers).
    """
    by_hash: dict[str, list[tuple[str, str, float]]] = {}
    for path in (source_a_jsonl, source_b_jsonl):
        if not path.exists():
            continue
        for rec in iter_jsonl(path):
            text = rec.get(text_field) or ""
            doc_id = rec.get(id_field)
            if not doc_id or not text:
                continue
            key = _sha256(_normalize_for_hash(text))
            score = richness_score(text, weights)
            by_hash.setdefault(key, []).append((doc_id, str(path.name), score))

    losers: set[str] = set()
    for cluster in by_hash.values():
        if len(cluster) < 2:
            continue
        sources_in_cluster = {item[1] for item in cluster}
        if len(sources_in_cluster) < 2:
            continue
        winner = max(cluster, key=lambda t: t[2])
        for doc_id, _src, _score in cluster:
            if doc_id != winner[0]:
                losers.add(doc_id)
    return losers


# ---------------------------------------------------------------------------- #
# Stats: per-phase counts, PII tallies, char/token summary                      #
# ---------------------------------------------------------------------------- #


def count_records_by_layer_source(
    path: Path,
    layer_field: str = "layer",
    source_field: str = "source",
) -> dict[str, dict[str, int]]:
    """Walk JSONL files at `path` (file or directory, recursive), return
    `{layer: {source: count}}`. Records missing `layer` / `source` go under
    `_unknown_`.
    """
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        files = sorted(q for q in p.rglob("*.jsonl") if q.is_file())

    out: dict[str, dict[str, int]] = {}
    for fp in files:
        for rec in iter_jsonl(fp):
            layer = str(rec.get(layer_field) or "_unknown_")
            source = str(rec.get(source_field) or "_unknown_")
            out.setdefault(layer, {}).setdefault(source, 0)
            out[layer][source] += 1
    return out


def aggregate_pii_audit(pii_dir: Path) -> dict[str, dict[str, int]]:
    """Read `<source>.audit.jsonl` files in `pii_dir`, return
    `{source: {entity_type: count}}`.
    """
    p = Path(pii_dir)
    out: dict[str, dict[str, int]] = {}
    if not p.exists():
        return out
    for fp in sorted(p.glob("*.audit.jsonl")):
        # Filename is "<source>.audit.jsonl"; strip the ".audit" suffix from the stem.
        source = fp.name[: -len(".audit.jsonl")]
        for rec in iter_jsonl(fp):
            ent = str(rec.get("entity_type") or "_unknown_")
            out.setdefault(source, {}).setdefault(ent, 0)
            out[source][ent] += 1
    return out


# Module-level state used by Pool worker init -- can't pass tokenizers across
# processes via pickle, so each worker loads its own. Globals are populated by
# `_token_pool_init` and read by `_count_one_file_for_pool`.
_TOK_WORKER_STATE: dict = {}


def _token_pool_init(tokenizer_id: str, hf_token_env: str) -> None:
    """Pool initializer: load the HF tokenizer once per worker process."""
    from transformers import AutoTokenizer  # type: ignore

    kwargs: dict = {"trust_remote_code": True}
    tok_env = os.getenv(hf_token_env, "")
    if tok_env:
        kwargs["token"] = tok_env
    _TOK_WORKER_STATE["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)


def _count_one_file_for_pool(args: tuple[str, str]) -> tuple[str, int, int]:
    """Pool worker fn: return (file_path, chars, tokens) for one curated JSONL.

    Reads the file inline (no IPC of large strings between processes); tokenizer
    is the per-worker global from `_TOK_WORKER_STATE`.
    """
    file_path, text_field = args
    tok = _TOK_WORKER_STATE.get("tokenizer")
    chars = tokens = 0
    for rec in iter_jsonl(Path(file_path)):
        text = rec.get(text_field) or ""
        if not text:
            continue
        chars += len(text)
        if tok is not None:
            try:
                tokens += len(tok.encode(text, add_special_tokens=False))
            except Exception:
                pass
    return file_path, chars, tokens


def compute_char_and_token_stats(
    curated_dir: Path,
    text_field: str,
    layer_field: str,
    source_field: str,
    tokenizer_id: str | None,
    hf_token_env: str,
    logger: Optional[logging.Logger] = None,
    workers: int | None = 32,
) -> dict[str, dict[str, dict]]:
    """Walk `<curated_dir>/<layer>/<source>.jsonl`, return per-(layer, source):

        {layer: {source: {"chars": int, "tokens": int | None}}}

    Tokenization is **file-level parallel** via :class:`multiprocessing.Pool`
    (each worker loads its own tokenizer instance, then iterates whole curated
    JSONL files independently). Bottleneck is the largest file, but everything
    else finishes in seconds. Single-threaded encoding takes 45-80 min on this
    corpus; with 32 workers it drops to 5-10 min.

    `tokens` is None if the tokenizer can't load (no HF auth, no `transformers`,
    network unavailable, gated model, etc.) -- in that case `chars` is still
    populated and a warning is logged.
    """
    p = Path(curated_dir)
    out: dict[str, dict[str, dict]] = {}
    if not p.exists():
        return out

    files = sorted(p.rglob("*.jsonl"))
    if not files:
        return out

    # Validate the tokenizer in the parent process so we can warn-and-fall-back
    # cleanly. Fan out to workers only if it loads here.
    tok_ok = False
    if tokenizer_id:
        try:
            from transformers import AutoTokenizer  # type: ignore

            kwargs: dict = {"trust_remote_code": True}
            tok_env = os.getenv(hf_token_env, "")
            if tok_env:
                kwargs["token"] = tok_env
            AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
            tok_ok = True
            if logger:
                logger.info(f"[tok] tokenizer ok ({tokenizer_id}); fanning out to {workers or 'auto'} workers")
        except Exception as e:
            if logger:
                logger.warning(
                    f"[tok] SKIP: {type(e).__name__}: {e} -- char counts only. "
                    f"Hint: set ${hf_token_env} (HuggingFace token with access to {tokenizer_id})."
                )

    # Aggregate per-file results back into per-(layer, source) entries. Layer +
    # source are derived from the curated path: <curated_dir>/<layer>/<source>.jsonl.
    if tok_ok:
        from multiprocessing import Pool

        n_workers = workers if workers and workers > 0 else max(1, (os.cpu_count() or 4) // 2)
        n_workers = min(n_workers, len(files))
        tasks = [(str(fp), text_field) for fp in files]
        with Pool(
            n_workers, initializer=_token_pool_init, initargs=(tokenizer_id, hf_token_env)
        ) as pool:
            for i, (fp_str, chars, tokens) in enumerate(
                pool.imap_unordered(_count_one_file_for_pool, tasks), 1
            ):
                fp = Path(fp_str)
                layer, source = fp.parent.name, fp.stem
                entry = out.setdefault(layer, {}).setdefault(source, {"chars": 0, "tokens": 0})
                entry["chars"] += chars
                entry["tokens"] += tokens
                if logger:
                    logger.info(
                        f"  [{i:>2d}/{len(files)}] {layer}/{source:35s}  "
                        f"chars={chars:>14,d}  tokens={tokens:>13,d}"
                    )
    else:
        # Char-only fallback (single-threaded, fast since no tokenizer call).
        for fp in files:
            layer, source = fp.parent.name, fp.stem
            chars = 0
            for rec in iter_jsonl(fp):
                chars += len(rec.get(text_field) or "")
            out.setdefault(layer, {}).setdefault(source, {"chars": chars, "tokens": None})

    return out


def filter_jsonl_by_id(
    in_path: Path,
    out_path: Path,
    id_field: str,
    drop_ids: set[str],
) -> tuple[int, int]:
    """Stream `in_path`, drop records whose id is in `drop_ids`, write to `out_path`.

    Returns (kept, dropped).
    """
    kept = 0
    dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in iter_jsonl(in_path):
            doc_id = rec.get(id_field)
            if doc_id and doc_id in drop_ids:
                dropped += 1
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped
