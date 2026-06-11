#!/usr/bin/env python3
"""Step 2 -- extract raw downloads under 1.data_download/data/raw/ into:

    data/cpt/<layer>/<source>.jsonl    clean prose, schema-aware per-source extraction
    data/sft/<source>.jsonl            rows preserved as JSON for Step 5 Data Designer
    data/transactional/<source>/...    raw files copied as-is

PDF extraction uses pypdfium2 (CPU-only, no RAG cluster needed). Bulk work is
parallelised across a process pool: PDF text extraction, then tokenisation.
On a 240-core box the entire 2,909-PDF corpus is processed in ~1-2 minutes.

Outputs:
    data/             the JSONLs + transactional copies
    summary.json      machine-readable run report (per-source stats + per-phase tokens)
    extraction.log    verbose stdout
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# ---- Paths + constants -----------------------------------------------------

ROOT      = Path("/data/swami/gsi-training")
RAW_ROOT  = ROOT / "1.data_download" / "data" / "raw"
WORK_ROOT = ROOT / "2.data_processing"
OUT_ROOT  = WORK_ROOT / "data"
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# Parallelism -- ProcessPoolExecutor bypasses the GIL for pypdfium2 + tokenizers.
# Sized for the host (240 cores, 2 TB RAM): plenty of headroom.
_NCPU       = os.cpu_count() or 16
PDF_WORKERS = min(64, max(8, _NCPU // 4))      # PDF extraction is CPU-bound, ~50-200 MB/worker
TOK_WORKERS = min(16, max(4, _NCPU // 16))     # Tokenizer ~200 MB resident per worker

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---- Routing ---------------------------------------------------------------

_LB = ["rule_qa", "sara_entailment", "sara_numeric", "hearsay", "consumer_contracts_qa",
       "contract_nli_confidentiality_of_agreement", "contract_nli_explicit_identification",
       "contract_nli_limited_use", "contract_nli_no_licensing",
       "contract_nli_notice_on_compelled_disclosure",
       "contract_nli_return_of_confidential_information",
       "supply_chain_disclosure_disclosed_accountability",
       "supply_chain_disclosure_disclosed_training",
       "supply_chain_disclosure_disclosed_verification",
       "supply_chain_disclosure_best_practice_accountability",
       "supply_chain_disclosure_best_practice_training"]

SOURCE_ROUTES = {
    # cpt/level_1 -- broad financial / regulatory register
    "edgar_corpus":                 ("cpt", "level_1"),
    "pile_of_law_sec":              ("cpt", "level_1"),
    "pile_of_law_cfr":              ("cpt", "level_1"),
    "pile_of_law_federal_register": ("cpt", "level_1"),
    "pile_of_law_uscode":           ("cpt", "level_1"),
    "pile_of_law_oig":              ("cpt", "level_1"),
    "pile_of_law_doj_guidance":     ("cpt", "level_1"),
    "uscode_house":                 ("cpt", "level_1"),
    # cpt/level_2 -- AML-specific
    "fincen_advisories":       ("cpt", "level_2"),
    "fincen_federal_register": ("cpt", "level_2"),
    "fincen_sar_reviews":      ("cpt", "level_2"),
    "fincen_enforcement":      ("cpt", "level_2"),
    "fincen_files":            ("cpt", "level_2"),
    "fatf_publications":       ("cpt", "level_2"),
    "ofac_guidance":           ("cpt", "level_2"),
    "courtlistener":           ("cpt", "level_2"),
    "caselaw_access_project":  ("cpt", "level_2"),
    # sft -- text Q&A / instruction-response
    "finqa": ("sft", None), "tat_qa": ("sft", None), "financebench": ("sft", None),
    "finance_instruct_500k": ("sft", None), "sarsum": ("sft", None), "ffiec_manual": ("sft", None),
    **{f"legalbench__{k}": ("sft", None) for k in _LB},
    # transactional -- toolset reference (Step 5 Data Designer reads natively)
    "amlgentex":                  ("transactional", None),
    "ibm_aml_transactions":       ("transactional", None),
    "enterprise_financial_crime": ("transactional", None),
    "cfpb_complaints":            ("transactional", None),
    "ofac_enforcement":           ("transactional", None),
}

FILE_OVERRIDES = {
    "fincen_files": {
        "data/download_data_fincen_files.zip": ("transactional", None),
    },
    "cfpb_complaints": {
        "complaints.csv.zip": ("__skip__", None),
    },
    "enterprise_financial_crime": {
        # The commercial-overview PDF is the vendor's own marketing brochure
        # describing the dataset; it contains no AML content. Previously
        # routed to cpt/level_2, where it appeared as a single 2 KB record of
        # dataset advertising copy in the Layer-2 AML-specific corpus. Skip.
        "dataset_commercial_overview.pdf":                     ("__skip__", None),
        "full_dataset/premium_module/aml_case_dossiers.jsonl": ("sft", None),
    },
}

# All PDF / row / file extensions go through DIRECT_DISPATCH now -- there is no
# RAG path.  PDFs use pypdfium2 (CPU, parallel).
DIRECT_EXTS    = {".parquet", ".csv", ".tsv", ".json", ".jsonl", ".txt", ".md",
                  ".html", ".htm", ".pdf"}
ZIP_EXTS       = {".zip"}
SKIP_EXTS      = {".gitignore", ".metadata", ".log", ".cache", ".lock"}
DEDUP_HTML_TXT = {"fincen_files"}
_ROW_BASED     = {".parquet", ".csv", ".tsv", ".json", ".jsonl"}


# ---- CPT extractors --------------------------------------------------------

def _ext_text_field(d):
    t = d.get("text")
    return t.strip() if isinstance(t, str) and t.strip() else None


def _ext_edgar(d):
    sections = [v.strip() for k, v in d.items()
                if k.startswith("section_") and isinstance(v, str) and v.strip()]
    if not sections:
        return None
    return (f"# 10-K Filing -- CIK {d.get('cik','?')}, Year {d.get('year','?')}\n\n"
            + "\n\n".join(sections))


CPT_EXTRACTORS = {
    "edgar_corpus":                 _ext_edgar,
    "pile_of_law_sec":              _ext_text_field,
    "pile_of_law_cfr":              _ext_text_field,
    "pile_of_law_federal_register": _ext_text_field,
    "pile_of_law_uscode":           _ext_text_field,
    "pile_of_law_oig":              _ext_text_field,
    "pile_of_law_doj_guidance":     _ext_text_field,
    "courtlistener":                _ext_text_field,
}


# ---- Discovery -------------------------------------------------------------

def list_files(src_dir):
    out = []
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(src_dir).parts):
            continue
        ext = p.suffix.lower()
        if ext in SKIP_EXTS:
            continue
        if ext in DIRECT_EXTS or ext in ZIP_EXTS:
            out.append((p, ext))
    return out


def file_destination(source, file_path, src_dir):
    rel = file_path.relative_to(src_dir).as_posix()
    for suffix, route in FILE_OVERRIDES.get(source, {}).items():
        if rel == suffix or rel.endswith("/" + suffix):
            return route
    return SOURCE_ROUTES.get(source)


def discover_units():
    units, skipped, unmapped = {}, [], []
    for phase in ("cpt", "sft"):
        phase_dir = RAW_ROOT / phase
        if not phase_dir.exists():
            continue
        if phase == "cpt":
            sources = [p for layer in sorted(phase_dir.iterdir()) if layer.is_dir()
                       for p in sorted(layer.iterdir()) if p.is_dir()]
        else:
            sources = [p for p in sorted(phase_dir.iterdir()) if p.is_dir()]
        for src_dir in sources:
            name = src_dir.name
            if name not in SOURCE_ROUTES:
                unmapped.append(name)
                continue
            files = list_files(src_dir)
            if name in DEDUP_HTML_TXT:
                html_stems = {p.with_suffix("") for p, e in files if e in (".html", ".htm")}
                kept = []
                for p, e in files:
                    if e == ".txt" and p.with_suffix("") in html_stems:
                        skipped.append((name, p.relative_to(src_dir).as_posix(),
                                        "duplicate of .html (preferred)"))
                    else:
                        kept.append((p, e))
                files = kept
            for path, ext in files:
                route = file_destination(name, path, src_dir)
                if route is None or route[0] == "__skip__":
                    skipped.append((name, path.relative_to(src_dir).as_posix(),
                                    "override:skip" if route else "unmapped"))
                    continue
                key = (name, route[0], route[1])
                units.setdefault(key, {"src_dir": src_dir, "files": []})["files"].append((path, ext))
    return units, skipped, unmapped


def assert_cpt_extractors_present(units):
    missing = [src for (src, top, _), info in units.items()
               if top == "cpt" and src not in CPT_EXTRACTORS
               and any(e in _ROW_BASED for _, e in info["files"])]
    if missing:
        raise RuntimeError(
            f"[CONFIG-ERROR] CPT sources without an extractor: {missing}\n"
            f"  Add to CPT_EXTRACTORS in run_extraction.py before retrying."
        )


# ---- Direct converters (parquet/csv/tsv/json/jsonl/txt/md/html/zip/PDF) ---

TEXT_HINTS = ("text", "content", "body", "narrative", "description", "answer",
              "plain_text", "investigator_notes", "resolution_notes", "assistant",
              "summary", "Consumer complaint narrative")


def _doc_id(p, src):  return p.relative_to(src).with_suffix("").as_posix()
def _orig(p, src):    return p.relative_to(src).as_posix()


def _normalize(v):
    if v is None:
        return None
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            return v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, dict):
        return v
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def _pick_text(d):
    for k in TEXT_HINTS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _looks_like_json_dict(s):
    if not isinstance(s, str):
        return False
    s = s.lstrip()
    if not s or s[0] != "{":
        return False
    try:
        return isinstance(json.loads(s), dict)
    except (json.JSONDecodeError, ValueError):
        return False


def _content_for(row, *, source, top, fn):
    if top == "cpt":
        if fn is None:
            raise RuntimeError(f"CPT row reached converter without content_fn (source bug)")
        text = fn(row)
        if not (isinstance(text, str) and text.strip()):
            return None
        if _looks_like_json_dict(text):
            raise AssertionError(
                f"CPT extractor for source={source!r} produced JSON-dict-shaped content "
                f"(EDGAR-bug class). Fix the extractor.\n  head: {text[:200]!r}"
            )
        return text
    return _pick_text(row) or json.dumps(row, ensure_ascii=False, default=str)


def _record(*, source, top, layer, doc_id, content, etype="text", page=None,
            ffmt=None, meta=None):
    m = dict(meta or {})
    if ffmt and "file_format" not in m:
        m["file_format"] = ffmt
    return {"source": source, "phase": top, "layer": layer, "doc_id": doc_id,
            "element_type": etype, "page": page, "content": content, "metadata": m}


def convert_parquet(p, source, top, layer, src, *, content_fn=None):
    df = pd.read_parquet(p)
    for i, row in df.iterrows():
        d = {k: _normalize(v) for k, v in row.to_dict().items()}
        c = _content_for(d, source=source, top=top, fn=content_fn)
        if c is None:
            continue
        yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                      etype="parquet_row", content=c, ffmt="parquet",
                      meta={"original_path": _orig(p, src), "row_index": int(i),
                            "row_fields": {k: (str(v)[:1000] if v is not None else None)
                                           for k, v in d.items()}})


def convert_csv(p, source, top, layer, src, sep=",", *, content_fn=None):
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter=sep)):
            d = dict(row)
            c = _content_for(d, source=source, top=top, fn=content_fn)
            if c is None:
                continue
            yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                          etype="tsv_row" if sep == "\t" else "csv_row",
                          content=c, ffmt="tsv" if sep == "\t" else "csv",
                          meta={"original_path": _orig(p, src), "row_index": i,
                                "row_fields": dict(row)})


def convert_tsv(p, source, top, layer, src, *, content_fn=None):
    yield from convert_csv(p, source, top, layer, src, "\t", content_fn=content_fn)


def convert_jsonl(p, source, top, layer, src, *, content_fn=None):
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = obj if isinstance(obj, dict) else {"value": obj}
            c = _content_for(dict(d), source=source, top=top, fn=content_fn)
            if c is None:
                continue
            yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                          etype="jsonl_row", content=c, ffmt="jsonl",
                          meta={"original_path": _orig(p, src), "row_index": i,
                                "row_fields": obj if isinstance(obj, dict) else {"value": obj}})


def convert_json(p, source, top, layer, src, *, content_fn=None):
    items = None
    with p.open("r", encoding="utf-8", errors="replace") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = None
    if data is None:
        items = []
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        items.append((ln, json.loads(line)))
                    except json.JSONDecodeError:
                        pass
    elif isinstance(data, list):
        items = list(enumerate(data))
    elif isinstance(data, dict):
        items = list(data.items())
    else:
        items = [(0, data)]
    for key, obj in items:
        d = obj if isinstance(obj, dict) else {"value": str(obj)[:2000]}
        c = _content_for(dict(d), source=source, top=top, fn=content_fn)
        if c is None:
            continue
        yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                      etype="json_record", content=c, ffmt="json",
                      meta={"original_path": _orig(p, src), "key": str(key),
                            "row_fields": obj if isinstance(obj, dict) else {"value": str(obj)[:2000]}})


def convert_txt(p, source, top, layer, src, *, content_fn=None):
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if text:
        yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                      content=text, ffmt=p.suffix.lstrip("."),
                      meta={"original_path": _orig(p, src)})


def convert_html(p, source, top, layer, src, *, content_fn=None):
    """HTML -> plain text via BeautifulSoup. CPU-only."""
    from bs4 import BeautifulSoup
    raw = p.read_bytes()
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    text = "\n".join(l.strip() for l in soup.get_text(separator="\n", strip=True).splitlines()
                     if l.strip())
    if not text:
        return
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                  content=text, ffmt=p.suffix.lstrip("."),
                  meta={"original_path": _orig(p, src), **({"title": title} if title else {})})


def convert_pdf(p, source, top, layer, src, *, content_fn=None):
    """PDF -> reading-order text via pypdfium2. One record per PDF, ALL pages
    concatenated as flowing prose. Sequential variant; bulk PDF extraction goes
    through `extract_pdfs_parallel` which uses a process pool."""
    text, status, n_pages = _extract_pdf_one(str(p))
    if status != "ok":
        return
    yield _record(source=source, top=top, layer=layer, doc_id=_doc_id(p, src),
                  content=text, ffmt="pdf",
                  meta={"original_path": _orig(p, src), "n_pages": n_pages})


def convert_zip(p, source, top, layer, src, *, content_fn=None):
    with tempfile.TemporaryDirectory(prefix="unzip_") as td:
        td_p = Path(td)
        with zipfile.ZipFile(p) as zf:
            zf.extractall(td_p)
        for inner in td_p.rglob("*"):
            if not inner.is_file():
                continue
            if any(part.startswith(".") or part.startswith("__MACOSX")
                   for part in inner.relative_to(td_p).parts):
                continue
            ext = inner.suffix.lower()
            doc_prefix = p.stem + "/" + inner.relative_to(td_p).with_suffix("").as_posix()
            if ext in DIRECT_DISPATCH:
                for rec in DIRECT_DISPATCH[ext](inner, source, top, layer, td_p, content_fn=content_fn):
                    rec["doc_id"] = doc_prefix
                    rec["metadata"]["original_path"] = (
                        p.relative_to(src).as_posix() + "!" + inner.relative_to(td_p).as_posix()
                    )
                    yield rec
            elif ext in ZIP_EXTS:
                yield from convert_zip(inner, source, top, layer, td_p, content_fn=content_fn)


DIRECT_DISPATCH = {
    ".parquet": convert_parquet, ".csv": convert_csv, ".tsv": convert_tsv,
    ".json": convert_json, ".jsonl": convert_jsonl, ".txt": convert_txt,
    ".md": convert_txt, ".html": convert_html, ".htm": convert_html,
    ".pdf": convert_pdf, ".zip": convert_zip,
}


# ---- pypdfium2 PDF extraction (parallel) -----------------------------------

def _extract_pdf_one(path_str, min_chars=100):
    """Worker-safe single-PDF extraction. Returns (text, status, n_pages)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return "", "pypdfium2_missing", 0
    try:
        doc = pdfium.PdfDocument(path_str)
    except Exception as e:
        return "", f"open_error:{type(e).__name__}", 0
    pages = []
    n_pages = 0
    try:
        n_pages = len(doc)
    except Exception:
        pass
    for page in doc:
        try:
            pages.append(page.get_textpage().get_text_range())
        except Exception:
            pages.append("")
    text = "\n\n".join(p for p in pages if p and p.strip())
    if len(text) < min_chars:
        return "", f"too_short:{len(text)}_chars", n_pages
    return text, "ok", n_pages


def _pdf_pool_worker(args):
    path_str, source, top, layer, src_dir_str, doc_id, original_path = args
    text, status, n_pages = _extract_pdf_one(path_str)
    if status != "ok":
        return None, status
    rec = _record(source=source, top=top, layer=layer, doc_id=doc_id,
                  content=text, ffmt="pdf",
                  meta={"original_path": original_path, "n_pages": n_pages})
    return rec, "ok"


def extract_pdfs_parallel(pdfs, source, top, layer, src):
    """Yield records for every PDF in `pdfs`, extracted in parallel via pypdfium2.
    `pdfs` is list of (Path, ext) like the rest of the orchestrator expects."""
    if not pdfs:
        return
    work = [(str(p), source, top, layer, str(src),
             _doc_id(p, src), _orig(p, src))
            for p, _ in pdfs]
    print(f"           [pypdfium2] {len(pdfs)} PDFs across {PDF_WORKERS} workers", flush=True)
    n_done = n_failed = 0
    fail_reasons = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=PDF_WORKERS) as ex:
        for rec, status in ex.map(_pdf_pool_worker, work, chunksize=4):
            n_done += 1
            if rec is None:
                n_failed += 1
                fail_reasons[status] = fail_reasons.get(status, 0) + 1
            else:
                yield rec
            if n_done % 200 == 0 or n_done == len(pdfs):
                rate = n_done / max(0.1, time.time() - t0)
                print(f"                 ... {n_done}/{len(pdfs)} done "
                      f"({n_failed} failed, {rate:.1f}/s)", flush=True)
    if fail_reasons:
        for reason, n in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"                 [WARN] {reason}: {n} PDFs", flush=True)


# ---- Per-unit orchestrator -------------------------------------------------

PROGRESS_EVERY = 10_000


def _copy_unit(source, top, layer, info):
    src = info["src_dir"]
    out_dir = OUT_ROOT / top / source
    out_dir.mkdir(parents=True, exist_ok=True)
    n, total_bytes, t0 = 0, 0, time.time()
    for fi, (p, _) in enumerate(info["files"], start=1):
        rel = p.relative_to(src)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        n += 1
        total_bytes += p.stat().st_size
    dt = time.time() - t0
    print(f"           TOTAL -> {n} file(s), {total_bytes/1e6:.1f} MB in {dt:.1f}s", flush=True)
    return {"source": source, "top": top, "layer": layer, "out": out_dir,
            "files_count": len(info["files"]),
            "files_mb": sum(p.stat().st_size for p, _ in info["files"]) / 1e6,
            "records": n, "duration_s": dt, "status": "written", "copy_only": True}


def extract_unit(source, top, layer, info):
    """Per-unit dispatch.
      transactional/  -> copy as-is
      cpt/* sft/*     -> direct converters; PDFs go through parallel pypdfium2
    """
    if top == "transactional":
        return _copy_unit(source, top, layer, info)

    out = (OUT_ROOT / top / layer / f"{source}.jsonl") if layer else (OUT_ROOT / top / f"{source}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    src = info["src_dir"]
    pdfs    = [(p, e) for p, e in info["files"] if e == ".pdf"]
    others  = [(p, e) for p, e in info["files"] if e in DIRECT_DISPATCH and e != ".pdf"]
    fn      = CPT_EXTRACTORS.get(source) if top == "cpt" else None

    files_mb = sum(p.stat().st_size for p, _ in info["files"]) / 1e6
    n_total, t_unit = 0, time.time()
    with out.open("w", encoding="utf-8") as fh:
        # ---- non-PDF files: parquet/csv/tsv/json/jsonl/txt/md/html/zip ----
        for fi, (p, e) in enumerate(others, start=1):
            print(f"           [{fi}/{len(others)}] {p.name} ({p.stat().st_size/1e6:.1f} MB)",
                  flush=True)
            ts, n_file = time.time(), 0
            try:
                kw = {"content_fn": fn} if e in _ROW_BASED else {}
                for rec in DIRECT_DISPATCH[e](p, source, top, layer, src, **kw):
                    c = rec.get("content")
                    if isinstance(c, str) and c.strip():
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_file += 1
                        if n_file % PROGRESS_EVERY == 0:
                            fh.flush()
                            rate = n_file / max(0.1, time.time() - ts)
                            print(f"                 ... {n_file:,} records ({rate:,.0f}/s)",
                                  flush=True)
            except AssertionError:
                raise
            except Exception as e2:
                print(f"           [ERR] {p.name}: {type(e2).__name__}: {e2}", flush=True)
            fh.flush()
            print(f"               -> {n_file:,} records in {time.time()-ts:.1f}s", flush=True)
            n_total += n_file

        # ---- PDFs: parallel pypdfium2 ----
        if pdfs:
            ts, n_pdf = time.time(), 0
            try:
                for rec in extract_pdfs_parallel(pdfs, source, top, layer, src):
                    c = rec.get("content")
                    if isinstance(c, str) and c.strip():
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_pdf += 1
            except Exception as e2:
                print(f"           [ERR] PDF parallel: {type(e2).__name__}: {e2}", flush=True)
            fh.flush()
            print(f"               -> {n_pdf:,} records from {len(pdfs)} PDFs "
                  f"in {time.time()-ts:.1f}s", flush=True)
            n_total += n_pdf

    dt = time.time() - t_unit
    print(f"           TOTAL -> {n_total:,} records in {dt:.1f}s", flush=True)
    return {"source": source, "top": top, "layer": layer, "out": out,
            "files_count": len(info["files"]), "files_mb": files_mb,
            "records": n_total, "duration_s": dt, "status": "written", "copy_only": False}


# ---- Validation ------------------------------------------------------------

def validate(results):
    issues, rows = [], []
    for r in results:
        out = r["out"]
        if r.get("copy_only"):
            if not out.exists() or not any(out.iterdir()):
                issues.append((r["source"], "copy folder missing/empty"))
                continue
            n_files = sum(1 for p in out.rglob("*") if p.is_file())
            mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
            rows.append((str(out.relative_to(WORK_ROOT)) + "/", "copy", f"{n_files} files / {mb:.1f} MB"))
            continue
        if not out.exists() or out.stat().st_size == 0:
            issues.append((r["source"], "jsonl missing/empty"))
            continue
        n, bad, first = 0, 0, None
        with out.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                    if first is None:
                        first = rec
                except json.JSONDecodeError:
                    bad += 1
        if bad:
            issues.append((r["source"], f"{bad} unparseable lines"))
        req = {"source", "phase", "doc_id", "element_type", "content", "metadata"}
        if first and not req.issubset(first.keys()):
            issues.append((r["source"], f"missing fields: {req - set(first.keys())}"))
        rows.append((str(out.relative_to(WORK_ROOT)), "jsonl", f"{n} records"))
    print(f"\n{'Path':<60} {'Kind':<6} {'Records / Files':>22}")
    print("-" * 92)
    for path, kind, val in sorted(rows):
        print(f"{path:<60} {kind:<6} {val:>22}")
    if issues:
        print("\nIssues:")
        for src, msg in issues:
            print(f"  [{src}] {msg}")
    else:
        print("\nAll outputs validated.")
    return issues


# ---- Token counting (parallel) --------------------------------------------

# Module-level globals so ProcessPoolExecutor workers can reuse a single tokenizer.
_TOK = None


def _tok_init():
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)


def _tok_count_worker(jsonl_path_str):
    n_rec = n_tok = 0
    batch = []
    with open(jsonl_path_str, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = rec.get("content")
            if isinstance(c, str) and c:
                batch.append(c); n_rec += 1
                if len(batch) >= 256:
                    enc = _TOK(batch, add_special_tokens=False, padding=False, truncation=False)
                    n_tok += sum(len(ids) for ids in enc["input_ids"])
                    batch.clear()
    if batch:
        enc = _TOK(batch, add_special_tokens=False, padding=False, truncation=False)
        n_tok += sum(len(ids) for ids in enc["input_ids"])
    return jsonl_path_str, n_rec, n_tok


def tokenize_summary():
    """Tokenise every cpt/* and sft/* JSONL in parallel. Returns (rows, totals)."""
    print("\n" + "=" * 80)
    print(f"Token counts (tokenizer={TOKENIZER}; counting `content` field only)")
    print("=" * 80)
    try:
        try:
            import transformers  # noqa
        except ImportError:
            print("[tok] installing transformers ...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "transformers>=4.45", "tokenizers>=0.20", "sentencepiece>=0.2"])
        # Probe tokenizer load up-front so workers don't all crash silently.
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    except Exception as e:
        print(f"[tok] SKIP: {type(e).__name__}: {e}")
        print("       Hint: huggingface-cli login (or set HF_TOKEN) and re-run tokenize_summary().")
        return [], {"cpt_level_1": 0, "cpt_level_2": 0, "sft": 0, "grand": 0}

    paths = []
    for phase in ("cpt", "sft"):
        ph = OUT_ROOT / phase
        if ph.exists():
            paths.extend(sorted(str(jp) for jp in ph.rglob("*.jsonl")))
    if not paths:
        return [], {"cpt_level_1": 0, "cpt_level_2": 0, "sft": 0, "grand": 0}

    print(f"[tok] tokenising {len(paths)} files across {TOK_WORKERS} workers", flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=TOK_WORKERS, initializer=_tok_init) as ex:
        for path, n_rec, n_tok in ex.map(_tok_count_worker, paths):
            rows.append((path, n_rec, n_tok))
            rel = Path(path).relative_to(WORK_ROOT)
            print(f"  {str(rel):<60} records={n_rec:>10}  tokens={n_tok:>14,}", flush=True)
    print(f"[tok] done in {time.time()-t0:.1f}s")

    totals = {"cpt_level_1": 0, "cpt_level_2": 0, "sft": 0}
    for path, _, n_tok in rows:
        rel = Path(path).relative_to(WORK_ROOT)
        parts = rel.parts
        if len(parts) >= 3 and parts[0] == "data" and parts[1] == "cpt":
            key = "cpt_level_1" if "level_1" in parts else "cpt_level_2"
            totals[key] += n_tok
        elif len(parts) >= 2 and parts[0] == "data" and parts[1] == "sft":
            totals["sft"] += n_tok
    totals["grand"] = totals["cpt_level_1"] + totals["cpt_level_2"] + totals["sft"]

    print(f"\n  CPT/level_1:    {totals['cpt_level_1']:>15,} tokens")
    print(f"  CPT/level_2:    {totals['cpt_level_2']:>15,} tokens")
    print(f"  SFT:            {totals['sft']:>15,} tokens")
    print(f"  GRAND total:    {totals['grand']:>15,} tokens")
    return rows, totals


# ---- summary.json (single output artefact) --------------------------------

def write_summary_json(*, units, skipped, all_results, tok_rows, tok_totals,
                       extraction_dt, vt_dt, total_dt, issues):
    """Single machine-readable run report. No markdown."""
    tok_by_path = {p: (n_rec, n_tok) for p, n_rec, n_tok in tok_rows}

    per_source = []
    for r in all_results:
        out = r["out"]
        out_rel = str(out.relative_to(WORK_ROOT)) + ("/" if r.get("copy_only") else "")
        rec = {
            "source": r["source"],
            "phase": r["top"],
            "layer": r["layer"],
            "files_count": r.get("files_count"),
            "files_mb": round(r.get("files_mb", 0), 2),
            "records": r["records"],
            "duration_s": round(r.get("duration_s", 0), 1),
            "output": out_rel,
            "copy_only": bool(r.get("copy_only")),
        }
        if not r.get("copy_only"):
            n_rec, n_tok = tok_by_path.get(str(out), (None, None))
            if n_tok is not None:
                rec["tokens"] = n_tok
        per_source.append(rec)

    bd = {("cpt", "level_1"): 0, ("cpt", "level_2"): 0,
          ("sft", None): 0, ("transactional", None): 0}
    for (_, t, l), _info in units.items():
        bd[(t, l)] = bd.get((t, l), 0) + 1

    cpt_l1_records = sum(r["records"] for r in all_results
                         if r["top"] == "cpt" and r["layer"] == "level_1" and not r.get("copy_only"))
    cpt_l2_records = sum(r["records"] for r in all_results
                         if r["top"] == "cpt" and r["layer"] == "level_2" and not r.get("copy_only"))
    sft_records = sum(r["records"] for r in all_results
                      if r["top"] == "sft" and not r.get("copy_only"))
    transactional_files = sum(r["records"] for r in all_results
                              if r.get("copy_only"))

    summary = {
        "tokenizer": TOKENIZER,
        "run_started_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "raw_root": str(RAW_ROOT),
        "out_root": str(OUT_ROOT),
        "discovery": {
            "total_units": len(units),
            "by_phase": {f"{t}/{l or '-'}": c for (t, l), c in bd.items()},
            "skipped_files_count": len(skipped),
            "skipped_files": [{"source": s, "file": f, "reason": w}
                              for s, f, w in skipped[:50]],
            "skipped_files_truncated": len(skipped) > 50,
        },
        "totals": {
            "records": {
                "cpt_level_1": cpt_l1_records,
                "cpt_level_2": cpt_l2_records,
                "sft": sft_records,
                "transactional_files": transactional_files,
            },
            "tokens": tok_totals,
        },
        "durations_s": {
            "extraction": round(extraction_dt, 1),
            "validate_and_tokenize": round(vt_dt, 1),
            "total": round(total_dt, 1),
        },
        "quality_gates": {
            "all_jsonls_parse": len([i for i in issues if "unparseable" in i[1]]) == 0,
            "no_missing_outputs": len([i for i in issues if "missing" in i[1]]) == 0,
            "all_cpt_extractors_registered": True,  # asserted at startup
            "no_json_dict_content_emitted": True,    # raised at write-time if violated
        },
        "per_source": per_source,
    }

    out_path = WORK_ROOT / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


# ---- Main ------------------------------------------------------------------

def wipe_outputs():
    for sub in ("cpt", "sft", "transactional"):
        if (OUT_ROOT / sub).exists():
            print(f"[wipe] {OUT_ROOT/sub}", flush=True)
            shutil.rmtree(OUT_ROOT / sub)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _ensure_pypdfium2():
    try:
        import pypdfium2  # noqa
    except ImportError:
        print("[deps] installing pypdfium2 ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdfium2>=4.30"])


def main():
    t_main = time.time()
    print("=" * 80)
    print("Step 2 -- Data Extraction & JSONL Conversion (pypdfium2, CPU-parallel)")
    print(f"  RAW_ROOT:    {RAW_ROOT}")
    print(f"  OUT_ROOT:    {OUT_ROOT}")
    print(f"  PDF workers: {PDF_WORKERS}    Token workers: {TOK_WORKERS}")
    print("=" * 80)

    _ensure_pypdfium2()
    wipe_outputs()

    units, skipped, unmapped = discover_units()
    assert_cpt_extractors_present(units)

    print(f"\nDiscovered {len(units)} extraction units")
    print(f"{'Top':<14} {'Layer':<10} {'Source':<55} {'Files':>6}  Extensions")
    print("-" * 110)
    for (n, t, l), info in sorted(units.items(),
                                  key=lambda x: (x[0][1], x[0][2] or "", x[0][0])):
        exts = sorted({e for _, e in info["files"]})
        print(f"{t:<14} {(l or '-'):<10} {n:<55} {len(info['files']):>6}  {','.join(exts)}")
    if skipped:
        print(f"\nSkipped files ({len(skipped)}):")
        for s, r, w in skipped[:20]:
            print(f"  [{s}] {r}  ({w})")
    if unmapped:
        print(f"\nUnmapped sources: {unmapped}")

    print(f"\nUnits: {len(units)}\n")

    # ---- Extraction pass (single pass; no RAG/direct split) ----
    print("=" * 80 + "\nExtraction pass\n" + "=" * 80)
    ex_t0 = time.time()
    all_results = []
    for (source, top, layer), info in sorted(units.items(),
                                             key=lambda x: (x[0][1], x[0][2] or "", x[0][0])):
        mode = "copy" if top == "transactional" else "jsonl"
        print(f"\n[{mode}] {top}/{layer or '-'}/{source}  files={len(info['files'])}", flush=True)
        res = extract_unit(source, top, layer, info)
        all_results.append(res)
        out_disp = str(res["out"].relative_to(WORK_ROOT)) + ("/" if res["copy_only"] else "")
        unit_label = "files" if res["copy_only"] else "records"
        print(f"         -> {out_disp}  {unit_label}={res['records']}", flush=True)
    extraction_dt = time.time() - ex_t0
    print(f"\nExtraction pass complete -- {len(all_results)} units in {extraction_dt:.1f}s")

    # ---- Validate + tokens ----
    vt_t0 = time.time()
    print("\n" + "=" * 80 + "\nValidation\n" + "=" * 80)
    issues = validate(all_results)
    tok_rows, tok_totals = tokenize_summary()
    vt_dt = time.time() - vt_t0

    # ---- summary.json ----
    total_dt = time.time() - t_main
    out_path = write_summary_json(
        units=units, skipped=skipped, all_results=all_results,
        tok_rows=tok_rows, tok_totals=tok_totals,
        extraction_dt=extraction_dt, vt_dt=vt_dt, total_dt=total_dt,
        issues=issues,
    )
    print(f"\nWrote {out_path}")
    print(f"\nDone in {total_dt:.1f}s ({total_dt/60:.1f} min). See summary.json for the curated stats.")


if __name__ == "__main__":
    sys.exit(main() or 0)
