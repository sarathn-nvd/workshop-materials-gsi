#!/usr/bin/env python3
"""Generate `data/final/distribution.md` describing the train + val split
in the same shape as `data/raw/distribution.md`, plus a per-stratum
sampling-rate verification table.

The output JSONL files only contain `messages` (metadata is stripped at
write time -- see stratified_split.py). To recover per-record metadata
for the distribution audit, we fingerprint each record on the SHA-1 of
its USER message content (records have unique user prompts), build a
fingerprint -> metadata index from `data/raw/`, then look each split
record up in that index.

Usage:
    python3 analyze_split.py \
      --raw_dir   data/raw \
      --split_dir data/final \
      --output    data/final/distribution.md
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def fp(record: dict) -> str:
    """SHA-1 fingerprint of the USER message content -- unique per record."""
    msgs = record.get("messages") or []
    user = ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            user = m.get("content", "")
            break
    return hashlib.sha1(user.encode("utf-8", errors="replace")).hexdigest()


def detect_label(record: dict) -> str:
    """Return 'POS' / 'NEG' / '_unknown' from a sar_judgment assistant message."""
    msgs = record.get("messages") or []
    last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else {}
    c = last.get("content", "") if isinstance(last, dict) else ""
    if '"is_suspicious": true' in c:
        return "POS"
    if '"is_suspicious": false' in c:
        return "NEG"
    try:
        v = json.loads(c).get("is_suspicious")
        return "POS" if v is True else ("NEG" if v is False else "_unknown")
    except Exception:
        return "_unknown"


def load_raw_index(raw_dir: str) -> Dict[str, dict]:
    """Build {fingerprint -> {metadata, label}} from the raw .jsonl files."""
    index: Dict[str, dict] = {}
    files = sorted(glob.glob(os.path.join(raw_dir, "*.jsonl")))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not isinstance(r, dict):
                    continue
                md = r.get("metadata") or {}
                index[fp(r)] = {
                    "task_type":   md.get("task_type") or "_unknown",
                    "typology":    md.get("typology") or "_unknown",
                    "source":      md.get("source")   or "_unknown",
                    "sar_variant": md.get("sar_variant") or "_unknown",
                    "label":       detect_label(r),
                }
    return index


def load_split(path: str, raw_index: Dict[str, dict]) -> List[dict]:
    """Read a split JSONL and attach metadata via the fingerprint index."""
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            md = raw_index.get(fp(r))
            if md is not None:
                rows.append(md)
    return rows


# ---------------------------------------------------------------------------
# Pretty markdown helpers
# ---------------------------------------------------------------------------

def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _share(n: int, total: int) -> str:
    return f"{(n / total * 100):.2f}%" if total else "-"


def md_section_header(title: str) -> str:
    return f"\n## {title}\n"


def md_kv_table(rows: List[Tuple[str, int]], total: int, key_label: str) -> str:
    lines = [f"| {key_label} | Records | Share |", "|---|---:|---:|"]
    for k, n in rows:
        lines.append(f"| {k} | {_fmt_int(n)} | {_share(n, total)} |")
    lines.append(f"| **Total** | **{_fmt_int(total)}** | 100.00% |")
    return "\n".join(lines) + "\n"


def md_split_kv_table(
    rows: List[Tuple[str, int, int]],
    total_train: int, total_val: int, key_label: str,
) -> str:
    lines = [
        f"| {key_label} | Train | Val | Total | Train % | Val % | Val of total | Val % of stratum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, tr, v in rows:
        tot = tr + v
        lines.append(
            f"| {k} | {_fmt_int(tr)} | {_fmt_int(v)} | {_fmt_int(tot)} | "
            f"{_share(tr, total_train)} | {_share(v, total_val)} | "
            f"{_share(tot, total_train + total_val)} | {_share(v, tot)} |"
        )
    lines.append(
        f"| **Total** | **{_fmt_int(total_train)}** | **{_fmt_int(total_val)}** | "
        f"**{_fmt_int(total_train + total_val)}** | 100.00% | 100.00% | 100.00% | "
        f"{_share(total_val, total_train + total_val)} |"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def by(field: str, rows: List[dict]) -> Counter:
    return Counter(r[field] for r in rows)


def by_pair(field_a: str, field_b: str, rows: List[dict]) -> Counter:
    return Counter((r[field_a], r[field_b]) for r in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",   required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--train_file", default="sft_mixed.chunk.00.jsonl")
    p.add_argument("--val_file",   default="sft_mixed.val.jsonl")
    p.add_argument("--output",    default=None,
                   help="Path to distribution.md (default: <split_dir>/distribution.md)")
    args = p.parse_args()

    out_path = args.output or os.path.join(args.split_dir, "distribution.md")

    print("Building raw fingerprint index...")
    idx = load_raw_index(args.raw_dir)
    print(f"  {_fmt_int(len(idx))} unique records indexed")

    train_path = os.path.join(args.split_dir, args.train_file)
    val_path   = os.path.join(args.split_dir, args.val_file)
    print(f"Loading train split: {train_path}")
    train = load_split(train_path, idx)
    print(f"  {_fmt_int(len(train))} train records joined to metadata")
    print(f"Loading val   split: {val_path}")
    val = load_split(val_path, idx)
    print(f"  {_fmt_int(len(val))} val records joined to metadata")

    train_size = sum(1 for _ in open(train_path)) if os.path.exists(train_path) else 0
    val_size   = sum(1 for _ in open(val_path))   if os.path.exists(val_path)   else 0
    if len(train) != train_size:
        print(f"  WARNING: {train_size - len(train)} train records did not match a raw fingerprint")
    if len(val) != val_size:
        print(f"  WARNING: {val_size   - len(val)} val   records did not match a raw fingerprint")

    # ----- begin building markdown -----
    md = []
    md.append("# SFT Corpus — Stratified Train/Val Distribution\n")
    md.append("Companion to `../raw/distribution.md`. Reports the realized "
              "distribution of the **stratified** split produced by "
              "`../stratified_split.py`. Every per-stratum table mirrors the "
              "categorical axes documented in the raw `distribution.md` so the "
              "two files can be diffed side by side.\n")
    md.append(f"\nGenerated from:\n"
              f"- raw:   `{os.path.abspath(args.raw_dir)}`\n"
              f"- split: `{os.path.abspath(args.split_dir)}`\n")

    # Files section
    md.append(md_section_header("Files"))
    train_bytes = os.path.getsize(train_path) if os.path.exists(train_path) else 0
    val_bytes   = os.path.getsize(val_path)   if os.path.exists(val_path)   else 0
    md.append(
        "| File | Records | Size |\n"
        "|---|---:|---:|\n"
        f"| `{args.train_file}` | {_fmt_int(len(train))} | {train_bytes/1024/1024:.1f} MB |\n"
        f"| `{args.val_file}` | {_fmt_int(len(val))} | {val_bytes/1024/1024:.1f} MB |\n"
        f"| **TOTAL** | **{_fmt_int(len(train) + len(val))}** | "
        f"**{(train_bytes + val_bytes)/1024/1024:.1f} MB** |\n"
    )

    # Top-level split (sar_judgment vs auxiliary_*)
    md.append(md_section_header("Top-level split (sar_judgment vs auxiliary_*)"))
    def grp_top(rows):
        c = Counter()
        for r in rows:
            if r["task_type"] == "sar_judgment":
                c["sar_judgment"] += 1
            else:
                c["auxiliary_*"] += 1
        return c
    tcnt = grp_top(train); vcnt = grp_top(val)
    rows = [(k, tcnt.get(k, 0), vcnt.get(k, 0)) for k in ("sar_judgment", "auxiliary_*")]
    md.append(md_split_kv_table(rows, len(train), len(val), "Group"))

    sar_tr = [r for r in train if r["task_type"] == "sar_judgment"]
    sar_val = [r for r in val   if r["task_type"] == "sar_judgment"]
    aux_tr = [r for r in train if r["task_type"] != "sar_judgment"]
    aux_val = [r for r in val   if r["task_type"] != "sar_judgment"]

    # Variant mix (sar_judgment only)
    md.append(md_section_header("Variant mix (`sar_judgment` only)"))
    keys = sorted(set(r["sar_variant"] for r in sar_tr + sar_val))
    rows = [(k, by("sar_variant", sar_tr).get(k, 0), by("sar_variant", sar_val).get(k, 0)) for k in keys]
    md.append(md_split_kv_table(rows, len(sar_tr), len(sar_val), "Variant"))

    # Label mix (sar_judgment only)
    md.append(md_section_header("Label mix (`sar_judgment` only)"))
    keys = ["POS", "NEG", "_unknown"]
    rows = [(k, by("label", sar_tr).get(k, 0), by("label", sar_val).get(k, 0)) for k in keys
            if (by("label", sar_tr).get(k, 0) + by("label", sar_val).get(k, 0)) > 0]
    md.append(md_split_kv_table(rows, len(sar_tr), len(sar_val), "Label"))

    # Typology distribution (sar_judgment only) -- with POS/NEG breakdowns
    md.append(md_section_header("Typology distribution (`sar_judgment` only)"))
    typs = sorted(set(r["typology"] for r in sar_tr + sar_val))
    lines = [
        "| Typology | Train | Val | Total | Train POS | Train NEG | Val POS | Val NEG | Val % of stratum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in typs:
        tr = [r for r in sar_tr if r["typology"] == t]
        vl = [r for r in sar_val if r["typology"] == t]
        tot = len(tr) + len(vl)
        tr_pos = sum(1 for r in tr if r["label"] == "POS")
        tr_neg = sum(1 for r in tr if r["label"] == "NEG")
        vl_pos = sum(1 for r in vl if r["label"] == "POS")
        vl_neg = sum(1 for r in vl if r["label"] == "NEG")
        lines.append(
            f"| {t} | {_fmt_int(len(tr))} | {_fmt_int(len(vl))} | {_fmt_int(tot)} | "
            f"{_fmt_int(tr_pos)} | {_fmt_int(tr_neg)} | {_fmt_int(vl_pos)} | {_fmt_int(vl_neg)} | "
            f"{_share(len(vl), tot)} |"
        )
    md.append("\n".join(lines) + "\n")

    # Source distribution (sar_judgment only)
    md.append(md_section_header("Source distribution (`sar_judgment` only)"))
    keys = sorted(set(r["source"] for r in sar_tr + sar_val))
    rows = [(k, by("source", sar_tr).get(k, 0), by("source", sar_val).get(k, 0)) for k in keys]
    md.append(md_split_kv_table(rows, len(sar_tr), len(sar_val), "Source"))

    # Auxiliary task mix
    md.append(md_section_header("Auxiliary task mix"))
    keys = sorted(set(r["task_type"] for r in aux_tr + aux_val))
    rows = [(k, by("task_type", aux_tr).get(k, 0), by("task_type", aux_val).get(k, 0)) for k in keys]
    md.append(md_split_kv_table(rows, len(aux_tr), len(aux_val), "Task"))

    # Auxiliary source distribution per task
    md.append(md_section_header("Auxiliary source distribution (per task)"))
    for task in keys:
        tr = [r for r in aux_tr if r["task_type"] == task]
        vl = [r for r in aux_val if r["task_type"] == task]
        md.append(f"\n### `{task}` ({_fmt_int(len(tr) + len(vl))} records)\n")
        srcs = sorted(set(r["source"] for r in tr + vl))
        rows_t = [(s, by("source", tr).get(s, 0), by("source", vl).get(s, 0)) for s in srcs]
        md.append(md_split_kv_table(rows_t, len(tr), len(vl), "Source"))

    # Stratification verification (per-stratum sampling rate)
    md.append(md_section_header("Stratification verification (val % per stratum)"))
    md.append(
        "Per-stratum val sampling rate target was 10% (per-stratum `ceil` so every "
        "non-empty stratum gets >=1 val record). Smallest cells over-sample slightly "
        "by design.\n\n"
    )
    # Build (stratum_str -> (train_n, val_n))
    strat_counts: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    def strat_key(r: dict) -> str:
        if r["task_type"] == "sar_judgment":
            return f"sar_judgment | {r['typology']} | {r['label']}"
        return f"{r['task_type']} | {r['source']}"
    for r in train:
        k = strat_key(r); t, v = strat_counts[k]; strat_counts[k] = (t + 1, v)
    for r in val:
        k = strat_key(r); t, v = strat_counts[k]; strat_counts[k] = (t, v + 1)
    lines = [
        "| Stratum | Train | Val | Total | Val % of stratum |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in sorted(strat_counts.keys()):
        t, v = strat_counts[k]
        tot = t + v
        lines.append(f"| {k} | {_fmt_int(t)} | {_fmt_int(v)} | {_fmt_int(tot)} | {_share(v, tot)} |")
    lines.append(
        f"| **TOTAL** | **{_fmt_int(len(train))}** | **{_fmt_int(len(val))}** | "
        f"**{_fmt_int(len(train) + len(val))}** | "
        f"**{_share(len(val), len(train) + len(val))}** |"
    )
    md.append("\n".join(lines) + "\n")

    # Schema reminder
    md.append(md_section_header("Schema in the split files"))
    md.append(
        "After stratified_split.py, only the `messages` field is retained -- "
        "metadata was stripped to match the `ChatDataset` input format used by the "
        "SFT recipe (`4.run_sft/4.run_sft/recipe_a100sxm-8.yaml`):\n\n"
        "```json\n"
        "{\n"
        "  \"messages\": [\n"
        "    {\"role\": \"system\",    \"content\": \"...\"},\n"
        "    {\"role\": \"user\",      \"content\": \"...\"},\n"
        "    {\"role\": \"assistant\", \"content\": \"...\"}\n"
        "  ]\n"
        "}\n"
        "```\n"
    )

    out = "\n".join(md)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
