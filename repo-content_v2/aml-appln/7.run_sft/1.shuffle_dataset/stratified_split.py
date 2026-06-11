#!/usr/bin/env python3
"""Stratified train/val split for the SFT corpus.

Replaces (or supplements) the un-stratified terashuf flow in shuffle.py with
metadata-aware sampling so that the val slice preserves the per-stratum
distribution of the training data:

  * sar_judgment records   -> stratify on (task_type='sar_judgment', typology, is_suspicious)
  * auxiliary_* records    -> stratify on (task_type, source)

Within each stratum we pick `ceil(stratum_size * --val_pct)` records for val
(at least 1 per non-empty stratum) and the rest for train. Each output split
is then globally shuffled (seeded RNG) and written with ONLY the `messages`
field -- the metadata is dropped so the output is directly compatible with
nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset.

Why ceil + max(1, ...): every stratum should be represented in val so the
in-training val signal isn't blind to small typologies (smurfing|NEG=54,
auxiliary_numeric|financebench=25, etc.). The aggregate val share lands a
hair over the requested pct (~10.03% for val_pct=0.10 on 35k records with
25 strata) -- desired behavior, not a bug.

Usage:
    python3 stratified_split.py \
      --input_dir   data/raw \
      --output_dir  data/final \
      --dataset_name sft_mixed \
      --val_pct     0.10 \
      --seed        42
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Stratum key
# ---------------------------------------------------------------------------

def _detect_is_suspicious(content: str) -> str:
    """Return 'POS' / 'NEG' / '_unknown' for a sar_judgment assistant message.

    Fast path: substring match on the canonical JSON shape produced by
    json.dumps. Fall back to a full json.loads if the shape differs.
    """
    if '"is_suspicious": true' in content:
        return "POS"
    if '"is_suspicious": false' in content:
        return "NEG"
    try:
        obj = json.loads(content)
        v = obj.get("is_suspicious")
        if v is True:
            return "POS"
        if v is False:
            return "NEG"
    except Exception:
        pass
    return "_unknown"


def stratum_key(record: dict) -> Tuple[str, ...]:
    """Compute a hashable stratum key from a record's metadata + content."""
    md = record.get("metadata") or {}
    task_type = md.get("task_type") or "_unknown"

    if task_type == "sar_judgment":
        typology = md.get("typology") or "_unknown"
        msgs = record.get("messages") or []
        last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else {}
        label = _detect_is_suspicious(last.get("content", ""))
        return ("sar_judgment", typology, label)

    # auxiliary_* (and any unknown task types) -- preserve source mix too.
    source = md.get("source") or "_unknown"
    return (task_type, source)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(input_dir: str, extension: str = ".jsonl") -> List[dict]:
    files = sorted(glob.glob(os.path.join(input_dir, f"*{extension}")))
    if not files:
        raise SystemExit(f"No {extension} files found in {input_dir!r}")
    records: List[dict] = []
    for path in files:
        before = len(records)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
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
                msgs = r.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    continue
                records.append(r)
        print(f"  loaded {len(records) - before:>7} valid records from {os.path.basename(path)}",
              file=sys.stderr)
    return records


def write_split(records: List[dict], indices: Iterable[int], out_path: str) -> int:
    """Write only the `messages` field per record (drop metadata)."""
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in indices:
            r = records[i]
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

def stratified_split(
    records: List[dict],
    val_pct: float,
    seed: int,
):
    rng = random.Random(seed)

    by_stratum: dict[Tuple, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_stratum[stratum_key(r)].append(i)

    train_idx: List[int] = []
    val_idx:   List[int] = []
    stratum_stats: List[Tuple[Tuple, int, int, int]] = []  # (stratum, total, val, train)

    # Sort strata by their string representation for deterministic logging.
    for stratum in sorted(by_stratum.keys(), key=lambda x: tuple(map(str, x))):
        ids = list(by_stratum[stratum])
        rng.shuffle(ids)
        if not ids:
            continue
        n_val = max(1, math.ceil(len(ids) * val_pct))
        n_val = min(n_val, len(ids))
        val_idx.extend(ids[:n_val])
        train_idx.extend(ids[n_val:])
        stratum_stats.append((stratum, len(ids), n_val, len(ids) - n_val))

    # Globally shuffle each output split so neighboring records aren't from
    # the same stratum (helps in-training val to see a representative micro-batch
    # immediately rather than 'all layering | POS' first).
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx, stratum_stats


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def write_log(out_path: str, stats, train_count: int, val_count: int,
              val_pct: float, seed: int, input_files: List[str]) -> None:
    total = train_count + val_count
    lines = [
        f"Stratified SFT split log",
        f"========================",
        f"seed:    {seed}",
        f"val_pct: {val_pct * 100:.2f}%  (per-stratum ceil; aggregate may exceed slightly)",
        f"input files:",
    ]
    for p in input_files:
        lines.append(f"  - {p}")
    lines += [
        "",
        f"{'stratum':<58} {'total':>7} {'val':>7} {'train':>7} {'val %':>6}",
        "-" * 92,
    ]
    for stratum, t, v, tr in stats:
        s = " | ".join(str(x) for x in stratum)
        pct = (v / t * 100) if t else 0.0
        lines.append(f"{s:<58} {t:>7} {v:>7} {tr:>7} {pct:>5.1f}%")
    lines += [
        "-" * 92,
        f"{'TOTAL':<58} {total:>7} {val_count:>7} {train_count:>7} "
        f"{(val_count / total * 100 if total else 0):>5.2f}%",
        "",
    ]
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Stratified train/val split for SFT chat-format JSONL data."
    )
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing raw .jsonl files (each record must have "
                        "messages + metadata).")
    p.add_argument("--output_dir", required=True,
                   help="Directory where {dataset_name}.chunk.00.jsonl and "
                        "{dataset_name}.val.jsonl will be written.")
    p.add_argument("--dataset_name", default="sft_mixed",
                   help="Output filename prefix. Default: sft_mixed.")
    p.add_argument("--extension",    default=".jsonl",
                   help="Input file extension to glob. Default: .jsonl")
    p.add_argument("--val_pct",      type=float, default=0.10,
                   help="Per-stratum validation share. Default: 0.10 (= 10%%).")
    p.add_argument("--seed",         type=int,   default=42,
                   help="RNG seed for reproducibility. Default: 42.")
    p.add_argument("--log_file",     default=None,
                   help="Where to write the per-stratum split log. Default: "
                        "<output_dir>/stratified_split.log")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(args.output_dir, "stratified_split.log")

    print(f"Reading records from {args.input_dir}...", file=sys.stderr)
    records = load_records(args.input_dir, args.extension)
    print(f"  total records loaded: {len(records)}", file=sys.stderr)

    train_idx, val_idx, stats = stratified_split(records, args.val_pct, args.seed)

    train_path = os.path.join(args.output_dir, f"{args.dataset_name}.chunk.00.jsonl")
    val_path   = os.path.join(args.output_dir, f"{args.dataset_name}.val.jsonl")
    n_train = write_split(records, train_idx, train_path)
    n_val   = write_split(records, val_idx,   val_path)
    print(f"Wrote {n_train:>7} train records -> {train_path}", file=sys.stderr)
    print(f"Wrote {n_val:>7} val   records -> {val_path}",   file=sys.stderr)

    files_glob = sorted(glob.glob(os.path.join(args.input_dir, f"*{args.extension}")))
    write_log(log_path, stats, n_train, n_val, args.val_pct, args.seed, files_glob)


if __name__ == "__main__":
    main()
