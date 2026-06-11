# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for Local JSONL processing
#
# Local fork of `../../../reference/1.shuffle_dataset/shuffle.py`. The only
# change vs the reference is the validation-extraction step: it now takes a
# percentage of total records (--val_pct, default 10) and computes the
# per-chunk pull dynamically, so the script works correctly on small corpora.
# The reference hardcodes `head -n 10000` per chunk, which would consume the
# entire training corpus for our 75K-record level_1 (and the 8K-record level_2
# in run-v2). See ../README.md "Validation extraction" for the full math.

import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

def run_command(command):
    print(f"Running: {command}")
    # added executable='/bin/bash' to ensure pipe handling works correctly across systems
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')


def _open_for_extension(path, extension):
    """Yield text lines from a JSONL file regardless of compression."""
    if extension.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if extension.endswith(".zst"):
        # Shell out to zstdcat to avoid pulling in a zstd python dep.
        proc = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE, text=True)
        return proc.stdout
    return open(path, "r", encoding="utf-8")


def scan_input_dir(input_dir, extension):
    """Single-pass scan of every input file.

    Returns:
        total_records (int)        -- number of JSONL lines
        total_chars (int)          -- sum of len(rec["text"]) across all records
        per_source_records (dict)  -- {source: count} discovered from rec["source"]
        per_source_chars (dict)    -- {source: char_total}
        files (list[str])          -- input files actually scanned

    The per-source breakdowns let the summary cross-check against the
    `--curator_summary` (which has exact tokens-per-source) and report
    per-source percentages of train/val without needing a second pass.
    """
    pattern = os.path.join(input_dir, "**", f"*{extension}")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No files matching {extension} under {input_dir} -- check --input_dir / --extension"
        )

    total_records = 0
    total_chars = 0
    per_source_records = defaultdict(int)
    per_source_chars = defaultdict(int)
    skipped = 0

    for f in files:
        with _open_for_extension(f, extension) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                text = rec.get("text", "") or ""
                source = rec.get("source", os.path.splitext(os.path.basename(f))[0])
                total_records += 1
                total_chars += len(text)
                per_source_records[source] += 1
                per_source_chars[source] += len(text)

    if skipped:
        print(f"  WARNING: skipped {skipped} malformed JSON lines during scan", file=sys.stderr)
    return total_records, total_chars, dict(per_source_records), dict(per_source_chars), files


def count_chars_in_jsonl(path, extension=".jsonl"):
    """Count records + sum of len(text) for one JSONL file."""
    records, chars = 0, 0
    with _open_for_extension(path, extension) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            chars += len(rec.get("text", "") or "")
    return records, chars


def load_curator_ratio(curator_summary_path, sources_seen):
    """Compute exact chars-per-token ratio by summing curator's per-source stats.

    The curator (`../../../1.data_curation/`) tokenized every record with the
    target FP8 tokenizer and recorded per-source `chars` and `tokens`. By
    summing both across the sources we actually saw in the input directory,
    we get the precise chars-per-token ratio for *this corpus* -- no need to
    re-tokenize 9 GB of text just to estimate train/val token counts.

    Returns (ratio, per_source_ratio) or (None, None) if the file is
    missing / unreadable.
    """
    if not curator_summary_path or not os.path.exists(curator_summary_path):
        return None, None
    try:
        with open(curator_summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: could not load curator summary {curator_summary_path}: {e}", file=sys.stderr)
        return None, None

    total_chars, total_tokens = 0, 0
    per_source_ratio = {}
    by_layer = summary.get("by_layer", {})
    for _layer, sources in by_layer.items():
        for source, stats in sources.items():
            if source not in sources_seen:
                continue
            c = stats.get("chars")
            t = stats.get("tokens")
            if c and t:
                total_chars += c
                total_tokens += t
                per_source_ratio[source] = c / t
    if total_tokens == 0:
        return None, None
    return total_chars / total_tokens, per_source_ratio


def write_summary(out_dir, args, scan_stats, val_stats, train_stats, ratio_info, val_actual, k_validation):
    """Write data/summary.json with record/char/token breakdowns for train + val.

    Values for `tokens` are *estimates* unless `--curator_summary` was provided.
    With the curator summary, the chars-per-token ratio is exact for our corpus
    and tokenizer, so the token estimates are accurate to within ~0.1% (the
    only error source is the tiny chars-per-token variance between train and
    val draws of the same source distribution).
    """
    total_records, total_chars, per_source_records, per_source_chars = scan_stats
    val_records, val_chars = val_stats
    train_records, train_chars = train_stats
    ratio, per_source_ratio = ratio_info

    def pct(part, whole):
        return round(100.0 * part / whole, 4) if whole else 0.0

    def tok(chars):
        return int(chars / ratio) if ratio else None

    summary = {
        "schema_version": "1.0",
        "dataset_name": args.dataset_name,
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(out_dir),
        "seed": args.seed,
        "nchunks": args.nchunks,
        "val_pct_target": args.val_pct,
        "val_per_chunk": k_validation,
        "tokenizer_for_token_estimate": (
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 (via curator summary)"
            if ratio else "n/a (no --curator_summary)"
        ),
        "chars_per_token_ratio": round(ratio, 6) if ratio else None,
        "totals": {
            "records": total_records,
            "chars": total_chars,
            "tokens_estimate": tok(total_chars),
        },
        "splits": {
            "train": {
                "records": train_records,
                "records_pct": pct(train_records, total_records),
                "chars": train_chars,
                "chars_pct": pct(train_chars, total_chars),
                "tokens_estimate": tok(train_chars),
                "tokens_pct": pct(train_chars, total_chars),
                "files": [f"{args.dataset_name}.chunk.{i:0{max(2, len(str(args.nchunks - 1)))}d}.jsonl"
                          for i in range(args.nchunks)],
            },
            "validation": {
                "records": val_records,
                "records_pct": pct(val_records, total_records),
                "chars": val_chars,
                "chars_pct": pct(val_chars, total_chars),
                "tokens_estimate": tok(val_chars),
                "tokens_pct": pct(val_chars, total_chars),
                "files": [f"{args.dataset_name}.val.jsonl"],
            },
        },
        "per_source": {
            source: {
                "records": records,
                "chars": per_source_chars.get(source, 0),
                "tokens_estimate": (
                    int(per_source_chars.get(source, 0) / per_source_ratio[source])
                    if per_source_ratio and source in per_source_ratio
                    else (int(per_source_chars.get(source, 0) / ratio) if ratio else None)
                ),
                "share_of_corpus_pct": pct(records, total_records),
            }
            for source, records in sorted(per_source_records.items())
        },
        "notes": [
            "Validation records are pulled deterministically (top-N from each chunk after global shuffle), "
            "so val is a uniform random sample of the corpus -- per-source share in val matches per-source "
            "share in the full corpus to within sqrt(N) variance.",
            "tokens_estimate uses the chars-per-token ratio measured by the curator on this exact corpus "
            "with the FP8 base tokenizer. Exact per-shard token counts are emitted by Stage 2 "
            "(preprocess_megatron_dataset.py) in `processed_data_*_stats.json`.",
        ],
    }

    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSummary written: {out_path}")
    print(f"  records: train={train_records:,} ({pct(train_records, total_records)}%) | "
          f"val={val_records:,} ({pct(val_records, total_records)}%)")
    print(f"  chars  : train={train_chars:,} ({pct(train_chars, total_chars)}%) | "
          f"val={val_chars:,} ({pct(val_chars, total_chars)}%)")
    if ratio:
        print(f"  tokens : train={tok(train_chars):,} ({pct(train_chars, total_chars)}%) | "
              f"val={tok(val_chars):,} ({pct(val_chars, total_chars)}%)  "
              f"(chars/token={ratio:.4f})")
    else:
        print(f"  tokens : not estimated (pass --curator_summary to enable)")

def setup_terashuf(work_dir):
    """
    Downloads and compiles terashuf in the working directory.
    """
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    # Ensure directory exists
    os.makedirs(work_dir, exist_ok=True)
    
    if not os.path.exists(terashuf_dir):
        run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir

def main(args):
    # 1. Setup Paths
    dataset_name = args.dataset_name
    input_dir = os.path.abspath(args.input_dir)
    
    # Default output dir to ./data/{dataset_name}_shuffled if not provided
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.join(os.getcwd(), "data", f"{dataset_name}_shuffled")

    work_dir = os.path.dirname(out_dir) # Use parent of out_dir for terashuf compilation
    os.makedirs(out_dir, exist_ok=True)

    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {out_dir}")

    # 2. Configuration for processing
    prefix = f"{dataset_name}.chunk."
    suffix = ".jsonl"  # Output format is always jsonl
    # Split suffix must be wide enough for requested number of chunks
    suffix_length = max(2, len(str(args.nchunks - 1)))
    
    # Determine how to read the input files based on extension
    input_extension = args.extension
    if input_extension.endswith(".zst"):
        cat_command = "zstdcat {} && echo" # echo ensures newline between files
    elif input_extension.endswith(".gz"):
        cat_command = "zcat {} && echo"
    else:
        cat_command = "cat {}"

    # 2b. Single scan over input: counts records + chars per source.
    # The reference hardcoded `head -n 10000` per chunk, which assumed a
    # multi-million-doc corpus (its translation/transliteration data was
    # ~10 M docs). Our level_1 has ~75 K docs; with 32 chunks * 10 K val
    # the validation file would consume the entire training set. Instead we
    # size the val slice from --val_pct (% of total records).
    #
    # We collect chars + per-source counts here so the summary at the end can
    # report exact char shares + (optional) token shares without re-scanning.
    print("Scanning input directory (counts records + chars per source)...")
    t0 = time.time()
    total_records, total_chars, per_source_records, per_source_chars, _input_files = (
        scan_input_dir(input_dir, args.extension)
    )
    print(f"  Scanned in {time.time() - t0:.1f}s: {total_records:,} records | "
          f"{total_chars:,} chars | {len(per_source_records)} sources")

    val_total = max(1, int(round(total_records * args.val_pct / 100.0)))
    k_validation = max(1, val_total // args.nchunks)
    val_actual = k_validation * args.nchunks  # what we'll actually pull (rounding artifact)

    # Guardrail: refuse to proceed if the val slice would eat more than half
    # the corpus -- almost always a misconfigured --val_pct or --nchunks.
    if val_actual > total_records // 2:
        raise ValueError(
            f"Validation slice ({val_actual:,}) > 50% of total records ({total_records:,}). "
            f"Lower --val_pct or --nchunks. Current: --val_pct={args.val_pct} "
            f"--nchunks={args.nchunks} -> {k_validation:,} records pulled per chunk."
        )

    print(
        f"  Validation target  : {val_total:,} records ({args.val_pct}%)\n"
        f"  Per-chunk pull     : {k_validation:,}  (chunks: {args.nchunks})\n"
        f"  Actual val size    : {val_actual:,}  (rounding: -{val_total - val_actual})\n"
        f"  Train remainder    : {total_records - val_actual:,} records"
    )

    # 3. Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    # 4. Set up environment variables for terashuf
    os.environ["MEMORY"] = f"{args.memory}"
    os.environ["SEED"] = f"{args.seed}"
    
    # Ensure the ulimit is high enough for opening many input files
    ulimit_cmd = "ulimit -n 100000"

    # 5. Build the pipeline command
    # Logic: Find files -> Cat them -> Terashuf (Shuffle) -> Split into chunks
    pipeline_cmd = (
        f"{ulimit_cmd} && "
        # `-L` makes find dereference symlinks during traversal so symlinked
        # input files (e.g. run-v2/1.shuffle_dataset/data/level_2_split/{native,replay}/*.jsonl)
        # are read as their target. Without -L, `-type f` excludes symlinks and
        # terashuf gets zero input -- silently producing empty chunk files.
        f"find -L {input_dir} -type f -name '*{input_extension}' -print0 | "
        f"xargs -0 -I {{}} sh -c '{cat_command}' | "
        f"{terashuf_executable} | "
        f"split -n r/{args.nchunks} -d --suffix-length {suffix_length} --additional-suffix {suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )

    print("Starting shuffling and splitting pipeline...")
    run_command(pipeline_cmd)

    # 6. Create validation set and remove lines from chunks
    print("Extracting validation set...")
    validation_file = f"{out_dir}/{dataset_name}.val{suffix}"
    
    # Clear validation file if it exists from a previous run to avoid appending duplicates
    if os.path.exists(validation_file):
        os.remove(validation_file)

    for i in range(args.nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:0{suffix_length}d}{suffix}"
        
        if not os.path.exists(chunk_file):
            print(f"Warning: Chunk file {chunk_file} not found. Skipping validation extraction for this chunk.")
            continue

        # Take top K lines for validation
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        # Remove those K lines from the file
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    # 7. Compute char + token stats for the val split, derive train stats
    # by subtraction, and write summary.json.
    print("\nMeasuring validation file...")
    val_records_actual, val_chars = count_chars_in_jsonl(validation_file, suffix)
    train_records = total_records - val_records_actual
    train_chars = total_chars - val_chars
    if val_records_actual != val_actual:
        print(
            f"  NOTE: val file has {val_records_actual:,} records vs targeted "
            f"{val_actual:,} (terashuf can drop empty lines or trailing junk; "
            f"this is normal at single-digit deltas)"
        )

    ratio_info = load_curator_ratio(args.curator_summary, set(per_source_records))
    if ratio_info[0] is None:
        print(
            f"  NOTE: --curator_summary not provided or unreadable; token counts in "
            f"summary.json will be null. Tried: {args.curator_summary!r}"
        )

    write_summary(
        out_dir, args,
        scan_stats=(total_records, total_chars, per_source_records, per_source_chars),
        val_stats=(val_records_actual, val_chars),
        train_stats=(train_records, train_chars),
        ratio_info=ratio_info,
        val_actual=val_actual,
        k_validation=k_validation,
    )

    print("\nAll tasks completed successfully!")
    print(f"Data available in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle and split local JSONL files.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing source .jsonl files")
    parser.add_argument("--dataset_name", type=str, default="custom_dataset", help="Name to prefix output files (e.g., 'fineweb')")
    parser.add_argument("--output_dir", type=str, default=None, help="Path where shuffled chunks will be saved")
    
    parser.add_argument("--extension", type=str, default=".jsonl", help="File extension to look for (e.g., .jsonl, .jsonl.zst)")
    parser.add_argument("--memory", type=float, default=8, help="Memory in GB for terashuf")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--nchunks", type=int, default=32, help="Number of chunks to split into")
    parser.add_argument(
        "--val_pct", type=float, default=10.0,
        help="Percentage of total records to pull off the top of every chunk for the "
             "validation file (default: 10.0). Total val size = round(total_records * val_pct/100), "
             "distributed evenly across all chunks. Errors out if val would exceed 50%% of corpus."
    )
    parser.add_argument(
        "--curator_summary",
        type=str,
        default="../../../1.data_curation/summary.json",
        help="Path (absolute or relative-to-cwd) to ../1.data_curation/summary.json. Used to "
             "compute the chars-per-token ratio for this corpus and tokenizer, which the script "
             "applies to per-split chars to estimate per-split token counts in summary.json. "
             "Pass an empty string or a non-existent path to skip token estimation."
    )

    args = parser.parse_args()

    # Basic check for Linux/Mac environment because of 'split', 'sed', 'cat', 'ulimit' usage
    if os.name == 'nt':
        print("Error: This script is designed for Linux/macOS environments (requires bash, sed, split, terashuf).")
        sys.exit(1)

    main(args)