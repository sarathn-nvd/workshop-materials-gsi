import argparse
import json
import glob
import os
import numpy as np
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math

# Global variable for worker process
worker_tokenizer = None

def init_worker(tokenizer_path):
    """Initialize the tokenizer in the worker process."""
    global worker_tokenizer
    try:
        worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer in worker: {e}")

def process_chunk(lines):
    """Process a list of JSON strings (lines)."""
    global worker_tokenizer
    if worker_tokenizer is None:
        return [], 0, 0
        
    token_counts = []
    local_tokens = 0
    local_records = 0
    
    has_chat_template = worker_tokenizer.chat_template is not None
    
    for line in lines:
        try:
            data = json.loads(line)
            messages = data.get("messages") or data.get("message")
            
            if messages is None:
                continue 

            count = 0
            if isinstance(messages, list):
                if has_chat_template:
                    try:
                        tokens = worker_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                        count = len(tokens)
                    except Exception:
                        text_content = ""
                        for msg in messages:
                            if isinstance(msg, dict) and 'content' in msg:
                                text_content += str(msg['content']) + "\n"
                        tokens = worker_tokenizer.encode(text_content, add_special_tokens=False)
                        count = len(tokens)
                else:
                    text_content = ""
                    for msg in messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            text_content += str(msg['content']) + "\n"
                    tokens = worker_tokenizer.encode(text_content, add_special_tokens=False)
                    count = len(tokens)
            elif isinstance(messages, str):
                tokens = worker_tokenizer.encode(messages, add_special_tokens=False)
                count = len(tokens)
            
            token_counts.append(count)
            local_tokens += count
            local_records += 1
            
        except Exception:
            continue
            
    return token_counts, local_records, local_tokens

def file_line_generator(file_paths, chunk_size=1000):
    """Yields chunks of lines from multiple files."""
    chunk = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if chunk:
        yield chunk

def count_total_lines(file_paths):
    """Quickly estimate or count total lines for progress bar."""
    total = 0
    # Using wc -l would be faster but this is python portable
    # For very large datasets, we might skip exact count or estimate
    print("Counting total lines for progress bar (this might take a moment)...")
    try:
        # Fast line count using buffer
        for fp in file_paths:
            with open(fp, "rb") as f:
                num_lines = sum(1 for _ in f)
            total += num_lines
    except Exception:
        pass
    return total

def main():
    parser = argparse.ArgumentParser(description="Analyze token statistics for JSONL datasets using multiprocessing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing jsonl files")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--output_file", type=str, default="corpus_details.log", help="Name of the output log file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: all available CPUs)")
    
    args = parser.parse_args()

    # Fixed chunk size for processing
    chunk_size = 1000

    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output_file)

    # Find JSONL files
    jsonl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"No jsonl files found in {args.input_dir}")
        return

    # Set workers to high count if not specified, but leave some room for system
    available_cpus = cpu_count()
    num_workers = args.workers if args.workers is not None else max(1, available_cpus - 4)
    
    print(f"Found {len(jsonl_files)} JSONL files.")
    print(f"System has {available_cpus} CPUs. Using {num_workers} workers.")
    
    total_lines = count_total_lines(jsonl_files)
    total_chunks = math.ceil(total_lines / chunk_size) if total_lines > 0 else None
    
    all_token_counts = []
    total_records = 0
    total_tokens = 0

    print(f"Processing with chunk size {chunk_size}...")

    # Use multiprocessing Pool
    with Pool(processes=num_workers, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
        # Create generator
        chunk_gen = file_line_generator(jsonl_files, chunk_size)
        
        # Use imap_unordered for better performance as order doesn't matter for stats
        # We assume total_chunks is accurate for the progress bar
        for result in tqdm(pool.imap_unordered(process_chunk, chunk_gen), total=total_chunks, desc="Processing chunks"):
            counts, recs, tokens = result
            
            if counts:
                all_token_counts.extend(counts)
                total_records += recs
                total_tokens += tokens

    if total_records == 0:
        print("No valid records found.")
        return

    print("Calculating statistics...")
    token_counts_arr = np.array(all_token_counts)
    avg_tokens = total_tokens / total_records
    
    # Sort for bucket analysis
    sorted_counts = np.sort(token_counts_arr)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
    perc_values = np.percentile(token_counts_arr, percentiles)
    
    # Calculate min/max for each percentile bucket
    bucket_edges = [0] + percentiles + [100]
    bucket_stats = []
    
    for i in range(len(bucket_edges) - 1):
        low_p = bucket_edges[i]
        high_p = bucket_edges[i + 1]
        
        low_idx = int(len(sorted_counts) * low_p / 100)
        high_idx = int(len(sorted_counts) * high_p / 100) - 1
        
        # Clamp indices
        low_idx = max(0, low_idx)
        high_idx = min(len(sorted_counts) - 1, high_idx)
        
        if low_idx <= high_idx:
            bucket_data = sorted_counts[low_idx:high_idx + 1]
            bucket_min = int(bucket_data.min())
            bucket_max = int(bucket_data.max())
            bucket_count = len(bucket_data)
            bucket_stats.append({
                "range": f"p{low_p}-p{high_p}" if high_p != 100 else f"p{low_p}-p100",
                "min": bucket_min,
                "max": bucket_max,
                "count": bucket_count
            })

    # Prepare output content
    output_lines = []
    output_lines.append("Corpus Statistics")
    output_lines.append("=================")
    output_lines.append(f"Total Tokens: {total_tokens}")
    output_lines.append(f"Total Records: {total_records}")
    output_lines.append(f"Average Tokens per Record: {avg_tokens:.2f}")
    output_lines.append(f"Min Tokens: {int(sorted_counts.min())}")
    output_lines.append(f"Max Tokens: {int(sorted_counts.max())}")
    output_lines.append("")
    output_lines.append("Token Count Percentiles:")
    
    for p, val in zip(percentiles, perc_values):
        key = f"p{p}" if p != 99.9 else "p99.9"
        output_lines.append(f"{key}: {val:.2f}")
    
    output_lines.append("")
    output_lines.append("Percentile Bucket Statistics (Min/Max tokens in each range):")
    output_lines.append("-" * 60)
    output_lines.append(f"{'Bucket':<15} {'Min Tokens':>12} {'Max Tokens':>12} {'Count':>12}")
    output_lines.append("-" * 60)
    
    for bucket in bucket_stats:
        output_lines.append(f"{bucket['range']:<15} {bucket['min']:>12} {bucket['max']:>12} {bucket['count']:>12}")
    
    output_lines.append("-" * 60)
    output_lines.append("")

    output_content = "\n".join(output_lines)
    
    with open(output_path, 'w') as f:
        f.write(output_content)
        
    print(output_content)
    print(f"Analysis complete. Results written to {output_path}")

if __name__ == "__main__":
    main()
