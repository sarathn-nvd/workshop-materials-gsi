import argparse
import os
import subprocess
import sys
import json

def run_command(command, verbose=True):
    if verbose:
        print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')

def setup_terashuf(work_dir):
    """
    Downloads and compiles terashuf in the working directory.
    """
    # Force terashuf to be in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    terashuf_dir = os.path.join(script_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    # Ensure directory exists
    
    if not os.path.exists(terashuf_dir):
        run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir

def main(args):
    # 1. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = args.dataset_name
    input_dir = os.path.abspath(args.input_dir)
    
    # Output directory for chunks: ./data
    out_dir = os.path.join(script_dir, "data")
    
    # Log file at script level
    log_file = os.path.join(script_dir, "shuffle_log.txt")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Log File: {log_file}")
    print(f"Terashuf Directory: {os.path.join(script_dir, 'terashuf')}")

    # 2. Configuration for processing
    prefix = f"{dataset_name}.chunk."
    suffix = ".jsonl"
    suffix_length = max(2, len(str(args.nchunks - 1)))
    
    # 3. Setup terashuf (force local dir)
    terashuf_dir = setup_terashuf(script_dir)
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    # 4. Set up environment variables for terashuf
    os.environ["MEMORY"] = f"{args.memory}"
    os.environ["SEED"] = f"{args.seed}"
    
    ulimit_cmd = "ulimit -n 100000"

    # 5. Pre-processing: Extract relevant keys and normalize

    extract_script = """
import sys, json, io

# Use a text wrapper that replaces bad chars instead of crashing
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for line in sys.stdin:
    try:
        line = line.strip()
        if not line: continue
        # Ignore lines that are clearly not JSON start/end (artifacts of parallel cat interleaving)
        if not (line.startswith('{') and line.endswith('}')): continue
        
        data = json.loads(line)
        if 'messages' in data and isinstance(data['messages'], list):
            print(json.dumps({'messages': data['messages']}, ensure_ascii=False))
    except Exception:
        continue
"""
    # Escape quotes for shell
    extract_script_cmd = f"python3 -c \"{extract_script}\""

    # Determine input extension handling from args
    input_extension = args.extension
    if input_extension.endswith(".zst"):
        cat_command = "zstdcat {} && echo" # echo ensures newline between files
    elif input_extension.endswith(".gz"):
        cat_command = "zcat {} && echo"
    else:
        cat_command = "cat {}"

    # 6. Build the pipeline command
    
    pipeline_cmd = (
        f"{ulimit_cmd} && "
        f"find {input_dir} -type f -name '*{input_extension}' -print0 | "
        f"xargs -0 -I {{}} sh -c '{cat_command}' | "
        f"{extract_script_cmd} | "
        f"{terashuf_executable} | "
        f"split -n r/{args.nchunks} -d --suffix-length {suffix_length} --additional-suffix {suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )

    print("Starting shuffling and splitting pipeline...")
    # Run the pipeline without printing the massive command to avoid log clutter
    run_command(pipeline_cmd, verbose=False)

    # 7. Create validation and test sets (SFT typically needs dedicated splits)
    print("Extracting validation and test sets...")
    validation_file = f"{out_dir}/{dataset_name}.val{suffix}"
    test_file = f"{out_dir}/{dataset_name}.test{suffix}"
    
    if os.path.exists(validation_file):
        os.remove(validation_file)
    if os.path.exists(test_file):
        os.remove(test_file)
    
    k_validation = args.val_samples_per_chunk
    k_test = args.test_samples_per_chunk
    total_to_extract = k_validation + k_test

    for i in range(args.nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:0{suffix_length}d}{suffix}"
        
        if not os.path.exists(chunk_file):
            print(f"Warning: Chunk file {chunk_file} not found. Skipping extraction for this chunk.")
            continue

        # Take first K lines for validation
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        
        # Take next M lines for test (lines k_validation+1 to k_validation+k_test)
        if k_test > 0:
            run_command(f"sed -n '{k_validation + 1},{total_to_extract}p' {chunk_file} >> {test_file}")
        
        # Remove all extracted lines (first total_to_extract lines) from the chunk
        run_command(f"sed -i '1,{total_to_extract}d' {chunk_file}")

    print("All tasks completed successfully!")
    print(f"Data available in: {out_dir}")

    # 8. Log line counts
    print(f"Calculating line counts and writing to {log_file}...")
    
    total_train = 0
    total_val = 0
    total_test = 0

    with open(log_file, "w") as f:
        f.write("Filename\tLine Count\n")
        f.write("-" * 30 + "\n")
        
        # Count validation file
        if os.path.exists(validation_file):
            # wc -l returns "  123 filename", using split to get just the number
            count_output = subprocess.check_output(f"wc -l {validation_file}", shell=True).decode().strip().split()[0]
            f.write(f"{os.path.basename(validation_file)}\t{count_output}\n")
            print(f"  {os.path.basename(validation_file)}: {count_output}")
            try:
                total_val += int(count_output)
            except ValueError:
                pass

        # Count test file
        if os.path.exists(test_file):
            count_output = subprocess.check_output(f"wc -l {test_file}", shell=True).decode().strip().split()[0]
            f.write(f"{os.path.basename(test_file)}\t{count_output}\n")
            print(f"  {os.path.basename(test_file)}: {count_output}")
            try:
                total_test += int(count_output)
            except ValueError:
                pass

        # Count chunks
        for i in range(args.nchunks):
            chunk_file = f"{out_dir}/{prefix}{i:0{suffix_length}d}{suffix}"
            if os.path.exists(chunk_file):
                count_output = subprocess.check_output(f"wc -l {chunk_file}", shell=True).decode().strip().split()[0]
                f.write(f"{os.path.basename(chunk_file)}\t{count_output}\n")
                print(f"  {os.path.basename(chunk_file)}: {count_output}")
                try:
                    total_train += int(count_output)
                except ValueError:
                    pass
        
        f.write("-" * 30 + "\n")
        f.write(f"Total Train Records\t{total_train}\n")
        f.write(f"Total Validation Records\t{total_val}\n")
        f.write(f"Total Test Records\t{total_test}\n")
        f.write(f"Grand Total\t{total_train + total_val + total_test}\n")

    print(f"Total Train: {total_train}")
    print(f"Total Val: {total_val}")
    print(f"Total Test: {total_test}")
    print("Logging complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle, normalize, and split SFT JSONL files.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing source .jsonl files")
    parser.add_argument("--dataset_name", type=str, default="sft_dataset", help="Name to prefix output files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path where shuffled chunks will be saved")
    
    parser.add_argument("--extension", type=str, default=".jsonl", help="File extension to look for (e.g., .jsonl, .jsonl.zst)")
    parser.add_argument("--memory", type=float, default=8, help="Memory in GB for terashuf")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--nchunks", type=int, default=1, help="Number of chunks to split into (Default 1 for SFT)")
    parser.add_argument("--val_samples_per_chunk", type=int, default=500, help="Number of samples to move to validation set per chunk")
    parser.add_argument("--test_samples_per_chunk", type=int, default=500, help="Number of samples to move to test set per chunk")

    args = parser.parse_args()


    if os.name == 'nt':
        print("Error: This script is designed for Linux/macOS environments.")
        sys.exit(1)

    main(args)

