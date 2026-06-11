import json
import glob
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import argparse

# Global tokenizer for worker processes
worker_tokenizer = None
MAX_TOKEN_LIMIT = 15000

def init_worker(tokenizer_path):
    """Initialize tokenizer in worker process."""
    global worker_tokenizer
    if tokenizer_path:
        from transformers import AutoTokenizer
        worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def count_tokens(messages):
    """Count tokens using chat template."""
    global worker_tokenizer
    if worker_tokenizer is None:
        return 0
    try:
        tokens = worker_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        return len(tokens)
    except Exception:
        return 0

def process_line(args):
    """Process a single line and return validation result."""
    line_num, line = args
    result = {
        "line_num": line_num,
        "valid": True,
        "line": line,
        "error": None,
        "keys": None,
        "msg_struct": None,
        "token_count": 0,
        "removed_reason": None
    }
    
    if not line.strip():
        result["valid"] = False
        result["error"] = "empty line"
        return result
    
    try:
        data = json.loads(line)
        result["keys"] = tuple(sorted(data.keys()))
        
        if 'messages' not in data:
            result["valid"] = False
            result["error"] = "no messages field"
            return result
            
        messages = data['messages']
        
        if not isinstance(messages, list):
            result["valid"] = False
            result["error"] = "messages is not a list"
            return result
            
        if not messages:
            result["valid"] = False
            result["error"] = "messages list is empty"
            return result
        
        # Check message structure
        msg_structs = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                result["valid"] = False
                result["error"] = f"message {i} is not a dict"
                return result
            if 'role' not in msg or 'content' not in msg:
                result["valid"] = False
                result["error"] = f"message {i} missing role or content"
                return result
            msg_keys = sorted(msg.keys())
            msg_structs.append(f"{msg.get('role', 'unknown')}:{tuple(msg_keys)}")
        
        result["msg_struct"] = tuple(msg_structs)
        
        # Check if last message is from assistant
        if messages[-1].get('role') != 'assistant':
            result["valid"] = False
            result["removed_reason"] = "last_message_not_assistant"
            return result
        
        # Check token count
        token_count = count_tokens(messages)
        result["token_count"] = token_count
        
        if token_count > MAX_TOKEN_LIMIT:
            result["valid"] = False
            result["removed_reason"] = "exceeds_token_limit"
            return result
            
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["error"] = f"JSON decode error: {str(e)}"
    
    return result

def process_file_chunk(lines_with_nums):
    """Process a chunk of lines."""
    return [process_line(args) for args in lines_with_nums]

def analyze_and_clean_jsonl_structure(directory, output_directory, tokenizer_path=None, num_workers=None):
    log_output = []
    
    def log(message):
        print(message)
        log_output.append(message)

    files = glob.glob(os.path.join(directory, "*.jsonl"))
    
    if not files:
        log(f"No jsonl files found in {directory}")
        return

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        log(f"Created output directory: {output_directory}")

    if tokenizer_path:
        log(f"Using tokenizer: {tokenizer_path}")
        log(f"Max token limit: {MAX_TOKEN_LIMIT}")
    else:
        log("[WARN] No tokenizer path provided. Token filtering will be skipped.")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    log(f"Using {num_workers} workers for parallel processing")

    log(f"\nFound {len(files)} JSONL files. Analyzing and cleaning...\n")

    global_structure_stats = defaultdict(list)
    global_message_structure_stats = defaultdict(list)
    file_cleaning_stats = {}
    
    # Global counters for removal reasons
    total_removed_not_assistant = 0
    total_removed_token_limit = 0
    total_removed_malformed = 0

    for file_path in sorted(files):
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_directory, file_name)
        log(f"Processing {file_name} -> {output_path}...")
        
        # Read all lines
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = [(i+1, line.strip()) for i, line in enumerate(f) if line.strip()]
        except Exception as e:
            log(f"  Error reading file: {e}")
            continue
        
        records_in_file = len(lines)
        log(f"  Read {records_in_file} records. Processing in parallel...")
        
        # Process in chunks using multiprocessing
        chunk_size = 1000
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        
        all_results = []
        with Pool(processes=num_workers, initializer=init_worker, initargs=(tokenizer_path,)) as pool:
            for chunk_results in pool.imap(process_file_chunk, chunks):
                all_results.extend(chunk_results)
        
        # Analyze results
        structure_map = defaultdict(int)
        message_structure_map = defaultdict(int)
        malformed_lines = []
        removed_not_assistant = 0
        removed_token_limit = 0
        records_kept = 0
        
        # Write valid records
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for result in all_results:
                if result["keys"]:
                    structure_map[result["keys"]] += 1
                if result["msg_struct"]:
                    message_structure_map[result["msg_struct"]] += 1
                
                if result["valid"]:
                    f_out.write(result["line"] + '\n')
                    records_kept += 1
                else:
                    if result["error"]:
                        malformed_lines.append((result["line_num"], result["error"]))
                    elif result["removed_reason"] == "last_message_not_assistant":
                        removed_not_assistant += 1
                    elif result["removed_reason"] == "exceeds_token_limit":
                        removed_token_limit += 1
        
        records_removed = records_in_file - records_kept
        
        # Update global counters
        total_removed_not_assistant += removed_not_assistant
        total_removed_token_limit += removed_token_limit
        total_removed_malformed += len(malformed_lines)
        
        file_cleaning_stats[file_name] = {
            "original": records_in_file,
            "removed": records_removed,
            "kept": records_kept,
            "removed_not_assistant": removed_not_assistant,
            "removed_token_limit": removed_token_limit,
            "removed_malformed": len(malformed_lines)
        }

        # Report findings for the file
        if malformed_lines:
            log(f"  [ERROR] Found {len(malformed_lines)} malformed lines:")
            for ln, err in malformed_lines[:5]:
                log(f"    Line {ln}: {err}")
            if len(malformed_lines) > 5:
                log(f"    ... and {len(malformed_lines) - 5} more.")
        
        if len(structure_map) == 0:
            log("  [WARN] File is empty or contains no valid JSON.")
        elif len(structure_map) == 1:
            keys = list(structure_map.keys())[0]
            log(f"  [OK] Consistent structure. Keys: {keys}")
            global_structure_stats[keys].append(file_name)
        else:
            log(f"  [WARN] Inconsistent structures found ({len(structure_map)} variants):")
            for keys, count in structure_map.items():
                log(f"    Keys: {keys} (Count: {count})")
                global_structure_stats[keys].append(f"{file_name} (partial)")
        
        # Report message structure findings
        if len(message_structure_map) == 0:
            log("  [WARN] No valid messages found.")
        elif len(message_structure_map) == 1:
            msg_struct = list(message_structure_map.keys())[0]
            log(f"  [OK] Consistent message structure: {msg_struct}")
            global_message_structure_stats[msg_struct].append(file_name)
        else:
            log(f"  [INFO] Multiple message structures found ({len(message_structure_map)} variants)")
            global_message_structure_stats["Mixed/Multiple"].append(file_name)

        log(f"  [CLEAN] Original: {records_in_file}, Kept: {records_kept}, Removed: {records_removed}")
        log(f"          - Removed (not ending with assistant): {removed_not_assistant}")
        log(f"          - Removed (exceeds {MAX_TOKEN_LIMIT} tokens): {removed_token_limit}")
        log(f"          - Removed (malformed): {len(malformed_lines)}")
        log("-" * 50)

    log("\n=== GLOBAL SUMMARY ===")
    for keys, filenames in global_structure_stats.items():
        log(f"\nStructure: {keys}")
        log(f"Found in {len(filenames)} files:")
        for fn in filenames:
            log(f"  - {fn}")

    log("\n=== MESSAGE STRUCTURE SUMMARY ===")
    for msg_struct, filenames in global_message_structure_stats.items():
        log(f"\nMessage Structure: {msg_struct}")
        log(f"Found in {len(filenames)} files:")
        for fn in filenames:
            log(f"  - {fn}")

    log("\n=== RECORD COUNTS & CLEANING STATS ===")
    total_original = 0
    total_kept = 0
    total_removed = 0
    for file_name in sorted(file_cleaning_stats.keys()):
        stats = file_cleaning_stats[file_name]
        log(f"{file_name}: Original={stats['original']}, Kept={stats['kept']}, Removed={stats['removed']}")
        total_original += stats['original']
        total_kept += stats['kept']
        total_removed += stats['removed']
    
    log(f"\n{'='*60}")
    log(f"TOTAL: Original={total_original}, Kept={total_kept}, Removed={total_removed}")
    log(f"{'='*60}")
    log(f"\nREMOVAL BREAKDOWN:")
    log(f"  - Records not ending with assistant: {total_removed_not_assistant}")
    log(f"  - Records exceeding {MAX_TOKEN_LIMIT} tokens: {total_removed_token_limit}")
    log(f"  - Malformed records: {total_removed_malformed}")
    log(f"{'='*60}")

    # Write to log file
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_and_clean_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_output))
    print(f"\nAnalysis and cleaning log written to {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and clean JSONL files for SFT training.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer for token counting")
    parser.add_argument("--max_tokens", type=int, default=15000,
                        help="Maximum token limit (default: 15000)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for cleaned files")
    
    args = parser.parse_args()

    # Update global max token limit
    MAX_TOKEN_LIMIT = args.max_tokens

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = args.input_dir or os.path.join(script_dir, "raw")
    output_dir = args.output_dir or os.path.join(script_dir, "fixed")

    if os.path.exists(target_dir):
        analyze_and_clean_jsonl_structure(
            target_dir, 
            output_dir, 
            tokenizer_path=args.tokenizer_path,
            num_workers=args.workers
        )
    else:
        print(f"Directory not found: {target_dir}")
