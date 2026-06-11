import argparse
import os
import json
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

# Defaults -- override via CLI flags. The tokenizer must come from the same model
# checkpoint that SFT will train from (i.e. the Phase-2 CPT LOWEST_VAL on this
# pipeline) so the token-budget filter matches the recipe's actual encoding.
DEFAULT_INPUT_DIR  = "../1.shuffle_dataset/data"
DEFAULT_OUTPUT_DIR = "./final_data"
DEFAULT_LOG_FILE   = "filtered_data.log"
DEFAULT_MODEL_PATH = "/workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated"
DEFAULT_MAX_LENGTH = 5120

INPUT_DIR  = DEFAULT_INPUT_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
LOG_FILE   = DEFAULT_LOG_FILE
MODEL_PATH = DEFAULT_MODEL_PATH
MAX_LENGTH = DEFAULT_MAX_LENGTH

# Setup Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

def setup_tokenizer():
    logging.info(f"Loading tokenizer from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise

def process_files():
    tokenizer = setup_tokenizer()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
    if not input_files:
        logging.error(f"No .jsonl files found in {INPUT_DIR}")
        return

    total_records = 0
    kept_records = 0
    dropped_records = 0
    
    max_len_global = 0
    longest_samples = []  # Store tuples (len, filename)
    
    logging.info(f"Found {len(input_files)} files to process.")
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(OUTPUT_DIR, filename)
        
        logging.info(f"Processing {filename}...")
        
        file_kept = 0
        file_dropped = 0
        
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in tqdm(fin, desc=filename):
                total_records += 1
                try:
                    data = json.loads(line)
                    
                    # Handle "messages" format (Chat)
                    text_content = ""
                    
                    if "messages" in data and isinstance(data["messages"], list):
                        for msg in data["messages"]:
                            content = msg.get("content", "")
                            if content:
                                text_content += str(content) + "\n"
                    
                    # Handle "translation/transliteration" format (Fallback)
                    elif "translation" in data or "transliteration" in data:
                        if "transliteration" in data:
                            text_content += str(data["transliteration"]) + "\n"
                        if "translation" in data:
                            text_content += str(data["translation"]) + "\n"
                    
                    else:
                        # Fallback: dump specific keys or just json dump
                        # But for now assume one of the above.
                        # If nothing found, length is 0
                        pass

                    if not text_content:
                        # Skip empty or unrecognizable records? 
                        # Or keep them if they are valid but empty?
                        # Let's count them as 0 length.
                        pass

                    # Tokenize combined text
                    # We use add_special_tokens=False to get raw content length.
                    # Real packing adds BOS/EOS/Role tokens, so we add buffer.
                    ids = tokenizer(text_content, add_special_tokens=False)["input_ids"]
                    current_sample_len = len(ids) + 100 # Buffer
                    
                    # Track statistics
                    max_len_global = max(max_len_global, current_sample_len)
                    
                    if current_sample_len > 8000:
                        longest_samples.append((current_sample_len, filename))
                        longest_samples.sort(reverse=True)
                        longest_samples = longest_samples[:10]
                    
                    if current_sample_len <= MAX_LENGTH:
                        fout.write(line)
                        kept_records += 1
                        file_kept += 1
                    else:
                        dropped_records += 1
                        file_dropped += 1
                        if dropped_records <= 5:
                            logging.info(f"Dropped sample in {filename} with length {current_sample_len}")
                            
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {filename}")
                    continue
        
        logging.info(f"Finished {filename}: Kept {file_kept}, Dropped {file_dropped}")

    logging.info("=" * 40)
    logging.info("PROCESSING COMPLETE")
    logging.info(f"Total Records: {total_records}")
    logging.info(f"Kept Records:  {kept_records}")
    logging.info(f"Dropped Records: {dropped_records}")
    logging.info(f"Drop Rate: {(dropped_records/total_records)*100:.2f}%")
    logging.info(f"Global Max Length Found: {max_len_global}")
    logging.info("Top Longest Samples:")
    for length, fname in longest_samples:
        logging.info(f" - {length} tokens in {fname}")
    logging.info("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter SFT JSONL records by token length using the target tokenizer."
    )
    parser.add_argument("--input_dir",  type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory of shuffled .jsonl files. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for filtered files. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--log_file",   type=str, default=DEFAULT_LOG_FILE,
                        help=f"Log file path. Default: {DEFAULT_LOG_FILE}")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=("Path or HF id of the model whose tokenizer to use. "
                              "Should match the SFT recipe's pretrained_model_name_or_path. "
                              f"Default: {DEFAULT_MODEL_PATH}"))
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=("Max token length to keep (records over this are dropped). "
                              "Should be ~packed_sequence_size in the SFT recipe. "
                              f"Default: {DEFAULT_MAX_LENGTH}"))
    args = parser.parse_args()

    INPUT_DIR  = args.input_dir
    OUTPUT_DIR = args.output_dir
    LOG_FILE   = args.log_file
    MODEL_PATH = args.model_path
    MAX_LENGTH = args.max_length

    # Re-bind the logger to honor a CLI-provided LOG_FILE.
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    process_files()
