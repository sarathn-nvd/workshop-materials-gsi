"""Tokenize a JSONL text corpus into a Megatron-Core indexed dataset (.bin/.idx).

This is the workshop's compact version of the production
`gsi-training/3.cpt/2.run_cpt/.../preprocess_megatron_dataset.py`. It uses the
*same* NeMo AutoModel API (`indexed_dataset.IndexedDatasetBuilder`) and the same
`--append-eod` document-boundary convention, trimmed to the essentials for a tiny
corpus (no sentence splitting, single worker).

Runs INSIDE the nvcr.io/nvidia/nemo-automodel:26.04 container (it imports
nemo_automodel). The CPT recipe's `MegatronPretraining` dataset then reads the
`<output-prefix>_text_document.{bin,idx}` files this produces.
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.megatron import indexed_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file, one {json_key: text} per line")
    ap.add_argument("--output-prefix", required=True, help="output path prefix (no suffix)")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer id or local path")
    ap.add_argument("--json-key", default="text")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    bin_path = f"{args.output_prefix}_text_document.bin"
    idx_path = f"{args.output_prefix}_text_document.idx"
    builder = indexed_dataset.IndexedDatasetBuilder(
        bin_path, dtype=indexed_dataset.DType.optimal_dtype(len(tok)),
    )

    n_docs, n_tokens = 0, 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)[args.json_key]
            ids = tok(text).input_ids
            if not ids:
                continue
            ids.append(tok.eos_token_id)          # --append-eod: mark the document boundary
            builder.add_document(ids, [len(ids)])
            n_docs += 1
            n_tokens += len(ids)
    builder.finalize(idx_path)
    print(f"wrote {n_docs} documents / {n_tokens} tokens -> {bin_path} (+ .idx)")


if __name__ == "__main__":
    main()
