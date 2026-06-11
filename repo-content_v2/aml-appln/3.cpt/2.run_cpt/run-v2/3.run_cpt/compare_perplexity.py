#!/usr/bin/env python3
"""Compute and compare perplexity of the base FP8 model vs the CPT-Phase-1
checkpoint on the held-out validation slice from Stage 1.

Adapted from ../../../reference/3.run_cpt/compare_perplexity.py:
    * Reads our `{text, source, id}` JSONL schema (the reference read keys
      `translation` and `transliteration`).
    * Stratifies per-source so we can see whether EDGAR over-fit while the
      smaller Pile-of-Law sources barely moved (the failure mode flagged in
      ../../README.md section 4.4 "Risks to Pre-mitigate" #3).
    * Defaults to `--use_bf16` autocast for the eval forward pass; FP8 inference
      via TE for ad-hoc perplexity isn't worth the wiring complexity, and BF16
      autocast on the FP8 weights gives equivalent loss values.

Usage:

    python3 compare_perplexity.py \
        --input     ../1.shuffle_dataset/data/level_1_shuffled/level_1.val.jsonl \
        --base_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
        --ft_model   ./checkpoints/LOWEST_VAL/model/consolidated \
        --batch_size 16 --max_len 1024 --warmup 2 --measure 50 --max_samples 10000 --use_bf16
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts_per_source(input_path: str, max_samples_per_source: int = 2000) -> Dict[str, List[str]]:
    """Load up to `max_samples_per_source` texts grouped by `source` field.

    Stratifying per source lets us report PPL deltas per dataset family
    (edgar_corpus vs pile_of_law_oig vs uscode_house, etc.). The reference
    pre-baked the keys `translation` and `transliteration` because its corpus
    had two known sub-tasks; ours has 8 sources so we discover them on the fly.
    """
    by_source: Dict[str, List[str]] = defaultdict(list)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get("text")
            source = obj.get("source", "_unknown")
            if isinstance(text, str) and text.strip() and len(by_source[source]) < max_samples_per_source:
                by_source[source].append(text.strip())
    return dict(by_source)


class TextDataset(Dataset):
    """Tokenize once on CPU and keep length-sorted sequences to minimize padding waste."""

    def __init__(self, texts: Iterable[str], tokenizer, max_len: int = 1024, min_len: int = 8):
        self.ids: List[torch.Tensor] = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_len, add_special_tokens=True)
            if len(enc["input_ids"]) >= min_len:
                self.ids.append(torch.tensor(enc["input_ids"], dtype=torch.long))
        self.ids.sort(key=len)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.ids[idx]


def collate_pad(batch: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)


def _primary_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
    return next(model.parameters()).device


@torch.inference_mode()
def evaluate(model, dataloader, pad_id: int, warmup: int, measure: int, use_bf16: bool) -> Tuple[float, float]:
    device = _primary_device(model)
    total_tokens, total_loss_weighted, measured = 0, 0.0, 0
    start_time = None

    for i, batch in enumerate(dataloader):
        if i >= warmup + measure:
            break

        batch = batch.to(device, non_blocking=True)
        attn = (batch != pad_id).to(device, non_blocking=True)
        labels = batch.clone()
        labels[batch == pad_id] = -100

        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch, attention_mask=attn, labels=labels)
        else:
            out = model(batch, attention_mask=attn, labels=labels)

        loss_val = float(out.loss.detach().cpu())
        num_tokens = int(attn.sum().item())

        if i >= warmup:
            if start_time is None:
                start_time = time.time()
            total_tokens += num_tokens
            total_loss_weighted += loss_val * num_tokens
            measured += 1
            if measured % 25 == 0:
                elapsed = time.time() - start_time
                print(f"  [{measured} batches] tokens/sec={total_tokens/elapsed:.1f}")

    elapsed = time.time() - start_time if start_time else 1e-9
    mean_loss = total_loss_weighted / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if not math.isnan(mean_loss) else float("nan")
    tps = total_tokens / elapsed if elapsed > 0 else float("nan")
    print(f"  Tokens/sec: {tps:.1f} | Mean loss: {mean_loss:.6f} | PPL: {ppl:.3f}")
    return mean_loss, ppl


def run_for_model(model_id: str, tokenizer, datasets: Dict[str, Dataset], args) -> Dict[str, Tuple[float, float]]:
    print(f"\n=== Loading model: {model_id} ===")
    dtype = torch.bfloat16 if args.use_bf16 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    primary = _primary_device(model)
    print("Primary device:", primary)

    pad_id = tokenizer.pad_token_id
    results: Dict[str, Tuple[float, float]] = {}
    for source_name, dataset in datasets.items():
        print(f"\n>>> Evaluating source={source_name} (n={len(dataset)})")
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=lambda b: collate_pad(b, pad_id),
        )
        results[source_name] = evaluate(model, dataloader, pad_id, args.warmup, args.measure, args.use_bf16)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL holdout from Stage 1 (e.g. ../1.shuffle_dataset/data/level_1_shuffled/level_1.val.jsonl)",
    )
    parser.add_argument("--base_model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    parser.add_argument("--ft_model", required=True,
                        help="Path to fine-tuned model (consolidated checkpoint dir with config.json)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Cap per source. With 8 sources the total eval set is ~8 * max_samples.")
    parser.add_argument("--use_bf16", action="store_true", help="Enable bfloat16 autocast on CUDA")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    texts_by_source = load_texts_per_source(args.input, max_samples_per_source=args.max_samples)
    print(f"Loaded sources: " + ", ".join(f"{s}={len(t)}" for s, t in sorted(texts_by_source.items())))
    if not texts_by_source:
        raise RuntimeError("No valid texts loaded -- check that --input is the right level_1.val.jsonl path")

    # Tokenizer: use base model tokenizer (shared with ft) for consistent evaluation.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(tokenizer.model_max_length, args.max_len)

    datasets = {
        source: TextDataset(texts, tokenizer, max_len=args.max_len)
        for source, texts in texts_by_source.items()
    }
    if any(len(ds) == 0 for ds in datasets.values()):
        empties = [s for s, ds in datasets.items() if len(ds) == 0]
        print(f"WARNING: empty dataset(s) after tokenization: {empties} -- skipping")
        datasets = {s: ds for s, ds in datasets.items() if len(ds) > 0}

    model_specs = [("base", args.base_model), ("ft", args.ft_model)]
    results = {label: run_for_model(model_id, tokenizer, datasets, args) for label, model_id in model_specs}

    print("\n======== Per-Source Perplexity Comparison ========")
    print(f"{'Source':<32} {'Base PPL':>12} {'FT PPL':>12} {'Δ %':>8}")
    print("-" * 68)
    for source in sorted(datasets.keys()):
        base_ppl = results["base"][source][1]
        ft_ppl = results["ft"][source][1]
        if base_ppl > 0 and not math.isnan(base_ppl) and not math.isnan(ft_ppl):
            delta_pct = (ft_ppl - base_ppl) / base_ppl * 100.0
            flag = "  <-- BAD" if delta_pct > 5 else (" <-- WIN" if delta_pct < -15 else "")
            print(f"{source:<32} {base_ppl:>12.3f} {ft_ppl:>12.3f} {delta_pct:>+7.1f}%{flag}")
        else:
            print(f"{source:<32} {base_ppl:>12.3f} {ft_ppl:>12.3f}  {'n/a':>7}")
    print("=" * 68)
    print("WIN  : per-source PPL drop >= 15% (CPT objective from ../../README.md section 4.4)")
    print("BAD  : per-source PPL increase > 5% (catastrophic forgetting on a held-back source)")


if __name__ == "__main__":
    main()
