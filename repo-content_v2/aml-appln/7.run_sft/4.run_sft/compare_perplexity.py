#!/usr/bin/env python3
"""Compare perplexity of the CPT base checkpoint vs the SFT checkpoint on the
held-out SFT validation JSONL (chat `messages` format).

Adapted from run-v2/3.run_cpt/compare_perplexity.py:
    * Reads `{messages: [{role, content}, ...]}` JSONL (not plain `text`).
    * Applies the same ChatML template used in recipe_a100sxm-8.yaml.
    * Stratifies by `task_type` parsed from the user turn (sar_judgment,
      auxiliary_citation, etc.) since the SFT val split has no `source` field.
    * By default masks loss to the final assistant turn only
      (`--mask_assistant_only`), matching in-training validation loss.

Usage (paths as seen inside the NeMo container with HOST_WORKSPACE mounted):

    python3 compare_perplexity.py \
        --input ../3.filter_dataset/final_data_clean/sft_mixed.val.jsonl \
        --base_model /workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated \
        --ft_model   ./checkpoints/LOWEST_VAL/model/consolidated \
        --batch_size 2 --max_len 4096 --warmup 2 --measure 50 \
        --max_samples 500 --use_bf16

Note: PPL on masked assistant tokens is a sanity check aligned with training val
loss. It does not replace task-quality eval (generation / harness / human review).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same ChatML template as recipe_a100sxm-8.yaml (inline because the CPT
# consolidated tokenizer may not ship chat_template in tokenizer_config.json).
CHATML_TEMPLATE = """{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}"""

START_OF_TURN_TOKEN = "<|im_start|>"


def _resolve_chat_template(tokenizer) -> None:
    """Ensure tokenizer has the ChatML template (from checkpoint .jinja or inline)."""
    if tokenizer.chat_template:
        return
    # Some consolidated checkpoints ship chat_template.jinja but not tokenizer_config entry.
    model_dir = getattr(tokenizer, "name_or_path", None)
    if model_dir and os.path.isdir(model_dir):
        jinja_path = os.path.join(model_dir, "chat_template.jinja")
        if os.path.isfile(jinja_path):
            with open(jinja_path, encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
            return
    tokenizer.chat_template = CHATML_TEMPLATE


def _extract_input_ids(tokenized) -> List[int]:
    """Normalize apply_chat_template output to a flat list of token ids."""
    if tokenized is None:
        return []
    if isinstance(tokenized, dict):
        ids = tokenized.get("input_ids", [])
    elif hasattr(tokenized, "input_ids"):
        ids = tokenized.input_ids
    else:
        ids = tokenized
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    # Some tokenizers return a batch dimension: [[...]]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _tokenized_chat_length(
    tokenizer, messages: List[dict], *, truncate: bool, max_len: int
) -> int:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if truncate:
        kwargs.update(truncation=True, max_length=max_len)
    return len(_extract_input_ids(tokenizer.apply_chat_template(messages, **kwargs)))


def _build_assistant_label_mask(
    tokenizer,
    messages: List[dict],
    input_ids: List[int],
    *,
    truncate: bool,
    max_len: int,
) -> List[int]:
    """Supervise all assistant turns (same strategy as NeMo format_chat_template)."""
    mask = [0] * len(input_ids)
    found = False
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        found = True
        start = _tokenized_chat_length(
            tokenizer, messages[:idx], truncate=truncate, max_len=max_len
        )
        end = _tokenized_chat_length(
            tokenizer, messages[: idx + 1], truncate=truncate, max_len=max_len
        )
        for pos in range(min(start, len(mask)), min(end, len(mask))):
            mask[pos] = 1
    if not found:
        return [1] * len(input_ids)
    return mask


def _infer_task_type(messages: List[dict]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            break
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and payload.get("task_type"):
                return str(payload["task_type"])
        except json.JSONDecodeError:
            pass
        break
    return "_unknown"


def load_conversations_per_task(
    input_path: str, max_samples_per_task: int
) -> Dict[str, List[List[dict]]]:
    by_task: Dict[str, List[List[dict]]] = defaultdict(list)
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
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            if not any(m.get("role") == "assistant" for m in messages):
                continue
            task = _infer_task_type(messages)
            if len(by_task[task]) < max_samples_per_task:
                by_task[task].append(messages)
    return dict(by_task)


class ChatDataset(Dataset):
    """Pre-tokenize chat conversations; keep length-sorted for less padding."""

    def __init__(
        self,
        conversations: Iterable[List[dict]],
        tokenizer,
        max_len: int,
        mask_assistant_only: bool,
        truncate: bool,
        min_len: int = 16,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        skipped = defaultdict(int)

        for messages in conversations:
            tmpl_kwargs = {
                "tokenize": True,
                "add_generation_prompt": False,
                "return_dict": True,
            }
            if truncate:
                tmpl_kwargs.update(truncation=True, max_length=max_len)
            tokenized = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
            input_ids = _extract_input_ids(tokenized)
            if len(input_ids) < min_len:
                skipped["too_short"] += 1
                continue
            if mask_assistant_only:
                mask = _build_assistant_label_mask(
                    tokenizer, messages, input_ids, truncate=truncate, max_len=max_len
                )
                labels_list = [tid if m else -100 for tid, m in zip(input_ids, mask)]
            else:
                labels_list = list(input_ids)
            if sum(1 for x in labels_list if x != -100) < min_len:
                skipped["too_few_label_tokens"] += 1
                continue
            input_ids_t = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels_list, dtype=torch.long)
            self.samples.append((input_ids_t, labels))

        if skipped:
            print(
                "Tokenization skips: "
                + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
            )
        self.samples.sort(key=lambda s: len(s[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_pad(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_labels[padded_ids == pad_id] = -100
    return padded_ids, padded_labels


def _primary_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
    return next(model.parameters()).device


@torch.inference_mode()
def evaluate(
    model,
    dataloader: DataLoader,
    pad_id: int,
    warmup: int,
    measure: int,
    use_bf16: bool,
) -> Tuple[float, float]:
    device = _primary_device(model)
    total_tokens, total_loss_weighted, measured = 0, 0.0, 0
    start_time = None

    for i, (batch, labels) in enumerate(dataloader):
        if i >= warmup + measure:
            break

        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn = (batch != pad_id).to(device, non_blocking=True)

        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch, attention_mask=attn, labels=labels)
        else:
            out = model(batch, attention_mask=attn, labels=labels)

        loss_val = float(out.loss.detach().cpu())
        num_tokens = int((labels != -100).sum().item())

        if i >= warmup:
            if start_time is None:
                start_time = time.time()
            total_tokens += num_tokens
            total_loss_weighted += loss_val * num_tokens
            measured += 1
            if measured % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  [{measured} batches] tokens/sec={total_tokens / elapsed:.1f}")

    elapsed = time.time() - start_time if start_time else 1e-9
    mean_loss = total_loss_weighted / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if not math.isnan(mean_loss) else float("nan")
    tps = total_tokens / elapsed if elapsed > 0 else float("nan")
    print(f"  Tokens/sec: {tps:.1f} | Mean loss: {mean_loss:.6f} | PPL: {ppl:.3f}")
    return mean_loss, ppl


def run_for_model(
    model_id: str,
    tokenizer,
    datasets: Dict[str, ChatDataset],
    args,
) -> Dict[str, Tuple[float, float]]:
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
    print("Primary device:", _primary_device(model))

    pad_id = tokenizer.pad_token_id
    results: Dict[str, Tuple[float, float]] = {}
    for task_name, dataset in datasets.items():
        print(f"\n>>> Evaluating task_type={task_name} (n={len(dataset)})")
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=lambda b: collate_pad(b, pad_id),
        )
        results[task_name] = evaluate(
            model, dataloader, pad_id, args.warmup, args.measure, args.use_bf16
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _build_tokenizer(base_model: str, max_len: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(tokenizer.model_max_length, max_len)
    _resolve_chat_template(tokenizer)
    if not tokenizer.chat_template:
        raise RuntimeError("Tokenizer has no chat_template; cannot evaluate chat SFT data.")
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="SFT validation JSONL with `messages` (e.g. ../3.filter_dataset/final_data_clean/sft_mixed.val.jsonl)",
    )
    parser.add_argument(
        "--base_model",
        default="/workspace/run-v2/3.run_cpt/checkpoints/LOWEST_VAL/model/consolidated",
        help="CPT checkpoint (pre-SFT); compared as 'base'",
    )
    parser.add_argument(
        "--ft_model",
        required=True,
        help="SFT checkpoint dir (e.g. ./checkpoints/LOWEST_VAL/model/consolidated)",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--max_len",
        type=int,
        default=8192,
        help="Max tokens per conversation when --truncate is set",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences to --max_len (off by default, matching SFT training)",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Cap per task_type. Total eval size is ~n_task_types * max_samples.",
    )
    parser.add_argument(
        "--mask_assistant_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask loss to final assistant turn only (matches in-training val; default on)",
    )
    parser.add_argument("--use_bf16", action="store_true", help="Enable bfloat16 autocast on CUDA")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    convos_by_task = load_conversations_per_task(args.input, max_samples_per_task=args.max_samples)
    print(
        "Loaded task types: "
        + ", ".join(f"{t}={len(c)}" for t, c in sorted(convos_by_task.items()))
    )
    if not convos_by_task:
        raise RuntimeError("No valid conversations loaded -- check --input path and JSONL schema")

    tokenizer = _build_tokenizer(args.base_model, args.max_len)

    datasets = {
        task: ChatDataset(
            convos,
            tokenizer,
            max_len=args.max_len,
            mask_assistant_only=args.mask_assistant_only,
            truncate=args.truncate,
        )
        for task, convos in convos_by_task.items()
    }
    if any(len(ds) == 0 for ds in datasets.values()):
        empties = [t for t, ds in datasets.items() if len(ds) == 0]
        print(f"WARNING: empty dataset(s) after tokenization: {empties} -- skipping")
        datasets = {t: ds for t, ds in datasets.items() if len(ds) > 0}
    if not datasets:
        raise RuntimeError(
            "No samples survived tokenization. If the current run shows this warning, "
            "stop it and re-run with the updated compare_perplexity.py."
        )

    print(
        f"Loss scope: {'assistant turns only' if args.mask_assistant_only else 'full sequence'} | "
        f"truncate={'on' if args.truncate else 'off (matches training)'}"
    )
    print("Tokenized sample counts: " + ", ".join(f"{t}={len(ds)}" for t, ds in sorted(datasets.items())))

    model_specs = [("base (CPT)", args.base_model), ("ft (SFT)", args.ft_model)]
    results = {
        label: run_for_model(model_id, tokenizer, datasets, args)
        for label, model_id in model_specs
    }

    print("\n======== Per-Task Perplexity Comparison (assistant-masked) ========")
    print(f"{'task_type':<28} {'Base PPL':>12} {'SFT PPL':>12} {'Δ %':>8}")
    print("-" * 64)
    for task in sorted(datasets.keys()):
        base_ppl = results["base (CPT)"][task][1]
        ft_ppl = results["ft (SFT)"][task][1]
        if base_ppl > 0 and not math.isnan(base_ppl) and not math.isnan(ft_ppl):
            delta_pct = (ft_ppl - base_ppl) / base_ppl * 100.0
            flag = "  <-- worse" if delta_pct > 5 else (" <-- better" if delta_pct < -5 else "")
            print(f"{task:<28} {base_ppl:>12.3f} {ft_ppl:>12.3f} {delta_pct:>+7.1f}%{flag}")
        else:
            print(f"{task:<28} {base_ppl:>12.3f} {ft_ppl:>12.3f}  {'n/a':>7}")
    print("=" * 64)
    print("For SFT, lower PPL on assistant tokens vs CPT base is expected.")
    print("Δ > +5% on a task_type may indicate regression on that task family.")


if __name__ == "__main__":
    main()
