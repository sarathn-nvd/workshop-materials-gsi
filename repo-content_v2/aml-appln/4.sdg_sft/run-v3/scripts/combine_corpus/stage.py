"""Combine-corpus stage - PLACEHOLDER.

Strategy doc explicitly marks this step as out of scope for the SFT pipeline.
The intended downstream behaviour (kept here for reference and for the
implementer once it's brought in scope):

    1. Concatenate `sar_judgment_non_auxillary_corpus.jsonl` + `auxiliary_corpus.jsonl`
       into `combined_corpus.jsonl` (= N records).
    2. Stratified split 90/10 train/test on
       (task_type x {sar_variant or "n/a"} x {label or "n/a"}) so every cell
       appears in both splits.
    3. Write `train.jsonl` and `test.jsonl`.
"""
from __future__ import annotations


def run(*, total_records: int, dry_run: bool = False) -> None:
    raise NotImplementedError(
        "combine_corpus is intentionally out of scope for the current pipeline. "
        "See `sft_data/combine_corpus/stage.py` docstring for the planned behaviour."
    )


if __name__ == "__main__":
    run(total_records=75000)
