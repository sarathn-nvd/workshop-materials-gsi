"""Single audit entry point — strategy-aligned 3-mode dispatcher.

Replaces the previous trio (`audit.py`, `audit_per_record_dump.py`,
`audit_judge_full.py`). All three sets of checks are still here; they're
implemented in private helper modules (`_audit_per_record.py`,
`_audit_judge.py`, `_audit_corpus.py`) and dispatched from this single
CLI.

Modes (subcommands):

  per-record   Run every per-record validator (RULE-3 .. RULE-8) on each
               record in both corpora and persist ONE LINE PER RECORD as
               JSONL so downstream tools (filter_and_cap.py, top-up
               planners, RL phase) can join on `record_id`. Fast (~30 s
               on 100K records); no LLM calls.

  judge        Run the strategy-aligned 7-bucket LLM judge on 100% of
               the corpus and persist per-record verdicts as JSONL.
               Compares pass-rates against the strategy targets
               (SDG_STRATEGY_SFT.md lines 922-929) and emits target
               alerts when buckets fall short. Slow (~70 min on 100K
               records); strategy-aligned LLM cost.

  corpus       Run the corpus-level audits: schema validity, RULE-1
               marginals (label / variant / typology floors), RULE-9
               source-cap + dedup, output-shape contract, and the 12
               semantic audits (A1-A12). Aggregates are written to a
               single JSON manifest. Medium speed (~5 min); no LLM
               calls.

  all          Run all three in sequence: per-record → judge → corpus.
               This is the standard post-pipeline audit flow.

Usage:

    cd /data/swami/gsi-training/4.sdg_sft

    # Run everything (recommended after a fresh main.py run):
    python -m scripts.audit all

    # Or invoke individual modes:
    python -m scripts.audit per-record
    python -m scripts.audit judge --buckets aux_stat,aux_cit
    python -m scripts.audit corpus --report data/manifests/audit_report.json
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _dispatch_per_record(rest: Sequence[str]) -> int:
    """Forward to the per-record verdict dump."""
    from scripts._audit_per_record import main as _per_record_main
    sys.argv = ["audit per-record"] + list(rest)
    return _per_record_main()


def _dispatch_judge(rest: Sequence[str]) -> int:
    """Forward to the LLM-judge audit."""
    from scripts._audit_judge import main as _judge_main
    sys.argv = ["audit judge"] + list(rest)
    _judge_main()
    return 0


def _dispatch_corpus(rest: Sequence[str]) -> int:
    """Forward to the corpus-level + semantic audits."""
    from scripts._audit_corpus import main as _corpus_main
    sys.argv = ["audit corpus"] + list(rest)
    return _corpus_main()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.audit",
        description="Single audit dispatcher: per-record, judge, corpus, or all.",
    )
    parser.add_argument(
        "mode",
        choices=["per-record", "judge", "corpus", "all"],
        help="Which audit to run.",
    )
    # Pass-through for mode-specific args (consumed by the dispatched module)
    args, rest = parser.parse_known_args()

    if args.mode == "per-record":
        return _dispatch_per_record(rest)
    if args.mode == "judge":
        return _dispatch_judge(rest)
    if args.mode == "corpus":
        return _dispatch_corpus(rest)
    # all — run all three sequentially
    rc = _dispatch_per_record(rest)
    if rc != 0:
        print(f"\n[audit] per-record exited with rc={rc}; continuing to judge.")
    rc = _dispatch_judge(rest)
    if rc != 0:
        print(f"\n[audit] judge exited with rc={rc}; continuing to corpus.")
    return _dispatch_corpus(rest)


if __name__ == "__main__":
    sys.exit(main())
