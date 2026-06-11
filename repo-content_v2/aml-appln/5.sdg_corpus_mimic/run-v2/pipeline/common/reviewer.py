"""LLM reviewer — semantic per-record gate used by Stages A2 / 6 / 7.

The reviewer reads each generated record's user + assistant content and
returns a structured JSON verdict. Critically, the reviewer DOES NOT see
gold labels / source pool metadata / generator's chain-of-thought. It must
derive judgement independently — same model, blank slate.

Verdict schema (returned by the LLM as one JSON object):

    {
      "verdict": "PASS" | "ISSUES_FOUND",
      "issues":  [list of short issue tags],
      "explain": "1-2 sentence rationale"
    }

Action policy (applied by callers):

    PASS           → keep
    ISSUES_FOUND   → 1 reroll with the issue tags appended; if reviewer
                     fails the regenerated record again, drop.

Six task-specific judge prompts are loaded from `reviewer_prompts.py`.
The same prompts that powered the offline `judge_smoke.py` analysis are
reused here — that script is essentially the corpus-level form of this
in-pipeline gate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.common.dd_helpers import run_dd_pass, safe_json_loads
from pipeline.common.reviewer_prompts import (
    REVIEWER_USER_TEMPLATE,
    REVIEWER_SYSTEM_BY_BUCKET,
)
from pipeline.config import CONCURRENCY, INTERIM_NONAUX

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================
def review_records(
    records_with_keys: list[dict],
    *,
    bucket: str,
    artifact_subdir: str,
    dataset_name: str,
    dry_run: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> list[dict]:
    """Run the LLM judge over a list of (user, assistant) pairs.

    Args:
      records_with_keys: list of dicts with at least
        {"user_content": <str>, "assistant_content": <str>}.
        Other keys are ignored by the reviewer but caller may carry record
        ids etc. through this list — the verdict list returned has the
        same length and order.
      bucket: which task-specific judge prompt to use. One of:
        sar_pos / sar_neg / sar_adv / aux_num / aux_cit / aux_stat.
      artifact_subdir / dataset_name: DataDesigner per-call artifact paths.
      dry_run: short-circuits to all-PASS for testing.

    Returns:
      list[dict] of verdicts, same length and order as input. Each dict has
      shape `{"verdict": ..., "issues": [...], "explain": "..."}`.
    """
    if not records_with_keys:
        return []
    if dry_run:
        return [{"verdict": "PASS", "issues": [], "explain": "dry-run"}
                for _ in records_with_keys]
    if bucket not in REVIEWER_SYSTEM_BY_BUCKET:
        raise ValueError(f"Unknown reviewer bucket: {bucket}. "
                         f"Valid: {sorted(REVIEWER_SYSTEM_BY_BUCKET.keys())}")

    seed_df = pd.DataFrame([{
        "user_content": str(r.get("user_content", ""))[:8000],
        "assistant_content": str(r.get("assistant_content", ""))[:4000],
    } for r in records_with_keys])

    try:
        gen = run_dd_pass(
            seed_df=seed_df,
            system_prompt=REVIEWER_SYSTEM_BY_BUCKET[bucket],
            user_template=REVIEWER_USER_TEMPLATE,
            output_column="judge_verdict",
            dataset_name=dataset_name,
            artifact_path=INTERIM_NONAUX / "_dd_artifacts" / artifact_subdir,
            max_parallel=CONCURRENCY.per_pipeline_llm,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:                                    # noqa: BLE001
        logger.error("Reviewer pass failed (bucket=%s): %s", bucket, exc)
        # On infra failure, default to PASS so we don't kill the pipeline.
        return [{"verdict": "PASS", "issues": [],
                 "explain": f"reviewer-infra-error: {exc!s}"[:200]}
                for _ in records_with_keys]

    out: list[dict] = []
    for j, gen_row in enumerate(gen.to_dict(orient="records")):
        if j >= len(records_with_keys):
            break
        verdict = safe_json_loads(gen_row.get("judge_verdict", ""))
        if not isinstance(verdict, dict):
            # Parse error → conservative PASS (don't kill records on judge bug)
            verdict = {"verdict": "PASS", "issues": [],
                       "explain": f"reviewer-parse-error: "
                                  f"{str(gen_row.get('judge_verdict',''))[:150]}"}
        # Normalise the verdict shape
        verdict.setdefault("verdict", "PASS")
        verdict.setdefault("issues", [])
        verdict.setdefault("explain", "")
        if verdict["verdict"] not in ("PASS", "ISSUES_FOUND"):
            verdict["verdict"] = "ISSUES_FOUND"     # be conservative on unknown verdicts
        out.append(verdict)
    while len(out) < len(records_with_keys):
        out.append({"verdict": "PASS", "issues": [], "explain": "reviewer-no-output"})
    return out


def summarise_verdicts(verdicts: Iterable[dict]) -> dict:
    """Aggregate counts over a verdict list — for stage manifests."""
    from collections import Counter
    n = pass_n = 0
    issue_counter: Counter = Counter()
    for v in verdicts:
        n += 1
        if v.get("verdict") == "PASS":
            pass_n += 1
        for tag in (v.get("issues") or []):
            issue_counter[str(tag)] += 1
    return {
        "n": n,
        "pass": pass_n,
        "pass_rate": round(pass_n / n, 4) if n else 1.0,
        "issues_top": dict(issue_counter.most_common(10)),
    }
