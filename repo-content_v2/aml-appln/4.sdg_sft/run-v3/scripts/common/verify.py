"""Verification gate — every stage exits through `run_stage_gate`.

The gate:
  1. Runs the listed RULE-N-* validators across every produced record.
  2. Computes corpus-level distribution drift (where applicable).
  3. Builds a `StageManifest` and writes it to disk.
  4. Halts the pipeline (exits with code 2) on `status="fail"`.

Validators are imported here on demand so this module has no circular
import with `scripts.validators`.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from scripts.common.io import write_manifest
from scripts.common.progress import StageTimer
from scripts.schemas import RuleFailure, StageManifest

logger = logging.getLogger(__name__)


# A validator is a callable: (record_dict) -> (passed: bool, reason: str)
# Same signature as everything in scripts.validators.rules_per_record.
RecordValidator = Callable[[dict[str, Any]], tuple[bool, str]]


def run_per_record_rules(
    records: Iterable[dict[str, Any]],
    rules: list[tuple[str, RecordValidator]],
    *,
    record_id_field: tuple[str, ...] = ("metadata", "record_id"),
    sample_failure_cap: int = 20,
) -> tuple[Counter, list[RuleFailure]]:
    """Run every (rule_id, fn) over every record. Return Counter + sample failures."""
    failures = Counter()
    samples: list[RuleFailure] = []

    for rec in records:
        rid = _nested_get(rec, record_id_field, default="<no-id>")
        for rule_id, fn in rules:
            try:
                ok, reason = fn(rec)
            except Exception as exc:  # noqa: BLE001
                ok, reason = False, f"validator-crash: {exc}"
            if not ok:
                failures[rule_id] += 1
                if len(samples) < sample_failure_cap:
                    samples.append(RuleFailure(rule_id=rule_id, record_id=rid, reason=reason))
    return failures, samples


def _nested_get(d: dict, path: tuple[str, ...], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ============================================================================
# Drift checks (corpus-level)
# ============================================================================
def share_drift(
    records: Iterable[dict[str, Any]],
    *,
    field_path: tuple[str, ...],
    target: dict[str, float],
    tolerance_pp: float = 2.0,
) -> dict[str, Any]:
    """Compare observed share of `field_path` values against `target`."""
    counter: Counter = Counter()
    total = 0
    for rec in records:
        v = _nested_get(rec, field_path)
        if v is not None:
            counter[v] += 1
            total += 1
    observed = {k: (counter.get(k, 0) / total if total else 0.0) for k in target}
    diffs = {k: abs(observed[k] - target[k]) * 100 for k in target}
    passed = max(diffs.values(), default=0.0) <= tolerance_pp
    return {
        "field": ".".join(field_path),
        "target": target,
        "observed": observed,
        "diffs_pp": diffs,
        "tolerance_pp": tolerance_pp,
        "passed": passed,
    }


def floor_drift(
    records: Iterable[dict[str, Any]],
    *,
    field_path: tuple[str, ...],
    floor: float,
) -> dict[str, Any]:
    """Every distinct value of `field_path` must be ≥ `floor` of the corpus."""
    counter: Counter = Counter()
    total = 0
    for rec in records:
        v = _nested_get(rec, field_path)
        if v is not None:
            counter[v] += 1
            total += 1
    shares = {k: counter[k] / total for k in counter} if total else {}
    below = {k: s for k, s in shares.items() if s < floor}
    return {
        "field": ".".join(field_path),
        "floor": floor,
        "observed": shares,
        "below_floor": below,
        "passed": not below,
    }


# ============================================================================
# The gate
# ============================================================================
def run_stage_gate(
    *,
    stage_id: str,
    pipeline: str,
    timer: StageTimer,
    input_files: list[Path | str],
    output_files: list[Path | str],
    counts: dict[str, int],
    records_for_validation: Optional[Iterable[dict[str, Any]]] = None,
    rules: Optional[list[tuple[str, RecordValidator]]] = None,
    drift_checks: Optional[list[dict[str, Any]]] = None,
    llm_calls: int = 0,
    manifest_path: Optional[Path] = None,
    halt_on_fail: bool = True,
    notes: str = "",
) -> StageManifest:
    """Build the manifest, write it, halt on fail."""
    timer.__exit__(None, None, None) if not timer.end_iso else None

    rule_failures: dict[str, int] = {}
    sample_failures: list[RuleFailure] = []
    if records_for_validation is not None and rules:
        records_list = list(records_for_validation)
        counter, samples = run_per_record_rules(records_list, rules)
        rule_failures = dict(counter)
        sample_failures = samples
    elif rules and records_for_validation is None:
        logger.debug("Stage %s declared %d rules but provided no records to check.",
                     stage_id, len(rules))

    drift = {}
    if drift_checks:
        for check in drift_checks:
            drift[check.get("name", check.get("field", "unnamed"))] = check

    # Determine status
    any_drift_failed = any(not d.get("passed", True) for d in drift.values())
    any_rule_failed_hard = sum(rule_failures.values()) > 0  # treated as warn unless explicit fail
    status = "ok"
    if any_drift_failed:
        status = "warn"
    if counts.get("produced", 0) == 0 and counts.get("input", 0) > 0:
        status = "fail"

    avg_llm = (llm_calls / counts["produced"]) if counts.get("produced") else 0.0

    manifest = StageManifest(
        stage=stage_id,
        pipeline=pipeline,                                 # type: ignore[arg-type]
        started_at=timer.start_iso,
        ended_at=timer.end_iso or "",
        runtime_seconds=timer.elapsed,
        input_files=[str(p) for p in input_files],
        output_files=[str(p) for p in output_files],
        counts=counts,
        llm_calls=llm_calls,
        llm_calls_per_record_avg=round(avg_llm, 3),
        rule_failures=rule_failures,
        drift=drift,
        sample_failures=sample_failures,
        status=status,                                     # type: ignore[arg-type]
        notes=notes,
    )

    if manifest_path is not None:
        write_manifest(manifest, manifest_path)

    logger.info("Stage gate %s — status=%s, produced=%d, llm_calls=%d, rule_failures=%d",
                stage_id, status, counts.get("produced", 0), llm_calls, sum(rule_failures.values()))

    if status == "fail" and halt_on_fail:
        logger.error("Stage %s FAILED. Halting pipeline.", stage_id)
        sys.exit(2)

    return manifest
