"""Stage A4 - Corpus consolidation + audit.

Final mechanical sweep + per-task share audit + output-shape contract test
against Stage 6's inline shape + dedup + write `auxiliary_corpus.jsonl`.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

from scripts.common.io import iter_jsonl, write_jsonl, write_json
from scripts.common.progress import StageTimer
from scripts.common.verify import run_stage_gate
from scripts.config import (
    AUX_TASK_SHARES, DEFAULT_SEED, FINAL_AUX,
    INTERIM_AUX, INTERIM_NONAUX, MANIFESTS_AUX, MANIFESTS_DIR,
)
from scripts.validators.rules_corpus import dedup_minhash, output_shape_contract_test

logger = logging.getLogger(__name__)

STAGE_ID = "stage_a4_audit"
INPUT_FILE = INTERIM_AUX / "stage_a3_assemble.jsonl"
NONAUX_INTERIM = INTERIM_NONAUX / "stage_8_adversarial.jsonl"
NONAUX_FALLBACK = INTERIM_NONAUX / "stage_7_assemble.jsonl"
FAILED_FILE = MANIFESTS_AUX.parent / "aux_failed_records.jsonl"
MANIFEST_FILE = MANIFESTS_AUX / f"{STAGE_ID}_manifest.json"


def _audit_task_share(records: list[dict], tolerance_pp: float = 2.0) -> dict:
    counter = Counter(r.get("metadata", {}).get("task_type") for r in records)
    total = sum(counter.values()) or 1
    obs = {t: counter.get(t, 0) / total for t in AUX_TASK_SHARES}
    diffs_pp = {t: abs(obs[t] - AUX_TASK_SHARES[t]) * 100 for t in AUX_TASK_SHARES}
    return {
        "rule": "AUX-TASK-MIX",
        "target": AUX_TASK_SHARES,
        "observed": obs,
        "diffs_pp": diffs_pp,
        "tolerance_pp": tolerance_pp,
        "passed": max(diffs_pp.values()) <= tolerance_pp,
    }


def _audit_per_source_cap(records: list[dict], cap: float = 0.60) -> dict:
    """No single source > 60% within its task type."""
    by_task: dict[str, Counter] = {}
    for r in records:
        md = r.get("metadata", {})
        task = md.get("task_type")
        src = md.get("source")
        if not task or not src:
            continue
        by_task.setdefault(task, Counter())[src] += 1
    above_cap = {}
    for task, counter in by_task.items():
        total = sum(counter.values()) or 1
        for src, cnt in counter.items():
            share = cnt / total
            if share > cap:
                above_cap[f"{task}/{src}"] = round(share, 4)
    return {
        "rule": "AUX-PER-TASK-SOURCE-CAP",
        "cap": cap,
        "above_cap": above_cap,
        "passed": not above_cap,
    }


def _sample_inline_findings(path) -> list[dict]:
    """Pull a sample of auxiliary_findings entries from the non-aux output."""
    if path is None or not path.exists():
        return []
    samples = []
    for rec in iter_jsonl(path):
        if len(samples) >= 200:
            break
        try:
            user = json.loads(rec["messages"][1]["content"])
        except Exception:  # noqa: BLE001
            continue
        aux = user.get("auxiliary_findings") or {}
        for kind in ("numeric", "citation", "statutory"):
            for entry in (aux.get(kind) or [])[:1]:
                samples.append({"kind": kind, "payload": entry})
                if len(samples) >= 200:
                    break
            if len(samples) >= 200:
                break
    return samples


def run(*, total_records, dry_run=False, seed=DEFAULT_SEED):
    timer = StageTimer().__enter__()
    if not INPUT_FILE.exists():
        logger.error("Stage A3 jsonl missing: %s", INPUT_FILE)
        raise FileNotFoundError(INPUT_FILE)

    records = list(iter_jsonl(INPUT_FILE))

    # V2: also include auxiliary_behavioral records produced by stage_a3b
    behav_path = INTERIM_AUX / "stage_a3b_behavioral.jsonl"
    if behav_path.exists():
        behav_records = list(iter_jsonl(behav_path))
        if behav_records:
            logger.info("Stage A4: appending %d auxiliary_behavioral records from %s",
                        len(behav_records), behav_path)
            records.extend(behav_records)

    # 1. Pydantic schema sweep — already done in A3 but re-run to be safe
    surviving: list[dict] = []
    failed: list[dict] = []
    for rec in records:
        # Light schema sanity: messages length + role names + valid JSON in [2]
        msgs = rec.get("messages") or []
        if (len(msgs) != 3 or msgs[0].get("role") != "system"
                or msgs[1].get("role") != "user" or msgs[2].get("role") != "assistant"):
            failed.append({"record_id": rec.get("metadata", {}).get("record_id"),
                           "rule_id": "SCHEMA-CHAT-SFT", "reason": "bad messages"})
            continue
        try:
            json.loads(msgs[2]["content"])
        except Exception as exc:  # noqa: BLE001
            failed.append({"record_id": rec.get("metadata", {}).get("record_id"),
                           "rule_id": "SCHEMA-ASSISTANT-JSON", "reason": str(exc)[:200]})
            continue
        surviving.append(rec)

    if failed:
        write_jsonl(failed, FAILED_FILE)

    # 2. Dedup
    deduped, dedup_audit = dedup_minhash(surviving)

    # 3. Audits
    audits = {
        "AUX-TASK-MIX": _audit_task_share(deduped),
        "AUX-PER-TASK-SOURCE-CAP": _audit_per_source_cap(deduped),
        "RULE-9-DEDUP": dedup_audit,
    }

    # 4. Output-shape contract test against Stage 6 inline findings
    nonaux_path = NONAUX_INTERIM if NONAUX_INTERIM.exists() else (
        NONAUX_FALLBACK if NONAUX_FALLBACK.exists() else None)
    inline_samples = _sample_inline_findings(nonaux_path)
    audits["AUX-OUTPUT-SHAPE-CONTRACT"] = output_shape_contract_test(deduped, inline_samples)

    # 5. Consolidate
    write_jsonl(deduped, FINAL_AUX)

    summary = {
        "build_id": f"sft_aux_run_{timer.start_iso}",
        "output_file": str(FINAL_AUX),
        "n_records": len(deduped),
        "audits": audits,
        "failed_records": {"count": len(failed)},
    }
    write_json(summary, MANIFESTS_DIR / "manifest_aux.json")

    manifest = run_stage_gate(
        stage_id=STAGE_ID, pipeline="aux", timer=timer,
        input_files=[INPUT_FILE], output_files=[FINAL_AUX],
        counts={"input": len(records), "surviving": len(surviving),
                "deduped": len(deduped), "produced": len(deduped),
                "failed_validation": len(failed)},
        rules=[], drift_checks=list(audits.values()),
        manifest_path=MANIFEST_FILE,
        notes=f"Aux corpus written to {FINAL_AUX}",
    )
    return manifest.model_dump()


if __name__ == "__main__":
    run(total_records=75000)
