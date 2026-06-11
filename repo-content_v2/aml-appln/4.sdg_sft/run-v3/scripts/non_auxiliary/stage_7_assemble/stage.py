"""Stage 7 - SAR output emission + chat-SFT assembly.

For each record:
  1. is_suspicious = label (mechanical)
  2. narrative: pass-through real text for Record_1/Record_4 bare;
     DD-generated for everything else; empty string for label=false.
  3. Build chat-SFT envelope: messages[system, user, assistant] + metadata.
"""
from __future__ import annotations

import json
import logging
import random
import re

import pandas as pd

from scripts.common.dd_helpers import run_dd_pass
from scripts.common.io import read_parquet, write_jsonl
from scripts.common.progress import StageTimer
from scripts.common.verify import run_stage_gate
from scripts.config import (
    CONCURRENCY, DEFAULT_SEED, INTERIM_NONAUX, MANIFESTS_NONAUX,
)
from scripts.non_auxiliary.stage_7_assemble.prompts import SAR_JUDGMENT_SYSTEM
from scripts.validators.rules_per_record import (
    rule_7_aug_cites, rule_7_bare_noleak, rule_7_english, rule_7_length,
    rule_7_neg_empty_narrative, rule_7_objectivity,
)

logger = logging.getLogger(__name__)

STAGE_ID = "stage_7_assemble"
INPUT_FILE = INTERIM_NONAUX / "stage_6_aux_findings.parquet"
OUTPUT_FILE = INTERIM_NONAUX / f"{STAGE_ID}.jsonl"
MANIFEST_FILE = MANIFESTS_NONAUX / f"{STAGE_ID}_manifest.json"


def _extract_must_cite(aux: dict | None) -> list[str]:
    """Pull the must-cite verbatim tokens out of auxiliary_findings.

    These are the exact phrases RULE-7-AUG-CITES requires the narrative to
    contain a 3-gram from. Surfacing them at the top of the user message
    makes the model significantly more likely to quote them verbatim
    instead of paraphrasing.
    """
    if not isinstance(aux, dict):
        return []
    out: list[str] = []
    for f in (aux.get("numeric") or []):
        ans = (f or {}).get("answer")
        if ans:
            out.append(f"[NUMERIC] {ans}")
    for f in (aux.get("citation") or []):
        ev = (f or {}).get("evidence_span")
        if ev:
            out.append(f"[CITATION] {ev}")
    for f in (aux.get("statutory") or []):
        rs = (f or {}).get("reasoning")
        if rs:
            out.append(f"[STATUTORY] {rs}")
    return out


def _build_user_content(row: dict) -> str:
    """Build the JSON-serialized user-message content."""
    aux_raw = row.get("auxiliary_findings")
    aux: dict | None = None
    if isinstance(aux_raw, str) and aux_raw.strip():
        try:
            aux = json.loads(aux_raw)
        except Exception:  # noqa: BLE001
            aux = None
    elif isinstance(aux_raw, dict):
        aux = aux_raw

    variant = row.get("aux_variant", "bare")

    bundle: dict[str, object] = {"task_type": "sar_judgment"}

    # For `augmented` records (NOT adversarial — those need the model to verify
    # rather than parrot), surface the must-cite tokens at the top of the JSON
    # so they're visually unmissable. RULE-7-AUG-CITES requires at least one
    # 3-gram from each of these in the narrative.
    if variant == "augmented" and aux is not None:
        must_cite = _extract_must_cite(aux)
        if must_cite:
            bundle["must_cite_verbatim"] = must_cite

    bundle["transactions"] = row.get("transactions") or []
    bundle["kyc_profile"] = {
        "entity_id": row["entity_id"],
        "entity_type": row["entity_type"],
        "expected_monthly_volume": int(row["expected_monthly_volume"]),
        "business_purpose": row["business_purpose"],
        "risk_rating": row["risk_rating"],
        "incorporation_jurisdiction": row["incorporation_jurisdiction"],
    }
    bundle["sanctions_pep_hits"] = row.get("sanctions_pep_hits") or []
    bundle["policy_excerpts"] = row.get("policy_excerpts") or []
    bundle["sop_excerpts"] = row.get("sop_excerpts") or []
    bundle["auxiliary_findings"] = aux

    return json.dumps(bundle, ensure_ascii=False, default=str)


def _stub_narrative(row: dict, rng: random.Random) -> str:
    """Deterministic narrative stub - real path invokes DataDesigner.

    Produces a regulator-grade-shaped narrative that meets RULE-7-LENGTH +
    RULE-7-OBJECTIVITY for testing. Real generation tunes per-typology.
    """
    if not row["label"]:
        return ""
    typology = row["typology"]
    entity_id = row["entity_id"]
    txs = row.get("transactions") or []
    n_tx = len(txs)
    total = sum(float(t.get("amount", 0)) for t in txs)
    declared = int(row["expected_monthly_volume"])
    ratio = (total / max(declared, 1))

    base = (
        f"Suspicious activity is identified for entity {entity_id} "
        f"({row['entity_archetype']}, {row['incorporation_jurisdiction']}, "
        f"declared monthly volume ${declared:,}). Between the investigation "
        f"window dates, the entity engaged in {n_tx} transactions totaling "
        f"${total:,.0f} - approximately {ratio:.1f}x the declared monthly "
        f"volume. The pattern is consistent with {typology.replace('_', ' ')} "
        f"as described in the supplied policy excerpts. The combined evidence "
        f"warrants filing a Suspicious Activity Report; investigator review "
        f"is recommended."
    )
    # Inject auxiliary citation tokens when augmented (so RULE-7-AUG-CITES passes)
    aux = row.get("auxiliary_findings")
    if isinstance(aux, dict):
        nums = aux.get("numeric") or []
        cits = aux.get("citation") or []
        stats = aux.get("statutory") or []
        addons = []
        if nums:
            addons.append(f" Auxiliary numeric: {nums[0]['answer']}.")
        if cits:
            span = cits[0].get("evidence_span", "")[:200]
            addons.append(f" Cited span: \"{span}\".")
        if stats:
            addons.append(f" Statutory: {stats[0]['reasoning'][:200]}")
        base += " ".join(addons)
    # Adversarial-aux: include detection marker
    if row.get("aux_variant") == "adversarial_aux":
        base += " Upon verification of the auxiliary findings against the underlying transactions, one finding appears inconsistent and was re-derived from the raw inputs."
    # Trim to 1000 chars
    return base[:1000]


def _build_record(row: dict, rng: random.Random) -> dict:
    user_content = _build_user_content(row)
    narrative = _stub_narrative(row, rng)
    output_obj = {
        "is_suspicious": bool(row["label"]),
        "suspicious_activity_report": narrative,
    }
    asst_content = json.dumps(output_obj, ensure_ascii=False)

    is_real_seed = (row.get("path") == "Record_1" and row.get("aux_variant") == "bare"
                    and bool(row["label"]))
    metadata = {
        "record_id": f"{row['path'].lower()}_{row.name if hasattr(row, 'name') else ''}",
        "phase": "sft",
        "source": row.get("source") or "unknown",
        "typology": row["typology"],
        "sar_variant": row["aux_variant"],
        "synthetic": not is_real_seed,
        "task_type": "sar_judgment",
        "surface_pattern": row["surface_pattern"],   # carry for Stage 9 RULE-1-FLOOR-NEAR-MISS
    }

    return {
        "messages": [
            {"role": "system", "content": SAR_JUDGMENT_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
        "metadata": metadata,
    }


def _llm_generate_narratives(targets: list[tuple[int, str]], dry_run: bool) -> dict[int, str]:
    """Run DD over (idx, user_content) pairs needing a narrative. Returns {idx: narrative_text}."""
    if not targets or dry_run:
        return {}
    seed_df = pd.DataFrame({
        "user_json": [u for _, u in targets],
    })
    try:
        gen = run_dd_pass(
            seed_df=seed_df,
            system_prompt=SAR_JUDGMENT_SYSTEM,
            user_template="{{ user_json }}",
            output_column="model_output",
            dataset_name="stage_7_narrative",
            artifact_path=INTERIM_NONAUX / "_dd_artifacts" / "stage_7",
            max_parallel=CONCURRENCY.per_pipeline_llm,
            max_tokens=2500,    # headroom for tightened prompt + JSON envelope + closing sentence
            temperature=0.7,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("DD pass (narrative) failed: %s", exc)
        return {}
    out: dict[int, str] = {}
    for j, gen_row in enumerate(gen.to_dict(orient="records")):
        if j >= len(targets):
            break
        idx, _ = targets[j]
        raw = (gen_row.get("model_output") or "").strip()
        narr = _extract_narrative_from_llm(raw)
        if narr:
            out[idx] = narr[:1000]
    return out


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)


def _extract_narrative_from_llm(raw: str) -> str:
    """Robustly extract `suspicious_activity_report` from an LLM response.

    Handles three observed shapes:
      1. Pure JSON object: `{"is_suspicious": ..., "suspicious_activity_report": "..."}`
      2. Markdown-fenced JSON: ` ```json\\n{...}\\n``` `
      3. Plain prose narrative (rare; treated as the narrative directly).
    """
    if not raw:
        return ""
    s = raw.strip()
    # Strip leading/trailing markdown fences if present
    s_unfenced = _FENCE_RE.sub("", s).strip()
    s_unfenced = _FENCE_RE.sub("", s_unfenced).strip()
    # Try to parse the unfenced version as JSON
    for candidate in (s_unfenced, s):
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(parsed, dict):
            v = parsed.get("suspicious_activity_report")
            if isinstance(v, str):
                return v.strip()
    # Last-resort: search for an embedded JSON object
    m = re.search(r'\{[^{}]*"suspicious_activity_report"\s*:\s*"((?:[^"\\]|\\.)*)"', s_unfenced, re.DOTALL)
    if m:
        try:
            return json.loads(f'"{m.group(1)}"').strip()
        except Exception:  # noqa: BLE001
            return m.group(1).strip()
    # If the model returned plain prose with no JSON structure, treat as narrative
    if not s_unfenced.startswith("{") and not s_unfenced.startswith("["):
        return s_unfenced[:1000]
    return ""


def run(*, total_records, dry_run=False, seed=DEFAULT_SEED):
    timer = StageTimer().__enter__()
    rng = random.Random(seed + 7)
    df = read_parquet(INPUT_FILE).reset_index(drop=True)

    # Pre-build user content for every record (deterministic)
    user_contents: list[str] = [_build_user_content(row.to_dict()) for _, row in df.iterrows()]

    # B10 fix: every positive gets a DD-generated narrative; the EFC seed
    # passthrough is dropped because EFC SAR text is templated ("The
    # institution identified a {typology} pattern involving N transfers
    # totaling X..."). The seed narrative is retained in metadata.seed_narrative
    # for traceability only — see strategy doc Stage 7 Step 2.
    targets: list[tuple[int, str]] = []
    for idx, row in df.iterrows():
        if not row["label"]:
            continue                                                                  # negatives → empty narrative
        targets.append((idx, user_contents[idx]))

    llm_narratives = _llm_generate_narratives(targets, dry_run=dry_run)
    llm_calls = len(llm_narratives)

    records = []
    n_dropped_no_llm_narrative = 0
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if not row_dict["label"]:
            narrative = ""
        elif idx in llm_narratives:
            narrative = llm_narratives[idx]
        else:
            # Change 3: LLM call failed for this positive — DROP the record.
            # Earlier draft fell back to `_stub_narrative` which emitted a
            # template "warrants filing" SAR regardless of whether the math
            # supported the conclusion. The LLM judge flagged 18 such records
            # as semantic false-positives. Better to ship fewer records than
            # to ship templated false-positives. The smoke audit will report
            # the drop count.
            n_dropped_no_llm_narrative += 1
            continue

        # Capture the seed narrative for traceability ONLY (not used as the
        # output narrative — see B10 fix). This lets downstream tools compare
        # the synthetic narrative to the real SAR text without contaminating
        # the training data with templated source text.
        seed_narrative = ""
        if (row_dict.get("path") in ("Record_1", "Record_4")
                and row_dict.get("source_payload")):
            try:
                payload = json.loads(row_dict["source_payload"])
                if isinstance(payload, dict):
                    sars = payload.get("sar_reports") or []
                    if sars:
                        seed_narrative = (sars[0].get("narrative") or "")[:1000]
                    if not seed_narrative:
                        seed_narrative = (payload.get("narrative") or payload.get("notes") or "")[:1000]
            except Exception:  # noqa: BLE001
                pass

        out_obj = {
            "is_suspicious": bool(row_dict["label"]),
            "suspicious_activity_report": narrative,
        }
        # `is_real_seed` is no longer true for any record (B10 fix) — all bare
        # positives are now DD-generated. The flag is retained as `False` for
        # backward-compat in metadata.
        is_real_seed = False
        rec = {
            "messages": [
                {"role": "system", "content": SAR_JUDGMENT_SYSTEM},
                {"role": "user", "content": user_contents[idx]},
                {"role": "assistant", "content": json.dumps(out_obj, ensure_ascii=False)},
            ],
            "metadata": {
                "record_id": f"{row_dict['path'].lower()}_{idx:07d}",
                "phase": "sft",
                "source": row_dict.get("source") or "unknown",
                "typology": row_dict["typology"],
                "sar_variant": row_dict["aux_variant"],
                "synthetic": True,                # all narratives are now DD-generated (B10)
                "task_type": "sar_judgment",
                "surface_pattern": row_dict["surface_pattern"],
                # B10: seed narrative retained for traceability only — never
                # used as the assistant content. Downstream tools can compare
                # synthetic vs. real-source text without contaminating training.
                "seed_narrative": seed_narrative if seed_narrative else None,
            },
        }
        records.append(rec)

    # =============================================================
    # Per-stage failure handling per SDG_STRATEGY_SFT.md Stage 7:
    #   - RULE-7-AUG-CITES fail → demote variant to bare
    #     (clear auxiliary_findings, regenerate narrative once with bare prompt)
    #   - RULE-7-LENGTH fail → re-roll once then keep (soft signal)
    #   - RULE-7-OBJECTIVITY / RULE-7-BARE-NOLEAK fail → re-roll up to 2× then drop
    #   - RULE-7-NEG-EMPTY-NARRATIVE fail → drop
    # =============================================================
    counters = {"demoted": 0, "rerolled_length": 0, "rerolled_obj_or_leak": 0,
                "dropped": 0, "extra_llm_calls": 0}

    def _apply_demote_to_bare(rec: dict) -> dict:
        """Strip auxiliary_findings + must_cite_verbatim and set sar_variant=bare."""
        try:
            user = json.loads(rec["messages"][1]["content"])
            user["auxiliary_findings"] = None
            user.pop("must_cite_verbatim", None)
            rec["messages"][1]["content"] = json.dumps(user, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            pass
        rec["metadata"]["sar_variant"] = "bare"
        return rec

    def _replace_narrative(rec: dict, new_narr: str) -> dict:
        out = json.loads(rec["messages"][2]["content"])
        out["suspicious_activity_report"] = new_narr
        rec["messages"][2]["content"] = json.dumps(out, ensure_ascii=False)
        return rec

    # ---- Pass 1: AUG-CITES failures → demote, then regenerate narrative once
    aug_cites_fail_idx: list[int] = []
    for i, rec in enumerate(records):
        ok, _reason = rule_7_aug_cites(rec)
        if not ok:
            aug_cites_fail_idx.append(i)

    drop_after_aug_demote: set[int] = set()
    if aug_cites_fail_idx:
        for i in aug_cites_fail_idx:
            _apply_demote_to_bare(records[i])
            counters["demoted"] += 1
        # Regenerate narratives once for demoted records (using bare-style content)
        regen_targets = [(i, records[i]["messages"][1]["content"]) for i in aug_cites_fail_idx]
        regen = _llm_generate_narratives(regen_targets, dry_run=dry_run)
        counters["extra_llm_calls"] += len(regen)
        for i in aug_cites_fail_idx:
            if i in regen:
                _replace_narrative(records[i], regen[i])
            else:
                # Change 3: regen failed — drop the record (no stub fallback)
                drop_after_aug_demote.add(i)

    # ---- Pass 2: LENGTH failures → one re-roll, then keep
    length_fail_idx = [i for i, rec in enumerate(records) if not rule_7_length(rec)[0]]
    if length_fail_idx:
        regen_targets = [(i, records[i]["messages"][1]["content"]) for i in length_fail_idx]
        regen = _llm_generate_narratives(regen_targets, dry_run=dry_run)
        counters["extra_llm_calls"] += len(regen)
        counters["rerolled_length"] += len(length_fail_idx)
        for i in length_fail_idx:
            if i in regen:
                _replace_narrative(records[i], regen[i])

    # ---- Pass 3: OBJECTIVITY / BARE-NOLEAK / ENGLISH failures → 2 re-rolls then drop
    def _obj_leak_fail(rec) -> bool:
        return (not rule_7_objectivity(rec)[0]
                or not rule_7_bare_noleak(rec)[0]
                or not rule_7_english(rec)[0])

    for attempt in range(2):
        bad_idx = [i for i, rec in enumerate(records) if _obj_leak_fail(rec)]
        if not bad_idx:
            break
        regen_targets = [(i, records[i]["messages"][1]["content"]) for i in bad_idx]
        regen = _llm_generate_narratives(regen_targets, dry_run=dry_run)
        counters["extra_llm_calls"] += len(regen)
        counters["rerolled_obj_or_leak"] += len(bad_idx)
        for i in bad_idx:
            if i in regen:
                _replace_narrative(records[i], regen[i])

    # ---- Pass 4: drop on remaining failures (NEG-EMPTY, persistent OBJ/LEAK,
    # and AUG-CITES regen failures from pass 1)
    final_records: list[dict] = []
    for i, rec in enumerate(records):
        if i in drop_after_aug_demote:
            counters["dropped"] += 1
            continue
        if not rule_7_neg_empty_narrative(rec)[0]:
            counters["dropped"] += 1
            continue
        if _obj_leak_fail(rec):
            counters["dropped"] += 1
            continue
        final_records.append(rec)

    # ========================================================================
    # Change 1 — In-pipeline LLM reviewer (Stage 7 narrative gate).
    # ========================================================================
    # Reviewer reads each record's user + assistant content and judges
    # whether the narrative is grounded, objective, free of fabrication /
    # off-topic citations / wrong-threshold reasoning. Three buckets per
    # variant × is_suspicious cell. Records that fail are dropped (no
    # re-roll — Stage 7 already does objectivity / length / aug-cites
    # re-rolls; the reviewer catches what those miss).
    n_reviewer_dropped_sar = {"sar_pos": 0, "sar_neg": 0, "sar_adv": 0}
    if final_records and not dry_run:
        from scripts.common.reviewer import review_records, summarise_verdicts

        # Bucket each record
        def _bucket_for(rec: dict) -> str:
            md = rec.get("metadata") or {}
            try:
                asst = json.loads(rec["messages"][2]["content"])
            except Exception:                            # noqa: BLE001
                asst = {}
            is_susp = bool(asst.get("is_suspicious"))
            variant = md.get("sar_variant", "")
            if variant == "adversarial_aux" and is_susp:
                return "sar_adv"
            return "sar_pos" if is_susp else "sar_neg"

        per_bucket: dict[str, list[tuple[int, dict]]] = {}
        for i, rec in enumerate(final_records):
            bucket = _bucket_for(rec)
            payload = {
                "user_content": rec["messages"][1]["content"][:8000],
                "assistant_content": rec["messages"][2]["content"][:4000],
            }
            per_bucket.setdefault(bucket, []).append((i, payload))

        verdicts_by_idx: dict[int, dict] = {}
        for bucket, items in per_bucket.items():
            payloads = [p for (_, p) in items]
            verdicts = review_records(
                payloads,
                bucket=bucket,
                artifact_subdir=f"stage7_reviewer_{bucket}",
                dataset_name=f"stage_7_reviewer_{bucket}",
                dry_run=False,
            )
            for (idx, _), v in zip(items, verdicts):
                verdicts_by_idx[idx] = v
            summary = summarise_verdicts(verdicts)
            logger.info(
                "Stage 7 reviewer (%s): pass_rate=%.1f%% (%d/%d) top_issues=%s",
                bucket, summary["pass_rate"] * 100, summary["pass"],
                summary["n"], summary["issues_top"],
            )

        # Apply verdicts: keep PASS, drop ISSUES_FOUND
        kept: list[dict] = []
        for i, rec in enumerate(final_records):
            v = verdicts_by_idx.get(i)
            if v is None or v.get("verdict") == "PASS":
                kept.append(rec)
            else:
                bucket = _bucket_for(rec)
                n_reviewer_dropped_sar[bucket] = n_reviewer_dropped_sar.get(bucket, 0) + 1
                # Annotate the dropped record's reason for traceability
                rec.setdefault("metadata", {})["reviewer_drop_reason"] = (
                    v.get("explain", "")[:200])
        final_records = kept

    write_jsonl(final_records, OUTPUT_FILE)

    # Final validation gate: validation should now pass on most rules
    rules = [
        ("RULE-7-NEG-EMPTY-NARRATIVE", rule_7_neg_empty_narrative),
        ("RULE-7-OBJECTIVITY", rule_7_objectivity),
        ("RULE-7-BARE-NOLEAK", rule_7_bare_noleak),
        ("RULE-7-AUG-CITES", rule_7_aug_cites),
        ("RULE-7-LENGTH", rule_7_length),
    ]
    manifest = run_stage_gate(
        stage_id=STAGE_ID, pipeline="nonaux", timer=timer,
        input_files=[INPUT_FILE], output_files=[OUTPUT_FILE],
        counts={"input": len(df), "produced": len(final_records),
                "llm_narratives": llm_calls,
                "extra_llm_calls": counters["extra_llm_calls"],
                "demoted_to_bare": counters["demoted"],
                "rerolled_length": counters["rerolled_length"],
                "rerolled_obj_or_leak": counters["rerolled_obj_or_leak"],
                "dropped": counters["dropped"],
                "dropped_no_llm_narrative": n_dropped_no_llm_narrative,
                "reviewer_dropped_total": sum(n_reviewer_dropped_sar.values()),
                **{f"reviewer_dropped_{k}": v for k, v in n_reviewer_dropped_sar.items()},
                "seed_passthrough": sum(1 for r in final_records
                                        if not r["metadata"]["synthetic"]
                                        and r["metadata"]["sar_variant"] == "bare"),
                "negatives": sum(1 for r in final_records
                                 if not json.loads(r["messages"][2]["content"])["is_suspicious"])},
        llm_calls=llm_calls + counters["extra_llm_calls"],
        records_for_validation=final_records, rules=rules,
        manifest_path=MANIFEST_FILE,
        notes=(f"Narrative via DD; demoted={counters['demoted']}, "
               f"length-reroll={counters['rerolled_length']}, "
               f"obj/leak-reroll={counters['rerolled_obj_or_leak']}, "
               f"dropped={counters['dropped']}."),
    )
    return manifest.model_dump()


if __name__ == "__main__":
    run(total_records=75000)
