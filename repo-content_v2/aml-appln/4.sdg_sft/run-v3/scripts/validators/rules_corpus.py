"""Corpus-level rules (RULE-1-* + RULE-9-DEDUP + RULE-9-SOURCE-CAP).

Aggregate computations that span the entire produced corpus. Called in
Stage 9 (non-aux) and Stage A4 (aux).
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================================
# RULE-1-MIX-LABEL — 45% positive / 55% negative within sar_judgment
# ============================================================================
def audit_mix_label(records: Iterable[dict[str, Any]], tolerance_pp: float = 2.0) -> dict[str, Any]:
    pos = neg = 0
    for r in records:
        out = r.get("messages", [{}, {}, {}])[2].get("content", "")
        try:
            parsed = json.loads(out) if isinstance(out, str) else out
        except Exception:  # noqa: BLE001
            parsed = {}
        if isinstance(parsed, dict):
            if parsed.get("is_suspicious"):
                pos += 1
            else:
                neg += 1
    total = pos + neg
    obs = {"positive": pos / total if total else 0.0, "negative": neg / total if total else 0.0}
    target = {"positive": 0.45, "negative": 0.55}
    diffs_pp = {k: abs(obs[k] - target[k]) * 100 for k in target}
    return {
        "rule": "RULE-1-MIX-LABEL",
        "target": target,
        "observed": obs,
        "tolerance_pp": tolerance_pp,
        "passed": max(diffs_pp.values()) <= tolerance_pp,
    }


# ============================================================================
# RULE-1-MIX-VARIANT — 70/25/5 augmented/bare/adversarial
# ============================================================================
def audit_mix_variant(records: Iterable[dict[str, Any]], tolerance_pp: float = 2.0) -> dict[str, Any]:
    """Variant-mix audit.

    The strategy doc target (70/25/5 augmented/bare/adversarial_aux) is
    unconditional, but it is only reachable on records that are *eligible* for
    aux findings (typology != none AND sufficient transactions/policy
    excerpts). Records with typology=none structurally cannot carry findings
    (RULE-5-NONE-EMPTY), so any augmented/adversarial_aux variant assignment
    on those records is forced to demote to bare in Stage 6.

    To make the audit faithful to the underlying generator, we report TWO views:
      - `observed_unconditional`: proportions over the full corpus (informational).
      - `observed_eligible`: proportions over records with typology != none.
    The `passed` flag is computed on the eligible view (the only one the
    sampler can actually control).
    """
    counter_all = Counter()
    counter_elig = Counter()
    total_all = total_elig = 0
    for r in records:
        v = r.get("metadata", {}).get("sar_variant")
        typ = r.get("metadata", {}).get("typology")
        if v in ("augmented", "bare", "adversarial_aux"):
            counter_all[v] += 1
            total_all += 1
            if typ and typ != "none":
                counter_elig[v] += 1
                total_elig += 1
    keys = ("augmented", "bare", "adversarial_aux")
    obs_all = {k: counter_all[k] / total_all if total_all else 0.0 for k in keys}
    obs_elig = {k: counter_elig[k] / total_elig if total_elig else 0.0 for k in keys}
    target = {"augmented": 0.70, "bare": 0.25, "adversarial_aux": 0.05}
    diffs_pp = {k: abs(obs_elig[k] - target[k]) * 100 for k in target}
    return {
        "rule": "RULE-1-MIX-VARIANT",
        "target": target,
        "observed": obs_elig,
        "observed_unconditional": obs_all,
        "eligible_records": total_elig,
        "total_records": total_all,
        "tolerance_pp": tolerance_pp,
        "passed": max(diffs_pp.values()) <= tolerance_pp,
        "note": ("Computed on records with typology != none (eligible for "
                 "auxiliary_findings per RULE-5-NONE-EMPTY). Records with "
                 "typology=none cannot carry findings and demote to bare."),
    }


# ============================================================================
# RULE-1-FLOOR-TYPOLOGY — every typology ≥ 3% of corpus
# ============================================================================
def audit_floor_typology(records: Iterable[dict[str, Any]], floor: float = 0.03) -> dict[str, Any]:
    counter = Counter()
    total = 0
    for r in records:
        typology = r.get("metadata", {}).get("typology")
        if typology and typology != "none":
            counter[typology] += 1
            total += 1
    shares = {k: counter[k] / total for k in counter} if total else {}
    below = {k: round(s, 4) for k, s in shares.items() if s < floor}
    return {
        "rule": "RULE-1-FLOOR-TYPOLOGY",
        "floor": floor,
        "observed_shares": shares,
        "below_floor": below,
        "passed": not below,
    }


# ============================================================================
# RULE-1-FLOOR-NEAR-MISS — ≥ 30% of negatives have surface_pattern=near_miss
# ============================================================================
def audit_floor_near_miss(records: Iterable[dict[str, Any]], floor: float = 0.30) -> dict[str, Any]:
    neg_count = near_miss_count = 0
    for r in records:
        out = r.get("messages", [{}, {}, {}])[2].get("content", "")
        try:
            parsed = json.loads(out) if isinstance(out, str) else out
        except Exception:  # noqa: BLE001
            parsed = {}
        is_susp = isinstance(parsed, dict) and parsed.get("is_suspicious")
        if is_susp:
            continue
        neg_count += 1
        # surface_pattern is in metadata if we propagated it; otherwise infer.
        sp = r.get("metadata", {}).get("surface_pattern", "direct")
        if sp == "near_miss":
            near_miss_count += 1
    obs = (near_miss_count / neg_count) if neg_count else 0.0
    return {
        "rule": "RULE-1-FLOOR-NEAR-MISS",
        "floor": floor,
        "observed": obs,
        "negatives": neg_count,
        "near_miss": near_miss_count,
        "passed": obs >= floor,
    }


# ============================================================================
# APPENDIX-G — every canonical typology has ≥ 1 POSITIVE record
#
# Closes the "0 structuring positives" gap that the previous floor-typology
# audit could miss (a typology can pass the share floor on negatives alone).
# A typology with zero positive records produces a model that cannot recognize
# that typology at runtime regardless of the share — block on this.
# ============================================================================
_CANONICAL_TYPOLOGIES = (
    "structuring", "smurfing", "layering", "trade_based_ml",
    "shell_company", "human_trafficking", "terrorist_financing",
    "elder_exploitation",
)

# Canonical regulatory frames the corpus emits. `te` is intentionally absent —
# strategy doc §4.2 drops it (0.3% share, 100% one-sided = label proxy).
_CANONICAL_FRAMES = (
    "layering_passthrough", "ctr_structuring", "tbml", "shell",
    "sanctions", "elder", "trafficking", "benign",
)


def audit_typology_positive_coverage(
    records: Iterable[dict[str, Any]],
    *,
    min_positive_per_typology: int = 1,
    typologies: Iterable[str] = _CANONICAL_TYPOLOGIES,
) -> dict[str, Any]:
    """Every typology must have ≥ N positive records (N proportional to N for
    smaller smoke runs; default = 1 per typology)."""
    pos_counter: Counter = Counter()
    for r in records:
        out = r.get("messages", [{}, {}, {}])[2].get("content", "")
        try:
            parsed = json.loads(out) if isinstance(out, str) else out
        except Exception:  # noqa: BLE001
            parsed = {}
        if not (isinstance(parsed, dict) and parsed.get("is_suspicious")):
            continue
        typology = r.get("metadata", {}).get("typology")
        if typology and typology != "none":
            pos_counter[typology] += 1
    missing = {t: pos_counter.get(t, 0) for t in typologies
               if pos_counter.get(t, 0) < min_positive_per_typology}
    return {
        "rule": "APPENDIX-G-TYPOLOGY-POS-COVERAGE",
        "min_positive_per_typology": min_positive_per_typology,
        "observed_positive_counts": dict(pos_counter),
        "missing": missing,
        "passed": not missing,
    }


# ============================================================================
# RULE-1-CAP-LAYERING — `_regulatory_frame=layering_passthrough` ≤ 25% (+3pp slack)
# of the SAR-judgment corpus. Strategy doc §4.2: layering was historically
# dominant and is the largest source of false positives in production.
# ============================================================================
def audit_cap_layering(
    records: Iterable[dict[str, Any]],
    *,
    cap: float = 0.25,
    tolerance_pp: float = 3.0,
) -> dict[str, Any]:
    total = 0
    n_layering = 0
    for r in records:
        u_content = r.get("messages", [{}, {}, {}])[1].get("content", "")
        try:
            uj = json.loads(u_content) if isinstance(u_content, str) else u_content
        except Exception:  # noqa: BLE001
            uj = {}
        if not isinstance(uj, dict) or uj.get("task_type") != "sar_judgment":
            continue
        # Frame in metadata first (where stage_1 stored it), fall back to
        # rule-layer guess if you ever re-add it to metadata.
        frame = (r.get("metadata") or {}).get("regulatory_frame")
        if not frame:
            sp = (r.get("metadata") or {}).get("semantic_profile") or {}
            frame = sp.get("regulatory_frame")
        if frame == "layering_passthrough":
            n_layering += 1
        total += 1
    share = n_layering / total if total else 0.0
    return {
        "rule": "RULE-1-CAP-LAYERING",
        "cap": cap,
        "tolerance_pp": tolerance_pp,
        "observed": round(share, 4),
        "n_layering": n_layering,
        "n_total_sar": total,
        "passed": share <= cap + tolerance_pp / 100,
    }


# ============================================================================
# RULE-1-FRAME-LABEL-BALANCE — no frame outside 30/70 ↔ 70/30 positive split.
# Strategy doc §4.2: every frame must carry minority-class examples to
# prevent the frame name from becoming a label proxy (the same disease
# `_decision_target` had). `benign` needs explicit positives; `sanctions`
# needs explicit negatives; etc.
# ============================================================================
def audit_frame_label_balance(
    records: Iterable[dict[str, Any]],
    *,
    min_minority_share: float = 0.30,
) -> dict[str, Any]:
    by_frame_label: dict[str, dict[bool, int]] = {}
    for r in records:
        u_content = r.get("messages", [{}, {}, {}])[1].get("content", "")
        out_content = r.get("messages", [{}, {}, {}])[2].get("content", "")
        try:
            uj = json.loads(u_content) if isinstance(u_content, str) else u_content
            aj = json.loads(out_content) if isinstance(out_content, str) else out_content
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(uj, dict) or uj.get("task_type") != "sar_judgment":
            continue
        meta = r.get("metadata") or {}
        frame = meta.get("regulatory_frame") or (meta.get("semantic_profile") or {}).get("regulatory_frame")
        if frame is None:
            continue
        is_susp = bool(aj.get("is_suspicious")) if isinstance(aj, dict) else False
        by_frame_label.setdefault(frame, {True: 0, False: 0})[is_susp] += 1

    failures = []
    per_frame = {}
    for frame, counts in by_frame_label.items():
        n_pos = counts[True]
        n_neg = counts[False]
        n_total = n_pos + n_neg
        if n_total == 0:
            continue
        minority_share = min(n_pos, n_neg) / n_total
        per_frame[frame] = {
            "n_pos": n_pos, "n_neg": n_neg,
            "minority_share": round(minority_share, 4),
            "passed": minority_share >= min_minority_share,
        }
        if minority_share < min_minority_share:
            failures.append(frame)
    return {
        "rule": "RULE-1-FRAME-LABEL-BALANCE",
        "min_minority_share": min_minority_share,
        "per_frame": per_frame,
        "frames_below_floor": failures,
        "passed": not failures,
    }


# ============================================================================
# RULE-1-NARR-LENGTH-MATCHED — positive and negative narrative length
# distributions must be within tolerance of each other so the model can't
# learn "short string = False" as a leaky shortcut. Strategy doc §4.2 + §6.1.
# ============================================================================
def audit_narrative_length_matched(
    records: Iterable[dict[str, Any]],
    *,
    median_tolerance_pct: float = 0.20,
) -> dict[str, Any]:
    pos_lens: list[int] = []
    neg_lens: list[int] = []
    for r in records:
        u_content = r.get("messages", [{}, {}, {}])[1].get("content", "")
        out_content = r.get("messages", [{}, {}, {}])[2].get("content", "")
        try:
            uj = json.loads(u_content) if isinstance(u_content, str) else u_content
            aj = json.loads(out_content) if isinstance(out_content, str) else out_content
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(uj, dict) or uj.get("task_type") != "sar_judgment":
            continue
        if not isinstance(aj, dict):
            continue
        narr = aj.get("suspicious_activity_report", "") or ""
        if aj.get("is_suspicious"):
            pos_lens.append(len(narr))
        else:
            neg_lens.append(len(narr))

    def _median(lst: list[int]) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    pos_med = _median(pos_lens)
    neg_med = _median(neg_lens)
    if pos_med == 0 or neg_med == 0:
        passed = False
        diff_pct = 1.0
    else:
        diff_pct = abs(pos_med - neg_med) / max(pos_med, neg_med)
        passed = diff_pct <= median_tolerance_pct
    return {
        "rule": "RULE-1-NARR-LENGTH-MATCHED",
        "median_tolerance_pct": median_tolerance_pct,
        "n_positive": len(pos_lens),
        "n_negative": len(neg_lens),
        "positive_median_chars": round(pos_med, 1),
        "negative_median_chars": round(neg_med, 1),
        "median_diff_pct": round(diff_pct, 4),
        "passed": passed,
    }


# ============================================================================
# RULE-1-NO-LEAKY-FIELDS — the SAR-judgment user message must NEVER contain
# `_decision_target`, `_regulatory_frame`, or `_typology_inferred`. These
# were rule-layer verdicts in earlier versions and caused 100%-follow-rate
# label leakage. Strategy doc §2.1 Rule A.
# ============================================================================
_FORBIDDEN_HINT_KEYS = ("_decision_target", "_regulatory_frame", "_typology_inferred")


def audit_no_leaky_fields(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    offenders = Counter()
    scanned = 0
    for r in records:
        u_content = r.get("messages", [{}, {}, {}])[1].get("content", "")
        try:
            uj = json.loads(u_content) if isinstance(u_content, str) else u_content
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(uj, dict) or uj.get("task_type") != "sar_judgment":
            continue
        scanned += 1
        for k in _FORBIDDEN_HINT_KEYS:
            if k in uj:
                offenders[k] += 1
    return {
        "rule": "RULE-1-NO-LEAKY-FIELDS",
        "forbidden_keys": list(_FORBIDDEN_HINT_KEYS),
        "n_sar_records_scanned": scanned,
        "offending_counts": dict(offenders),
        "passed": not offenders,
    }


# ============================================================================
# RULE-9-SOURCE-CAP — no single source > 30% of corpus
# ============================================================================
def audit_source_cap(records: Iterable[dict[str, Any]], cap: float = 0.30) -> dict[str, Any]:
    counter = Counter()
    total = 0
    for r in records:
        src = r.get("metadata", {}).get("source")
        if src:
            counter[src] += 1
            total += 1
    shares = {k: counter[k] / total for k in counter} if total else {}
    above = {k: round(s, 4) for k, s in shares.items() if s > cap}
    return {
        "rule": "RULE-9-SOURCE-CAP",
        "cap": cap,
        "observed_shares": shares,
        "above_cap": above,
        "passed": not above,
    }


# ============================================================================
# RULE-9-DEDUP — MinHash 0.85 over messages content
# ============================================================================
def dedup_minhash(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deduped records + audit dict. Removes near-duplicates (Jaccard ≥ threshold).

    Signature: `(user_content + assistant_content)`. The system prompt is
    intentionally excluded — it's a single constant (~3KB) shared by every
    sar_judgment record, contributing zero discriminative signal but
    inflating the Jaccard estimate so much that genuinely different records
    cross the 0.85 threshold.

    Negative records (`is_suspicious=false`) have an identical empty-narrative
    assistant content by design; their meaningful content lives in the user
    bundle (transactions + KYC). For these we hash only the user content.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed — RULE-9-DEDUP skipped.")
        return records, {"rule": "RULE-9-DEDUP", "passed": True, "skipped": True,
                         "removed": 0, "kept": len(records)}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices: list[int] = []
    removed: int = 0

    def _signature_text(rec: dict[str, Any]) -> str:
        """Build the MinHash signature text per record type.

        Strategy intent (`RULE-9-DEDUP`): catch genuine duplicates at Jaccard
        ≥ threshold while ignoring incidental overlap of shared input
        passages or templated system prompts. Signatures are tuned per record
        type so the hashed text is the *discriminative* part of the record:

          - sar_judgment positive: user bundle + assistant SAR narrative
          - sar_judgment negative: user bundle only (assistant is the
            constant empty-narrative envelope)
          - auxiliary_*: assistant content only — the question + answer +
            evidence_span / calculation / reasoning IS the unique signal.
            The user side replays the (long, often-shared) source passage,
            which inflates Jaccard to falsely flag distinct Q/A pairs as
            duplicates.

        The threshold (default 0.85) and num_perm are unchanged from the
        strategy spec.
        """
        msgs = rec.get("messages") or []
        # Skip the system prompt (msgs[0]) — it's templated, never discriminative.
        user = msgs[1].get("content", "") if len(msgs) >= 2 else ""
        asst = msgs[2].get("content", "") if len(msgs) >= 3 else ""

        task_type = (rec.get("metadata") or {}).get("task_type") or ""
        if task_type.startswith("auxiliary_"):
            # Aux records: assistant content carries the unique Q/A signal.
            return asst

        # sar_judgment branch
        try:
            asst_obj = json.loads(asst) if isinstance(asst, str) else asst
            is_susp = isinstance(asst_obj, dict) and bool(asst_obj.get("is_suspicious"))
            if not is_susp:
                # Negatives: assistant is identical empty-narrative envelope.
                return user
        except Exception:  # noqa: BLE001
            pass
        return user + " || " + asst

    for idx, rec in enumerate(records):
        content = _signature_text(rec)
        shingles = {content[i:i + shingle_size] for i in range(0, max(0, len(content) - shingle_size + 1))}
        if not shingles:
            keep_indices.append(idx)
            continue
        m = MinHash(num_perm=num_perm)
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        key = f"r{idx}"
        if lsh.query(m):
            removed += 1
            continue
        lsh.insert(key, m)
        keep_indices.append(idx)

    kept = [records[i] for i in keep_indices]
    audit = {
        "rule": "RULE-9-DEDUP",
        "threshold": threshold,
        "num_perm": num_perm,
        "input": len(records),
        "kept": len(kept),
        "removed": removed,
        "removed_pct": round(removed / max(len(records), 1), 4),
        "passed": True,
        "signature": "per-record-type: aux=assistant; nonaux-pos=user+assistant; nonaux-neg=user (system always excluded)",
    }
    return kept, audit


# ============================================================================
# Aux-pipeline contract test (Stage A4): aux record output ≡ Stage 6 inline shape
# ============================================================================
def output_shape_contract_test(
    aux_records: list[dict[str, Any]],
    sample_inline_findings: list[dict[str, Any]],
    *,
    sample_n: int = 200,
) -> dict[str, Any]:
    """Sample N records from each side and assert both deserialize into the same schema."""
    from scripts.schemas import CitationFinding, NumericFinding, StatutoryFinding

    schema_for_kind = {
        "auxiliary_numeric": NumericFinding,
        "auxiliary_citation": CitationFinding,
        "auxiliary_statutory": StatutoryFinding,
    }
    failures: list[str] = []
    checked = 0

    # Side 1: aux records
    for rec in aux_records[:sample_n]:
        tt = rec.get("metadata", {}).get("task_type")
        sch = schema_for_kind.get(tt)
        if sch is None:
            continue
        out = rec.get("messages", [{}, {}, {}])[2].get("content", "")
        try:
            parsed = json.loads(out) if isinstance(out, str) else out
            sch(**parsed)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"aux-record {rec.get('metadata', {}).get('record_id')} "
                            f"task={tt}: {exc}")
        checked += 1

    # Side 2: inline findings (e.g., from a sar_judgment record's auxiliary_findings)
    for entry in sample_inline_findings[:sample_n]:
        kind = entry.get("kind")  # caller tags each item with which sub-array it came from
        payload = entry.get("payload")
        sch = {"numeric": NumericFinding, "citation": CitationFinding,
               "statutory": StatutoryFinding}.get(kind)
        if sch is None or payload is None:
            continue
        try:
            sch(**payload)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"inline {kind}: {exc}")
        checked += 1

    return {
        "rule": "AUX-OUTPUT-SHAPE-CONTRACT",
        "samples_checked": checked,
        "failures": len(failures),
        "sample_failures": failures[:10],
        "passed": not failures,
    }


# ============================================================================
# Appendix-G unified corpus blocking gate
#
# Per training_strategy.md Appendix G, the construction pipeline emits a
# `phase_stats.json` and BLOCKS if any of the following gates fail:
#   - Per-typology floor (audit_floor_typology)
#   - Per-typology positive coverage (audit_typology_positive_coverage) — NEW
#   - Pos/neg ratio (audit_mix_label)
#   - Variant ratio (audit_mix_variant)
#   - Source contribution cap (audit_source_cap)
#   - Aux task balance — handled in Stage A4
#   - Schema validity — handled per-record at every stage
#
# This function is called from Stage 9 (non-aux). For smoke-scale runs
# (N < 1000) gates are RELAXED proportionally — a 500-record smoke can't
# satisfy "≥ 1.5K records per typology" and shouldn't be expected to.
# ============================================================================
def corpus_blocking_gate(
    records: list[dict[str, Any]],
    *,
    n_target: int = 75000,
    smoke_relaxation: bool = True,
) -> dict[str, Any]:
    """Run all blocking gates per Appendix G; return composite verdict.

    The composite `blocked` boolean is True iff at least one of the below
    gates fails. The pipeline orchestrator should treat True as a hard fail
    and exit non-zero with the diagnostics in `failed_gates`.
    """
    n_recs = len(records)

    # Smoke relaxation: scale per-typology positive minimum by run size.
    # For N >= 5000, demand ≥ 50 positives per typology.
    # For smaller runs, demand at least 1 (so small smoke runs still surface
    # the "0 positives in a typology" problem).
    if smoke_relaxation and n_recs < 1000:
        min_pos = 1
    else:
        min_pos = max(1, int(n_recs * 0.005))     # ~0.5% of corpus each

    gates: list[dict[str, Any]] = []
    gates.append(audit_mix_label(records))
    gates.append(audit_mix_variant(records))
    gates.append(audit_floor_typology(records, floor=0.01 if n_recs < 1000 else 0.03))
    gates.append(audit_typology_positive_coverage(records, min_positive_per_typology=min_pos))
    gates.append(audit_source_cap(records, cap=0.30))
    # Near-miss floor is best-effort; warn-only at smoke scale (often there
    # aren't enough negatives to compute a stable share). We still surface it.
    gates.append(audit_floor_near_miss(records, floor=0.30 if n_recs >= 1000 else 0.0))

    failed = [g for g in gates if not g.get("passed", True)]
    return {
        "n_records": n_recs,
        "n_target": n_target,
        "smoke_relaxation": smoke_relaxation,
        "min_pos_per_typology": min_pos,
        "gates": gates,
        "failed_gates": [g["rule"] for g in failed],
        "blocked": bool(failed),
    }
