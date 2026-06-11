"""Per-record validators (RULE-3-* through RULE-8-*).

Each validator has the signature `(record: dict) -> (passed: bool, reason: str)`.

`record` is a chat-SFT record dict (post Stage 7 assembly) for narrative-level
rules, OR a partial record for stage-internal calls. The validator inspects
whichever fields exist and returns True if not applicable.
"""
from __future__ import annotations

import json
import re
from typing import Any

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None  # noqa: N816

from pipeline.schemas import (
    AuxiliaryFindings,
    CitationFinding,
    KYCProfile,
    NumericFinding,
    SARJudgmentInput,
    StatutoryFinding,
)


# ============================================================================
# Helpers
# ============================================================================
_NUM_PAT = re.compile(r"-?\$?\s*\d{1,3}(?:[,\.]\d{3})*(?:\.\d+)?")
_TX_INDEX_PAT = re.compile(r"transactions?\s*\[\s*(\d+)\s*(?:\.\.|-)\s*(\d+)\s*\]"
                           r"|transactions?\s*\[\s*([\d,\s]+?)\s*\]")


def _extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    out: list[float] = []
    for m in _NUM_PAT.finditer(text):
        raw = m.group(0).replace("$", "").replace(",", "").replace(" ", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _parse_tx_indices(evidence: str) -> list[int]:
    """Parse strings like 'transactions[0..5]' or 'transactions[2,4,7]'."""
    if not evidence:
        return []
    indices: list[int] = []
    for m in _TX_INDEX_PAT.finditer(evidence):
        if m.group(1) is not None and m.group(2) is not None:
            try:
                a, b = int(m.group(1)), int(m.group(2))
                indices.extend(range(min(a, b), max(a, b) + 1))
            except ValueError:
                pass
        elif m.group(3) is not None:
            for tok in re.split(r"[,\s]+", m.group(3)):
                if tok.isdigit():
                    indices.append(int(tok))
    return sorted(set(indices))


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _approx_equal(a: float, b: float, *, rel_tol: float = 0.05, abs_tol: float = 1.0) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def _get_input(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the 'input' bundle from a chat-SFT record's user message."""
    msgs = record.get("messages") or []
    if len(msgs) >= 2:
        user_content = msgs[1].get("content", "")
        try:
            parsed = json.loads(user_content) if isinstance(user_content, str) else user_content
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            return {}
    # Fallback for partial / pre-assembly records
    return record.get("input", {}) if isinstance(record.get("input"), dict) else {}


def _get_output(record: dict[str, Any]) -> dict[str, Any]:
    msgs = record.get("messages") or []
    if len(msgs) >= 3:
        asst = msgs[2].get("content", "")
        try:
            parsed = json.loads(asst) if isinstance(asst, str) else asst
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            return {"suspicious_activity_report": asst}
    return record.get("output", {}) if isinstance(record.get("output"), dict) else {}


# ============================================================================
# RULE-3-* — Stage 1 hard rules (already enforced at sample time, but checked here)
# ============================================================================
ARCHETYPE_TYPOLOGY_COMPAT: dict[str, set[str]] = {
    "structuring": {
        "individual — wage earner", "individual — small business owner", "individual — retiree (age 65+)",
        "retail business — cash-intensive (jewelry/precious metals)",
        "retail business — cash-intensive (restaurant/hospitality)",
        "retail business — cash-intensive (laundromat/car wash)",
        "retail business — cash-intensive (convenience/grocery)",
        "money services business",
    },
    "smurfing": {
        "individual — wage earner", "individual — small business owner", "individual — retiree (age 65+)",
        "retail business — cash-intensive (jewelry/precious metals)",
        "retail business — cash-intensive (restaurant/hospitality)",
        "retail business — cash-intensive (laundromat/car wash)",
        "retail business — cash-intensive (convenience/grocery)",
        "money services business",
    },
    "layering": {
        "broker / dealer", "money services business", "crypto-exchange / VASP",
        "shell holding (offshore)", "shell holding (domestic)", "import-export firm",
        "individual — wage earner", "individual — small business owner",
    },
    "trade_based_ml": {
        "import-export firm", "broker / dealer",
        "retail business — cash-intensive (convenience/grocery)",
    },
    "shell_company": {
        "shell holding (offshore)", "shell holding (domestic)", "professional services gatekeeper",
    },
    "human_trafficking": {
        "individual — wage earner", "individual — small business owner",
    },
    "terrorist_financing": {
        "individual — wage earner", "NGO / charity",
    },
    "elder_exploitation": {
        "individual — retiree (age 65+)",
    },
    # `none` is always compatible
}


def rule_3_compat(rec: dict) -> tuple[bool, str]:
    typology = rec.get("typology") or rec.get("metadata", {}).get("typology")
    archetype = rec.get("entity_archetype") or rec.get("metadata", {}).get("entity_archetype")
    if typology in (None, "none") or archetype is None:
        return True, ""
    allowed = ARCHETYPE_TYPOLOGY_COMPAT.get(typology, set())
    if not allowed:
        return True, ""
    if archetype in allowed:
        return True, ""
    return False, f"archetype '{archetype}' incompatible with typology '{typology}'"


def rule_3_near_miss_benign(rec: dict) -> tuple[bool, str]:
    sp = rec.get("surface_pattern") or rec.get("metadata", {}).get("surface_pattern")
    label = rec.get("label")
    if label is None:
        out = _get_output(rec)
        label = out.get("is_suspicious")
    if sp == "near_miss" and label is True:
        return False, "surface_pattern=near_miss must imply label=false"
    return True, ""


def rule_3_none_benign(rec: dict) -> tuple[bool, str]:
    typology = rec.get("typology") or rec.get("metadata", {}).get("typology")
    label = rec.get("label")
    if label is None:
        out = _get_output(rec)
        label = out.get("is_suspicious")
    if typology == "none" and label is True:
        return False, "typology=none must imply label=false"
    return True, ""


# ============================================================================
# RULE-4-COUNTERPARTY-OVERLAP
# ============================================================================
def rule_4_counterparty_overlap(rec: dict) -> tuple[bool, str]:
    inp = _get_input(rec)
    hits = inp.get("sanctions_pep_hits") or []
    txs = inp.get("transactions") or []
    if not hits:
        return True, ""
    counterparties = " || ".join((t.get("counterparty") or "").lower() for t in txs)
    for h in hits:
        name = (h.get("name") or "").lower()
        if not name:
            continue
        if name not in counterparties:
            if fuzz is None:
                return False, f"hit name '{h.get('name')}' not in transactions[].counterparty"
            best = max((fuzz.partial_ratio(name, (t.get("counterparty") or "").lower())
                        for t in txs), default=0)
            if best < 85:
                return False, f"hit name '{h.get('name')}' not in transactions[].counterparty (fuzz={best})"
    return True, ""


# ============================================================================
# RULE-5-NONE-EMPTY
# ============================================================================
def rule_5_none_empty(rec: dict) -> tuple[bool, str]:
    typology = rec.get("typology") or rec.get("metadata", {}).get("typology")
    if typology != "none":
        return True, ""
    inp = _get_input(rec)
    if (inp.get("policy_excerpts") or []):
        return False, "typology=none but policy_excerpts non-empty"
    if (inp.get("sop_excerpts") or []):
        return False, "typology=none but sop_excerpts non-empty"
    return True, ""


# ============================================================================
# RULE-6-* — auxiliary_findings consistency
# ============================================================================
def rule_6_num_sum(rec: dict) -> tuple[bool, str]:
    inp = _get_input(rec)
    aux = inp.get("auxiliary_findings") or {}
    nums = aux.get("numeric") or []
    txs = inp.get("transactions") or []
    if not nums or not txs:
        return True, ""
    for f in nums:
        idxs = _parse_tx_indices(f.get("evidence", ""))
        if not idxs:
            continue
        try:
            actual_sum = sum(float(txs[i].get("amount", 0)) for i in idxs if 0 <= i < len(txs))
        except Exception as exc:  # noqa: BLE001
            return False, f"sum failed: {exc}"
        nums_in = _extract_numbers(f.get("calculation", "") + " " + f.get("answer", ""))
        if not nums_in:
            continue
        if not any(_approx_equal(c, actual_sum) for c in nums_in):
            return False, (f"numeric.calculation/answer does not contain Σ={actual_sum:.2f} "
                           f"of transactions{idxs}; saw {nums_in[:5]}")
    return True, ""


def rule_6_cit_verbatim(rec: dict) -> tuple[bool, str]:
    inp = _get_input(rec)
    aux = inp.get("auxiliary_findings") or {}
    cits = aux.get("citation") or []
    excerpts = inp.get("policy_excerpts") or []
    if not cits:
        return True, ""
    haystack = " || ".join(_normalize_ws(e.get("text", "")) for e in excerpts)
    for f in cits:
        span = _normalize_ws(f.get("evidence_span", ""))
        if not span:
            return False, "citation.evidence_span empty"
        if span in haystack:
            continue
        if fuzz is not None:
            best = max((fuzz.partial_ratio(span, _normalize_ws(e.get("text", "")))
                        for e in excerpts), default=0)
            if best >= 90:
                continue
            return False, f"evidence_span not in any policy_excerpt (best fuzz={best})"
        return False, "evidence_span not a substring of any policy_excerpt"
    return True, ""


TYPOLOGY_TO_STATUTE: dict[str, str] = {
    "structuring": "5324",
    "smurfing": "5324",
    "shell_company": "1010.230",
    "terrorist_financing": "2339B",
    "trade_based_ml": "5318(g)",
    "layering": "5318(g)",
    "human_trafficking": "5318(g)",
    "elder_exploitation": "5318(g)",
}


def rule_6_stat_label(rec: dict) -> tuple[bool, str]:
    inp = _get_input(rec)
    aux = inp.get("auxiliary_findings") or {}
    stats = aux.get("statutory") or []
    if not stats:
        return True, ""
    out = _get_output(rec)
    is_susp = bool(out.get("is_suspicious"))
    expected_label = "entailment" if is_susp else "contradiction"
    typology = rec.get("metadata", {}).get("typology") or rec.get("typology")
    expected_token = TYPOLOGY_TO_STATUTE.get(typology) if typology else None

    sar_variant = rec.get("metadata", {}).get("sar_variant")
    if sar_variant == "adversarial_aux":
        return True, ""  # exempt

    for f in stats:
        label = f.get("label", "")
        if expected_token and expected_token not in (f.get("reasoning") or ""):
            return False, f"statutory.reasoning missing statute token '{expected_token}'"
        if label != expected_label and label != "neutral":
            return False, f"statutory.label='{label}' inconsistent with is_suspicious={is_susp}"
    return True, ""


# ============================================================================
# RULE-7-* — narrative-level
# ============================================================================
OBJECTIVITY_DENYLIST = re.compile(
    r"\b(definitely|clearly\s+laundering|obviously|without\s+(?:a\s+|any\s+)?doubt|"
    r"certainly|is\s+guilty\s+of|beyond\s+doubt|blatant(?:ly)?)\b",
    re.IGNORECASE,
)
BARE_LEAKAGE = re.compile(
    r"as\s+computed\s+above|per\s+the\s+cited\s+section|as\s+derived\s+earlier|"
    r"per\s+auxiliary_findings",
    re.IGNORECASE,
)


def rule_7_neg_empty_narrative(rec: dict) -> tuple[bool, str]:
    """Deprecated alias kept for compatibility — delegates to
    :func:`rule_7_narrative_nonempty`. v3 requires non-empty narratives for
    both classes; the old name is preserved so external callers (audit
    reports, manifests) don't break."""
    return rule_7_narrative_nonempty(rec)


def rule_7_narrative_nonempty(rec: dict) -> tuple[bool, str]:
    """v3 contract: `suspicious_activity_report` must be non-empty for BOTH
    classes. Strategy doc §2.1 Rule B.

    Earlier versions required empty-string on negatives, which
    (a) gave the model no negative-class reasoning supervision and
    (b) created a label-by-length leak. v3 requires a grounded rationale
    on every record regardless of verdict.

    Scope: SAR-judgment records only. Auxiliary records (numeric / citation /
    statutory / behavioral) carry no `suspicious_activity_report` field by
    design, so this rule short-circuits to PASS for them rather than
    falsely flagging the entire aux corpus as narrative-empty. The
    audit-harness used to run this rule on every record indiscriminately,
    which produced a 100% hard-fail rate on aux. This guard restores the
    rule's intended scope.
    """
    md = rec.get("metadata") or {}
    if md.get("task_type") != "sar_judgment":
        return True, ""
    out = _get_output(rec)
    narr = (out.get("suspicious_activity_report") or "").strip()
    if not narr:
        return False, "narrative empty (v3 requires non-empty narrative for both classes)"
    return True, ""


# ---------------------------------------------------------------------------
# RULE-7-NEG-DISPOSITION-MARKER — negative narratives must name the surface
# red flag and a disambiguating reason. Strategy doc §4.2 (no-leak invariants)
# and §5.2 Stage 7. Cheap-to-check heuristic: at least one of a curated set
# of "disposition phrases" must appear in the narrative. The LLM judge in
# audit.py picks up the harder cases (sycophancy, ungrounded claims).
# ---------------------------------------------------------------------------
_DISPOSITION_MARKERS = (
    "consistent with",     "within the declared",  "common-name",
    "verified",            "historical pattern",   "no sar warranted",
    "not actionable",      "does not warrant",     "does not rise to",
    "resolved",            "explained by",         "warrants no filing",
    "no filing warranted", "no further action",    "routine",
    "expected for",        "appropriate for",      "in line with",
)


def rule_7_neg_disposition_marker(rec: dict) -> tuple[bool, str]:
    out = _get_output(rec)
    if out.get("is_suspicious") is not False:
        return True, ""
    narr = (out.get("suspicious_activity_report") or "").lower()
    if not narr:
        # rule_7_narrative_nonempty will catch this; don't double-report.
        return True, ""
    if any(m in narr for m in _DISPOSITION_MARKERS):
        return True, ""
    return False, "negative narrative missing disposition marker (e.g. 'consistent with', 'no SAR warranted', 'within the declared')"


# ---------------------------------------------------------------------------
# RULE-7-NO-LEAKY-HINTS — per-record check that the SAR-judgment user
# message contains none of the forbidden hint fields. The corpus-level audit
# in rules_corpus.py covers aggregate counts; this per-record check fires
# inline in the Stage 7 gate so an offending record can't slip through.
# ---------------------------------------------------------------------------
_FORBIDDEN_HINT_KEYS = ("_decision_target", "_regulatory_frame", "_typology_inferred")


def rule_7_no_leaky_hints(rec: dict) -> tuple[bool, str]:
    inp = _get_input(rec)
    offenders = [k for k in _FORBIDDEN_HINT_KEYS if k in inp]
    if offenders:
        return False, f"user message contains forbidden hint field(s): {offenders}"
    return True, ""


def rule_7_objectivity(rec: dict) -> tuple[bool, str]:
    out = _get_output(rec)
    narr = out.get("suspicious_activity_report") or ""
    m = OBJECTIVITY_DENYLIST.search(narr)
    if m:
        return False, f"narrative contains denylist phrase: '{m.group(0)}'"
    return True, ""


def rule_7_bare_noleak(rec: dict) -> tuple[bool, str]:
    """Bare variant must not cite or paraphrase auxiliary-finding content
    (since the bare user message has `auxiliary_findings=null`). v3 applies
    this to BOTH classes — even a negative narrative on a bare variant
    cannot reference findings that aren't in the bundle.
    """
    md = rec.get("metadata", {})
    if md.get("sar_variant") != "bare":
        return True, ""
    inp = _get_input(rec)
    if inp.get("auxiliary_findings"):
        return False, "bare variant must have auxiliary_findings null"
    out = _get_output(rec)
    narr = out.get("suspicious_activity_report") or ""
    m = BARE_LEAKAGE.search(narr)
    if m:
        return False, f"bare narrative leaks: '{m.group(0)}'"
    return True, ""


def rule_7_aug_cites(rec: dict) -> tuple[bool, str]:
    md = rec.get("metadata", {})
    if md.get("sar_variant") != "augmented":
        return True, ""
    out = _get_output(rec)
    # v3: AUG-CITES enforces verbatim citation of must-cite phrases from
    # auxiliary findings, but only on POSITIVES. Negatives are also
    # grounded in v3 but the grounding pattern is different — a negative
    # narrative cites bundle *disambiguators* (KYC, expected volume,
    # historical pattern), not the aux findings. Negative narratives are
    # checked by rule_7_neg_disposition_marker instead.
    if out.get("is_suspicious") is False:
        return True, ""
    narr = _normalize_ws(out.get("suspicious_activity_report") or "")
    if not narr:
        return False, "augmented variant must have non-empty narrative"
    inp = _get_input(rec)
    aux = inp.get("auxiliary_findings") or {}
    for kind, key in [("numeric", "answer"), ("citation", "evidence_span"),
                      ("statutory", "reasoning")]:
        for f in (aux.get(kind) or []):
            phrase = _normalize_ws(f.get(key, ""))
            if not phrase:
                continue
            tokens = phrase.split()
            if len(tokens) < 3:
                if phrase not in narr:
                    return False, f"narrative missing {kind}.{key} '{phrase[:80]}'"
                continue
            trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
            if not any(tg in narr for tg in trigrams[:8]):
                return False, f"narrative contains no 3-gram from {kind}.{key}"
    return True, ""


def rule_7_length(rec: dict) -> tuple[bool, str]:
    """Narrative length must be in [NARRATIVE_LENGTH_MIN, NARRATIVE_LENGTH_MAX]
    for BOTH classes (v3 change — matches application.md line 246).
    """
    from pipeline.config import NARRATIVE_LENGTH_MAX, NARRATIVE_LENGTH_MIN
    out = _get_output(rec)
    narr = out.get("suspicious_activity_report") or ""
    n = len(narr)
    if not (NARRATIVE_LENGTH_MIN <= n <= NARRATIVE_LENGTH_MAX):
        return False, f"narrative length {n} outside [{NARRATIVE_LENGTH_MIN}, {NARRATIVE_LENGTH_MAX}]"
    return True, ""


# ============================================================================
# RULE-8-ADV-DETECT
# ============================================================================
ADV_DETECT_PAT = re.compile(
    r"inconsisten[ct]|discrepancy|contradicts?|"
    r"upon\s+(?:re-?)?(?:verification|review)|"
    r"re-?compute[ds]?|re-?deriv(?:e[ds]?|ation)",
    re.IGNORECASE,
)


def rule_8_adv_detect(rec: dict) -> tuple[bool, str]:
    md = rec.get("metadata", {})
    if md.get("sar_variant") != "adversarial_aux":
        return True, ""
    out = _get_output(rec)
    # v3: adversarial-aux records can be either polarity. On positives the
    # model must explicitly detect + re-derive (the original signal). On
    # negatives the model's disposition reasoning IS the detection — it
    # explains why the bundle (including the misleading aux finding) does
    # not warrant a SAR. We check the detection marker on positives only;
    # negatives are covered by rule_7_neg_disposition_marker which already
    # forces the negative narrative to name the surface signal it resolves.
    if out.get("is_suspicious") is False:
        return True, ""
    narr = out.get("suspicious_activity_report") or ""
    if not ADV_DETECT_PAT.search(narr):
        return False, "adversarial narrative missing detection marker (inconsistent / re-derive / etc.)"
    return True, ""


# ============================================================================
# RULE-AUX-NO-GOLD-LEAK (v3) — prompt-time guard for teacher LLM calls
#
# When the SDG teacher (gemma) generates an auxiliary finding during Stage 6
# (inline) or Stage A2 (standalone), the prompt context it receives must
# contain ONLY the bundle evidence (transactions, KYC, policy excerpt,
# statute, fact pattern, question). It must NEVER contain the gold typology,
# gold label, regulatory_frame, or any other rule-layer-derived signal.
#
# This is the Rule C invariant from SDG_STRATEGY_SFT.md §2.1. Violating it
# causes teacher-side label leakage: the aux text perfectly correlates with
# the gold label, and the downstream SAR-judgment model learns to trust aux
# blindly. At inference the live aux skill has no gold and would never
# produce text that correlates — the trained model then under-uses raw
# evidence.
#
# This validator runs BEFORE the LLM call, inspecting the prompt context
# dict. There is no post-hoc test that can reliably detect a leaky aux text
# (the leak is statistical, not syntactic). Catching it at the prompt level
# is the only reliable enforcement point.
# ============================================================================
_GOLD_LEAK_KEYS_AUX_PROMPT = frozenset((
    # Direct labels
    "typology", "typology_gold", "_typology_inferred", "typology_inferred",
    "label", "label_gold", "is_suspicious", "sar_verdict",
    # Rule-layer derivations
    "regulatory_frame", "frame_gold", "_regulatory_frame",
    "decision_target", "_decision_target",
    # Semantic profile (may carry typology / frame)
    "semantic_profile",
    # Reference-patterns / typology hints
    "_reference_patterns", "typology_hint", "frame_hint",
    # Source-data target labels
    "_paired_pattern_ids",   # carries pattern IDs that signal typology
    "_source_typology",
))


def rule_aux_no_gold_leak(prompt_ctx: dict) -> tuple[bool, str]:
    """Verify that an aux-generation prompt context contains no gold-leak keys.

    Called by Stage 6 (`stage_6_aux_findings`) and Stage A2 (`stage_a2_generate`)
    BEFORE the DataDesigner LLM call. Returns False (FATAL) if any forbidden
    key is present.

    The check is structural — it inspects the top-level keys of the prompt
    context dict. Nested gold leakage inside a string value (e.g. typology
    name embedded in a `passage` field) cannot be detected here; the
    generator must avoid that in its own prompt construction (e.g.
    `passage_render.py` must not include the typology in the rendered text).

    Args:
      prompt_ctx: the dict passed to DataDesigner as `seed_columns` / prompt
        variables for an aux-task generation call. Top-level key names only.

    Returns:
      (passed, reason): passed=False ⇒ FATAL leak; abort generation.
    """
    if not isinstance(prompt_ctx, dict):
        return True, ""
    offenders = sorted(set(prompt_ctx.keys()) & _GOLD_LEAK_KEYS_AUX_PROMPT)
    if offenders:
        return False, f"aux prompt context contains gold-leak keys: {offenders}"
    return True, ""


def assert_aux_prompt_clean(prompt_ctx: dict, *, stage_id: str = "?", record_id: str = "?") -> None:
    """Raise immediately if `prompt_ctx` carries any gold-leak key.

    Convenience wrapper for stage code: call this right before the LLM
    invocation. Any leak is a generator bug and must abort the pipeline.
    """
    passed, reason = rule_aux_no_gold_leak(prompt_ctx)
    if not passed:
        raise RuntimeError(
            f"[RULE-AUX-NO-GOLD-LEAK] stage={stage_id} record_id={record_id}: {reason}"
        )


# ============================================================================
# Pydantic schema check (catch-all)
# ============================================================================
def schema_check(rec: dict) -> tuple[bool, str]:
    """Validate the chat-SFT envelope against `ChatSFTRecord`."""
    from pipeline.schemas import ChatSFTRecord  # local import to avoid cycles
    try:
        ChatSFTRecord(**rec)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"pydantic: {exc}"


# ============================================================================
# Rule registries — passed to `verify.run_stage_gate`
# ============================================================================
RULES_STAGE_4 = [
    ("RULE-4-COUNTERPARTY-OVERLAP", rule_4_counterparty_overlap),
]

RULES_STAGE_5 = [
    ("RULE-5-NONE-EMPTY", rule_5_none_empty),
]

RULES_STAGE_6 = [
    ("RULE-6-NUM-SUM", rule_6_num_sum),
    ("RULE-6-CIT-VERBATIM", rule_6_cit_verbatim),
    ("RULE-6-STAT-LABEL", rule_6_stat_label),
]

# RULE-7-ENGLISH (Change 5): the SAR narrative must be in English. Foreign-
# language tokens leak in when the citation evidence_span itself was non-
# English (observed in 0.6% of corpus during smoke audit). The check is a
# lightweight regex against known Spanish/French/Portuguese/German AML
# terms — broad enough to flag the leakage we saw, narrow enough not to
# false-positive on transliterated proper nouns (e.g., "Société Générale"
# survives because we only match content words, not entity names).
_FOREIGN_AML_TOKENS = re.compile(
    r"\b(disfraz|fondos|derivados|actividades|ilegales|"
    r"argent|blanchiment|operations?\s+suspectes|"
    r"lavado|dinero|operaci[óo]n|"
    r"riciclaggio|denaro|"
    r"geldw[äa]sche|verdacht)\b",
    re.IGNORECASE,
)


def rule_7_english(rec: dict) -> tuple[bool, str]:
    """Narrative must be predominantly English (no foreign-language AML tokens)."""
    out = _get_output(rec)
    narr = out.get("suspicious_activity_report") or ""
    m = _FOREIGN_AML_TOKENS.search(narr)
    if m:
        return False, f"narrative contains foreign-language AML token: '{m.group(0)}'"
    return True, ""


RULES_STAGE_7 = [
    # v3 narrative-quality rules (both classes carry grounded narratives)
    ("RULE-7-NARR-NONEMPTY", rule_7_narrative_nonempty),
    ("RULE-7-NEG-DISPOSITION-MARKER", rule_7_neg_disposition_marker),
    ("RULE-7-NO-LEAKY-HINTS", rule_7_no_leaky_hints),
    # Carried forward from earlier versions
    ("RULE-7-OBJECTIVITY", rule_7_objectivity),
    ("RULE-7-BARE-NOLEAK", rule_7_bare_noleak),
    ("RULE-7-AUG-CITES", rule_7_aug_cites),
    ("RULE-7-LENGTH", rule_7_length),
    ("RULE-7-ENGLISH", rule_7_english),
]

RULES_STAGE_8 = [
    ("RULE-8-ADV-DETECT", rule_8_adv_detect),
]

ALL_PER_RECORD_RULES = (
    [("RULE-3-COMPAT", rule_3_compat),
     ("RULE-3-NEAR-MISS-BENIGN", rule_3_near_miss_benign),
     ("RULE-3-NONE-BENIGN", rule_3_none_benign)]
    + RULES_STAGE_4
    + RULES_STAGE_5
    + RULES_STAGE_6
    + RULES_STAGE_7
    + RULES_STAGE_8
    + [("PYDANTIC-CHAT-SFT", schema_check)]
)
