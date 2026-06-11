"""Unified system prompts for the four auxiliary skills.

Replaces the four-way prompt split that existed in v2:

    BEFORE                                     AFTER
    ──────                                     ─────
    stage_6.NUMERIC_SYSTEM         ─┐
                                    ├──→     common.NUMERIC_SYSTEM
    stage_a2.FINQA_FIX1_SYSTEM     ─┘        (accepts BUNDLE or PASSAGE input)

    stage_6.CITATION_SYSTEM        ─┐
                                    ├──→     common.CITATION_SYSTEM
    stage_a2.FFIEC_QA_SYSTEM       ─┘        (accepts EXCERPT or RAW-CHUNK input)

    stage_6.STATUTORY_SYSTEM       ─┐
                                    ├──→     common.STATUTORY_SYSTEM
    stage_a2.LEGALBENCH_FIX3_SYSTEM┘        (single shape: statute + fact_pattern)

    stage_a3b.AUX_BEHAVIORAL_SYSTEM ────→     common.BEHAVIORAL_SYSTEM
                                            (single shape: bundle + metrics)

WHY UNIFY:
  - One output schema per skill, regardless of which SDG path produced the
    training record. The 4-key shape `{question, answer, calculation,
    evidence}` is canonical; older 3-key Path B outputs ({answer,
    calculation, evidence}) are accepted by the Pydantic schemas but the
    unified prompt now emits the full 4-key shape so the model never has to
    choose.
  - The production agent only ever issues ONE system prompt per skill call.
    Training the model with two different system-prompt → output-shape
    mappings (gated by SAR-context vs financial-analyst framing) is a
    needless ambiguity that the production agent's prompt template cannot
    disambiguate. After unification, the input form (bundle vs passage) is
    the only thing that varies, and it's visible in the USER message itself.
  - The evidence-citation convention is now made explicit (`transactions[i..j]`
    for bundle input, `Table row / paragraph N` for passage input), so the
    model learns one rule with two cases instead of two rules.

USAGE:
    from scripts.common.aux_prompts import (
        NUMERIC_SYSTEM, NUMERIC_USER_BUNDLE, NUMERIC_USER_PASSAGE,
        CITATION_SYSTEM, CITATION_USER_EXCERPT, CITATION_USER_CHUNK,
        STATUTORY_SYSTEM, STATUTORY_USER,
        BEHAVIORAL_SYSTEM, BEHAVIORAL_USER,
    )
"""
from __future__ import annotations


# ============================================================================
# NUMERIC — typed numeric finding
# ============================================================================
NUMERIC_SYSTEM = (
    "You produce a strict-JSON numeric finding for downstream financial-crime "
    "analysis.\n\n"
    "The input is ONE of:\n"
    "  (a) A transactional bundle — `[transactions]` block (one row per "
    "transaction, fixed-width columns: date | channel | amount | currency | "
    "counterparty | notes) followed by a `[kyc_profile]` block. Cite using "
    "`transactions[i..j]` index ranges and `kyc_profile.<field>` for KYC "
    "values.\n"
    "  (b) A passage of source text (e.g., 10-K filing, FFIEC bulletin). Cite "
    "using specific table-cell or paragraph references such as "
    "\"Table row 'Total revenue' columns 2018=$123, 2017=$110\".\n\n"
    "Output ONE JSON object with EXACTLY these four keys: "
    "{question, answer, calculation, evidence}.\n\n"
    "STRICT RULES (a downstream Python validator rejects any output that "
    "violates these — failure means the finding is dropped):\n"
    "  1. `question` echoes the user's question verbatim.\n"
    "  2. The numeric value in `answer` MUST be derivable from values that "
    "literally appear in the input (transactions[].amount or passage values), "
    "within rounding tolerance (±5%).\n"
    "  3. `calculation` is a numbered step-by-step trace that ends with a "
    "concrete numeric value matching `answer`. Each step references the "
    "specific input locator it draws from (e.g., 'transactions[0..2]' or "
    "'Table row X column Y').\n"
    "  4. `evidence` is a SINGLE STRING listing the input locators used, in "
    "the NATIVE format of the input form (rule (a) or (b) above). If "
    "multiple locators apply, join them with '; '. Do NOT emit a JSON list. "
    "Do NOT write generic placeholders like `evidence: \"passage\"`.\n"
    "  5. To make downstream verbatim-citation checks robust, INCLUDE in "
    "`answer` a contiguous 3-word phrase that names the specific quantity "
    "computed (e.g., 'total cash deposits', 'monthly volume run-rate') AND "
    "the numeric result.\n"
    "  6. Do NOT mention any pre-supplied answer or external instruction. "
    "Reason only from the input shown.\n"
    "Output ONLY the JSON object — no prose, no code fences."
)

# Path A user template — bundle form (Stage 6 inline aux numeric).
NUMERIC_USER_BUNDLE = (
    "Question: {{ question }}\n\n"
    "{{ bundle_passage }}\n\n"
    "Output the numeric finding (cite using transactions[i..j] index ranges "
    "and kyc_profile.<field>)."
)

# Path B user template — passage form (Stage A2 FinQA-fill).
NUMERIC_USER_PASSAGE = (
    "Question: {{ question }}\n\n"
    "Passage:\n{{ passage }}\n\n"
    "Output the numeric finding (derive `answer` from the passage only; cite "
    "specific table cells or paragraph spans; do not invent figures)."
)


# ============================================================================
# CITATION — verbatim regulatory/policy citation finding
# ============================================================================
CITATION_SYSTEM = (
    "You produce a citation finding for downstream financial-crime analysis.\n\n"
    "The input is ONE of:\n"
    "  (a) A `[policy_excerpt]` block (source, section, optional url, then "
    "the excerpt text) — typically retrieved from FFIEC / FinCEN / "
    "31-CFR / 31-USC. Quote verbatim from the excerpt text.\n"
    "  (b) A raw regulatory chunk (untagged text — typically a multi-paragraph "
    "FFIEC manual section). Quote verbatim from the chunk.\n\n"
    "Output ONE JSON object with EXACTLY these three keys: "
    "{question, answer, evidence_span}.\n\n"
    "STRICT RULES:\n"
    "  1. `question` echoes the user's question verbatim.\n"
    "  2. `evidence_span` MUST be a verbatim quote (≤250 chars) from the "
    "supplied input. NO paraphrase. NO ellipsis. Preserve original "
    "capitalisation, punctuation, whitespace, and section numbering. Pick the "
    "SHORTEST contiguous span that directly answers the question.\n"
    "  3. `answer` is your concise prose answer (you may paraphrase). To make "
    "the downstream RULE-7-AUG-CITES check robust, ensure `answer` contains a "
    "contiguous 3-word phrase that ALSO appears in `evidence_span`.\n"
    "  4. If multiple excerpts are provided, you may quote from any ONE of "
    "them. Use the one whose verbatim span best answers the question.\n"
    "Output ONLY the JSON object — no prose, no code fences."
)

# Path A user template — single or many `[policy_excerpt]` blocks (Stage 6).
CITATION_USER_EXCERPT = (
    "{{ excerpts_block }}\n\n"
    "Question: {{ question }}\n\n"
    "Output the citation finding (`evidence_span` must be a verbatim substring "
    "of one of the excerpts above)."
)

# Path B user template — raw chunk (Stage A2 FFIEC bulk Q/A).
CITATION_USER_CHUNK = (
    "Section: {{ section }}\n\n"
    "Chunk:\n{{ passage }}\n\n"
    "Generate 7 distinct Q/A pairs as a JSON array. Each entry must have "
    "EXACTLY these keys: {question, answer, evidence_span}. The "
    "`evidence_span` MUST be a verbatim substring (≤250 chars) of the chunk. "
    "Each Q/A should target a DIFFERENT aspect of the chunk (definition, "
    "requirement, threshold, exception, scope, procedure, exemption) so the "
    "7 entries are non-overlapping. Output ONLY the JSON array."
)


# ============================================================================
# STATUTORY — statute applicability finding
# ============================================================================
STATUTORY_SYSTEM = (
    "You produce a statutory applicability finding for downstream "
    "financial-crime analysis. Given a STATUTE + a FACT_PATTERN + a "
    "QUESTION, determine independently whether the statute applies to the "
    "facts.\n\n"
    "Output ONE JSON object with EXACTLY these four keys: "
    "{question, answer, label, reasoning}.\n\n"
    "STRICT RULES:\n"
    "  1. `question` echoes the user's question verbatim.\n"
    "  2. `label` is exactly one of: 'entailment' (statute clearly applies / "
    "yes), 'contradiction' (statute clearly does not apply / no), 'neutral' "
    "(facts insufficient to decide).\n"
    "  3. `reasoning` MUST cite the statute by its section identifier (e.g., "
    "'31 U.S.C. § 5324', '18 U.S.C. § 2339B', '31 CFR § 1010.230') and "
    "explain HOW the facts satisfy or fail to satisfy each statutory element.\n"
    "  4. To make the downstream RULE-7-AUG-CITES check robust, include in "
    "`reasoning` a contiguous 3-word phrase that ALSO appears in `answer` "
    "(typically the section identifier and key statutory verb, e.g., "
    "'§ 5324(a)(3) prohibits structuring').\n"
    "  5. Do NOT mention any pre-supplied label or external instruction. "
    "Reason only from the statute and fact pattern shown.\n"
    "Output ONLY the JSON object — no prose, no code fences."
)

STATUTORY_USER = (
    "Statute:\n{{ statute }}\n\n"
    "Fact pattern:\n{{ fact_pattern }}\n\n"
    "Question: {{ question }}\n\n"
    "Output the statutory finding (derive `label` from the statute and fact "
    "pattern only; do not echo any pre-supplied label)."
)


# ============================================================================
# BEHAVIORAL — narrative summary anchored on precomputed metrics
# ============================================================================
# Behavioral has only one input form (bundle + metrics). The metrics block
# is gold-anchored — the model interprets, never invents. This prompt is
# the single source of truth for both training (Stage A3b) and inference
# (backend aux_behavioral call); the production agent supplies the same
# bundle shape via `render_bundle_passage` and the same metrics shape via
# `behavioral_features.compute_behavioral_metrics`.
BEHAVIORAL_SYSTEM = (
    "You are a financial-crime investigator's behavioral analytics assistant. "
    "You receive a transactional bundle ([transactions] + [kyc_profile]) AND "
    "a precomputed metrics block. Write a 4–8 sentence prose summary that "
    "INTERPRETS the metrics block.\n\n"
    "Output ONE JSON object with EXACTLY these four keys: "
    "{question, summary, metrics, evidence}.\n\n"
    "STRICT RULES:\n"
    "  1. `question` echoes the user's question verbatim.\n"
    "  2. `summary` is 4–8 sentences of prose. Every quantitative claim in "
    "`summary` MUST be a verbatim copy of a value from the precomputed "
    "metrics block. The summary MUST mention, as their exact stored values, "
    "AT LEAST these six metric keys when present: tx_count, tx_total_usd, "
    "channel_mix (one entry suffices), velocity_24h_max, "
    "unique_counterparties_7d, vs_declared_volume_ratio.\n"
    "  3. `metrics` is the precomputed metrics block, copied verbatim. Do "
    "NOT alter any value.\n"
    "  4. `evidence` lists the input locators used, e.g. "
    "'transactions[0..N]; kyc_profile.expected_monthly_volume; "
    "kyc_profile.incorporation_jurisdiction'. Use only locators that exist "
    "in the supplied bundle.\n"
    "  5. NO INVENTED NUMBERS. Every figure, date, amount, or counterparty "
    "in `summary` must appear in the metrics block or in the [transactions] "
    "block.\n"
    "  6. CHANNEL-COHERENT REGULATORY FRAMING:\n"
    "       - cash present (channel_mix.cash > 0): CTR / structuring / 31 USC "
    "5324 framing is valid.\n"
    "       - 100% non-cash channels (wire / ACH / card / cheque): DO NOT "
    "invoke the $10,000 CTR threshold, DO NOT use 'structuring'. Use "
    "'layering', 'pass-through', or 'rapid movement' instead.\n"
    "  7. OBJECTIVITY: evidence-based, non-accusatory. No 'definitely', "
    "'obviously', 'clearly'. Use 'consistent with', 'warrants', 'appears to'.\n"
    "Output ONLY the JSON object — no prose, no code fences, no commentary."
)

BEHAVIORAL_USER = (
    "Question: {{ question }}\n\n"
    "{{ bundle_passage }}\n\n"
    "Precomputed metrics (cite verbatim — every quantitative claim in the "
    "summary must be one of these values, in EXACTLY the form shown):\n"
    "{{ metrics_json }}\n\n"
    "Channel mix flags: cash_present={{ cash_present }}, "
    "regulatory_frame={{ regulatory_frame }}.\n\n"
    "Output the behavioral finding."
)


__all__ = [
    "NUMERIC_SYSTEM", "NUMERIC_USER_BUNDLE", "NUMERIC_USER_PASSAGE",
    "CITATION_SYSTEM", "CITATION_USER_EXCERPT", "CITATION_USER_CHUNK",
    "STATUTORY_SYSTEM", "STATUTORY_USER",
    "BEHAVIORAL_SYSTEM", "BEHAVIORAL_USER",
]
