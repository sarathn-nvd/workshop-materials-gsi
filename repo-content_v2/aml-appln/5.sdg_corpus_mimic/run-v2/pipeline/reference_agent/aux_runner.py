"""Runtime aux runner — invokes the 4 auxiliary_* tasks in parallel.

Each aux call uses a task-specific passage built from the gathered tool
outputs (transactions, kyc, policy_excerpts, statute_text). The runner
returns the raw LLM responses; gating (input availability + schema parse
+ LLM-as-Judge) happens in aux_gate.py.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass

from pipeline.reference_agent.llm_client import chat
from pipeline.reference_agent.prompts import (
    AUX_BEHAVIORAL_SYSTEM_PROMPT,
    AUX_CITATION_SYSTEM_PROMPT,
    AUX_NUMERIC_SYSTEM_PROMPT,
    AUX_STATUTORY_SYSTEM_PROMPT,
)
from pipeline.schemas import (
    KYCProfile,
    PolicyExcerpt,
    SanctionsHit,
    SOPExcerpt,
    Transaction,
    Typology,
)

logger = logging.getLogger("pipeline.reference_agent.aux_runner")


# Per-typology statute lookup (matches SDG_STRATEGY_SFT §Stage 6 statutory table).
STATUTE_BY_TYPOLOGY: dict[str, tuple[str, str]] = {
    "structuring":        ("5324",      "31 U.S.C. § 5324 — Structuring transactions to evade reporting requirements"),
    "smurfing":           ("5324",      "31 U.S.C. § 5324 — Structuring transactions to evade reporting requirements"),
    "shell_company":      ("1010.230",  "31 CFR § 1010.230 — Beneficial-ownership requirements for legal entity customers"),
    "terrorist_financing":("2339B",     "18 U.S.C. § 2339B — Providing material support or resources to designated foreign terrorist organizations"),
    "trade_based_ml":     ("5318(g)",   "31 U.S.C. § 5318(g) — Suspicious activity reporting requirements"),
    "layering":           ("5318(g)",   "31 U.S.C. § 5318(g) — Suspicious activity reporting requirements"),
    "human_trafficking":  ("5318(g)",   "31 U.S.C. § 5318(g) — Suspicious activity reporting requirements"),
    "elder_exploitation": ("5318(g)",   "31 U.S.C. § 5318(g) — Suspicious activity reporting requirements"),
}


# Per-typology numeric question.
NUMERIC_QUESTION_BY_TYPOLOGY: dict[str, str] = {
    "structuring": "Sum the cash deposits in the investigation window and compare to the KYC declared monthly cash volume.",
    "smurfing":    "Sum the inbound transactions from distinct sender accounts and compare to KYC declared monthly volume.",
    "layering":    "Compute the total volume passing through any detected cycle; compare to KYC declared monthly volume.",
    "trade_based_ml": "Compute the over- or under-invoicing ratio (sum of trade-finance wires / declared trade volume).",
    "shell_company": "Sum round-number wires through the entity; compare to KYC declared monthly volume.",
    "human_trafficking": "Sum small cash withdrawals; report locations and a per-week rate.",
    "terrorist_financing": "Sum inbound wires from distinct senders; report the share routed outbound to corridor jurisdictions.",
    "elder_exploitation": "Compute the escalation rate of wires to any newly-added payee.",
}


@dataclass
class AuxResponse:
    task: str           # one of behavioral / numeric / citation / statutory
    requested: bool     # whether we actually issued the LLM call
    raw_text: str | None
    passage: str
    question: str
    latency_ms: float
    err: str | None = None


# ============================================================================
# Passage assembly
# ============================================================================
def _render_tx_table(txs: list[Transaction]) -> str:
    if not txs:
        return "(no transactions)"
    lines = ["date,amount,currency,counterparty,channel,notes"]
    for t in txs[:50]:   # cap
        lines.append(
            f"{t.date},{t.amount:.2f},{t.currency},\"{t.counterparty[:60]}\","
            f"{t.channel},\"{t.notes[:40]}\""
        )
    return "\n".join(lines)


def _render_kyc(kyc: KYCProfile) -> str:
    return (
        f"entity_id: {kyc.entity_id}\n"
        f"entity_type: {kyc.entity_type}\n"
        f"expected_monthly_volume: {kyc.expected_monthly_volume}\n"
        f"business_purpose: {kyc.business_purpose}\n"
        f"risk_rating: {kyc.risk_rating}\n"
        f"incorporation_jurisdiction: {kyc.incorporation_jurisdiction}\n"
    )


def _behavioral_passage(txs: list[Transaction], kyc: KYCProfile) -> str:
    return (
        "## KYC profile\n"
        + _render_kyc(kyc)
        + "\n## Transactions\n"
        + _render_tx_table(txs)
    )


def _numeric_passage(txs: list[Transaction], kyc: KYCProfile) -> str:
    return _behavioral_passage(txs, kyc)


def _citation_passage(excerpt: PolicyExcerpt) -> str:
    return (
        f"## Policy excerpt\n"
        f"source: {excerpt.source}\n"
        f"section: {excerpt.section}\n"
        f"url: {excerpt.url or ''}\n\n"
        f"{excerpt.text}\n"
    )


def _statutory_passage(typology: Typology, txs: list[Transaction], kyc: KYCProfile) -> str:
    if typology not in STATUTE_BY_TYPOLOGY:
        return ""
    _, statute_text = STATUTE_BY_TYPOLOGY[typology]
    fact_pattern = (
        f"Entity {kyc.entity_id} ({kyc.entity_type}, {kyc.incorporation_jurisdiction}, "
        f"declared monthly volume ${kyc.expected_monthly_volume:,.0f}, "
        f"business purpose: {kyc.business_purpose[:120]}). "
        f"Observed {len(txs)} transactions in the investigation window."
    )
    return (
        f"## Statute\n{statute_text}\n\n"
        f"## Fact pattern\n{fact_pattern}\n\n"
        f"## Transactions (first 20)\n"
        + _render_tx_table(txs[:20])
    )


# ============================================================================
# Aux call helpers
# ============================================================================
def _call_one(
    task: str,
    system: str,
    passage: str,
    question: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> AuxResponse:
    """Issue one LLM call. Returns AuxResponse with raw_text or err."""
    user = f"{passage}\n\n## Question\n{question}"
    try:
        resp = chat(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return AuxResponse(
            task=task, requested=True, raw_text=resp.text,
            passage=passage, question=question,
            latency_ms=resp.latency_ms,
        )
    except Exception as e:
        logger.warning("aux call %s failed: %s", task, e)
        return AuxResponse(
            task=task, requested=True, raw_text=None,
            passage=passage, question=question,
            latency_ms=0.0, err=str(e),
        )


def run_all_aux_calls(
    *,
    typology: Typology,
    transactions: list[Transaction],
    kyc_profile: KYCProfile,
    sanctions_hits: list[SanctionsHit],            # noqa: ARG001 (carried for symmetry)
    policy_excerpts: list[PolicyExcerpt],
    sop_excerpts: list[SOPExcerpt],                # noqa: ARG001
    max_workers: int = 4,
) -> list[AuxResponse]:
    """Invoke all 4 aux tasks in parallel. Returns raw responses (gating later).

    Per the strategy doc §2.4, we always invoke all 4 — input-availability
    guards in aux_gate decide whether to USE each response.
    """
    tasks = []

    # 1. behavioral
    tasks.append(("behavioral", _behavioral_passage(transactions, kyc_profile),
                  "Provide a behavioral summary with the metrics described in the schema."))

    # 2. numeric
    q = NUMERIC_QUESTION_BY_TYPOLOGY.get(typology,
        "Summarize transaction volumes and compare to KYC declared monthly volume.")
    tasks.append(("numeric", _numeric_passage(transactions, kyc_profile), q))

    # 3. citation
    if policy_excerpts:
        passage = _citation_passage(policy_excerpts[0])
        section = policy_excerpts[0].section[:60] or "the passage"
        q = f"What does {section} say about the activity described?"
    else:
        passage = "(no policy excerpts available)"
        q = "No policy excerpts were retrieved; respond accordingly."
    tasks.append(("citation", passage, q))

    # 4. statutory
    if typology in STATUTE_BY_TYPOLOGY:
        statute_id = STATUTE_BY_TYPOLOGY[typology][0]
        passage = _statutory_passage(typology, transactions, kyc_profile)
        q = f"Does the conduct described fall within {statute_id}?"
    else:
        passage = ""
        q = "No statute mapped for this typology; respond accordingly."
    tasks.append(("statutory", passage, q))

    # Map to system prompts
    systems = {
        "behavioral": AUX_BEHAVIORAL_SYSTEM_PROMPT,
        "numeric":    AUX_NUMERIC_SYSTEM_PROMPT,
        "citation":   AUX_CITATION_SYSTEM_PROMPT,
        "statutory":  AUX_STATUTORY_SYSTEM_PROMPT,
    }

    # Fire all 4 in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_call_one, name, systems[name], passage, question)
            for name, passage, question in tasks
        ]
        results = [f.result() for f in futures]
    return results
