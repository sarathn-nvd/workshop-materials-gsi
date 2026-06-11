"""Runtime typology classifier (§2.3 of the strategy doc).

Pure function, no LLM. Takes the outputs of Tools 1/2/3 (the typology-
independent inputs) and predicts ONE typology + a short activity descriptor.
The descriptor is used as a similarity-rerank hint for Tool 4 retrieval.

The classifier IS a retrieval prior — wrong-prior cases still produce
reasonable SARs at the final sar_judgment step because the model has learned
to handle imperfect aux/policy alignments during SFT (adversarial_aux
variant; see SDG_STRATEGY_SFT §Stage 8).

Accuracy target on seeded demo cases: >= 70% top-1 typology match
(MRULE-N-CLASSIFIER-COVERAGE).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Iterable

from pipeline.schemas import KYCProfile, SanctionsHit, Transaction, Typology


# ============================================================================
# Helpers
# ============================================================================
def _channel_mix(txs: list[Transaction]) -> dict[str, float]:
    if not txs:
        return {}
    counts = Counter(t.channel for t in txs)
    n = len(txs)
    return {k: counts[k] / n for k in counts}


def _amounts(txs: list[Transaction]) -> list[float]:
    return [t.amount for t in txs]


def _is_sub_ctr(amount: float) -> bool:
    return 5_000.0 <= amount < 10_000.0


def _count_sub_ctr(txs: list[Transaction]) -> int:
    return sum(1 for t in txs if t.channel == "cash" and _is_sub_ctr(t.amount))


def _date_span_days(txs: list[Transaction]) -> int:
    if len(txs) < 2:
        return 0
    dates = []
    for t in txs:
        try:
            dates.append(datetime.fromisoformat(t.date[:10]))
        except ValueError:
            continue
    if len(dates) < 2:
        return 0
    return (max(dates) - min(dates)).days


def _unique_counterparties(txs: list[Transaction]) -> int:
    return len({t.counterparty for t in txs})


# Offshore / domestic-opaque jurisdictions used in the shell-company rule.
_OFFSHORE = {"BVI", "Cayman", "Panama", "Bahamas", "Bermuda", "Marshall Islands"}
_DOMESTIC_OPAQUE = {"Delaware-US", "Nevada-US", "Wyoming-US", "South Dakota-US", "New Mexico-US"}
_HIGH_RISK_CORRIDORS = {"Hong Kong", "UAE", "Singapore", "Cyprus", "Pakistan", "Yemen", "Somalia"}


# ============================================================================
# Main entry point
# ============================================================================
def classify_typology(
    transactions: Iterable[Transaction] | list[dict],
    kyc_profile: KYCProfile | dict,
    sanctions_pep_hits: Iterable[SanctionsHit] | list[dict],
    trigger_summary: str | None = None,
) -> tuple[Typology, str]:
    """Predict a typology hypothesis from raw tool outputs.

    Returns:
        (typology, activity_descriptor)

    Priority order — first rule to fire wins. The intent matches §2.3 of
    the strategy doc.
    """
    # Normalize inputs to Pydantic models so downstream uses .attr access.
    txs: list[Transaction] = []
    for t in transactions or []:
        if isinstance(t, Transaction):
            txs.append(t)
        elif isinstance(t, dict):
            # Tool 1 rows carry extra sidecar columns; strip them.
            d = {k: t[k] for k in ("date", "amount", "currency", "counterparty",
                                    "channel", "notes") if k in t}
            d.setdefault("notes", "")
            txs.append(Transaction.model_validate(d))

    kyc: KYCProfile = (
        kyc_profile if isinstance(kyc_profile, KYCProfile)
        else KYCProfile.model_validate(kyc_profile)
    )

    hits: list[SanctionsHit] = []
    for h in sanctions_pep_hits or []:
        if isinstance(h, SanctionsHit):
            hits.append(h)
        elif isinstance(h, dict):
            hits.append(SanctionsHit.model_validate(h))

    # ----- Rule 1: Sanctions override (highest priority) ----------------
    strong_hits = [h for h in hits if h.match_score >= 0.6]
    if strong_hits:
        if kyc.business_purpose.lower().find("charity") >= 0 or \
           kyc.business_purpose.lower().find("ngo") >= 0:
            return (
                "terrorist_financing",
                f"Sanctions counterparty hit on NGO-archetype outbound wire ({strong_hits[0].name})",
            )
        if kyc.incorporation_jurisdiction in _OFFSHORE | _DOMESTIC_OPAQUE:
            return (
                "shell_company",
                f"Sanctions counterparty hit on offshore-incorporated entity ({strong_hits[0].name})",
            )
        return (
            "layering",
            f"Sanctions counterparty hit on wire flow ({strong_hits[0].name})",
        )

    # ----- Rule 2: archetype-locked patterns -----------------------------
    bp_lower = kyc.business_purpose.lower()
    is_retiree = "retiree" in bp_lower or "retired" in bp_lower or "pension" in bp_lower
    is_ngo = "ngo" in bp_lower or "charity" in bp_lower or "donation" in bp_lower

    if txs and kyc.entity_type == "individual" and is_retiree:
        # Look for escalating wires (3+ wires to same counterparty, amounts increasing)
        wires = [t for t in txs if t.channel == "wire"]
        if len(wires) >= 3:
            cp_counts = Counter(t.counterparty for t in wires)
            top_cp, top_n = cp_counts.most_common(1)[0]
            if top_n >= 3:
                amts = [t.amount for t in wires if t.counterparty == top_cp]
                if amts == sorted(amts) and amts[-1] >= 2 * amts[0]:
                    return (
                        "elder_exploitation",
                        f"Retiree with {len(amts)} escalating wires to a single payee ({top_cp})",
                    )

    if txs and is_ngo:
        in_wires = [t for t in txs if t.channel in {"wire", "ach"}]
        if len(in_wires) >= 6 and _unique_counterparties(in_wires) >= 5:
            corridor_outs = [
                t for t in txs
                if t.channel == "wire"
                and any(c in t.counterparty for c in _HIGH_RISK_CORRIDORS)
            ]
            if corridor_outs:
                return (
                    "terrorist_financing",
                    f"Fan-in pattern at NGO with {len(corridor_outs)} corridor-bound wires",
                )

    if not txs:
        return ("none", "No transactions in investigation window")

    # ----- Rule 3: cash-heavy → structuring / smurfing ------------------
    cmix = _channel_mix(txs)
    if cmix.get("cash", 0.0) >= 0.5:
        sub_ctr_count = _count_sub_ctr(txs)
        if sub_ctr_count >= 3 and _date_span_days(txs) <= 14:
            uniq_cps = _unique_counterparties(
                [t for t in txs if t.channel == "cash" and _is_sub_ctr(t.amount)]
            )
            if uniq_cps >= 3:
                return (
                    "smurfing",
                    f"{sub_ctr_count} sub-$10K cash deposits from {uniq_cps} sender accounts in 14 days",
                )
            return (
                "structuring",
                f"{sub_ctr_count} sub-$10K cash deposits across branches in 14 days",
            )
        return (
            "structuring",
            f"Cash-heavy activity ({sub_ctr_count} sub-$10K of {len(txs)} txns)",
        )

    # ----- Rule 4: wire-heavy + cycle / peeling -------------------------
    if cmix.get("wire", 0.0) >= 0.6:
        # Cycle detection (loose): A→B and B→A both appear
        wire_pairs = {(t.counterparty, t.notes[:30]) for t in txs if t.channel == "wire"}
        # Heuristic: if many transactions sharing same counterparty AND amounts
        # decrease step by step (peeling chain), call it layering.
        amts = sorted(_amounts(txs), reverse=True)
        if len(amts) >= 3 and all(amts[i] >= amts[i+1] for i in range(len(amts)-1)):
            return (
                "layering",
                f"Peel-chain pattern with {len(amts)} descending wire amounts",
            )
        return (
            "layering",
            f"Wire-heavy activity ({len(txs)} txns; channel mix {cmix})",
        )

    # ----- Rule 5: TBML — large round wires + import/export archetype ---
    if "import" in bp_lower or "export" in bp_lower or "trading" in bp_lower:
        big_wires = [t for t in txs if t.channel == "wire" and t.amount >= 100_000]
        if big_wires:
            return (
                "trade_based_ml",
                f"{len(big_wires)} large outbound wires from import-export entity",
            )

    # ----- Rule 6: shell company — round-number wires + opaque juris ----
    if kyc.incorporation_jurisdiction in _OFFSHORE | _DOMESTIC_OPAQUE:
        big_wires = [t for t in txs if t.channel == "wire" and t.amount >= 100_000]
        rounded = sum(1 for t in big_wires if t.amount % 10_000 == 0)
        if rounded >= 2:
            return (
                "shell_company",
                f"{rounded} round-number wires through opaque-jurisdiction entity",
            )

    # ----- Residual ------------------------------------------------------
    return ("none", f"No suspicious pattern (channel mix {cmix})")
