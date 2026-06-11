"""Rule-based typology classifier — pure Python, no LLM.

Picks ONE typology from {structuring, smurfing, layering, trade_based_ml,
shell_company, human_trafficking, terrorist_financing, elder_exploitation,
none} based on the (transactions, kyc, sanctions_hits) bundle. The
classification is a retrieval prior — wrong-prior cases still produce
reasonable SARs because the trained model has been hardened against
adversarial aux during SFT.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable

from aml_app.common.schemas import (
    KYCProfile,
    SanctionsHit,
    Transaction,
    Typology,
)
from aml_app.common.semantic_profile import filter_confident_hits


_OFFSHORE = {"BVI", "Cayman", "Panama", "Bahamas", "Bermuda", "Marshall Islands"}
_DOMESTIC_OPAQUE = {"Delaware-US", "Nevada-US", "Wyoming-US", "South Dakota-US", "New Mexico-US"}
_HIGH_RISK_CORRIDORS = {"Hong Kong", "UAE", "Singapore", "Cyprus", "Pakistan", "Yemen", "Somalia"}


def _normalize_txs(transactions: Iterable[Transaction] | list[dict]) -> list[Transaction]:
    out: list[Transaction] = []
    for t in transactions or []:
        if isinstance(t, Transaction):
            out.append(t)
        elif isinstance(t, dict):
            d = {k: t[k] for k in ("date", "amount", "currency", "counterparty",
                                   "channel", "notes") if k in t}
            d.setdefault("notes", "")
            out.append(Transaction.model_validate(d))
    return out


def _normalize_kyc(kyc_profile: KYCProfile | dict) -> KYCProfile:
    return (
        kyc_profile if isinstance(kyc_profile, KYCProfile)
        else KYCProfile.model_validate(kyc_profile)
    )


def _normalize_hits(sanctions_pep_hits: Iterable[SanctionsHit] | list[dict]) -> list[SanctionsHit]:
    out: list[SanctionsHit] = []
    for h in sanctions_pep_hits or []:
        if isinstance(h, SanctionsHit):
            out.append(h)
        elif isinstance(h, dict):
            out.append(SanctionsHit.model_validate(h))
    return out


def _channel_mix(txs: list[Transaction]) -> dict[str, float]:
    if not txs:
        return {}
    c = Counter(t.channel for t in txs)
    n = len(txs)
    return {k: c[k] / n for k in c}


def _is_sub_ctr(amount: float) -> bool:
    return 5_000.0 <= amount < 10_000.0


def _count_sub_ctr_cash(txs: list[Transaction]) -> int:
    return sum(1 for t in txs if t.channel == "cash" and _is_sub_ctr(t.amount))


def _date_span_days(txs: list[Transaction]) -> int:
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


def classify_typology(
    transactions: Iterable[Transaction] | list[dict],
    kyc_profile: KYCProfile | dict,
    sanctions_pep_hits: Iterable[SanctionsHit] | list[dict],
    trigger_summary: str | None = None,  # noqa: ARG001 (kept for symmetry)
) -> tuple[Typology, str]:
    """Predict a typology + short activity descriptor."""
    txs = _normalize_txs(transactions)
    kyc = _normalize_kyc(kyc_profile)
    hits = _normalize_hits(sanctions_pep_hits)

    # Rule 1 \u2014 sanctions override. Requires a "confident" hit:
    #   - match_score >= 0.90 AND
    #   - NOT a common-name PEP (see semantic_profile.is_common_name_pep)
    # On the demo corpus, ~95% of stand-alone common-name PEP hits at
    # score=1.0 were false positives (Michael Brown, Patricia Smith on
    # benign wage-earner counterparties); requiring a distinctive name
    # keeps the real OFAC corporate hits while dropping that noise.
    strong_hits = filter_confident_hits(
        [h.model_dump() for h in hits], min_score=0.90,
    )
    if strong_hits:
        # Re-build as SanctionsHit for the descriptor name below.
        strong_hits = [SanctionsHit.model_validate(h) for h in strong_hits]
        bp = kyc.business_purpose.lower()
        if "charity" in bp or "ngo" in bp:
            return ("terrorist_financing",
                    f"Sanctions counterparty hit on NGO entity ({strong_hits[0].name})")
        if kyc.incorporation_jurisdiction in _OFFSHORE | _DOMESTIC_OPAQUE:
            return ("shell_company",
                    f"Sanctions counterparty hit on offshore-incorporated entity ({strong_hits[0].name})")
        return ("layering",
                f"Sanctions counterparty hit on wire flow ({strong_hits[0].name})")

    bp_lower = kyc.business_purpose.lower()
    is_retiree = "retiree" in bp_lower or "retired" in bp_lower or "pension" in bp_lower
    is_ngo = "ngo" in bp_lower or "charity" in bp_lower or "donation" in bp_lower

    # Rule 2a — elder exploitation
    if txs and kyc.entity_type == "individual" and is_retiree:
        wires = [t for t in txs if t.channel == "wire"]
        if len(wires) >= 3:
            cp_counts = Counter(t.counterparty for t in wires)
            top_cp, top_n = cp_counts.most_common(1)[0]
            if top_n >= 3:
                amts = [t.amount for t in wires if t.counterparty == top_cp]
                if amts == sorted(amts) and amts[-1] >= 2 * amts[0]:
                    return ("elder_exploitation",
                            f"Retiree with {len(amts)} escalating wires to ({top_cp})")

    # Rule 2b — NGO fan-in + corridor outflow
    if txs and is_ngo:
        in_wires = [t for t in txs if t.channel in {"wire", "ach"}]
        if len(in_wires) >= 6 and _unique_counterparties(in_wires) >= 5:
            corridor_outs = [
                t for t in txs
                if t.channel == "wire"
                and any(c in t.counterparty for c in _HIGH_RISK_CORRIDORS)
            ]
            if corridor_outs:
                return ("terrorist_financing",
                        f"Fan-in NGO with {len(corridor_outs)} corridor-bound wires")

    if not txs:
        return ("none", "No transactions in investigation window")

    cmix = _channel_mix(txs)

    # Rule 3 — cash-heavy → structuring / smurfing
    if cmix.get("cash", 0.0) >= 0.5:
        sub_ctr_count = _count_sub_ctr_cash(txs)
        if sub_ctr_count >= 3 and _date_span_days(txs) <= 14:
            uniq_cps = _unique_counterparties(
                [t for t in txs if t.channel == "cash" and _is_sub_ctr(t.amount)]
            )
            if uniq_cps >= 3:
                return ("smurfing",
                        f"{sub_ctr_count} sub-$10K cash deposits from {uniq_cps} sender accounts in 14 days")
            return ("structuring",
                    f"{sub_ctr_count} sub-$10K cash deposits across branches in 14 days")
        return ("structuring",
                f"Cash-heavy activity ({sub_ctr_count} sub-$10K of {len(txs)} txns)")

    # Rule 4 \u2014 wire-heavy patterns. Three structural sub-signals; bare
    # "wire-heavy" alone is NOT layering (most clean wage earners have
    # wire/ach-heavy normal activity with 30+ distinct counterparties).
    if cmix.get("wire", 0.0) >= 0.6:
        # 4a) Descending peel-chain pattern \u2014 classic layering.
        amts = sorted([t.amount for t in txs], reverse=True)
        if len(amts) >= 3 and all(amts[i] >= amts[i + 1] for i in range(len(amts) - 1)):
            return ("layering",
                    f"Peel-chain pattern with {len(amts)} descending wire amounts")
        # 4b) Concentrated flow: small number of txns to few counterparties.
        cp_counts = Counter(t.counterparty for t in txs if t.counterparty)
        n_unique = len(cp_counts)
        if cp_counts:
            top_n = cp_counts.most_common(1)[0][1]
            top_share = top_n / len(txs)
        else:
            top_share = 0.0
        if len(txs) <= 8 and (top_share >= 0.5 or n_unique <= 3):
            return ("layering",
                    f"Concentrated wire flow ({len(txs)} txns, top counterparty share {top_share:.0%})")
        # 4c) Round-number large wires (shell / passthrough signature).
        big_wires = [t for t in txs if t.channel in ("wire", "transfer")
                     and t.amount >= 50_000]
        rounded = sum(1 for t in big_wires
                      if t.amount > 0 and t.amount % 1_000 == 0)
        if rounded >= 2:
            return ("layering",
                    f"{rounded} round-number large wires (potential passthrough)")
        # Fall through \u2014 normal high-volume diversified activity.

    # Rule 5 — trade-based ML
    if "import" in bp_lower or "export" in bp_lower or "trading" in bp_lower:
        big_wires = [t for t in txs if t.channel == "wire" and t.amount >= 100_000]
        if big_wires:
            return ("trade_based_ml",
                    f"{len(big_wires)} large outbound wires from import-export entity")

    # Rule 6 — shell company
    if kyc.incorporation_jurisdiction in _OFFSHORE | _DOMESTIC_OPAQUE:
        big_wires = [t for t in txs if t.channel == "wire" and t.amount >= 100_000]
        rounded = sum(1 for t in big_wires if t.amount % 10_000 == 0)
        if rounded >= 2:
            return ("shell_company",
                    f"{rounded} round-number wires through opaque-jurisdiction entity")

    return ("none", f"No suspicious pattern (channel mix {cmix})")
