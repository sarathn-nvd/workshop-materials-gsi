#!/usr/bin/env python3
"""Seed the miniature SDG input corpus under ``data/input/``.

This is the M6 analogue of M5's ``seed_raw_corpus.py``. It writes a small,
self-contained input set that *mirrors the shape* of the production SDG inputs
in ``gsi-training/2.data_processing/data/{sft,transactional}`` — but hand-authored
so the workshop runs without the 1 TB+ production corpus or any external path.

Layout (mirrors the two production source families):

    data/input/
      sft/                      <- derived from 2.data_processing/data/sft
        case_drivers.jsonl        one row per SAR case to generate (Stage 1 drivers)
        reference_narratives.jsonl SARSum-style narratives (Stage 7 style refs)
        policy_excerpts.jsonl     FFIEC / FinCEN / statute excerpts (grounding + aux citation)
        aux_passages.jsonl        numeric / citation / statutory aux source passages
      transactional/            <- derived from 2.data_processing/data/transactional
        sanctions_watchlist.jsonl OFAC/PEP-style names (Stage 4 screening)

Idempotent: safe to re-run. Committed to the repo so the notebook runs offline.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "input"

# Target number of SAR cases to drive Track A. Each case fans out into KYC +
# transactions + SAR-judgment LLM calls against the local Nemotron Nano teacher
# NIM (~3 calls/case, no rate limit). 48 is sized for the 90-min hands-on;
# raise it for a larger corpus, lower it for a quicker smoke run.
N_CASES = 48

# Nine canonical typologies (production set), plus the benign "none".
TYPOLOGIES = [
    "structuring", "smurfing", "layering", "trade_based_ml", "shell_company",
    "human_trafficking", "terrorist_financing", "elder_exploitation", "none",
]

# typology -> regulatory frame the narrative should derive (NOT leaked to the model)
FRAME = {
    "structuring": "ctr", "smurfing": "ctr", "layering": "layering_passthrough",
    "trade_based_ml": "tbml", "shell_company": "shell", "human_trafficking": "trafficking",
    "terrorist_financing": "trafficking", "elder_exploitation": "elder", "none": "benign",
}

# typology -> a plausible entity archetype + entity type
ARCHETYPE = {
    "structuring": ("retail_business_laundromat", "business"),
    "smurfing": ("cash_intensive_convenience", "business"),
    "layering": ("import_export_firm", "business"),
    "trade_based_ml": ("import_export_firm", "business"),
    "shell_company": ("shell_holding_offshore", "business"),
    "human_trafficking": ("individual_wage_earner", "individual"),
    "terrorist_financing": ("money_services_business", "business"),
    "elder_exploitation": ("individual_retiree", "individual"),
    "none": ("individual_wage_earner", "individual"),
}


# ---------------------------------------------------------------------------- #
# SARSum-style reference narratives (modeled on 2.data_processing/data/sft/sarsum.jsonl)
# Format mirrors the real "Pattern Identified: ... Decision: ..." structure.
# Used by Stage 7 as STYLE templates only (never copied verbatim).
# ---------------------------------------------------------------------------- #
REFERENCE_NARRATIVES = [
    {"typology": "structuring", "is_suspicious": True,
     "notes": "Pattern Identified: Multiple cash deposits just below the $10,000 CTR threshold across several branches in one week. Decision: Suspicious. The deposit sequence is consistent with structuring to evade currency transaction reporting under 31 U.S.C. § 5324; the activity does not match the customer's declared retail volume."},
    {"typology": "layering", "is_suspicious": True,
     "notes": "Pattern Identified: Rapid pass-through of incoming wires to multiple offshore counterparties within 48 hours, leaving negligible residual balance. Decision: Suspicious. The movement pattern is consistent with layering and obscures the origin of funds; volume preserved across the chain indicates pass-through rather than genuine commercial settlement."},
    {"typology": "elder_exploitation", "is_suspicious": True,
     "notes": "Pattern Identified: An elderly customer initiated several large wires to a newly-added payee inconsistent with prior history. Decision: Suspicious. The activity is consistent with elder financial exploitation per FinCEN Advisory FIN-2022-A002; the new payee and abrupt change in behavior warrant escalation."},
    {"typology": "none", "is_suspicious": False,
     "notes": "Pattern Identified: Regular monthly invoices matching the customer's consulting business and declared volume. Decision: Not concerning. Activity is consistent with the declared business purpose and expected monthly volume; no red flags identified and no SAR is warranted."},
    {"typology": "none", "is_suspicious": False,
     "notes": "Pattern Identified: Sanctions screening produced a common-name match on a counterparty with no corroborating identifiers. Decision: Not suspicious. The match appears to be a common-name false positive with no DOB or jurisdiction corroboration; no other red flag in the transaction set, so no SAR is warranted."},
    {"typology": "trade_based_ml", "is_suspicious": True,
     "notes": "Pattern Identified: Wire payments for imported goods materially exceed a plausible market value for the stated commodity. Decision: Suspicious. The over-invoicing pattern is consistent with trade-based money laundering per FATF guidance; third-party payments further obscure the value transfer."},
]


# ---------------------------------------------------------------------------- #
# Policy / SOP excerpts (modeled on FFIEC manual + FinCEN advisories + statutes;
# 2.data_processing/data/sft/ffiec_manual.jsonl). Used for grounding (Stage 5)
# and as the source text for the auxiliary CITATION task.
# ---------------------------------------------------------------------------- #
POLICY_EXCERPTS = [
    {"source": "FinCEN", "section": "31 U.S.C. 5324",
     "text": "No person shall, for the purpose of evading the currency transaction reporting requirements, cause or attempt to cause a domestic financial institution to fail to file a report required under section 5313(a), including by breaking up a single transaction above the reporting threshold into two or more transactions — a practice commonly referred to as structuring."},
    {"source": "FFIEC", "section": "Structuring red flags",
     "text": "Common indicators of structuring include multiple cash deposits or withdrawals conducted within a short period in amounts just below the $10,000 currency transaction report threshold, deposits made at multiple branches on the same day, and cash activity inconsistent with the customer's stated business or occupation."},
    {"source": "FATF", "section": "Trade-Based Money Laundering",
     "text": "Trade-based money laundering exploits the complexity of international trade to disguise the movement of illicit value. Typical techniques include over- and under-invoicing of goods and services, multiple invoicing of the same goods, and misrepresentation of the quality or quantity of traded commodities."},
    {"source": "FinCEN", "section": "FIN-2022-A002 Elder Financial Exploitation",
     "text": "Financial institutions should be alert to behavioral indicators of elder financial exploitation, including a sudden increase in wire transfers or withdrawals inconsistent with the customer's historical pattern, the addition of new and previously unknown payees, and transactions that appear to be directed by a third party."},
    {"source": "OFAC", "section": "31 CFR 501",
     "text": "U.S. persons must block property and reject transactions involving individuals or entities on the Specially Designated Nationals list. Screening should consider name variations and the strength of identifying information; a name match alone, absent corroborating identifiers, may constitute a false positive subject to disposition."},
    {"source": "FFIEC", "section": "Layering and pass-through",
     "text": "Layering involves moving illicit funds through a series of transactions to distance them from their source. Indicators include rapid movement of funds through accounts with little or no apparent business purpose, pass-through activity in which incoming funds are quickly transferred out, and the use of multiple jurisdictions."},
]


# ---------------------------------------------------------------------------- #
# Auxiliary source passages — three task types (numeric / citation / statutory),
# modeled on finqa (numeric table text), ffiec_manual (citation), and
# legalbench__sara (statute + fact pattern). Used by Track B.
# ---------------------------------------------------------------------------- #
AUX_PASSAGES = [
    # numeric — finqa-style financial table rendered as text
    {"task_type": "auxiliary_numeric",
     "question": "What is the total of the three cash deposits in the investigation window?",
     "passage": "[transactions] 2026-03-02 | cash | 9,400 USD | branch_A\n2026-03-04 | cash | 9,650 USD | branch_B\n2026-03-05 | cash | 9,300 USD | branch_A\n[kyc_profile] declared monthly cash volume: 12,000 USD"},
    {"task_type": "auxiliary_numeric",
     "question": "By what ratio does the total wire volume exceed the declared monthly volume?",
     "passage": "[transactions] 2026-04-10 | wire | 48,000 USD | OFFS_12\n2026-04-11 | wire | 45,500 USD | OFFS_19\n[kyc_profile] expected_monthly_volume: 35,000 USD"},
    # citation — FFIEC chunk; the model must quote a verbatim span
    {"task_type": "auxiliary_citation",
     "question": "What deposit pattern does the FFIEC manual list as a structuring indicator?",
     "passage": "Common indicators of structuring include multiple cash deposits or withdrawals conducted within a short period in amounts just below the $10,000 currency transaction report threshold, deposits made at multiple branches on the same day, and cash activity inconsistent with the customer's stated business or occupation."},
    {"task_type": "auxiliary_citation",
     "question": "According to FATF, what are typical trade-based money laundering techniques?",
     "passage": "Trade-based money laundering exploits the complexity of international trade to disguise the movement of illicit value. Typical techniques include over- and under-invoicing of goods and services, multiple invoicing of the same goods, and misrepresentation of the quality or quantity of traded commodities."},
    # statutory — statute + fact pattern (legalbench sara style)
    {"task_type": "auxiliary_statutory",
     "statute": "31 U.S.C. § 5324(a)(3): No person shall, for the purpose of evading the reporting requirements of section 5313(a), structure or assist in structuring, or attempt to structure or assist in structuring, any transaction with one or more domestic financial institutions.",
     "fact_pattern": "Over four days, a customer made five cash deposits of $9,200-$9,700 each at three different branches of the same bank. The customer's declared business is a small flower shop with expected monthly cash receipts of $6,000.",
     "question": "Does 31 U.S.C. § 5324(a)(3) apply to this fact pattern?"},
    {"task_type": "auxiliary_statutory",
     "statute": "31 U.S.C. § 5324(a)(3): No person shall, for the purpose of evading the reporting requirements of section 5313(a), structure ... any transaction with one or more domestic financial institutions.",
     "fact_pattern": "A licensed check-cashing business deposits $9,500 in cash each business day. Its money services business registration and declared daily cash receipts of roughly $9,000-$10,000 are on file and consistent with the activity.",
     "question": "Does 31 U.S.C. § 5324(a)(3) apply to this fact pattern?"},
]


# ---------------------------------------------------------------------------- #
# Sanctions / PEP watchlist (modeled on 2.data_processing/data/transactional/
# ofac_enforcement/targets.simple.csv). Mix of real-style sanctioned entities and
# common-name noise so Stage 4 produces both true hits and false positives.
# ---------------------------------------------------------------------------- #
SANCTIONS_WATCHLIST = [
    {"name": "Al-Barakaat Group of Companies", "list": "OFAC", "kind": "true_target"},
    {"name": "Tornado Cash", "list": "OFAC", "kind": "true_target"},
    {"name": "Hydra Market", "list": "OFAC", "kind": "true_target"},
    {"name": "Michael Brown", "list": "OpenSanctions", "kind": "common_name_noise"},
    {"name": "John Smith", "list": "OpenSanctions", "kind": "common_name_noise"},
    {"name": "Maria Garcia", "list": "OpenSanctions", "kind": "common_name_noise"},
]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.relative_to(ROOT)}  ({len(rows)} records)")


def build_case_drivers() -> list[dict]:
    """Stage-1 drivers: one row per SAR case, balanced ~45% positive / 55% negative,
    with a typology mix and a near-miss fraction among negatives (mirrors the
    production distribution targets, scaled down).
    """
    random.seed(7)
    rows: list[dict] = []
    # eligible (non-none) typologies drive positives; "none" drives clean negatives
    eligible = [t for t in TYPOLOGIES if t != "none"]
    n_pos = round(N_CASES * 0.45)
    for i in range(N_CASES):
        positive = i < n_pos
        if positive:
            typ = eligible[i % len(eligible)]
            surface = "active_pattern"
        else:
            # ~30% of negatives are "near miss" (look risky, resolve benign)
            if (i - n_pos) % 3 == 0:
                typ = eligible[i % len(eligible)]   # near-miss: risky typology, benign label
                surface = "near_miss"
            else:
                typ = "none"
                surface = "clean"
        arche, etype = ARCHETYPE[typ]
        rows.append({
            "case_id": f"CASE-{i:04d}",
            "entity_id": f"SYN_{i:04d}",
            "entity_archetype": arche,
            "entity_type": etype,
            "typology": typ,
            "regulatory_frame": FRAME[typ],     # internal only — must NOT leak into user msg
            "is_suspicious_target": positive,   # gold label — used to check the LLM verdict
            "severity": "heavy" if positive else "light",
            "surface_pattern": surface,
            "incorporation_jurisdiction": "US" if random.random() < 0.8 else "intl",
            "expected_monthly_volume": random.choice([6000, 12000, 35000, 80000, 150000]),
        })
    random.shuffle(rows)
    return rows


def main() -> None:
    print(f"Seeding miniature SDG input under {INPUT}")
    _write(INPUT / "sft" / "case_drivers.jsonl", build_case_drivers())
    _write(INPUT / "sft" / "reference_narratives.jsonl", REFERENCE_NARRATIVES)
    _write(INPUT / "sft" / "policy_excerpts.jsonl", POLICY_EXCERPTS)
    _write(INPUT / "sft" / "aux_passages.jsonl", AUX_PASSAGES)
    _write(INPUT / "transactional" / "sanctions_watchlist.jsonl", SANCTIONS_WATCHLIST)
    print("Done.")


if __name__ == "__main__":
    main()
