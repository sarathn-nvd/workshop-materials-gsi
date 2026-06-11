"""Minimal SOP corpus builder — no LLM, no DataDesigner.

Produces 8 typology SOPs (variant 1 only) anchored on FFIEC + FinCEN
keyword-matched passages. Each SOP has 6 sections (Investigation Steps,
Escalation Criteria, Documentation Requirements, Filing Decision, Tools
and Systems, References) populated with templated content keyed by
typology.

This is the workshop-demo substitute for the full DataDesigner-driven SOP
synthesis. The full version (3 variants × LLM-generated content) lives in
pipeline.tools_prep.synthesize_sops and requires the data_designer
package; this minimal builder needs nothing beyond the policy corpus.

Writes 8 markdown files at config.SFT_TOOL_SOPS_DIR:
    structuring_v1.md, smurfing_v1.md, layering_v1.md, ...
"""
from __future__ import annotations

import logging
from pathlib import Path

from pipeline.common.typology_keywords import TYPOLOGY_KEYWORDS
from pipeline.config import SFT_TOOL_SOPS_DIR

logger = logging.getLogger("pipeline.tools_prep.build_sops_minimal")


# Friendly typology names for prose.
_FRIENDLY = {
    "structuring": "sub-CTR cash structuring",
    "smurfing": "smurfing (multi-actor sub-threshold deposits)",
    "layering": "layering (pass-through / cycle / peeling)",
    "trade_based_ml": "trade-based money laundering",
    "shell_company": "shell-company pass-through",
    "human_trafficking": "human-trafficking-related financial flows",
    "terrorist_financing": "terrorist financing",
    "elder_exploitation": "elder financial exploitation",
}


# Per-typology investigation steps. Keep concise; the model has seen
# similar SOP templates in 24 SFT-synthesized SOP markdowns (when those
# are built). Here we provide a single variant per typology that's
# faithful to the SDG_STRATEGY_SFT.md §Stage 5 SOP shape.
_INVESTIGATION_STEPS = {
    "structuring": (
        "1. Pull all cash transactions for the entity in the prior 90 days.\n"
        "2. Identify sub-$10,000 cash deposits made within 14 days across multiple "
        "branches or distinct counterparties.\n"
        "3. Confirm the declared business purpose against observed cash run-rate.\n"
        "4. Aggregate per the BSA aggregation rule (single business day, same person).\n"
        "5. Escalate to SAR review if the pattern persists across ≥3 sub-CTR deposits."
    ),
    "smurfing": (
        "1. Pull all cash and ACH transactions for the entity in the prior 90 days.\n"
        "2. Identify fan-in patterns: ≥3 distinct sender accounts converging on a "
        "single beneficiary with sub-$10,000 amounts.\n"
        "3. Map sender-account graph; flag clusters that share counterparty names "
        "or branches.\n"
        "4. Cross-reference with KYC declared volume.\n"
        "5. Escalate to SAR review if fan-in pattern exceeds ≥4 unique senders."
    ),
    "layering": (
        "1. Pull all wire and ACH transactions for the entity in the prior 90 days.\n"
        "2. Detect cycles (A→B→C→A) and peel chains (descending amounts hop-by-hop) "
        "in the counterparty graph.\n"
        "3. Compute total volume into/out of cycle members; compare to declared volume.\n"
        "4. Identify high-risk jurisdiction nexus (FATF blacklist, sanctioned corridors).\n"
        "5. Escalate to SAR review if cycle exists or peel-chain >3 hops with ≥3% trim."
    ),
    "trade_based_ml": (
        "1. Pull all wire transactions tagged as trade-finance or accompanied by "
        "import/export references.\n"
        "2. Compare invoice values against stated market values; compute over/under-"
        "invoicing ratio.\n"
        "3. Identify phantom-shipments indicators (matching wire pairs to/from same "
        "trading partner without supporting cargo).\n"
        "4. Cross-reference with sanctions and dual-use-goods watchlists.\n"
        "5. Escalate when over/under-invoicing ratio ∉ [0.5, 2.0]."
    ),
    "shell_company": (
        "1. Pull beneficial-ownership records and verify ultimate beneficial owner (UBO).\n"
        "2. Pull all transactions; identify round-number wires in/out without "
        "operating-expense pattern.\n"
        "3. Confirm declared business purpose against absence of payroll, rent, utilities.\n"
        "4. Check incorporation jurisdiction against opaque-jurisdiction list "
        "(BVI, Cayman, Panama, Delaware-US, Nevada-US, Wyoming-US).\n"
        "5. Escalate to SAR review if UBO unverifiable or pass-through pattern clear."
    ),
    "human_trafficking": (
        "1. Pull transactions; identify cash withdrawals at locations near transit "
        "hubs (airports, bus terminals, hotels in known corridors).\n"
        "2. Look for small recurring outbound wires to corridor jurisdictions.\n"
        "3. Cross-reference counterparty list with NGO advisories on trafficking "
        "indicators (FinCEN FIN-2014-A008, FIN-2020-A008).\n"
        "4. Confirm KYC archetype consistency (wage-earner vs observed activity).\n"
        "5. Escalate to SAR review even on weak signals; trafficking SARs use a "
        "lower threshold."
    ),
    "terrorist_financing": (
        "1. Pull transactions; identify fan-in inbound wires (≥6 distinct senders, "
        "small amounts) with subsequent outbound wires to corridor jurisdictions.\n"
        "2. Screen all counterparties against OFAC SDGT list, UN 1267 Sanctions "
        "Committee list, and EU consolidated sanctions list.\n"
        "3. Check entity archetype — NGO/charity profile elevates priority.\n"
        "4. Cross-reference with 314(a) requests.\n"
        "5. Escalate to SAR review immediately on any sanctions hit ≥0.6."
    ),
    "elder_exploitation": (
        "1. Confirm beneficiary age (≥65 per FinCEN Advisory FIN-2022-A002).\n"
        "2. Pull transactions; identify newly-added payees followed by escalating "
        "wire amounts (e.g., $2K → $5K → $15K → $25K).\n"
        "3. Check for sudden changes in account-management contacts (new authorized "
        "user, recent power-of-attorney filing).\n"
        "4. Cross-reference with caregiver / financial-advisor counterparty roles.\n"
        "5. Escalate to SAR review; consider concurrent APS (Adult Protective "
        "Services) referral."
    ),
}


_ESCALATION = {
    "structuring": "Escalate when ≥3 sub-CTR deposits within 14 days OR observed/declared cash ratio ≥3×.",
    "smurfing": "Escalate when ≥4 distinct senders fan into a single beneficiary at sub-CTR amounts.",
    "layering": "Escalate on any detected cycle OR peel chain ≥3 hops with ≥3% trim per hop.",
    "trade_based_ml": "Escalate when over/under-invoicing ratio ∉ [0.5, 2.0] OR phantom-shipment indicator present.",
    "shell_company": "Escalate on unverified UBO OR pass-through pattern OR opaque-jurisdiction incorporation.",
    "human_trafficking": "Escalate on any indicator — trafficking SARs use a lower threshold per FIN-2014-A008.",
    "terrorist_financing": "Escalate on any sanctions hit ≥0.6 OR NGO fan-in to corridor jurisdiction.",
    "elder_exploitation": "Escalate on escalating wires to new payee OR sudden account-management change.",
}


def _build_one(typology: str) -> str:
    friendly = _FRIENDLY[typology]
    kws = ", ".join(TYPOLOGY_KEYWORDS.get(typology, []))
    return f"""# SOP — {friendly}

## Investigation Steps

{_INVESTIGATION_STEPS[typology]}

## Escalation Criteria

{_ESCALATION[typology]}

## Documentation Requirements

For each {typology} investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more {typology} indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with {friendly}: {kws}.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
"""


def build() -> int:
    SFT_TOOL_SOPS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for typology in _INVESTIGATION_STEPS.keys():
        path = SFT_TOOL_SOPS_DIR / f"{typology}_v1.md"
        path.write_text(_build_one(typology), encoding="utf-8")
        n += 1
        logger.info("Wrote %s", path)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = build()
    print(f"Built {n} SOP markdown files in {SFT_TOOL_SOPS_DIR}")
