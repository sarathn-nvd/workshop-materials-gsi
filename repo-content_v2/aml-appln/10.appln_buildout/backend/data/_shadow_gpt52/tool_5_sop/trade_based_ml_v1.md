# SOP — trade-based money laundering

## Investigation Steps

1. Pull all wire transactions tagged as trade-finance or accompanied by import/export references.
2. Compare invoice values against stated market values; compute over/under-invoicing ratio.
3. Identify phantom-shipments indicators (matching wire pairs to/from same trading partner without supporting cargo).
4. Cross-reference with sanctions and dual-use-goods watchlists.
5. Escalate when over/under-invoicing ratio ∉ [0.5, 2.0].

## Escalation Criteria

Escalate when over/under-invoicing ratio ∉ [0.5, 2.0] OR phantom-shipment indicator present.

## Documentation Requirements

For each trade_based_ml investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more trade_based_ml indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with trade-based money laundering: trade-based, TBML, over-invoicing, under-invoicing, phantom shipments, trade misinvoicing, import-export, 5318(g).

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
