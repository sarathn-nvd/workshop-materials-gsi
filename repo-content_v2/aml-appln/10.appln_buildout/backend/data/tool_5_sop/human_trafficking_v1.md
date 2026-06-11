# SOP — human-trafficking-related financial flows

## Investigation Steps

1. Pull transactions; identify cash withdrawals at locations near transit hubs (airports, bus terminals, hotels in known corridors).
2. Look for small recurring outbound wires to corridor jurisdictions.
3. Cross-reference counterparty list with NGO advisories on trafficking indicators (FinCEN FIN-2014-A008, FIN-2020-A008).
4. Confirm KYC archetype consistency (wage-earner vs observed activity).
5. Escalate to SAR review even on weak signals; trafficking SARs use a lower threshold.

## Escalation Criteria

Escalate on any indicator — trafficking SARs use a lower threshold per FIN-2014-A008.

## Documentation Requirements

For each human_trafficking investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more human_trafficking indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with human-trafficking-related financial flows: human trafficking, trafficking, FIN-2014-A008, 5318(g), labor exploitation, victims.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
