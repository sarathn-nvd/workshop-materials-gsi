# SOP — sub-CTR cash structuring

## Investigation Steps

1. Pull all cash transactions for the entity in the prior 90 days.
2. Identify sub-$10,000 cash deposits made within 14 days across multiple branches or distinct counterparties.
3. Confirm the declared business purpose against observed cash run-rate.
4. Aggregate per the BSA aggregation rule (single business day, same person).
5. Escalate to SAR review if the pattern persists across ≥3 sub-CTR deposits.

## Escalation Criteria

Escalate when ≥3 sub-CTR deposits within 14 days OR observed/declared cash ratio ≥3×.

## Documentation Requirements

For each structuring investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more structuring indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with sub-CTR cash structuring: structuring, structure, sub-CTR, sub-threshold, currency transaction report, $10,000, 5324, FIN-2014-A005, split, deposits below.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
