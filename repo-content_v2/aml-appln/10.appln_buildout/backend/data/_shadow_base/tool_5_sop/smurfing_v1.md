# SOP — smurfing (multi-actor sub-threshold deposits)

## Investigation Steps

1. Pull all cash and ACH transactions for the entity in the prior 90 days.
2. Identify fan-in patterns: ≥3 distinct sender accounts converging on a single beneficiary with sub-$10,000 amounts.
3. Map sender-account graph; flag clusters that share counterparty names or branches.
4. Cross-reference with KYC declared volume.
5. Escalate to SAR review if fan-in pattern exceeds ≥4 unique senders.

## Escalation Criteria

Escalate when ≥4 distinct senders fan into a single beneficiary at sub-CTR amounts.

## Documentation Requirements

For each smurfing investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more smurfing indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with smurfing (multi-actor sub-threshold deposits): smurfing, smurf, multiple actors, scatter, fan-in, sub-threshold deposits, 5324.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
