# SOP — shell-company pass-through

## Investigation Steps

1. Pull beneficial-ownership records and verify ultimate beneficial owner (UBO).
2. Pull all transactions; identify round-number wires in/out without operating-expense pattern.
3. Confirm declared business purpose against absence of payroll, rent, utilities.
4. Check incorporation jurisdiction against opaque-jurisdiction list (BVI, Cayman, Panama, Delaware-US, Nevada-US, Wyoming-US).
5. Escalate to SAR review if UBO unverifiable or pass-through pattern clear.

## Escalation Criteria

Escalate on unverified UBO OR pass-through pattern OR opaque-jurisdiction incorporation.

## Documentation Requirements

For each shell_company investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more shell_company indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with shell-company pass-through: shell company, shell entity, beneficial owner, beneficial ownership, 1010.230, BVI, Cayman, Panama, nominee director.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
