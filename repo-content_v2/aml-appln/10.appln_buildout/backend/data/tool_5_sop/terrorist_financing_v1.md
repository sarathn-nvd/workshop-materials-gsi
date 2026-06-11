# SOP — terrorist financing

## Investigation Steps

1. Pull transactions; identify fan-in inbound wires (≥6 distinct senders, small amounts) with subsequent outbound wires to corridor jurisdictions.
2. Screen all counterparties against OFAC SDGT list, UN 1267 Sanctions Committee list, and EU consolidated sanctions list.
3. Check entity archetype — NGO/charity profile elevates priority.
4. Cross-reference with 314(a) requests.
5. Escalate to SAR review immediately on any sanctions hit ≥0.6.

## Escalation Criteria

Escalate on any sanctions hit ≥0.6 OR NGO fan-in to corridor jurisdiction.

## Documentation Requirements

For each terrorist_financing investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more terrorist_financing indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with terrorist financing: terrorist financing, terrorism, 2339B, IEEPA, 1705, designated, material support, OFAC, SDN.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
