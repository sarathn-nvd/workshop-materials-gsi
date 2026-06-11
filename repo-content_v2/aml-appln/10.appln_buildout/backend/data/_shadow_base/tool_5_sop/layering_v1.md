# SOP — layering (pass-through / cycle / peeling)

## Investigation Steps

1. Pull all wire and ACH transactions for the entity in the prior 90 days.
2. Detect cycles (A→B→C→A) and peel chains (descending amounts hop-by-hop) in the counterparty graph.
3. Compute total volume into/out of cycle members; compare to declared volume.
4. Identify high-risk jurisdiction nexus (FATF blacklist, sanctioned corridors).
5. Escalate to SAR review if cycle exists or peel-chain >3 hops with ≥3% trim.

## Escalation Criteria

Escalate on any detected cycle OR peel chain ≥3 hops with ≥3% trim per hop.

## Documentation Requirements

For each layering investigation, the case file must include:
- Transaction list with dates, amounts, channels, and counterparties.
- KYC profile snapshot at investigation time.
- Sanctions / PEP screening results for all counterparties.
- Behavioral metrics computation showing observed-vs-declared volume.
- Cited regulatory sections from FFIEC / FATF / FinCEN advisories.
- Investigator notes with timestamped reasoning.

## Filing Decision

A SAR is warranted when:
- The pattern matches one or more layering indicators above.
- Evidence is documented and reproducible from the Tool 1 / Tool 2 / Tool 3 outputs.
- No benign explanation reconciles the observed activity with the declared business purpose.

When in doubt, file. Per 31 U.S.C. § 5318(g), the bank has safe-harbor protection for good-faith filings.

## Tools and Systems

- Transaction DB query: filter by entity_id + 90-day window.
- KYC store lookup: confirm declared business purpose and risk_rating.
- Sanctions API: screen each counterparty (RapidFuzz threshold ≥ 0.55).
- Policy RAG: retrieve cited sections for the typology before drafting narrative.

## References

Keywords commonly associated with layering (pass-through / cycle / peeling): layering, peeling, peel chain, nested, intermediary accounts, FATF Recommendation 10, FATF Recommendation 11, 5318(g), cycle.

Primary regulatory anchors: FFIEC BSA/AML Examination Manual, FATF Recommendations,
FinCEN Advisories. Specific section identifiers should be drawn from `policy_excerpts[]`
returned by Tool 4 at investigation time — do not cite identifiers absent from the
retrieved excerpts.
