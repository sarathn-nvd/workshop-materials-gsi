"""Stage 2 — `business_purpose` LLM-generation prompt."""

BUSINESS_PURPOSE_SYSTEM = (
    "You write the `business_purpose` field for a KYC profile in a financial "
    "institution's Customer Due Diligence record. The field is one sentence (15-50 "
    "words) describing what the entity does, declared revenue/volume markers, and "
    "any cash-receipt characteristics relevant to AML risk. Output ONLY the "
    "sentence — no preamble, no quotes."
)

BUSINESS_PURPOSE_USER = (
    "Entity archetype: {{ entity_archetype }}\n"
    "Entity type:      {{ entity_type }}\n"
    "Jurisdiction:     {{ incorporation_jurisdiction }}\n"
    "Expected monthly volume: ${{ expected_monthly_volume }}\n"
    "Typology hypothesis (for narrative consistency): {{ typology }}\n\n"
    "Write the business_purpose sentence."
)
