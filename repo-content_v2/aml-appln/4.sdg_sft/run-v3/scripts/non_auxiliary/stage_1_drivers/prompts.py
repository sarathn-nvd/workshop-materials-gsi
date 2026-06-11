"""Stage 1 has no LLM call — placeholder for symmetry with other stages.

The Pool_4 (SARSum) typology classifier IS an LLM call but is invoked from
the Stage 1 source-extraction projector when records on the SARSum path are
encountered. The system prompt for that classifier lives below.
"""

SARSUM_TYPOLOGY_CLASSIFY_SYSTEM = (
    "You classify free-text Suspicious Activity Report (SAR) narratives into one "
    "of nine canonical typologies: structuring, smurfing, layering, "
    "trade_based_ml, shell_company, human_trafficking, terrorist_financing, "
    "elder_exploitation, none. "
    "Output ONLY the typology label as a single lowercase token."
)

SARSUM_TYPOLOGY_CLASSIFY_USER = (
    "SAR narrative:\n{{ notes }}\n\n"
    "Classify into one of: structuring, smurfing, layering, trade_based_ml, "
    "shell_company, human_trafficking, terrorist_financing, elder_exploitation, none.\n"
    "Output the label only (one token)."
)
