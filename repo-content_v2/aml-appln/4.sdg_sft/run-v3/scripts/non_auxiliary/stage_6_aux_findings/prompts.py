"""Stage 6 - prompts for the three auxiliary findings.

v3 update — all four aux skills (numeric, citation, statutory, behavioral)
now share a SINGLE system prompt per skill, defined in
`scripts.common.aux_prompts`. Stage 6 (Path A — inline aux findings on SAR
records) and Stage A2 (Path B — standalone aux training records) import
the same system prompts; only their USER templates differ to carry the
appropriate input form. See `common/aux_prompts.py` for rationale.

This module remains as a thin re-export shim for backward compatibility
and to keep typology-specific lookup tables (NUMERIC_QUESTIONS,
TYPOLOGY_STATUTES) local to Stage 6.
"""

# Re-export the unified system prompts + Stage 6's user templates.
from scripts.common.aux_prompts import (
    NUMERIC_SYSTEM,
    NUMERIC_USER_BUNDLE as NUMERIC_USER,
    CITATION_SYSTEM,
    CITATION_USER_EXCERPT as CITATION_USER,
    STATUTORY_SYSTEM,
    STATUTORY_USER,
)


# Per-typology numeric question template (Stage 6 Step 3)
NUMERIC_QUESTIONS = {
    "structuring": "Sum the cash deposits in the investigation window and compare to the KYC declared monthly cash volume.",
    "smurfing":    "Sum the deposits across involved actors and compare to declared monthly volume.",
    "trade_based_ml": "Compute the over/under-invoicing ratio between the wire amount and a plausible market value of the stated goods.",
    "layering":    "Trace the volume preserved across the cycle / peeling chain.",
    "shell_company": "Compute the ratio of inflow to declared operating activity.",
    "human_trafficking": "Compute the cash withdrawal pattern at transit hubs.",
    "terrorist_financing": "Compute the inbound aggregation pattern from many sources.",
    "elder_exploitation": "Compute the escalation rate of wires to the new payee.",
    "none": "Compute the relevant volume / ratio summary for the activity.",
}

# Typology -> statute lookup (Stage 6 Step 3)
TYPOLOGY_STATUTES = {
    "structuring": ("31 U.S.C. \u00a7 5324", "31 U.S.C. \u00a7 5324(a) - No person shall, for the purpose of evading the reporting requirements of section 5313(a), structure or assist in structuring any transaction with one or more domestic financial institutions."),
    "smurfing":    ("31 U.S.C. \u00a7 5324", "31 U.S.C. \u00a7 5324(a) - same as structuring; multi-actor variant."),
    "shell_company": ("31 CFR \u00a7 1010.230", "31 CFR \u00a7 1010.230 - Beneficial ownership requirements for legal entity customers."),
    "terrorist_financing": ("18 U.S.C. \u00a7 2339B", "18 U.S.C. \u00a7 2339B - Material support to a foreign terrorist organization."),
    "trade_based_ml": ("31 U.S.C. \u00a7 5318(g)", "31 U.S.C. \u00a7 5318(g) - SAR filing obligations."),
    "layering": ("31 U.S.C. \u00a7 5318(g)", "31 U.S.C. \u00a7 5318(g) - SAR filing obligations."),
    "human_trafficking": ("31 U.S.C. \u00a7 5318(g)", "31 U.S.C. \u00a7 5318(g) - SAR filing obligations per FIN-2014-A008."),
    "elder_exploitation": ("31 U.S.C. \u00a7 5318(g)", "31 U.S.C. \u00a7 5318(g) - SAR filing obligations per FIN-2022-A002."),
    "none": (None, None),
}
