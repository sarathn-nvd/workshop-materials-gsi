"""Typology keyword dictionary — single source of truth for typology→keyword mapping.

Auto-extracted by pipeline.bootstrap from
4.sdg_sft/scripts/tools_prep/build_policy_chunks.py at bootstrap time.
Re-run bootstrap to refresh.
"""
from __future__ import annotations

TYPOLOGY_KEYWORDS: dict[str, list[str]] = {
    "structuring": [
        "structuring", "structure", "sub-CTR", "sub-threshold", "currency transaction report",
        "$10,000", "5324", "FIN-2014-A005", "split", "deposits below",
    ],
    "smurfing": [
        "smurfing", "smurf", "multiple actors", "scatter", "fan-in",
        "sub-threshold deposits", "5324",
    ],
    "layering": [
        "layering", "peeling", "peel chain", "nested", "intermediary accounts",
        "FATF Recommendation 10", "FATF Recommendation 11", "5318(g)", "cycle",
    ],
    "trade_based_ml": [
        "trade-based", "TBML", "over-invoicing", "under-invoicing", "phantom shipments",
        "trade misinvoicing", "import-export", "5318(g)",
    ],
    "shell_company": [
        "shell company", "shell entity", "beneficial owner", "beneficial ownership",
        "1010.230", "BVI", "Cayman", "Panama", "nominee director",
    ],
    "human_trafficking": [
        "human trafficking", "trafficking", "FIN-2014-A008", "5318(g)",
        "labor exploitation", "victims",
    ],
    "terrorist_financing": [
        "terrorist financing", "terrorism", "2339B", "IEEPA", "1705",
        "designated", "material support", "OFAC", "SDN",
    ],
    "elder_exploitation": [
        "elder financial exploitation", "elder abuse", "FIN-2022-A002",
        "senior", "retiree", "fiduciary", "5318(g)",
    ],
}
