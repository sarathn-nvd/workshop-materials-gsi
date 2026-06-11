"""Stage A2 - LLM-fill prompts (3 modes).

v3 update — all four aux skills now share a SINGLE system prompt per skill,
defined in `scripts.common.aux_prompts`. Stage A2 re-uses those system
prompts; only the local USER templates differ to carry Stage A2's input
forms (FinQA SEC passage, FFIEC chunk, LegalBench statute+fact_pattern).

This module remains as a thin re-export shim for backward compatibility.

Anti-leak history:
  - FinQA Fix 1 — gold answer held back; Stage A3 post-validates derived
    answer vs gold within rounding tolerance and drops mismatches.
  - LegalBench Fix 3 — gold label held back; Stage A3 post-validates
    derived label vs gold and drops mismatches.
  - FFIEC Q/A — special case: LLM generates BOTH the question and the
    answer (there is no pre-existing question for FFIEC chunks).
"""

# Re-export the unified system prompts. Stage A2 maps Path B (passage)
# user templates onto them.
from scripts.common.aux_prompts import (
    NUMERIC_SYSTEM as FINQA_FIX1_SYSTEM,
    NUMERIC_USER_PASSAGE as FINQA_FIX1_USER,
    CITATION_SYSTEM as FFIEC_QA_SYSTEM,
    CITATION_USER_CHUNK as FFIEC_QA_USER,
    STATUTORY_SYSTEM as LEGALBENCH_FIX3_SYSTEM,
    STATUTORY_USER as LEGALBENCH_FIX3_USER,
)
