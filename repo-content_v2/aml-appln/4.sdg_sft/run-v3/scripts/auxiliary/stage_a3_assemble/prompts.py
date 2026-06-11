"""Stage A3 - task-keyed system prompts (one per auxiliary task type).

v3 update — re-exports the unified system prompts from
`scripts.common.aux_prompts`. These are the SAME strings the backend's
`aml_app.skills.prompts.SYSTEM_PROMPT_BY_TASK` ships at inference time.

The SFT chat record carries the unified system prompt VERBATIM (Stage A3
assembles `messages=[{role:"system", content: SYSTEM_BY_TASK[tt]}, ...]`),
so a model trained on this corpus and a model invoked by the backend see
byte-identical system prompts.
"""

from scripts.common.aux_prompts import (
    NUMERIC_SYSTEM as AUXILIARY_NUMERIC_SYSTEM,
    CITATION_SYSTEM as AUXILIARY_CITATION_SYSTEM,
    STATUTORY_SYSTEM as AUXILIARY_STATUTORY_SYSTEM,
)

SYSTEM_BY_TASK = {
    "auxiliary_numeric":   AUXILIARY_NUMERIC_SYSTEM,
    "auxiliary_citation":  AUXILIARY_CITATION_SYSTEM,
    "auxiliary_statutory": AUXILIARY_STATUTORY_SYSTEM,
}
