"""System + user prompts for auxiliary_behavioral standalone records.

v3 update — re-export the unified BEHAVIORAL system prompt from
`scripts.common.aux_prompts`. Stage A3b is the only place this prompt is
used during training; the production agent uses the same prompt at inference
time via the `auxiliary_behavioral` skill registration. The user template is
re-exported under the old `AUX_BEHAVIORAL_USER_TEMPLATE` name for backward
compatibility but Stage A3b should switch to importing `BEHAVIORAL_USER`
directly going forward.
"""
from __future__ import annotations

from scripts.common.aux_prompts import (
    BEHAVIORAL_SYSTEM as AUX_BEHAVIORAL_SYSTEM,
    BEHAVIORAL_USER as AUX_BEHAVIORAL_USER_TEMPLATE,
)
