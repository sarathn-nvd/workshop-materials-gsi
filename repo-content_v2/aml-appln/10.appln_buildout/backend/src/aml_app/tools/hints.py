"""compute_hints — pure-Python tool the orchestrator calls after fetching
raw inputs. Returns the typology hypothesis + semantic profile + (legacy)
decision target.

**v2 (aligned with SFT v3.1)**: the values returned here are used
INTERNALLY by the orchestrator only:

  * `typology_inferred` — routes Tool 4 (policy retrieval) and Tool 5
    (SOP lookup) by typology; picks the per-typology statute + numeric
    question for the auxiliary skill calls.
  * `regulatory_frame` — informs aux skill prompt routing.
  * `decision_target` — **deprecated** for bundle injection; preserved
    in the trace metadata for audit + A/B against the legacy 10-key
    shape. The v3.1-trained model derives its verdict from the bundle
    evidence; the SAR caller does NOT inject this value into the user
    message.
  * `activity_descriptor` — a short human-readable label, persisted
    in the trace metadata.
  * `semantic_profile` — the full profile dict, persisted for trace
    audit + per-cohort scoring (`MRULE-N-CLASSIFIER-COVERAGE`).

The orchestrator (`workflow/investigate_case.py`) is the SINGLE caller
of this leaf. The values it produces MUST NOT appear in the SAR user
message (`skills/sar_caller.py::build_user_message`). The bundle-shape
assertion in that module enforces the contract at build time.
"""
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from aml_app.common.schemas import KYCProfile, SanctionsHit, Transaction
from aml_app.common.semantic_profile import compute_semantic_profile, decision_target_from
from aml_app.common.typology_classifier import classify_typology


logger = logging.getLogger("aml_app.tools.hints")


# ---------------------------------------------------------------------------
# I/O shapes
# ---------------------------------------------------------------------------
class ComputeHintsInput(BaseModel):
    transactions: list[dict] = Field(default_factory=list)
    kyc_profile: dict = Field(default_factory=dict)
    sanctions_pep_hits: list[dict] = Field(default_factory=list)
    trigger_summary: str = ""


class ComputeHintsOutput(BaseModel):
    typology_inferred:   str
    regulatory_frame:    str
    decision_target:     str
    activity_descriptor: str
    semantic_profile:    dict


class ComputeHintsConfig(FunctionBaseConfig, name="compute_hints"):
    description: str = "Deterministic typology + regulatory_frame + decision_target derivation."


# ---------------------------------------------------------------------------
# Registered NAT function
# ---------------------------------------------------------------------------
@register_function(config_type=ComputeHintsConfig)
async def compute_hints_fn(config: ComputeHintsConfig, builder: Builder):
    async def _run(args: ComputeHintsInput) -> dict:
        if args is None:
            args = ComputeHintsInput()

        typology_inferred, activity_descriptor = classify_typology(
            args.transactions,
            args.kyc_profile,
            args.sanctions_pep_hits,
            args.trigger_summary,
        )

        profile = compute_semantic_profile(
            args.transactions,
            args.kyc_profile,
            typology_inferred,
            args.sanctions_pep_hits,
        )

        # Legacy naive decision target — retained for trace audit; not
        # injected into the SAR user message (v3.1 contract).
        decision_target = decision_target_from(typology_inferred)

        return {
            "typology_inferred":   typology_inferred,
            "regulatory_frame":    profile.regulatory_frame,
            "decision_target":     decision_target,
            "activity_descriptor": activity_descriptor,
            "semantic_profile":    profile.model_dump(),
        }

    yield FunctionInfo.from_fn(
        _run,
        description=("Compute typology hypothesis + semantic profile + "
                     "legacy decision target (internal routing only)."),
        input_schema=ComputeHintsInput,
    )
