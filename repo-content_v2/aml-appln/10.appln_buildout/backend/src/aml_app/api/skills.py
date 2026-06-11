"""Skill playground routes \u2014 thin wrappers around the trained-model leaf
calls so the frontend can drive each skill in isolation.

These reuse the registered `aux_*_call` and `sar_judgment_caller` functions
via the NAT builder, so they share the exact same prompts, parsing, and
schema validation as the workflow.
"""
import os
from collections.abc import AsyncGenerator
from typing import Optional

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.function import FunctionBaseConfig


# ---------------------------------------------------------------------------
# Playgrounds for each auxiliary skill
# ---------------------------------------------------------------------------
class SkillPassageInput(BaseModel):
    """Generic passage+question playground input.

    Statutory accepts `passage` for backwards compat (we map it into a
    pre-rendered statute+fact_pattern), or use SkillStatutoryInput for
    the canonical SFT shape with separate fields."""
    passage: str
    question: str = ""


class SkillStatutoryInput(BaseModel):
    """Canonical SFT shape for the statutory playground."""
    statute: str
    fact_pattern: str
    question: str = ""


class SkillBehavioralConfig(FunctionBaseConfig, name="skill_behavioral"):
    aux_function: FunctionRef = Field(
        ..., description="Name of the registered aux_call function for behavioral.")


@register_function(config_type=SkillBehavioralConfig)
async def skill_behavioral(config: SkillBehavioralConfig, builder: Builder):
    fn = await builder.get_function(config.aux_function)
    async def _run(args: SkillPassageInput) -> dict:
        return await fn.ainvoke({"passage": args.passage, "question": args.question})
    yield FunctionInfo.from_fn(_run, description="Behavioral skill playground.",
                                input_schema=SkillPassageInput)


class SkillNumericConfig(FunctionBaseConfig, name="skill_numeric"):
    aux_function: FunctionRef = Field(...)


@register_function(config_type=SkillNumericConfig)
async def skill_numeric(config: SkillNumericConfig, builder: Builder):
    fn = await builder.get_function(config.aux_function)
    async def _run(args: SkillPassageInput) -> dict:
        return await fn.ainvoke({"passage": args.passage, "question": args.question})
    yield FunctionInfo.from_fn(_run, description="Numeric skill playground.",
                                input_schema=SkillPassageInput)


class SkillCitationConfig(FunctionBaseConfig, name="skill_citation"):
    aux_function: FunctionRef = Field(...)


@register_function(config_type=SkillCitationConfig)
async def skill_citation(config: SkillCitationConfig, builder: Builder):
    fn = await builder.get_function(config.aux_function)
    async def _run(args: SkillPassageInput) -> dict:
        return await fn.ainvoke({"passage": args.passage, "question": args.question})
    yield FunctionInfo.from_fn(_run, description="Citation skill playground.",
                                input_schema=SkillPassageInput)


class SkillStatutoryConfig(FunctionBaseConfig, name="skill_statutory"):
    aux_function: FunctionRef = Field(...)


@register_function(config_type=SkillStatutoryConfig)
async def skill_statutory(config: SkillStatutoryConfig, builder: Builder):
    fn = await builder.get_function(config.aux_function)
    async def _run(args: SkillStatutoryInput) -> dict:
        return await fn.ainvoke({"statute": args.statute,
                                 "fact_pattern": args.fact_pattern,
                                 "question": args.question})
    yield FunctionInfo.from_fn(_run, description="Statutory skill playground.",
                                input_schema=SkillStatutoryInput)


# ---------------------------------------------------------------------------
# Standalone SAR call (full hand-built bundle)
# ---------------------------------------------------------------------------
class SkillSarInput(BaseModel):
    """7-key SAR bundle, matching the SFT contract.

    No `regulatory_frame`, `typology_inferred`, or `decision_target`
    fields — those v1-era hint fields have been retired. The trained
    model derives the verdict from the evidence keys alone.
    """
    transactions: list[dict] = Field(default_factory=list)
    kyc_profile: dict
    sanctions_pep_hits: list[dict] = Field(default_factory=list)
    policy_excerpts: list[dict] = Field(default_factory=list)
    sop_excerpts: list[dict] = Field(default_factory=list)
    auxiliary_findings: Optional[dict] = None


class SkillSarConfig(FunctionBaseConfig, name="skill_sar"):
    sar_function: FunctionRef = Field(
        ..., description="Name of the registered sar_judgment_caller function.")


@register_function(config_type=SkillSarConfig)
async def skill_sar(config: SkillSarConfig, builder: Builder):
    fn = await builder.get_function(config.sar_function)
    async def _run(args: SkillSarInput) -> dict:
        return await fn.ainvoke(args.model_dump())
    yield FunctionInfo.from_fn(_run, description="SAR skill playground (full bundle).",
                                input_schema=SkillSarInput)
