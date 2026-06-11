"""aux_call — leaf NAT function that wraps one call to the trained
Custom Task NIM for one auxiliary task type.

Each instance is registered under `functions:` with a fixed `task_type`
(one of auxiliary_behavioral / auxiliary_numeric / auxiliary_citation /
auxiliary_statutory). The ReAct specialist sub-agent invokes it after
rendering the appropriate passage; this function:

  1) Posts {system: <SFT-time prompt>, user: <rendered passage + question>}
     to the trained NIM via the LangChain wrapper.
  2) Parses the JSON response (tolerant: extracts the first JSON object
     even if wrapped in markdown fences).
  3) Validates against the corresponding Pydantic *Finding model and
     returns the validated dict.

Behavior is intentionally minimal: render → call → parse → validate.
Any free-text reasoning lives in the specialist sub-agent above, not
inside this leaf.
"""
import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from aml_app.common.schemas import (
    BehavioralFinding, CitationFinding, NumericFinding, StatutoryFinding,
)
from aml_app.skills.prompts import SYSTEM_PROMPT_BY_TASK, _maybe_no_think


logger = logging.getLogger("aml_app.skills.aux_call")


# ---------------------------------------------------------------------------
# I/O shapes
# ---------------------------------------------------------------------------
class AuxCallInput(BaseModel):
    """User-message payload the orchestrator passes in.

    The SFT corpus uses different user-message shapes per task type:
      - behavioral / numeric / citation:
            {"task_type":"<aux_type>", "passage":"...", "question":"..."}
      - statutory:
            {"task_type":"auxiliary_statutory",
             "statute":"...", "fact_pattern":"...", "question":"..."}

    We accept either input convention: for statutory the orchestrator
    supplies `statute` + `fact_pattern`; for the others it supplies
    `passage`. The leaf builds the matching JSON object accordingly.
    """
    passage: str      = Field(default="", description="Pre-rendered passage (behavioral / numeric / citation).")
    statute: str      = Field(default="", description="Statute text (statutory only).")
    fact_pattern: str = Field(default="", description="Fact pattern (statutory only).")
    question: str     = Field(default="", description="Task-specific question.")


_TASK_TYPES = Literal[
    "auxiliary_behavioral", "auxiliary_numeric",
    "auxiliary_citation", "auxiliary_statutory",
]


_MODEL_FOR_TASK: dict[str, type[BaseModel]] = {
    "auxiliary_behavioral": BehavioralFinding,
    "auxiliary_numeric":    NumericFinding,
    "auxiliary_citation":   CitationFinding,
    "auxiliary_statutory":  StatutoryFinding,
}


class AuxCallConfig(FunctionBaseConfig, name="aux_call"):
    """One auxiliary skill call, parameterized by task_type."""
    task_type: _TASK_TYPES = Field(..., description="Which auxiliary task to invoke.")
    llm_name: LLMRef = Field(..., description="LLM binding for the trained model.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=64, le=8192)
    max_thinking_tokens: int = Field(
        default=0,
        description=(
            "Cap reasoning tokens for reasoning-capable Nemotron NIMs "
            "(e.g. nemotron-3-nano-30b-a3b). The trained Custom Task "
            "NIM ignores this. 0 = no cap; the trained custom model "
            "never reasons anyway. Set to e.g. 256 on a reasoning base "
            "model to trim wasted generation time."
        ),
    )


# ---------------------------------------------------------------------------
# JSON extraction (tolerant — handles markdown fences, <think> blocks)
# ---------------------------------------------------------------------------
_JSON_OBJECT_RE   = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CLOSING_THINK_RE = re.compile(r"</think>", re.IGNORECASE)


def filter_think_tokens(text: str) -> str:
    """Strip <think>...</think> blocks and any prose before a stray </think>."""
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    m = _CLOSING_THINK_RE.search(text)
    if m:
        text = text[m.end():]
    return text.strip()


def extract_json(text: str):
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    text = filter_think_tokens(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Registered NAT function
# ---------------------------------------------------------------------------
@register_function(config_type=AuxCallConfig)
async def aux_call(config: AuxCallConfig, builder: Builder):
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = _maybe_no_think(SYSTEM_PROMPT_BY_TASK[config.task_type])
    result_model = _MODEL_FOR_TASK[config.task_type]

    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )

    async def _call(args: AuxCallInput) -> dict:
        if args is None:
            args = AuxCallInput()

        # Build the user-message JSON body in the SFT-time shape per task type.
        if config.task_type == "auxiliary_statutory":
            user_obj = {
                "task_type":    config.task_type,
                "statute":      args.statute,
                "fact_pattern": args.fact_pattern,
                "question":     args.question,
            }
        else:
            user_obj = {
                "task_type": config.task_type,
                "passage":   args.passage,
                "question":  args.question,
            }
        user_text = json.dumps(user_obj)

        # Generation params. /no_think + max_thinking_tokens are
        # honored when the underlying LLM is a reasoning-capable Nemotron.
        gen_kwargs = {
            "temperature": config.temperature,
            "max_tokens":  config.max_tokens,
        }
        # Optional override via env var (used by compare_endpoints.py)
        if os.environ.get("NAT_AML_NO_THINK", "").strip().lower() in ("1", "true", "yes", "y"):
            # In addition to the system-prompt marker (set in prompts.py),
            # also pass `enable_thinking: False` to NIMs that honor it.
            gen_kwargs.setdefault("extra_body", {})
            gen_kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        thinking_budget = int(os.environ.get("NAT_AML_THINKING_BUDGET",
                                              config.max_thinking_tokens) or 0)
        if thinking_budget > 0:
            gen_kwargs.setdefault("extra_body", {})["max_thinking_tokens"] = thinking_budget

        try:
            bound = llm.bind(**gen_kwargs)
        except Exception:
            bound = llm

        messages = [SystemMessage(content=system_prompt),
                    HumanMessage(content=user_text)]
        try:
            resp = await bound.ainvoke(messages)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

        raw_text = getattr(resp, "content", None) or str(resp)
        parsed = extract_json(raw_text)
        if parsed is None:
            logger.warning(
                "aux_call (%s) returned no JSON; raw[:200]=%s",
                config.task_type, (raw_text or "")[:200],
            )
            return {"error": "no_json", "raw": raw_text[:500] if raw_text else ""}

        # Validate against the typed finding model.
        try:
            validated = result_model.model_validate(parsed)
            return validated.model_dump()
        except ValidationError as e:
            logger.warning(
                "aux_call (%s) failed schema validation: %s", config.task_type, e,
            )
            return {"error": "schema_validation_failed",
                    "details": str(e)[:300], "parsed": parsed}

    yield FunctionInfo.from_fn(
        _call,
        description=(
            f"Invoke the trained AML model with task_type={config.task_type}. "
            f"Returns a typed {_MODEL_FOR_TASK[config.task_type].__name__} dict."
        ),
        input_schema=AuxCallInput,
    )
