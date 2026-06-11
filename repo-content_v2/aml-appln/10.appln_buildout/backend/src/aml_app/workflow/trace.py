"""CaseTrace \u2014 the per-case audit artifact persisted at the end of every
investigation. Read by the cockpit / replay / eval / disposition views.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("aml_app.workflow.trace")


class CaseTrace(BaseModel):
    """Per-case full audit artifact."""
    model_config = ConfigDict(extra="allow")

    case_id: str
    alert_id: str = ""
    entity_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    wall_clock_ms: float = 0.0

    # Inputs sent to the trained model
    transactions: list[dict] = Field(default_factory=list)
    kyc_profile: Optional[dict] = None
    sanctions_pep_hits: list[dict] = Field(default_factory=list)
    policy_excerpts: list[dict] = Field(default_factory=list)
    sop_excerpts: list[dict] = Field(default_factory=list)
    semantic_profile: Optional[dict] = None
    typology_hypothesis: str = "none"
    activity_descriptor: str = ""
    decision_target: str = "not_suspicious"

    # Planner output
    planner_plan: Optional[str] = None

    # Orchestrator tool-call trace
    orchestrator_calls: list[dict] = Field(default_factory=list)

    # Aux pipeline audit
    aux_responses_raw: dict[str, Any] = Field(default_factory=dict)
    aux_gate_decisions: list[dict] = Field(default_factory=list)
    auxiliary_findings: Optional[dict] = None

    # Final SAR output
    sar_user_message: str = ""
    sar_raw_text: str = ""
    sar_output: Optional[dict] = None
    sar_parse_error: Optional[str] = None
    sar_is_suspicious: Optional[bool] = None
    sar_narrative: str = ""

    # Metadata
    judge_enabled: bool = True
    error: Optional[str] = None


def write_trace(trace: CaseTrace | dict, traces_dir: Path) -> Path:
    """Persist one trace to ./<traces_dir>/<case_id>.json. Returns the path."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(trace, CaseTrace):
        obj = trace.model_dump()
        case_id = trace.case_id
    else:
        obj = dict(trace)
        case_id = obj.get("case_id", "unknown")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", case_id)
    path = traces_dir / f"{safe}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, default=str, indent=2),
                    encoding="utf-8")
    logger.info("Trace written: %s", path)
    return path


def read_trace(case_id: str, traces_dir: Path) -> Optional[dict]:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", case_id)
    path = traces_dir / f"{safe}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
