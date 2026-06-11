"""Step 5 — Wire Tool 3 / 4 / 5 service stubs.

Emits the configuration the future agentic application will use to spin
up its Tool 3 / 4 / 5 services:

  - Verifies the on-disk artifacts each tool will read from exist.
  - Emits a `tools_endpoints.yaml` describing the canonical endpoint
    contract (input/output shape per tool).
  - Emits per-tool `service_config.yaml` (Docker-deploy ready).

No data is generated; no LLM calls. The agentic pipeline itself is NOT
part of this package — these configs are produced for whoever builds it.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import yaml

from pipeline.config import (
    FINAL_DIR,
    MANIFESTS_DIR,
    POOLS,
    SFT_TOOL_POLICY_CHUNKS,
    SFT_TOOL_SOPS_DIR,
    TOOL_1_DIR,
    TOOL_2_DIR,
    TOOL_3_DIR,
    TOOL_3_PORT,
    TOOL_4_DIR,
    TOOL_4_PORT,
    TOOL_5_DIR,
    TOOL_5_PORT,
)

logger = logging.getLogger("pipeline.steps.step_5_wire_services")


def _check(path, label: str) -> tuple[bool, str]:
    ok = path.exists()
    return ok, f"{label}: {'OK' if ok else 'MISSING'} ({path})"


def run(*, seed: int) -> None:  # noqa: ARG001
    checks: list[tuple[bool, str]] = []
    checks.append(_check(TOOL_1_DIR / "transactions.parquet", "Tool 1 transactions parquet"))
    checks.append(_check(TOOL_2_DIR / "entities.parquet", "Tool 2 entities parquet"))
    checks.append(_check(POOLS.ofac_targets, "Tool 3 OFAC targets"))
    checks.append(_check(POOLS.pep_names, "Tool 3 PEP names"))
    checks.append(_check(SFT_TOOL_POLICY_CHUNKS, "Tool 4 policy chunks parquet"))
    checks.append(_check(SFT_TOOL_SOPS_DIR, "Tool 5 SOPs directory"))

    for ok, msg in checks:
        (logger.info if ok else logger.warning)(msg)
    all_ok = all(c[0] for c in checks)

    # Emit tool_3 service config (FastAPI mock REST over OpenSanctions snapshot)
    (TOOL_3_DIR / "service_config.yaml").write_text(yaml.safe_dump({
        "host": "0.0.0.0",
        "port": TOOL_3_PORT,
        "sources": [
            {"path": str(POOLS.ofac_targets), "list_tag": "OFAC"},
            {"path": str(POOLS.pep_names), "list_tag": "OpenSanctions"},
        ],
        "match": {
            "algorithm": "rapidfuzz.token_set_ratio",
            "min_score": 0.55,
            "country_boost": 1.05,
        },
    }, sort_keys=False))

    # Emit tool_4 retrieval_api.yaml
    (TOOL_4_DIR / "retrieval_api.yaml").write_text(yaml.safe_dump({
        "host": "0.0.0.0",
        "port": TOOL_4_PORT,
        "chunks_parquet": str(SFT_TOOL_POLICY_CHUNKS),
        "default_k": 4,
        "text_truncate_chars": 1500,
        "section_truncate_chars": 200,
        "typology_keyword_dict_module": "pipeline.common.typology_keywords",
    }, sort_keys=False))

    # Emit tool_5 service config
    (TOOL_5_DIR / "service_config.yaml").write_text(yaml.safe_dump({
        "host": "0.0.0.0",
        "port": TOOL_5_PORT,
        "sop_dir": str(SFT_TOOL_SOPS_DIR),
        "section_weights": {
            "Investigation Steps": 0.70,
            "Escalation Criteria": 0.15,
            "Documentation Requirements": 0.10,
            "Filing Decision": 0.025,
            "Tools and Systems": 0.025,
            "References": 0.0,
        },
        "text_truncate_chars": 1500,
    }, sort_keys=False))

    # Emit unified tools_endpoints.yaml — canonical contract the future
    # agentic application implements. Function signatures listed here are
    # the canonical Python-level contract; the agentic app can serve them
    # in-process, as FastAPI services, or via Postgres / pgvector — as long
    # as the input/output shapes match these signatures.
    (FINAL_DIR / "tools_endpoints.yaml").write_text(yaml.safe_dump({
        "tool_1": {
            "name": "transactions_db",
            "input": "(entity_id: str, window_start: ISO date, window_end: ISO date)",
            "output": "list[Transaction]  # see pipeline/schemas.py",
        },
        "tool_2": {
            "name": "kyc_store",
            "input": "(entity_id: str)",
            "output": "KYCProfile  # see pipeline/schemas.py",
        },
        "tool_3": {
            "name": "sanctions_pep_screen",
            "input": "(counterparty_name: str, country?: str, min_score: float = 0.55)",
            "output": "list[SanctionsHit]  # see pipeline/schemas.py",
        },
        "tool_4": {
            "name": "policy_rag",
            "input": "(typology: str, k: int = 4, activity_descriptor?: str)",
            "output": "list[PolicyExcerpt]  # see pipeline/schemas.py",
        },
        "tool_5": {
            "name": "sop_service",
            "input": "(typology: str, variant: int = 1, section?: str)",
            "output": "list[SOPExcerpt]  # see pipeline/schemas.py",
        },
        "deployable_endpoints": {
            "tool_3": f"http://<host>:{TOOL_3_PORT}/screen",
            "tool_4": f"http://<host>:{TOOL_4_PORT}/retrieve",
            "tool_5": f"http://<host>:{TOOL_5_PORT}/sop",
        },
    }, sort_keys=False))

    manifest = {
        "step": 5,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "all_artifacts_present": all_ok,
        "artifact_checks": [msg for _, msg in checks],
        "service_configs_written": [
            str(TOOL_3_DIR / "service_config.yaml"),
            str(TOOL_4_DIR / "retrieval_api.yaml"),
            str(TOOL_5_DIR / "service_config.yaml"),
            str(FINAL_DIR / "tools_endpoints.yaml"),
        ],
    }
    with (MANIFESTS_DIR / "step_5_services.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
