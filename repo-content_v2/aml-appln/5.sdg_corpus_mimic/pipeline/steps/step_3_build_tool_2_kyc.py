"""Step 3 — Build Tool 2 KYC store (fresh synthesis).

Generates ~2K KYC entities by SAMPLING from the SFT-derived distribution
tables (per-archetype jurisdiction weights, volume bands, risk-rating
distribution) — but the specific entity content is NEWLY SYNTHESIZED so
it is held out from the SFT training set.

Step 6 will ADD ~75 more entities (suspicious + near-miss) on top of these.

No SFT bundle content is extracted into this store.

Outputs:
    data/final/prod_mimic/tool_2_kyc/entities.parquet  (~2000 clean entities)
    data/final/prod_mimic/tool_2_kyc/schema.json
    data/final/prod_mimic/tool_2_kyc/stats.json
    data/final/prod_mimic/manifests/step_3_tool_2_kyc.json
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
from pydantic import ValidationError

from pipeline.common.kyc_samplers import ARCHETYPES, sample_kyc_for_archetype
from pipeline.config import (
    ENTITY_PREFIXES,
    MANIFESTS_DIR,
    TARGET_ENTITIES,
    TOOL_2_DIR,
)
from pipeline.schemas import KYCProfile

logger = logging.getLogger("pipeline.steps.step_3_build_tool_2_kyc")


# Archetype-mix target for the clean population (mirrors a plausible bank
# customer mix; same archetype catalog as SFT Stage 1, weighted by typical
# observability).
_ARCHETYPE_MIX = {
    "individual_wage_earner":              0.30,
    "individual_small_business_owner":     0.12,
    "individual_retiree_65+":              0.08,
    "retail_business_jewelry":             0.02,
    "retail_business_restaurant":          0.04,
    "retail_business_laundromat":          0.02,
    "retail_business_convenience":         0.05,
    "retail_services_non_cash":            0.14,
    "import_export_firm":                  0.04,
    "broker_dealer":                       0.02,
    "money_services_business":             0.03,
    "shell_holding_offshore":              0.01,
    "shell_holding_domestic":              0.02,
    "professional_services_gatekeeper":    0.03,
    "crypto_exchange_vasp":                0.01,
    "ngo_charity":                         0.07,
}


def _new_entity_id() -> str:
    return ENTITY_PREFIXES["synthetic"] + uuid.uuid4().hex[:8]


def _sample_archetype(rng: random.Random) -> str:
    pop, w = zip(*_ARCHETYPE_MIX.items())
    return rng.choices(pop, weights=w, k=1)[0]


def _generate_clean_entity(rng: random.Random) -> dict:
    archetype = _sample_archetype(rng)
    kyc = sample_kyc_for_archetype(archetype, rng, label_positive=False)
    kyc["entity_id"] = _new_entity_id()
    kyc["source_pool"] = "synth_clean"
    kyc["_archetype"] = archetype
    return kyc


def _validate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errs: list[str] = []
    keep = []
    for r in df.itertuples(index=False):
        d = {k: getattr(r, k) for k in KYCProfile.model_fields.keys()
             if hasattr(r, k)}
        try:
            KYCProfile.model_validate(d)
            keep.append(True)
        except ValidationError as e:
            errs.append(f"{getattr(r, 'entity_id', '?')}: {str(e)[:140]}")
            keep.append(False)
    return df[keep].reset_index(drop=True), errs


def run(*, seed: int) -> None:
    rng = random.Random(seed)
    n_clean = TARGET_ENTITIES

    logger.info("Synthesizing %d clean KYC entities (fresh; no SFT content)", n_clean)
    rows = [_generate_clean_entity(rng) for _ in range(n_clean)]
    df = pd.DataFrame(rows)

    surviving, errs = _validate_rows(df)
    if errs:
        logger.warning("Dropped %d rows failing KYCProfile validation", len(errs))
        for e in errs[:5]:
            logger.warning("  failure: %s", e)

    # Reorder columns: KYCProfile fields first, then sidecars
    kyc_cols = list(KYCProfile.model_fields.keys())
    sidecar = [c for c in surviving.columns if c not in kyc_cols]
    surviving = surviving[kyc_cols + sidecar]

    out_path = TOOL_2_DIR / "entities.parquet"
    surviving.to_parquet(out_path, index=False)
    logger.info("Wrote Tool 2 KYC store: %s (%d rows)", out_path, len(surviving))

    with (TOOL_2_DIR / "schema.json").open("w") as fh:
        json.dump(KYCProfile.model_json_schema(), fh, indent=2)

    stats = {
        "n_total": int(len(surviving)),
        "n_clean": int(len(surviving)),
        "archetype_distribution": Counter(surviving["_archetype"]).most_common(),
        "entity_type_distribution": surviving["entity_type"].value_counts().to_dict(),
        "risk_rating_distribution": surviving["risk_rating"].value_counts().to_dict(),
        "jurisdiction_top10": surviving["incorporation_jurisdiction"]
            .value_counts().head(10).to_dict(),
        "expected_monthly_volume_p50": float(surviving["expected_monthly_volume"].median()),
        "expected_monthly_volume_p95": float(
            surviving["expected_monthly_volume"].quantile(0.95)
        ),
    }
    with (TOOL_2_DIR / "stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    manifest = {
        "step": 3,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "synthesis_mode": "fresh (no SFT content extraction)",
        "n_total": stats["n_total"],
        "n_validation_failures": len(errs),
        "validation": ("MRULE-1-KYC-SCHEMA: 100%" if not errs
                       else f"MRULE-1-KYC-SCHEMA: {len(errs)} failures"),
        "output_path": str(out_path),
    }
    with (MANIFESTS_DIR / "step_3_tool_2_kyc.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
