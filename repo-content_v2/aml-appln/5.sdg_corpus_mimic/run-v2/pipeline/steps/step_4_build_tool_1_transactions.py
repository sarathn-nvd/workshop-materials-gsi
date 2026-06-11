"""Step 4 — Build Tool 1 transactions store (fresh synthesis).

For each entity in Tool 2 (all clean at this step), synthesize benign
transactions from the per-archetype channel-mix / amount-band sampler.
NO content extraction from SFT bundles.

Step 6 will later append the suspicious + near-miss seeded entities' txns.

Outputs:
    data/final/prod_mimic/tool_1_transactions/transactions.parquet
    data/final/prod_mimic/tool_1_transactions/schema.json
    data/final/prod_mimic/tool_1_transactions/stats.json
    data/final/prod_mimic/manifests/step_4_tool_1.json
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

from pipeline.common.typology_patterns import generate_pattern
from pipeline.config import (
    MANIFESTS_DIR,
    TOOL_1_DIR,
    TOOL_2_DIR,
)
from pipeline.schemas import Transaction

logger = logging.getLogger("pipeline.steps.step_4_build_tool_1_transactions")


def _tx_count_for_archetype(archetype: str, rng: random.Random) -> int:
    """Plausible 90-day transaction count per archetype."""
    if archetype.startswith("individual_retiree"):
        return rng.randint(15, 35)
    if archetype.startswith("individual"):
        return rng.randint(20, 60)
    if archetype.startswith("retail_business"):
        return rng.randint(40, 120)
    if archetype.startswith("retail_services"):
        return rng.randint(25, 70)
    if archetype == "import_export_firm":
        return rng.randint(30, 80)
    if archetype == "broker_dealer":
        return rng.randint(50, 150)
    if archetype == "money_services_business":
        return rng.randint(40, 100)
    if archetype.startswith("shell_holding"):
        return rng.randint(5, 20)
    if archetype == "professional_services_gatekeeper":
        return rng.randint(20, 60)
    if archetype == "crypto_exchange_vasp":
        return rng.randint(50, 150)
    if archetype == "ngo_charity":
        return rng.randint(20, 60)
    return rng.randint(20, 60)


def _validate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errs: list[str] = []
    keep = []
    schema_fields = set(Transaction.model_fields.keys())
    for r in df.itertuples(index=False):
        d = {k: getattr(r, k) for k in schema_fields if hasattr(r, k)}
        try:
            Transaction.model_validate(d)
            keep.append(True)
        except ValidationError as e:
            errs.append(f"{getattr(r,'transaction_id','?')}: {str(e)[:140]}")
            keep.append(False)
    return df[keep].reset_index(drop=True), errs


def run(*, seed: int) -> None:
    rng = random.Random(seed)

    entities_path = TOOL_2_DIR / "entities.parquet"
    if not entities_path.exists():
        raise RuntimeError(f"Run step 3 first; {entities_path} missing.")
    entities_df = pd.read_parquet(entities_path)
    logger.info("Tool 2 has %d entities; synthesizing benign transactions for each",
                len(entities_df))

    rows: list[dict] = []
    for r in entities_df.itertuples(index=False):
        eid = r.entity_id
        archetype = getattr(r, "_archetype", "retail_services_non_cash")
        kyc_dict = {
            "entity_id": eid,
            "entity_type": r.entity_type,
            "expected_monthly_volume": float(r.expected_monthly_volume),
            "business_purpose": r.business_purpose,
            "risk_rating": r.risk_rating,
            "incorporation_jurisdiction": r.incorporation_jurisdiction,
        }
        n = _tx_count_for_archetype(archetype, rng)
        # Clean synthesis path
        txs = generate_pattern("none", kyc_dict, rng)
        # Trim or extend to land near the archetype's target count
        if len(txs) > n:
            txs = txs[:n]
        for idx, t in enumerate(txs):
            rows.append({
                "transaction_id": f"TX_{eid}_{idx:04d}",
                "entity_id": eid,
                **t,
                "source_pool": "synth_clean",
                "typology_tag": None,
            })

    df = pd.DataFrame(rows)
    logger.info("Synthesized %d transactions across %d entities",
                len(df), df["entity_id"].nunique())

    surviving, errs = _validate_rows(df)
    if errs:
        logger.warning("Dropped %d rows failing Transaction validation", len(errs))
        for e in errs[:3]:
            logger.warning("  failure: %s", e)

    # Order columns deterministically: schema first, then sidecars
    schema_cols = ["date", "amount", "currency", "counterparty", "channel", "notes"]
    extras = [c for c in surviving.columns if c not in schema_cols]
    surviving = surviving[schema_cols + extras]

    out_path = TOOL_1_DIR / "transactions.parquet"
    surviving.to_parquet(out_path, index=False)
    logger.info("Wrote Tool 1 transactions store: %s (%d rows)", out_path, len(surviving))

    with (TOOL_1_DIR / "schema.json").open("w") as fh:
        json.dump(Transaction.model_json_schema(), fh, indent=2)

    stats = {
        "n_total": int(len(surviving)),
        "n_entities_covered": int(surviving["entity_id"].nunique()),
        "channel_mix": surviving["channel"].value_counts(normalize=True).round(3).to_dict(),
        "currency_top10": surviving["currency"].value_counts().head(10).to_dict(),
        "amount_p50": float(surviving["amount"].median()),
        "amount_p95": float(surviving["amount"].quantile(0.95)),
        "amount_mean": float(surviving["amount"].mean()),
        "date_min": str(surviving["date"].min()),
        "date_max": str(surviving["date"].max()),
        "source_pool_mix": surviving["source_pool"].value_counts().to_dict(),
        "n_with_typology_tag": int(surviving["typology_tag"].notna().sum()),
        "txns_per_entity_p50": float(
            surviving.groupby("entity_id").size().quantile(0.50)
        ),
        "txns_per_entity_p95": float(
            surviving.groupby("entity_id").size().quantile(0.95)
        ),
    }
    with (TOOL_1_DIR / "stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    manifest = {
        "step": 4,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "synthesis_mode": "fresh (no SFT bundle flatten)",
        "n_transactions": stats["n_total"],
        "n_entities_covered": stats["n_entities_covered"],
        "n_validation_failures": len(errs),
        "channel_mix": stats["channel_mix"],
        "date_range": [stats["date_min"], stats["date_max"]],
        "validation": ("MRULE-1-TX-SCHEMA: 100%" if not errs
                       else f"MRULE-1-TX-SCHEMA: {len(errs)} failures"),
        "output_path": str(out_path),
    }
    with (MANIFESTS_DIR / "step_4_tool_1.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
