"""Step 2 — SFT entity inventory + clean-pool inventory.

Reads the SFT non-aux corpus (24,487 records), groups by `entity_id`, and
writes a Parquet inventory that downstream steps query.

  outputs
    data/processed/sft_entity_inventory.parquet
        one row per unique entity_id, columns:
          entity_id, source, n_bundles_seen, typology_seen (list[str]),
          label_seen (list[bool]), kyc_profile_json, tx_count_total,
          tx_date_min, tx_date_max, channels_present (list[str]),
          currencies_present (list[str])

    data/processed/clean_pool_inventory.parquet
        one row per EFC entities_master entity NOT already in the SFT
        inventory, with `kyc_bucket ∈ {standard, elevated}` (i.e. clean).
        Columns: entity_id (EFC_ prefixed), entity_type, country,
                 kyc_bucket, source_pool ("efc_clean").

    data/final/prod_mimic/manifests/step_2_inventory.json
        counts + summary.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from pipeline.config import (
    ENTITY_PREFIXES,
    MANIFESTS_DIR,
    POOLS,
    PROCESSED_DIR,
    SFT_CORPUS_NONAUX_JSONL,
)

logger = logging.getLogger("pipeline.steps.step_2_inventory")


# ============================================================================
# SFT inventory
# ============================================================================
def _build_sft_inventory() -> pd.DataFrame:
    """One row per unique entity_id across the SFT non-aux corpus."""
    if not SFT_CORPUS_NONAUX_JSONL.exists():
        raise FileNotFoundError(
            f"SFT corpus not found at {SFT_CORPUS_NONAUX_JSONL}. "
            f"Confirm 4.sdg_sft has been generated."
        )

    by_entity: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "n_bundles_seen": 0,
        "sources": set(),
        "typologies": set(),
        "labels": set(),
        "kyc_profile": None,            # keep first seen
        "tx_count": 0,
        "tx_dates": [],                  # min/max derived at end
        "channels": set(),
        "currencies": set(),
    })

    n_lines = 0
    with SFT_CORPUS_NONAUX_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            rec = json.loads(line)
            user = json.loads(rec["messages"][1]["content"])
            assistant = json.loads(rec["messages"][2]["content"])
            md = rec.get("metadata", {}) or {}

            kyc = user.get("kyc_profile") or {}
            entity_id = kyc.get("entity_id")
            if not entity_id:
                continue

            row = by_entity[entity_id]
            row["n_bundles_seen"] += 1
            row["sources"].add(md.get("source", "?"))
            row["typologies"].add(md.get("typology", "none"))
            row["labels"].add(bool(assistant.get("is_suspicious", False)))
            if row["kyc_profile"] is None:
                row["kyc_profile"] = kyc

            for t in user.get("transactions", []) or []:
                row["tx_count"] += 1
                d = t.get("date")
                if d:
                    row["tx_dates"].append(d)
                ch = t.get("channel")
                if ch:
                    row["channels"].add(ch)
                cur = t.get("currency")
                if cur:
                    row["currencies"].add(cur)

    logger.info("SFT corpus: read %d records, %d unique entities", n_lines, len(by_entity))

    rows: list[dict[str, Any]] = []
    for entity_id, agg in by_entity.items():
        dates = sorted(agg["tx_dates"])
        rows.append({
            "entity_id": entity_id,
            "source": next(iter(agg["sources"])) if len(agg["sources"]) == 1
                      else "+".join(sorted(agg["sources"])),
            "n_bundles_seen": agg["n_bundles_seen"],
            "typology_seen": sorted(agg["typologies"]),
            "label_seen": sorted(agg["labels"]),
            "kyc_profile_json": json.dumps(agg["kyc_profile"]),
            "tx_count_total": agg["tx_count"],
            "tx_date_min": dates[0] if dates else None,
            "tx_date_max": dates[-1] if dates else None,
            "channels_present": sorted(agg["channels"]),
            "currencies_present": sorted(agg["currencies"]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Clean-pool inventory
# ============================================================================
def _build_clean_pool_inventory(sft_entity_ids: set[str]) -> pd.DataFrame:
    """Enumerate EFC entities not present in the SFT inventory.

    We use the EFC pool as the primary clean-fill source: it has ~70K
    entities with kyc_bucket ∈ {standard, elevated, enhanced, prohibited}.
    We keep only standard/elevated rows for clean filler.
    """
    if not POOLS.efc_entities_master.exists():
        logger.warning(
            "EFC entities_master.csv not found at %s — clean pool inventory will be empty",
            POOLS.efc_entities_master,
        )
        return pd.DataFrame(columns=[
            "entity_id", "entity_type", "country", "kyc_bucket", "source_pool",
        ])

    df = pd.read_csv(POOLS.efc_entities_master, low_memory=False)
    # Normalize column names — EFC pool has `entity_id, entity_name, entity_type,
    # country, kyc_bucket, ...`. Accept lowercase variants too.
    df.columns = [c.strip().lower() for c in df.columns]
    expected = {"entity_id", "entity_type", "country", "kyc_bucket"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(
            f"EFC entities_master.csv missing required columns: {missing}. "
            f"Got columns: {list(df.columns)[:20]}"
        )

    # Namespace
    pref = ENTITY_PREFIXES["efc"]
    df["entity_id"] = pref + df["entity_id"].astype(str)
    df = df[~df["entity_id"].isin(sft_entity_ids)]
    df = df[df["kyc_bucket"].isin(["standard", "elevated"])]
    df["source_pool"] = "efc_clean"

    return df[["entity_id", "entity_type", "country", "kyc_bucket", "source_pool"]]


# ============================================================================
# Step driver
# ============================================================================
def run(*, seed: int) -> None:  # noqa: ARG001
    sft_df = _build_sft_inventory()
    sft_path = PROCESSED_DIR / "sft_entity_inventory.parquet"
    sft_df.to_parquet(sft_path, index=False)
    logger.info("Wrote SFT inventory: %s (%d rows)", sft_path, len(sft_df))

    sft_ids = set(sft_df["entity_id"].tolist())
    clean_df = _build_clean_pool_inventory(sft_ids)
    clean_path = PROCESSED_DIR / "clean_pool_inventory.parquet"
    clean_df.to_parquet(clean_path, index=False)
    logger.info("Wrote clean pool inventory: %s (%d rows)", clean_path, len(clean_df))

    summary = {
        "step": 2,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "sft_entity_inventory": {
            "path": str(sft_path),
            "n_rows": int(len(sft_df)),
            "n_bundles_total": int(sft_df["n_bundles_seen"].sum()),
            "n_transactions_total": int(sft_df["tx_count_total"].sum()),
        },
        "clean_pool_inventory": {
            "path": str(clean_path),
            "n_rows": int(len(clean_df)),
        },
    }
    manifest_path = MANIFESTS_DIR / "step_2_inventory.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote step manifest: %s", manifest_path)
