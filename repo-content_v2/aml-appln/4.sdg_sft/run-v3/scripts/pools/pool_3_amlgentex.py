"""Pool_3 - AMLGentex synthetic transactions.

Spatial CSVs + temporal parquet under `synthetic/`:
  - spatial/{accounts, transactions, alert_models, normal_models, degree}.csv
  - temporal/tx_log.parquet

`alert_models.csv` has multiple rows per `modelID` (one row per participating
account). `temporal/tx_log.parquet` carries the actual transaction stream
with a `patternID` column that joins to `modelID`.

Implementation notes (after profiling on-disk data):

- Earlier draft iterated `alert_models` row-by-row, producing 70 unique
  modelIDs duplicated 9x = 617 records. Now we GROUP by `modelID` and emit
  one cluster per unique pattern (~70 clusters).
- Earlier draft tried to filter `spatial/transactions.csv` by `modelID` but
  that file has columns `(id, src, dst, ttype)` and no `modelID` column - so
  every cluster ended up with 0 transactions. The real transaction stream is
  in `temporal/tx_log.parquet` keyed by `patternID`.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from scripts.config import POOLS

logger = logging.getLogger(__name__)


# AMLGentex alert_models.type → canonical typology
GENTEX_TYPE_MAP: dict[str, str] = {
    "fan_out": "structuring",
    "fan_in": "smurfing",
    "cycle": "layering",
    "stack": "layering",
    "bipartite": "layering",
    "gather_scatter": "layering",
    "scatter_gather": "smurfing",
}


_TX_LOG_CACHE: Optional[pd.DataFrame] = None


def _load_tx_log() -> pd.DataFrame:
    """Load `temporal/tx_log.parquet` once; subsequent calls reuse the cache."""
    global _TX_LOG_CACHE
    if _TX_LOG_CACHE is not None:
        return _TX_LOG_CACHE
    parquet_path = POOLS.pool_3_amlgentex / "temporal" / "tx_log.parquet"
    if not parquet_path.exists():
        logger.warning("AMLGentex tx_log.parquet missing: %s", parquet_path)
        _TX_LOG_CACHE = pd.DataFrame()
        return _TX_LOG_CACHE
    _TX_LOG_CACHE = pd.read_parquet(parquet_path)
    return _TX_LOG_CACHE


def _project_transactions(tx_subset: pd.DataFrame, max_tx: int = 30) -> list[dict]:
    """Project a subset of tx_log rows onto the canonical transaction schema."""
    if tx_subset.empty:
        return []
    rows = tx_subset.head(max_tx).to_dict(orient="records")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "transaction_id": f"gentex_tx_{int(r.get('step', i)):06d}_{i:03d}",
            "amount": float(r.get("amount", 0.0)),
            "currency": "USD",                     # AMLGentex doesn't carry currency
            "timestamp": str(r.get("step", "")),   # `step` is the synthetic time index
            "channel": str(r.get("type", "")).lower(),
            "sender_account": str(r.get("nameOrig", "")),
            "receiver_account": str(r.get("nameDest", "")),
            "country_origin": "synthetic",
            "country_destination": "synthetic",
            "pattern_type": str(r.get("modelType", "")),
        })
    return out


def load(max_rows: int | None = None) -> pd.DataFrame:
    """Load Pool_3 → DataFrame of cluster records (one per unique modelID)."""
    base = POOLS.pool_3_amlgentex
    sp = base / "spatial"
    if not (sp / "accounts.csv").exists():
        logger.warning("Pool_3 dir missing: %s", base)
        return pd.DataFrame()

    alert_models_path = sp / "alert_models.csv"
    if not alert_models_path.exists():
        logger.warning("alert_models.csv missing - Pool_3 returns empty.")
        return pd.DataFrame()

    alert_models = pd.read_csv(alert_models_path)
    if "modelID" not in alert_models.columns or "type" not in alert_models.columns:
        logger.warning("alert_models.csv lacks expected columns; Pool_3 returns empty.")
        return pd.DataFrame()

    # Aggregate by modelID - first row per modelID gives the canonical type;
    # also count participating accounts for context.
    grouped = (
        alert_models.groupby("modelID")
        .agg(
            type=("type", "first"),
            n_accounts=("accountID", "count"),
        )
        .reset_index()
    )

    tx_log = _load_tx_log()
    has_tx_log = (not tx_log.empty) and ("patternID" in tx_log.columns)
    if not has_tx_log:
        logger.warning(
            "AMLGentex tx_log.parquet missing or lacks patternID; "
            "clusters will have empty transactions[] (Stage 3 LLM-fills)."
        )

    out: list[dict] = []
    n_remapped_to_layering = 0
    for _, cluster in grouped.iterrows():
        model_id = int(cluster["modelID"])
        model_type = str(cluster["type"]).lower()
        typology = GENTEX_TYPE_MAP.get(model_type, "none")

        if has_tx_log:
            tx_subset = tx_log[tx_log["patternID"] == model_id]
            tx_list = _project_transactions(tx_subset)
        else:
            tx_list = []

        # Option 3: structuring/smurfing are CASH-specific by statutory
        # definition (CTR / 31 USC 5324). If the AMLGentex cluster was
        # mapped to structuring/smurfing but its transactions contain no
        # cash channel, remap to "layering" so the SAR narrative model
        # does not learn to apply CTR rules to TRANSFER-only flows.
        # AMLGentex tx_log `type` is one of TRANSFER / INITALBALANCE / CASH
        # (lowercased into the projected `channel` field).
        if typology in ("structuring", "smurfing") and tx_list:
            has_cash = any("cash" in str(t.get("channel", "")).lower() for t in tx_list)
            if not has_cash:
                typology = "layering"
                n_remapped_to_layering += 1

        n_tx = len(tx_list)
        if n_tx <= 5:
            severity = "light"
        elif n_tx <= 15:
            severity = "medium"
        else:
            severity = "heavy"

        out.append({
            "cluster_id": f"gentex_{model_id:06d}",
            "source": "amlgentex",
            "scenario_native": model_type,
            "typology": typology,
            "label": typology != "none",
            "severity": severity,
            "transactions": tx_list,
            "kyc_seed": {
                "entity_id": f"gentex_cluster_{model_id}",
                "n_participating_accounts": int(cluster["n_accounts"]),
            },
        })
        if max_rows and len(out) >= max_rows:
            break

    df = pd.DataFrame(out)
    if not df.empty:
        n_with_tx = (df["transactions"].apply(len) > 0).sum()
        logger.info(
            "pool_3_amlgentex.load -> %d unique clusters (%d with transactions from tx_log; "
            "remapped %d structuring/smurfing -> layering due to no cash channel; "
            "typology dist: %s)",
            len(df), n_with_tx, n_remapped_to_layering, df["typology"].value_counts().to_dict(),
        )
    else:
        logger.info("pool_3_amlgentex.load -> 0 clusters")
    return df
