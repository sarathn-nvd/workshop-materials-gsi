"""Pool_1 — Enterprise FC.

Two equivalent ingestion paths exist:
  (a) The pre-bundled SFT JSONL at `data/sft/enterprise_financial_crime.jsonl`
      where each row already carries the joined case + transactions + emails +
      sar_reports payload.
  (b) The raw transactional dirs at `data/transactional/enterprise_financial_crime/full_dataset/`
      which we'd need to join ourselves.

We use (a) by default — same data, easier ingest. The raw path is available
for richer joins if (a) is found insufficient.
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

import pandas as pd

from scripts.config import POOLS

logger = logging.getLogger(__name__)


# Pool_1 native scenario → canonical typology mapping (per strategy doc Stage 1 §2A)
EFC_SCENARIO_MAP: dict[str, str] = {
    "structuring": "structuring",
    "structuring_smurfing": "structuring",
    "smurfing": "smurfing",
    "layering": "layering",
    "cross_border_layering": "layering",
    "rapid_movement": "layering",
    "round_tripping": "layering",
    "crypto_conversion": "layering",
    "shell_company": "shell_company",
    "shell_company_transfer": "shell_company",
    "trade_based_ml": "trade_based_ml",
    "trade_based_laundering": "trade_based_ml",
    "tbml": "trade_based_ml",
    "human_trafficking": "human_trafficking",
    "trafficking": "human_trafficking",
    "terrorist_financing": "terrorist_financing",
    "terrorism_financing": "terrorist_financing",
    "elder_exploitation": "elder_exploitation",
    "elder_fraud": "elder_exploitation",
    "sanctions_evasion": "layering",  # mapped to layering with sanctions hit fired
}


def _iter_bundle() -> Iterator[dict]:
    """Stream the pre-bundled SFT JSONL one row at a time."""
    path = POOLS.pool_1_efc_sft_bundle
    if not path.exists():
        logger.warning("Pool_1 bundle missing: %s", path)
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                payload = json.loads(row.get("content") or "{}")
            except json.JSONDecodeError:
                payload = {}
            if not payload:
                continue
            yield payload


def _has_cash_channel_efc(transactions: list[dict]) -> bool:
    """EFC transactions are dicts with a `channel` key (e.g. wire, ach, cash,
    atm, crypto). Returns True iff at least one transaction is on a cash
    channel.
    """
    for tx in transactions or []:
        ch = str(tx.get("channel", "")).strip().lower()
        if "cash" in ch or "atm" in ch or "currency" in ch:
            return True
    return False


def load(max_rows: int | None = None) -> pd.DataFrame:
    """Load Pool_1 into a DataFrame projected to the canonical schema.

    Columns:
      case_id, scenario_native, typology, label, severity_native,
      transactions (list[dict]), kyc_seed (dict), narrative (str),
      sar_outcome (str)
    """
    out: list[dict] = []
    n_remapped_to_layering = 0
    for payload in _iter_bundle():
        case = payload.get("case", {})
        case_id = case.get("case_id")
        if not case_id:
            continue
        scenario = (case.get("scenario") or "").lower()
        typology = EFC_SCENARIO_MAP.get(scenario, "none")
        # `final_decision` ∈ {"escalated_for_review", "filed", "no_action"}; treat
        # filed/escalated as positive label.
        label = case.get("final_decision") in {"escalated_for_review", "filed"} \
            or any(s.get("outcome") == "filed" for s in payload.get("sar_reports", []))

        txs = payload.get("transactions") or []
        sar_reports = payload.get("sar_reports") or []
        narrative = sar_reports[0]["narrative"] if sar_reports else ""

        # Option 3: structuring / smurfing typologies are CASH-specific by
        # statutory definition (CTR / 31 USC 5324). EFC has many "structuring"
        # cases whose underlying transactions are all wires/ACH — labeling
        # them as structuring would teach the SAR model to apply the $10K
        # CTR threshold to non-cash flows, which is regulatorily wrong and
        # creates internal-contradiction failures downstream. Remap to
        # "layering" when no cash channel is present.
        if typology in ("structuring", "smurfing"):
            if not _has_cash_channel_efc(txs):
                typology = "layering"
                n_remapped_to_layering += 1

        # Derive severity bucket from transaction count
        n_tx = len(txs)
        if n_tx <= 5:
            severity = "light"
        elif n_tx <= 15:
            severity = "medium"
        else:
            severity = "heavy"

        # KYC seed — Pool_1 doesn't carry per-case KYC; entity_id is the sender account
        first_tx = txs[0] if txs else {}
        kyc_seed = {
            "entity_id": case.get("linked_accounts", "").split(",")[0].strip("[]\"") or first_tx.get("sender_account", "UNKNOWN"),
            "country": first_tx.get("country_origin"),
        }

        out.append({
            "case_id": case_id,
            "source": "enterprise_fc",
            "scenario_native": scenario,
            "typology": typology,
            "label": label,
            "severity": severity,
            "transactions": txs,
            "kyc_seed": kyc_seed,
            "narrative": narrative,
            "sar_outcome": sar_reports[0].get("outcome") if sar_reports else None,
        })
        if max_rows and len(out) >= max_rows:
            break
    df = pd.DataFrame(out)
    logger.info(
        "pool_1_efc.load → %d rows (remapped %d structuring/smurfing → "
        "layering due to no cash channel)",
        len(df), n_remapped_to_layering,
    )
    return df
