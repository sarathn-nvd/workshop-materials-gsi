"""Shared helpers for the L2 prototype generators.

Kept intentionally minimal — direct OpenAI-compatible LLM calls (no
DataDesigner overhead for 5-record prototype runs), and a few utility
functions for projecting EFC raw payloads into the canonical bundle shape.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

import requests

from scripts.config import LLM_API_KEY, LLM_ENDPOINT, LLM_MODEL, POOLS

logger = logging.getLogger(__name__)


# ============================================================================
# Direct LLM call (bypasses DataDesigner for prototype simplicity)
# ============================================================================
def chat_completion(
    *,
    system: str,
    user: str,
    temperature: float = 0.5,
    max_tokens: int = 1500,
    timeout: int = 180,
) -> str:
    """Synchronous chat-completion via the local NIM / vLLM endpoint."""
    url = f"{LLM_ENDPOINT.rstrip('/')}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def safe_json_loads(s: str) -> dict | list | None:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    s = (s or "").strip()
    for fence in ("```json", "```JSON", "```"):
        if s.startswith(fence):
            s = s[len(fence):].strip()
            if s.endswith("```"):
                s = s[:-3].strip()
            break
    try:
        return json.loads(s)
    except Exception:
        # Try to extract the first JSON object from the text
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


# ============================================================================
# EFC bundle projection (raw → canonical schema)
# ============================================================================
_EFC_SCENARIO_TO_TYPOLOGY = {
    "structuring":              "structuring",
    "structuring_smurfing":     "structuring",
    "smurfing":                 "smurfing",
    "layering":                 "layering",
    "rapid_movement":           "layering",
    "round_tripping":           "layering",
    "cross_border_layering":    "layering",
    "crypto_conversion":        "layering",
    "shell_company":            "shell_company",
    "shell_company_transfer":   "shell_company",
    "trade_based_ml":           "trade_based_ml",
    "trade_based_laundering":   "trade_based_ml",
    "tbml":                     "trade_based_ml",
    "human_trafficking":        "human_trafficking",
    "trafficking":              "human_trafficking",
    "terrorist_financing":      "terrorist_financing",
    "terrorism_financing":      "terrorist_financing",
    "elder_exploitation":       "elder_exploitation",
    "elder_fraud":              "elder_exploitation",
    "sanctions_evasion":        "layering",
}


def iter_efc_bundles(max_cases: int | None = None) -> Iterator[dict]:
    """Stream parsed EFC SFT bundles."""
    path: Path = POOLS.pool_1_efc_sft_bundle
    with path.open("r", encoding="utf-8") as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                payload = json.loads(row.get("content") or "{}")
            except Exception:
                continue
            if not payload or "case" not in payload or "transactions" not in payload:
                continue
            yield payload
            n += 1
            if max_cases and n >= max_cases:
                return


def project_efc_to_bundle(payload: dict) -> tuple[list[dict], dict, str, str]:
    """Project an EFC raw payload into (transactions[], kyc_profile, typology,
    case_id) tuples in the canonical schema.

    The transactions list keeps both the canonical fields needed by the SAR
    narrative AND the extra columns needed by the feature computer
    (sender_account, country_origin, country_destination).
    """
    case = payload.get("case", {})
    case_id = str(case.get("case_id", "UNKNOWN"))
    src_scenario = (case.get("scenario") or "").lower().strip()
    typology = _EFC_SCENARIO_TO_TYPOLOGY.get(src_scenario, "none")

    raw_txs = payload.get("transactions") or []
    txs: list[dict] = []
    for t in raw_txs:
        txs.append({
            "date": str(t.get("timestamp", ""))[:10],
            "amount": float(t.get("amount", 0.0) or 0.0),
            "currency": t.get("currency", "USD"),
            "counterparty": t.get("receiver_account", ""),
            "channel": "wire",   # EFC bundle is wire-only by source convention
            "notes": "",
            # Extra columns for behavioral feature computer:
            "country_origin": t.get("country_origin", ""),
            "country_destination": t.get("country_destination", ""),
            "sender_account": t.get("sender_account", ""),
            "receiver_account": t.get("receiver_account", ""),
            "risk_score": t.get("risk_score", 0.0),
        })

    # Synthesize a minimal KYC. The real pipeline LLM-generates business_purpose;
    # for the prototype we use a templated string from the trigger event.
    linked = (case.get("linked_accounts") or "").strip("[]\"' ")
    entity_id = linked.split(",")[0].strip("\"' ") if linked else "AC_UNKNOWN"
    country = (txs[0].get("country_origin") if txs else "US") or "US"
    trigger = case.get("trigger_event") or "transactional review"
    kyc = {
        "entity_id": entity_id,
        "entity_type": "business",
        "expected_monthly_volume": 50000.0,
        "business_purpose": (
            f"The entity operates a business in {country} with declared monthly "
            f"transaction volume of $50,000. Activity profile under review for: "
            f"{trigger}."
        ),
        "risk_rating": "medium",
        "incorporation_jurisdiction": country,
    }
    return txs, kyc, typology, case_id


# ============================================================================
# IBM AML bundle projection (Pool_2)
# ============================================================================
_IBM_PATTERN_TO_TYPOLOGY = {
    "FAN-OUT": "structuring",       # cash-channel will trigger remap below
    "FAN-IN": "smurfing",
    "GATHER-SCATTER": "smurfing",
    "SCATTER-GATHER": "structuring",
    "STACK": "layering",
    "CYCLE": "layering",
    "BIPARTITE": "layering",
    "RANDOM": "none",
}

_IBM_TX_COLUMNS = (
    "timestamp", "from_bank", "from_acct", "to_bank", "to_acct",
    "amount", "currency", "recv_amount", "recv_currency",
    "channel", "is_laundering",
)


def _parse_ibm_tx(line: str) -> dict | None:
    cols = line.split(",")
    if len(cols) < 10:
        return None
    try:
        amount = float(cols[5])
    except (ValueError, IndexError):
        return None
    return {
        "date": cols[0][:10],
        "amount": amount,
        "currency": cols[6].strip() if len(cols) > 6 else "USD",
        "counterparty": cols[4].strip() if len(cols) > 4 else "",
        "channel": cols[9].strip().lower() if len(cols) > 9 else "wire",
        "notes": "",
        "country_origin": "US",
        "country_destination": "US",
        "sender_account": cols[2].strip() if len(cols) > 2 else "",
        "receiver_account": cols[4].strip() if len(cols) > 4 else "",
        "risk_score": 0.0,
    }


def iter_ibm_bundles(max_cases: int | None = None) -> Iterator[dict]:
    """Stream IBM AML pattern blocks as `payload` dicts compatible with
    project_ibm_to_bundle(). Yields `{case: {scenario, case_id}, transactions: [...]}`.
    """
    from scripts.pools.pool_2_ibm import _parse_patterns_txt
    from scripts.config import POOLS

    base = POOLS.pool_2_ibm
    if not base.exists():
        return
    blocks = _parse_patterns_txt(base / "HI-Small_Patterns.txt")
    n = 0
    for i, block in enumerate(blocks):
        scenario = block["pattern_type"].lower()
        # Parse raw CSV strings into canonical tx dicts
        txs: list[dict] = []
        for raw in block.get("transactions") or []:
            tx = _parse_ibm_tx(raw)
            if tx:
                txs.append(tx)
        if len(txs) < 2:
            continue
        yield {
            "case": {
                "case_id": f"ibm_{i:06d}",
                "scenario": scenario,
                "trigger_event": f"IBM-AML pattern: {block['pattern_type']}",
                "linked_accounts": txs[0].get("sender_account", "") if txs else "",
            },
            "transactions": [
                {**tx, "timestamp": tx["date"]} for tx in txs    # iter expects timestamp
            ],
        }
        n += 1
        if max_cases and n >= max_cases:
            return


def project_ibm_to_bundle(payload: dict) -> tuple[list[dict], dict, str, str]:
    """Project an IBM payload into (transactions[], kyc_profile, typology, case_id)."""
    case = payload.get("case", {})
    case_id = str(case.get("case_id", "ibm_unknown"))
    scenario = (case.get("scenario") or "").upper().rstrip(":").strip()
    typology_native = _IBM_PATTERN_TO_TYPOLOGY.get(scenario, "none")

    raw_txs = payload.get("transactions") or []
    txs: list[dict] = []
    has_cash = False
    for t in raw_txs:
        tx = {
            "date": t.get("date") or str(t.get("timestamp", ""))[:10],
            "amount": float(t.get("amount", 0.0) or 0.0),
            "currency": t.get("currency", "USD"),
            "counterparty": t.get("counterparty", "") or t.get("receiver_account", ""),
            "channel": (t.get("channel", "wire") or "wire").lower(),
            "notes": "",
            "country_origin": t.get("country_origin", "US"),
            "country_destination": t.get("country_destination", "US"),
            "sender_account": t.get("sender_account", ""),
            "receiver_account": t.get("receiver_account", ""),
            "risk_score": float(t.get("risk_score", 0.0) or 0.0),
        }
        if "cash" in tx["channel"]:
            has_cash = True
        txs.append(tx)

    # Mirror the cash-channel remap logic from pool_2_ibm.load() so the
    # behavioral typology is consistent with the SAR narrative typology
    # (no CTR framing for non-cash structuring).
    typology = typology_native
    if typology in ("structuring", "smurfing") and not has_cash:
        typology = "layering"

    entity = txs[0].get("sender_account", "") if txs else "ibm_unknown"
    kyc = {
        "entity_id": entity or f"ibm_block_{case_id}",
        "entity_type": "business",
        "expected_monthly_volume": 50000.0,
        "business_purpose": (
            f"Activity classified by IBM-AML pattern detector as '{scenario}'. "
            "Profile under behavioral review."
        ),
        "risk_rating": "medium",
        "incorporation_jurisdiction": "US",
    }
    return txs, kyc, typology, case_id


# ============================================================================
# IBM ACCOUNT-LEVEL bundle iterator (uses raw HI-Small_Trans.csv)
#
# The pattern-block iterator (above) only yields the ~327 BEGIN/END laundering
# attempts in `HI-Small_Patterns.txt`. The raw transaction file has 5M+ rows
# across 497K accounts — far more behavioral signal. This iterator groups
# `HI-Small_Trans.csv` by sender account and yields one bundle per entity
# meeting the (min_tx, max_tx) filter.
#
# Records produced this way are unlabeled by pattern type; typology is
# inferred at projection time via `compute_semantic_profile()` (same as the
# pattern-block path). This is fine for behavioral training: the metrics
# block is gold-anchored regardless of pattern label.
# ============================================================================
_IBM_TRANS_DF_CACHE: "object" = None


def _load_ibm_trans_df():
    """Load HI-Small_Trans.csv into a DataFrame once, cached for reuse."""
    global _IBM_TRANS_DF_CACHE
    if _IBM_TRANS_DF_CACHE is not None:
        return _IBM_TRANS_DF_CACHE
    import pandas as pd
    from scripts.config import POOLS

    csv_path = POOLS.pool_2_ibm / "HI-Small_Trans.csv"
    if not csv_path.exists():
        logger.warning("IBM HI-Small_Trans.csv not found at %s", csv_path)
        _IBM_TRANS_DF_CACHE = pd.DataFrame()
        return _IBM_TRANS_DF_CACHE
    logger.info("Loading IBM HI-Small_Trans.csv (~5M rows) — first call, then cached")
    df = pd.read_csv(csv_path, dtype={
        "Timestamp": str, "From Bank": str, "Account": str,
        "To Bank": str, "Account.1": str,
        "Amount Received": float, "Receiving Currency": str,
        "Amount Paid": float, "Payment Currency": str,
        "Payment Format": str, "Is Laundering": int,
    })
    _IBM_TRANS_DF_CACHE = df
    logger.info("IBM Trans loaded: %d rows", len(df))
    return _IBM_TRANS_DF_CACHE


def iter_ibm_account_bundles(
    min_tx: int = 5,
    max_tx: int = 30,
    max_cases: int | None = None,
) -> Iterator[dict]:
    """Yield IBM bundles formed by grouping HI-Small_Trans.csv by sender account.

    Each yielded bundle has:
      case.case_id    = "ibm_acct_<sender>"
      case.scenario   = "" (typology inferred downstream from semantic profile)
      transactions[]  = up to `max_tx` rows for that sender, projected to the
                        canonical {date, amount, currency, channel, ...} dict
    """
    df = _load_ibm_trans_df()
    if df.empty:
        return
    counts = df["Account"].value_counts()
    qualifying = counts[(counts >= min_tx) & (counts <= max_tx)]
    if qualifying.empty:
        return
    grouped = df[df["Account"].isin(qualifying.index)].groupby("Account", sort=False)
    n = 0
    for sender, group in grouped:
        rows = group.head(max_tx).to_dict("records")
        txs = []
        for r in rows:
            try:
                amt_paid = float(r.get("Amount Paid") or 0.0)
            except (ValueError, TypeError):
                amt_paid = 0.0
            ts = str(r.get("Timestamp", ""))[:10]
            channel_raw = str(r.get("Payment Format", "") or "wire").lower()
            # Map IBM "Payment Format" → canonical channel
            if "cash" in channel_raw:
                channel = "cash"
            elif "ach" in channel_raw or "credit" in channel_raw:
                channel = "ach"
            elif "wire" in channel_raw:
                channel = "wire"
            elif "cheque" in channel_raw or "check" in channel_raw:
                channel = "cheque"
            else:
                channel = "wire"
            cur = (str(r.get("Payment Currency", "USD") or "USD")
                   .replace("US Dollar", "USD").replace(" ", ""))
            txs.append({
                "date": ts or "2022-09-01",
                "amount": amt_paid,
                "currency": cur or "USD",
                "counterparty": str(r.get("Account.1", "")),
                "channel": channel,
                "notes": "",
                "country_origin": "US",
                "country_destination": "US",
                "sender_account": str(sender),
                "receiver_account": str(r.get("Account.1", "")),
                "risk_score": 0.0,
            })
        if len(txs) < min_tx:
            continue
        yield {
            "case": {
                "case_id": f"ibm_acct_{sender}",
                "scenario": "",  # typology inferred from semantic profile
                "trigger_event": "IBM-AML entity-level transactional review",
            },
            "transactions": [{**tx, "timestamp": tx["date"]} for tx in txs],
        }
        n += 1
        if max_cases and n >= max_cases:
            return


# ============================================================================
# AMLGentex ACCOUNT-LEVEL bundle iterator (uses raw tx_log.parquet)
#
# The cluster iterator (below) yields one bundle per modelID (~58 clusters).
# The raw `tx_log.parquet` has 679K transactions across 10K accounts — far
# more behavioral signal. This iterator groups by `nameOrig` and yields one
# bundle per qualifying account.
# ============================================================================
_AMLGENTEX_TX_LOG_CACHE: "object" = None


def _load_amlgentex_tx_log_df():
    """Load tx_log.parquet once, cached."""
    global _AMLGENTEX_TX_LOG_CACHE
    if _AMLGENTEX_TX_LOG_CACHE is not None:
        return _AMLGENTEX_TX_LOG_CACHE
    import pandas as pd
    from scripts.config import POOLS

    parquet_path = POOLS.pool_3_amlgentex / "temporal" / "tx_log.parquet"
    if not parquet_path.exists():
        logger.warning("AMLGentex tx_log.parquet not found at %s", parquet_path)
        _AMLGENTEX_TX_LOG_CACHE = pd.DataFrame()
        return _AMLGENTEX_TX_LOG_CACHE
    logger.info("Loading AMLGentex tx_log.parquet — first call, then cached")
    _AMLGENTEX_TX_LOG_CACHE = pd.read_parquet(parquet_path)
    logger.info("AMLGentex tx_log loaded: %d rows", len(_AMLGENTEX_TX_LOG_CACHE))
    return _AMLGENTEX_TX_LOG_CACHE


def iter_amlgentex_account_bundles(
    min_tx: int = 3,
    max_tx: int = 30,
    max_cases: int | None = None,
) -> Iterator[dict]:
    """Yield AMLGentex bundles formed by grouping tx_log.parquet by nameOrig."""
    df = _load_amlgentex_tx_log_df()
    if df.empty or "nameOrig" not in df.columns:
        return
    counts = df["nameOrig"].value_counts()
    qualifying = counts[(counts >= min_tx) & (counts <= max_tx)]
    if qualifying.empty:
        return
    grouped = df[df["nameOrig"].isin(qualifying.index)].groupby("nameOrig", sort=False)
    n = 0
    for sender, group in grouped:
        rows = group.head(max_tx).to_dict("records")
        txs = []
        for r in rows:
            ch = str(r.get("type", "") or "wire").lower()
            if "cash" in ch:
                channel = "cash"
            elif "transfer" in ch:
                channel = "wire"
            elif "initalbalance" in ch:
                continue   # skip non-economic-event opening-balance rows
            else:
                channel = "wire"
            txs.append({
                "date": "2025-01-01",   # AMLGentex has step (synthetic time index), not real dates
                "amount": float(r.get("amount", 0.0) or 0.0),
                "currency": "USD",
                "counterparty": str(r.get("nameDest", "")),
                "channel": channel,
                "notes": "",
                "country_origin": "synthetic",
                "country_destination": "synthetic",
                "sender_account": str(sender),
                "receiver_account": str(r.get("nameDest", "")),
                "risk_score": 0.0,
            })
        if len(txs) < min_tx:
            continue
        yield {
            "case": {
                "case_id": f"gentex_acct_{sender}",
                "scenario": "",
                "trigger_event": "AMLGentex entity-level transactional review",
            },
            "transactions": [{**tx, "timestamp": tx["date"]} for tx in txs],
        }
        n += 1
        if max_cases and n >= max_cases:
            return


# ============================================================================
# AMLGentex bundle projection (Pool_3)
# ============================================================================
_GENTEX_TYPE_TO_TYPOLOGY = {
    "fan_out": "structuring",
    "fan_in": "smurfing",
    "scatter_gather": "smurfing",
    "gather_scatter": "structuring",
    "stack": "layering",
    "cycle": "layering",
    "bipartite": "layering",
    "peeling_chain": "layering",
}


def iter_amlgentex_bundles(max_cases: int | None = None) -> Iterator[dict]:
    """Stream AMLGentex cluster records (one per unique modelID) as `payload`
    dicts compatible with project_amlgentex_to_bundle().
    """
    from scripts.pools.pool_3_amlgentex import load as load_amlgentex

    df = load_amlgentex()
    if df.empty:
        return
    n = 0
    for _, row in df.iterrows():
        txs = row.get("transactions") or []
        if len(txs) < 2:
            continue
        yield {
            "case": {
                "case_id": str(row.get("cluster_id") or row.get("block_id") or f"gentex_{n:06d}"),
                "scenario": str(row.get("scenario_native", "")).lower(),
                "trigger_event": f"AMLGentex cluster: {row.get('scenario_native', '')}",
            },
            "transactions": txs,
        }
        n += 1
        if max_cases and n >= max_cases:
            return


def project_amlgentex_to_bundle(payload: dict) -> tuple[list[dict], dict, str, str]:
    """Project an AMLGentex payload into (transactions[], kyc_profile, typology, case_id)."""
    case = payload.get("case", {})
    case_id = str(case.get("case_id", "gentex_unknown"))
    scenario = (case.get("scenario") or "").lower()
    typology_native = _GENTEX_TYPE_TO_TYPOLOGY.get(scenario, "none")

    raw_txs = payload.get("transactions") or []
    txs: list[dict] = []
    has_cash = False
    for t in raw_txs:
        ch = (t.get("channel", "wire") or "wire").lower()
        if "cash" in ch:
            has_cash = True
        txs.append({
            "date": str(t.get("timestamp", ""))[:10] or "2025-01-01",
            "amount": float(t.get("amount", 0.0) or 0.0),
            "currency": t.get("currency", "USD"),
            "counterparty": t.get("receiver_account", ""),
            "channel": ch,
            "notes": "",
            "country_origin": t.get("country_origin", "synthetic"),
            "country_destination": t.get("country_destination", "synthetic"),
            "sender_account": t.get("sender_account", ""),
            "receiver_account": t.get("receiver_account", ""),
            "risk_score": 0.0,
        })

    typology = typology_native
    if typology in ("structuring", "smurfing") and not has_cash:
        typology = "layering"

    kyc = {
        "entity_id": txs[0].get("sender_account", "") if txs else f"gentex_{case_id}",
        "entity_type": "business",
        "expected_monthly_volume": 50000.0,
        "business_purpose": (
            f"AMLGentex synthetic entity under behavioral review for "
            f"'{scenario}' pattern."
        ),
        "risk_rating": "medium",
        "incorporation_jurisdiction": "synthetic",
    }
    return txs, kyc, typology, case_id


def render_bundle_passage(transactions: list[dict], kyc: dict) -> str:
    """Render a (transactions, kyc) bundle as a structured text passage.

    This is the same shape NAT will assemble at runtime when invoking the
    auxiliary_behavioral or sar_judgment task. Keep formatting deterministic
    (no LLM in the loop).
    """
    lines: list[str] = ["[transactions]"]
    for t in transactions:
        lines.append(
            f"{t['date']} {t['channel']:<5} {float(t['amount']):>14,.2f} {t['currency']:<10} "
            f"{t.get('sender_account','')[-12:]:>12} → {t.get('receiver_account','')[-12:]:<12} "
            f"({t.get('country_origin','')}→{t.get('country_destination','')})"
        )
    lines.append("")
    lines.append("[kyc_profile]")
    for k in ("entity_id", "entity_type", "expected_monthly_volume",
              "business_purpose", "risk_rating", "incorporation_jurisdiction"):
        lines.append(f"{k}: {kyc.get(k, '')}")
    return "\n".join(lines)
