"""Pool_2 — IBM AML transactions.

Three CSVs:
  - `HI-Small_Trans.csv`: transactions
  - `HI-Small_accounts.csv`: account → entity name + bank metadata
  - `HI-Small_Patterns.txt`: BEGIN/END pattern blocks tagging tx clusters

The strategy doc maps Pattern types → canonical typologies (Stage 1 §2B).
"""
from __future__ import annotations

import logging

import pandas as pd

from scripts.config import POOLS

logger = logging.getLogger(__name__)


# IBM AML pattern type → canonical typology
IBM_PATTERN_MAP: dict[str, str] = {
    "FAN-OUT": "structuring",
    "FAN-IN": "smurfing",
    "GATHER-SCATTER": "smurfing",
    "SCATTER-GATHER": "structuring",
    "STACK": "layering",
    "CYCLE": "layering",
    "BIPARTITE": "layering",
    "RANDOM": "none",
}


def _parse_patterns_txt(path) -> list[dict]:
    """Parse `HI-Small_Patterns.txt` into list of {pattern_type, transactions[]}."""
    blocks: list[dict] = []
    cur_type: str | None = None
    cur_txs: list[str] = []
    if not path.exists():
        return blocks
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT - "):
                # Some pattern names in the file have a trailing colon
                # ("FAN-OUT:", "CYCLE:", etc.) and some don't ("STACK",
                # "BIPARTITE", "SCATTER-GATHER"). Strip both whitespace and
                # any trailing punctuation so IBM_PATTERN_MAP keys all match.
                cur_type = line.split(" - ")[-1].strip().rstrip(":").strip()
                cur_txs = []
            elif line.startswith("END LAUNDERING ATTEMPT - "):
                if cur_type:
                    blocks.append({"pattern_type": cur_type, "transactions": cur_txs})
                cur_type = None
                cur_txs = []
            elif cur_type:
                cur_txs.append(line)
    return blocks


def load(max_rows: int | None = None) -> pd.DataFrame:
    """Load Pool_2 → DataFrame of pattern-block records.

    Columns:
      block_id, source, scenario_native, typology, label, severity,
      transactions (list[dict]), kyc_seed (dict)
    """
    base = POOLS.pool_2_ibm
    if not base.exists():
        logger.warning("Pool_2 dir missing: %s", base)
        return pd.DataFrame()

    accounts = pd.read_csv(base / "HI-Small_accounts.csv") if (base / "HI-Small_accounts.csv").exists() else pd.DataFrame()
    blocks = _parse_patterns_txt(base / "HI-Small_Patterns.txt")

    out: list[dict] = []
    n_dropped_suspicious_random = 0
    n_remapped_to_layering = 0
    for i, block in enumerate(blocks):
        scenario = block["pattern_type"].lower()
        typology = IBM_PATTERN_MAP.get(block["pattern_type"], "none")
        # Change 7: filter RANDOM blocks whose transactions look suspicious.
        # IBM's "RANDOM" pattern is intended to label benign / non-laundering
        # transactions, but the LLM judge flagged 30 of these as actually
        # showing structuring behaviour (multiple sub-$10K amounts). Drop
        # these so they don't contaminate Record_5 negatives.
        if typology == "none":
            sus = _looks_suspicious_pattern(block["transactions"])
            if sus:
                n_dropped_suspicious_random += 1
                continue
        # Option 3: structuring / smurfing typologies are CASH-specific by
        # statutory definition (31 USC 5324: "structure or assist in
        # structuring any transaction" — CTR is cash-only). If the IBM
        # pattern is mapped to structuring or smurfing but the transactions
        # have NO cash channel, the source label is misaligned with the
        # actual activity — remap to "layering". This avoids training the
        # SAR narrative model to apply CTR rules to wire/ACH transfers.
        if typology in ("structuring", "smurfing"):
            if not _has_cash_channel_ibm(block["transactions"]):
                typology = "layering"
                n_remapped_to_layering += 1
        n_tx = len(block["transactions"])
        if n_tx <= 5:
            severity = "light"
        elif n_tx <= 15:
            severity = "medium"
        else:
            severity = "heavy"
        out.append({
            "block_id": f"ibm_{i:06d}",
            "source": "ibm_aml",
            "scenario_native": scenario,
            "typology": typology,
            "label": typology != "none",
            "severity": severity,
            "transactions": block["transactions"],          # raw CSV-style strings
            "kyc_seed": {"entity_id": f"ibm_block_{i}"},
        })
        if max_rows and len(out) >= max_rows:
            break
    df = pd.DataFrame(out)
    logger.info(
        "pool_2_ibm.load → %d pattern blocks (dropped %d 'RANDOM' blocks "
        "with suspicious-looking content; remapped %d structuring/smurfing "
        "blocks to 'layering' due to no cash channel)",
        len(df), n_dropped_suspicious_random, n_remapped_to_layering,
    )
    return df


def _has_cash_channel_ibm(transactions: list[str]) -> bool:
    """IBM transactions are CSV strings:
    timestamp,from_bank,from_acct,to_bank,to_acct,amount,currency,recv_amount,
    recv_currency,channel,is_laundering
    Field index 9 is the channel (e.g. ACH, Cash, Wire, Cheque, Credit Card).
    Returns True iff at least one transaction uses a cash channel.
    """
    for tx in transactions:
        cols = tx.split(",")
        if len(cols) < 10:
            continue
        ch = cols[9].strip().lower()
        if "cash" in ch:
            return True
    return False


def _looks_suspicious_pattern(transactions: list[str]) -> bool:
    """Heuristic: a RANDOM block 'looks suspicious' if it contains ≥3 amounts
    within $500 of the $10,000 CTR threshold (i.e., $9,500–$10,000) or if
    the amounts are unnaturally clustered (low coefficient of variation).
    """
    if not transactions or len(transactions) < 3:
        return False
    amounts: list[float] = []
    for line in transactions:
        try:
            cols = line.split(",")
            if len(cols) >= 6:
                amount = float(cols[5])
                amounts.append(amount)
        except (ValueError, IndexError):
            continue
    if len(amounts) < 3:
        return False
    near_threshold = sum(1 for a in amounts if 9500 <= a < 10000)
    if near_threshold >= 3:
        return True
    # Many identical amounts (suggests structuring)
    same_amount_count = max((amounts.count(a) for a in set(amounts)), default=0)
    if same_amount_count >= 3 and len(set(amounts)) <= max(2, len(amounts) // 3):
        return True
    return False
