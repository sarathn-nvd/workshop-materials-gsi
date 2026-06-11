"""Deterministic behavioral feature computer.

Pure function — no LLM, no I/O. Produces a `BehavioralMetrics` block
over (transactions[], kyc_profile). Used by:
- compute_hints (via semantic_profile) for the typology / regulatory-frame
  derivation
- the behavioral specialist sub-agent (as a sanity-check reference)
- the analytics dashboard's per-entity behavioral summary route
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from aml_app.common.schemas import BehavioralMetrics, KYCProfile, Transaction


# ---------------------------------------------------------------------------
# Static reference tables
# ---------------------------------------------------------------------------
_FX_TO_USD: dict[str, float] = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CHF": 1.13, "JPY": 0.0067,
    "CNY": 0.14, "Yuan": 0.14, "AUD": 0.66, "Australian Dollar": 0.66,
    "CAD": 0.74, "Canadian Dollar": 0.74, "Swiss Franc": 1.13,
    "Euro": 1.08, "US Dollar": 1.0,
}

_COUNTRY_RISK: dict[str, float] = {
    # FATF Black List
    "KP": 1.00, "IR": 1.00, "MM": 0.95,
    # FATF Grey List
    "CD": 0.85, "MZ": 0.80, "SY": 0.85, "YE": 0.80, "VE": 0.75,
    "JM": 0.65, "BG": 0.55, "AL": 0.55, "BB": 0.55, "BF": 0.65,
    "KH": 0.60, "JO": 0.60, "ML": 0.65, "MA": 0.55, "NI": 0.55,
    "PH": 0.65, "SN": 0.55, "SS": 0.75, "TR": 0.65,
    "AE": 0.55, "UG": 0.55,
    # Offshore / secrecy
    "KY": 0.60, "VG": 0.60, "BS": 0.55, "BZ": 0.55, "CY": 0.50,
    "LU": 0.40, "MT": 0.45, "PA": 0.65,
    # Sanctioned regions
    "RU": 0.85, "BY": 0.80, "CU": 0.85,
    # Lower risk
    "US": 0.10, "GB": 0.10, "DE": 0.10, "FR": 0.10, "NL": 0.10,
    "IT": 0.15, "ES": 0.15, "JP": 0.10, "AU": 0.10, "CA": 0.10,
    "CH": 0.20, "IE": 0.15, "BE": 0.15, "PT": 0.15, "AT": 0.15,
    "FI": 0.10, "SE": 0.10, "NO": 0.10, "DK": 0.10,
    "SG": 0.20, "HK": 0.30,
    "synthetic": 0.10, "": 0.0, "UNKNOWN": 0.10,
}


def country_risk(code: str) -> float:
    if not code:
        return 0.0
    # Match by ISO code prefix only (e.g. "US-NY" → "US").
    head = code.strip().split("-")[0].upper()
    return _COUNTRY_RISK.get(head, _COUNTRY_RISK.get(code.strip().upper(), 0.20))


def _to_usd(amount: float, currency: str) -> float:
    if not currency:
        return float(amount)
    rate = _FX_TO_USD.get(currency, _FX_TO_USD.get(currency.upper(), 1.0))
    return float(amount) * rate


_CASH_CHANNELS = {"cash", "atm", "currency"}


def normalize_channel(raw: str) -> str:
    """Map source-pool channel strings into the canonical schema enum."""
    c = (raw or "").strip().lower()
    if not c:
        return "wire"
    if c in {"cheque", "check"}:
        return "cheque"
    if c in {"transfer", "internal"}:
        return "wire"
    if c in {"atm", "currency"}:
        return "cash"
    if c in {"wire", "ach", "cash", "card", "crypto"}:
        return c
    return "wire"


# ---------------------------------------------------------------------------
# Feature computer
# ---------------------------------------------------------------------------
def compute_behavioral_metrics(
    transactions: Iterable[dict | Transaction],
    kyc_profile: dict | KYCProfile,
    window_days: int = 90,
) -> BehavioralMetrics:
    """Compute the 10-field BehavioralMetrics block over a bundle."""
    txs: list[dict] = []
    for t in transactions:
        if isinstance(t, Transaction):
            d = t.model_dump()
        elif isinstance(t, dict):
            d = dict(t)
        else:
            continue
        d["channel"] = normalize_channel(str(d.get("channel", "")))
        txs.append(d)

    if isinstance(kyc_profile, KYCProfile):
        kyc = kyc_profile.model_dump()
    else:
        kyc = dict(kyc_profile or {})

    if not txs:
        return BehavioralMetrics(
            tx_count=0, tx_total_usd=0.0, channel_mix={},
            velocity_24h_max=0, velocity_24h_avg_30d=0.0,
            unique_counterparties_7d=0, amount_z_score_max=0.0,
            country_risk_max=0.0, loop_detected=False,
            vs_declared_volume_ratio=0.0,
        )

    df = pd.DataFrame(txs)
    df["amount_usd"] = [
        abs(_to_usd(float(t.get("amount", 0.0) or 0.0),
                    str(t.get("currency", "USD") or "USD")))
        for t in txs
    ]
    date_col = df["date"] if "date" in df.columns else df.get("timestamp", "")
    df["ts"] = pd.to_datetime(date_col, errors="coerce", format="mixed")
    has_ts = df["ts"].notna().any()

    tx_count = int(len(df))
    tx_total_usd = float(df["amount_usd"].sum())
    cmix = _channel_mix_frac(df["channel"].tolist())

    if has_ts:
        velocity_24h_max = _velocity_24h_max(df)
        velocity_24h_avg_30d = _velocity_24h_avg_30d(df)
        unique_counterparties_7d = _unique_counterparties_7d(df)
    else:
        velocity_24h_max = tx_count
        velocity_24h_avg_30d = float(tx_count) / 30.0
        unique_counterparties_7d = int(df["counterparty"].nunique())

    amount_z_score_max = _amount_z_score_max(df)
    country_risk_max = _country_risk_max(df, kyc)
    loop_detected = _loop_detected(df, kyc)
    vs_declared_volume_ratio = _vs_declared_volume_ratio(df, kyc, window_days)

    return BehavioralMetrics(
        tx_count=tx_count,
        tx_total_usd=round(tx_total_usd, 2),
        channel_mix=cmix,
        velocity_24h_max=velocity_24h_max,
        velocity_24h_avg_30d=round(velocity_24h_avg_30d, 2),
        unique_counterparties_7d=unique_counterparties_7d,
        amount_z_score_max=round(amount_z_score_max, 2),
        country_risk_max=round(country_risk_max, 2),
        loop_detected=loop_detected,
        vs_declared_volume_ratio=round(vs_declared_volume_ratio, 2),
    )


def _channel_mix_frac(channels: list[str]) -> dict[str, float]:
    if not channels:
        return {}
    c = Counter(channels)
    n = len(channels)
    return {k: round(v / n, 3) for k, v in c.most_common()}


def _velocity_24h_max(df: pd.DataFrame) -> int:
    if df.empty or df["ts"].isna().all():
        return 0
    s = df.set_index("ts").sort_index()
    counts = s.assign(c=1)["c"].rolling("24h").sum()
    return int(counts.max() or 0)


def _velocity_24h_avg_30d(df: pd.DataFrame) -> float:
    if df.empty or df["ts"].isna().all():
        return 0.0
    span_days = max(
        1.0,
        (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0,
    )
    span_days = min(span_days, 30.0)
    return float(len(df)) / span_days


def _unique_counterparties_7d(df: pd.DataFrame) -> int:
    if df.empty or df["ts"].isna().all():
        return int(df["counterparty"].nunique())
    s = df.set_index("ts").sort_index()
    last_7d = s[s.index >= (s.index.max() - pd.Timedelta(days=7))]
    return int(last_7d["counterparty"].nunique())


def _amount_z_score_max(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 3:
        return 0.0
    amounts = df["amount_usd"].astype(float)
    mu = float(amounts.mean())
    sd = float(amounts.std(ddof=0))
    if sd <= 0.0:
        return 0.0
    z = (amounts - mu) / sd
    return float(z.abs().max())


def _country_risk_max(df: pd.DataFrame, kyc: dict) -> float:
    candidates: list[float] = []
    for col in ("country_origin", "country_destination", "country"):
        if col in df.columns:
            candidates.extend(country_risk(str(c)) for c in df[col].dropna().tolist())
    candidates.append(country_risk(str(kyc.get("incorporation_jurisdiction", ""))))
    return max(candidates) if candidates else 0.0


def _loop_detected(df: pd.DataFrame, kyc: dict) -> bool:
    if df.empty or "counterparty" not in df.columns:
        return False
    entity_id = str(kyc.get("entity_id", "")).strip()
    if not entity_id:
        return False
    if "sender_account" not in df.columns or "receiver_account" not in df.columns:
        return False
    out_to = set(
        df.loc[df["sender_account"].astype(str) == entity_id, "receiver_account"].astype(str)
    )
    in_from = set(
        df.loc[df["receiver_account"].astype(str) == entity_id, "sender_account"].astype(str)
    )
    return len(out_to & in_from) > 0


def _vs_declared_volume_ratio(df: pd.DataFrame, kyc: dict, window_days: int) -> float:
    declared = float(kyc.get("expected_monthly_volume", 0.0) or 0.0)
    if declared <= 0.0 or df.empty:
        return 0.0
    if df["ts"].notna().any():
        span_days = max(
            1.0,
            (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0,
        )
    else:
        span_days = float(window_days)
    monthly_rate = float(df["amount_usd"].sum()) * (30.0 / span_days)
    return abs(monthly_rate) / declared
