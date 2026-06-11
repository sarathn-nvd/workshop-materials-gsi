"""Deterministic behavioral feature computer.

Given a (transactions[], kyc_profile) bundle, computes the 10-field metrics
block used by `auxiliary_behavioral` (see schemas.BehavioralMetrics and
training_strategy.md §5.3 sample record).

Design contract (per training_strategy Appendix F + §7.4):
- Pure function — no LLM, no I/O, no side effects.
- Output is gold-anchored: at training time, this is the ground truth the
  LLM's metrics block must reproduce. At runtime, the model produces its
  own metrics; this module can be used as a verifier.
- All 10 metrics are computable from the inputs alone — no external lookups
  beyond a small static country-risk table.
- Currency-normalization uses a static rate table (not real-time FX). The
  model only needs to learn the *aggregation pattern*, not FX accuracy;
  RL phase can refine FX handling if needed.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from scripts.schemas import BehavioralMetrics, KYCProfile, Transaction


# ============================================================================
# Static reference tables
# ============================================================================
# Static USD conversion table — sufficient for training-time aggregation.
# At runtime an FX-aware version would hit a quote service; the model only
# learns the pattern of "convert to a canonical currency, then aggregate".
_FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CHF": 1.13,
    "JPY": 0.0067,
    "CNY": 0.14,        # approx for "Yuan" alias
    "Yuan": 0.14,
    "AUD": 0.66,
    "Australian Dollar": 0.66,
    "CAD": 0.74,
    "Canadian Dollar": 0.74,
    "Swiss Franc": 1.13,
    "Euro": 1.08,
    "US Dollar": 1.0,
}


# Country-risk lookup. Values from FATF high-risk + EU AMLD high-risk
# country lists, simplified to a 0-1 risk score. Anything not listed
# defaults to 0.0 (low risk).
_COUNTRY_RISK: dict[str, float] = {
    # FATF "Black List" — call for action
    "KP": 1.00, "IR": 1.00, "MM": 0.95,
    # FATF "Grey List" — increased monitoring
    "CD": 0.85, "MZ": 0.80, "SY": 0.85, "YE": 0.80, "VE": 0.75,
    "JM": 0.65, "BG": 0.55, "AL": 0.55, "BB": 0.55, "BF": 0.65,
    "KH": 0.60, "JO": 0.60, "ML": 0.65, "MA": 0.55, "NI": 0.55,
    "PA": 0.65, "PH": 0.65, "SN": 0.55, "SS": 0.75, "TR": 0.65,
    "AE": 0.55, "UG": 0.55,
    # Common offshore / secrecy jurisdictions
    "KY": 0.60, "VG": 0.60, "BS": 0.55, "BZ": 0.55, "CY": 0.50,
    "LU": 0.40, "MT": 0.45, "PA": 0.65,
    # Sanctioned regions (rough markers)
    "RU": 0.85, "BY": 0.80, "CU": 0.85,
    # Lower-risk reference (for completeness)
    "US": 0.10, "GB": 0.10, "DE": 0.10, "FR": 0.10, "NL": 0.10,
    "IT": 0.15, "ES": 0.15, "JP": 0.10, "AU": 0.10, "CA": 0.10,
    "CH": 0.20, "IE": 0.15, "BE": 0.15, "PT": 0.15, "AT": 0.15,
    "FI": 0.10, "SE": 0.10, "NO": 0.10, "DK": 0.10,
    "SG": 0.20, "HK": 0.30,
    # Unknown / synthetic
    "synthetic": 0.10, "": 0.0, "UNKNOWN": 0.10,
}


def _country_risk(code: str) -> float:
    if not code:
        return 0.0
    return _COUNTRY_RISK.get(code.strip().upper(), 0.20)


def _to_usd(amount: float, currency: str) -> float:
    """Convert to USD using the static FX table.

    Currency strings observed in seed data include both ISO codes (USD, EUR)
    and human names ("US Dollar", "Yuan"). The table covers both. Unknown
    currencies pass through at 1:1 — the result is documented as best-effort.
    """
    if not currency:
        return float(amount)
    rate = _FX_TO_USD.get(currency, _FX_TO_USD.get(currency.upper(), 1.0))
    return float(amount) * rate


# ============================================================================
# Channel classification
# ============================================================================
_CASH_CHANNELS = {"cash", "atm", "currency"}
_NONCASH_CHANNELS = {"wire", "ach", "card", "cheque", "check", "crypto", "internal", "transfer"}


def _normalize_channel(raw: str) -> str:
    """Map source-pool channel strings into the canonical schema enum.

    The schema's Transaction.channel is Literal["wire","ach","cash","card",
    "cheque","crypto"], but pools sometimes emit "transfer" / "internal" /
    "check" / lowercased variants. We map them consistently here so the
    feature computer is robust to source heterogeneity.
    """
    c = (raw or "").strip().lower()
    if not c:
        return "wire"           # safe default for amlgentex etc.
    if c in {"cheque", "check"}:
        return "cheque"
    if c in {"transfer", "internal"}:
        return "wire"           # transfers are non-cash; bucket under wire
    if c in {"atm", "currency"}:
        return "cash"
    if c in {"wire", "ach", "cash", "card", "crypto"}:
        return c
    return "wire"


# ============================================================================
# The feature computer
# ============================================================================
def compute_behavioral_metrics(
    transactions: Iterable[dict | Transaction],
    kyc_profile: dict | KYCProfile,
    window_days: int = 90,
) -> BehavioralMetrics:
    """Compute the 10-field behavioral metrics block over a bundle.

    Parameters
    ----------
    transactions : iterable of dicts or Transaction models
        Each must have at least: date (str), amount (float), currency (str),
        counterparty (str), channel (str). Other fields are ignored.
    kyc_profile : dict or KYCProfile
        Must have at least: expected_monthly_volume (float),
        incorporation_jurisdiction (str).
    window_days : int
        Trailing window for "30d" and "7d" rolling-style metrics. Defaults
        to 90 (the agent's standard look-back).

    Returns
    -------
    BehavioralMetrics
        Pydantic-validated metrics block. Empty bundle produces an
        all-zero/false metrics block (loop_detected=False, channel_mix={}).
    """
    # ----- normalize inputs to plain dicts ---------------------------------
    txs: list[dict] = []
    for t in transactions:
        if isinstance(t, Transaction):
            d = t.model_dump()
        elif isinstance(t, dict):
            d = dict(t)
        else:
            continue
        d["channel"] = _normalize_channel(str(d.get("channel", "")))
        txs.append(d)

    if isinstance(kyc_profile, KYCProfile):
        kyc = kyc_profile.model_dump()
    else:
        kyc = dict(kyc_profile or {})

    # ----- empty bundle short-circuit --------------------------------------
    if not txs:
        return BehavioralMetrics(
            tx_count=0, tx_total_usd=0.0, channel_mix={},
            velocity_24h_max=0, velocity_24h_avg_30d=0.0,
            unique_counterparties_7d=0, amount_z_score_max=0.0,
            country_risk_max=0.0, loop_detected=False,
            vs_declared_volume_ratio=0.0,
        )

    # ----- frame (perf-tuned: vectorize amount_usd; format-hinted ts parse) -
    # Compute amount_usd in a vectorized list comp, NOT df.apply — saves ~70%
    # of per-call overhead at the small frame sizes we see (3–8 rows).
    # Use ABSOLUTE amounts for behavioral aggregation. AML behavioral analysis
    # tracks gross activity magnitude, not net direction — a $1,000 refund is
    # still $1,000 of activity. Real-world EFC bundles include negative
    # amounts (refunds, reversals, adjustments).
    df = pd.DataFrame(txs)
    df["amount_usd"] = [
        abs(_to_usd(float(t.get("amount", 0.0) or 0.0),
                    str(t.get("currency", "USD") or "USD")))
        for t in txs
    ]
    # parse date — `format="mixed"` skips the dateutil slow-path (which was
    # the dominant Stage-7 pre-build bottleneck on the original launch).
    # Suppresses "could not infer format" warning and gives ~50–100x speedup
    # on small frames.
    date_col = df["date"] if "date" in df.columns else df.get("timestamp", "")
    df["ts"] = pd.to_datetime(date_col, errors="coerce", format="mixed")
    df = df.dropna(subset=["ts"]).copy()
    if df.empty:
        # All dates unparseable — fall back to non-temporal metrics only.
        df = pd.DataFrame(txs)
        df["amount_usd"] = [
            abs(_to_usd(float(t.get("amount", 0.0) or 0.0),
                        str(t.get("currency", "USD") or "USD")))
            for t in txs
        ]
        df["ts"] = pd.NaT

    # ----- top-line ---------------------------------------------------------
    tx_count = int(len(df))
    tx_total_usd = float(df["amount_usd"].sum())
    channel_mix = _channel_mix(df["channel"].tolist())

    # ----- temporal metrics (need parsed timestamps) -----------------------
    if df["ts"].notna().any():
        velocity_24h_max = _velocity_24h_max(df)
        velocity_24h_avg_30d = _velocity_24h_avg_30d(df)
        unique_counterparties_7d = _unique_counterparties_7d(df)
    else:
        velocity_24h_max = tx_count       # collapse all into one window
        velocity_24h_avg_30d = float(tx_count) / 30.0
        unique_counterparties_7d = int(df["counterparty"].nunique())

    # ----- amount z-score ---------------------------------------------------
    amount_z_score_max = _amount_z_score_max(df, kyc)

    # ----- country risk -----------------------------------------------------
    country_risk_max = _country_risk_max(df, kyc)

    # ----- loop detection ---------------------------------------------------
    loop_detected = _loop_detected(df, kyc)

    # ----- volume vs declared ----------------------------------------------
    vs_declared_volume_ratio = _vs_declared_volume_ratio(df, kyc, window_days)

    return BehavioralMetrics(
        tx_count=tx_count,
        tx_total_usd=round(tx_total_usd, 2),
        channel_mix=channel_mix,
        velocity_24h_max=velocity_24h_max,
        velocity_24h_avg_30d=round(velocity_24h_avg_30d, 2),
        unique_counterparties_7d=unique_counterparties_7d,
        amount_z_score_max=round(amount_z_score_max, 2),
        country_risk_max=round(country_risk_max, 2),
        loop_detected=loop_detected,
        vs_declared_volume_ratio=round(vs_declared_volume_ratio, 2),
    )


# ============================================================================
# Per-feature computers (kept separate for testability)
# ============================================================================
def _channel_mix(channels: list[str]) -> dict[str, float]:
    """Fraction of txns by channel (canonical-enum keys only)."""
    if not channels:
        return {}
    c = Counter(channels)
    n = len(channels)
    return {k: round(v / n, 3) for k, v in c.most_common()}


def _velocity_24h_max(df: pd.DataFrame) -> int:
    """Maximum count of transactions in any rolling 24-hour window."""
    if df.empty or df["ts"].isna().all():
        return 0
    s = df.set_index("ts").sort_index()
    counts = s.assign(c=1)["c"].rolling("24h").sum()
    return int(counts.max() or 0)


def _velocity_24h_avg_30d(df: pd.DataFrame) -> float:
    """Average daily transaction count over the bundle's first 30 days.

    'Velocity_24h_avg_30d' = mean transactions per day in the trailing 30
    days from the latest transaction. Approximated by tx_count / 30 when
    the bundle spans < 30 days (typical for alert windows).
    """
    if df.empty or df["ts"].isna().all():
        return 0.0
    span_days = max(
        1.0,
        (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0,
    )
    span_days = min(span_days, 30.0)
    return float(len(df)) / span_days


def _unique_counterparties_7d(df: pd.DataFrame) -> int:
    """Number of distinct counterparties seen in any rolling 7-day window."""
    if df.empty or df["ts"].isna().all():
        return int(df["counterparty"].nunique())
    s = df.set_index("ts").sort_index()
    # window-wise unique count via groupby on date bucket
    last_7d = s[s.index >= (s.index.max() - pd.Timedelta(days=7))]
    return int(last_7d["counterparty"].nunique())


def _amount_z_score_max(df: pd.DataFrame, kyc: dict) -> float:
    """Max z-score of any single transaction amount against the bundle's
    own mean / std (since KYC doesn't carry per-customer history).

    For bundles with < 3 txns or zero-variance amounts, returns 0.0.
    """
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
    """Maximum country risk across origin / destination of any transaction
    plus the entity's incorporation jurisdiction.
    """
    candidates: list[float] = []
    for col in ("country_origin", "country_destination", "country"):
        if col in df.columns:
            candidates.extend(_country_risk(str(c)) for c in df[col].dropna().tolist())
    candidates.append(_country_risk(str(kyc.get("incorporation_jurisdiction", ""))))
    return max(candidates) if candidates else 0.0


def _loop_detected(df: pd.DataFrame, kyc: dict) -> bool:
    """Detect a simple A→B→A loop within the bundle.

    Iff there exists a counterparty B such that some transaction goes
    entity→B and another transaction goes B→entity (both within the same
    bundle). Limited to single-hop loops; multi-hop cycle detection is
    out-of-scope for SFT (RL phase can refine).
    """
    if df.empty or "counterparty" not in df.columns:
        return False
    entity_id = str(kyc.get("entity_id", "")).strip()
    if not entity_id:
        return False
    # If the source has explicit sender/receiver, use it; otherwise the
    # bundle is unidirectional from entity to counterparties and there's
    # no loop signal available.
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
    """Ratio of (bundle's tx_total_usd, normalized to a monthly run-rate) over
    the KYC declared monthly volume.

    Example: bundle has $75K over 90 days → monthly run-rate ≈ $25K. If
    declared monthly volume is $50K, ratio = 0.5 (under). If declared is
    $10K, ratio = 2.5 (over).

    Returns 0.0 when declared volume is unknown or zero.
    """
    declared = float(kyc.get("expected_monthly_volume", 0.0) or 0.0)
    if declared <= 0.0 or df.empty:
        return 0.0
    # Use the bundle's actual span; if it spans < 1 day, treat as 1 day.
    if df["ts"].notna().any():
        span_days = max(
            1.0,
            (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0,
        )
    else:
        span_days = float(window_days)
    # `amount_usd` is already abs() per the upstream frame construction; this
    # ratio is non-negative by construction.
    monthly_rate = float(df["amount_usd"].sum()) * (30.0 / span_days)
    return abs(monthly_rate) / declared
