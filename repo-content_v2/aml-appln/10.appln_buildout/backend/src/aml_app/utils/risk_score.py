"""Derived risk score \u2014 hand-tuned weighted blend.

Inputs: kyc.risk_rating, behavioral metrics (z-score, country risk, volume
ratio), sanctions hit count. Output: 0-100.
"""
from __future__ import annotations


_RR_WEIGHTS = {"low": 5, "medium": 25, "high": 50, "enhanced": 75, "prohibited": 95}


def risk_score(
    kyc_risk_rating: str,
    n_sanctions_hits: int = 0,
    country_risk_max: float = 0.0,
    vs_declared_volume_ratio: float = 0.0,
    amount_z_score_max: float = 0.0,
) -> int:
    base = _RR_WEIGHTS.get(kyc_risk_rating, 25)
    sanctions = min(40, 15 * n_sanctions_hits)
    geo = int(country_risk_max * 30)
    volume = 0
    if vs_declared_volume_ratio >= 4:
        volume = 25
    elif vs_declared_volume_ratio >= 2:
        volume = 15
    elif vs_declared_volume_ratio >= 1.5:
        volume = 8
    z = 0
    if amount_z_score_max >= 4:
        z = 15
    elif amount_z_score_max >= 2.5:
        z = 8

    score = base + sanctions + geo + volume + z
    return int(max(0, min(100, score)))
