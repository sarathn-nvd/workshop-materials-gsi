"""Tests for the aux_gate logic.

We exercise input-guard + schema-validation deterministically. The judge
call is bypassed by going straight against the underlying helpers (the
NAT-registered function requires a builder; here we test the rules
directly).
"""
from __future__ import annotations

from aml_app.gating.aux_gate import _input_guard, _validate
from aml_app.skills.prompts import STATUTE_BY_TYPOLOGY


def test_input_guard_behavioral_needs_tx():
    ok, _ = _input_guard("behavioral", n_tx=0, n_pol=0, typology="structuring")
    assert ok is False
    ok, _ = _input_guard("behavioral", n_tx=1, n_pol=0, typology="structuring")
    assert ok is True


def test_input_guard_citation_needs_policy():
    ok, _ = _input_guard("citation", n_tx=10, n_pol=0, typology="structuring")
    assert ok is False
    ok, _ = _input_guard("citation", n_tx=10, n_pol=1, typology="structuring")
    assert ok is True


def test_input_guard_statutory_typology_must_have_statute():
    for typ in STATUTE_BY_TYPOLOGY:
        ok, _ = _input_guard("statutory", n_tx=1, n_pol=0, typology=typ)
        assert ok is True
    ok, reason = _input_guard("statutory", n_tx=1, n_pol=0, typology="none")
    assert ok is False
    assert "statute" in reason


def test_validate_passes_well_formed_behavioral():
    payload = {
        "question": "",
        "summary": "ok",
        "metrics": {
            "tx_count": 1, "tx_total_usd": 100.0,
            "channel_mix": {"cash": 1.0},
            "velocity_24h_max": 1, "velocity_24h_avg_30d": 1.0,
            "unique_counterparties_7d": 1, "amount_z_score_max": 0.0,
            "country_risk_max": 0.1, "loop_detected": False,
            "vs_declared_volume_ratio": 0.0,
        },
        "evidence": "transactions[0]",
    }
    out, err = _validate("behavioral", payload)
    assert err is None
    assert out is not None
    assert out["metrics"]["tx_count"] == 1


def test_validate_accepts_numeric_with_defaults():
    """As long as `answer` is present, the other fields default to '' so
    a brief but coherent finding is not dropped."""
    out, err = _validate("numeric", {"answer": "x"})
    assert err is None
    assert out is not None
    assert out["answer"] == "x"
    assert out["calculation"] == ""
    assert out["evidence"] == ""


def test_validate_rejects_numeric_missing_answer():
    """`answer` remains required; without it the finding is dropped."""
    out, err = _validate("numeric", {"calculation": "1+1=2"})
    assert out is None
    assert err and "answer" in err


def test_validate_rejects_bad_statutory_label():
    out, err = _validate("statutory", {
        "answer": "yes", "label": "MAYBE", "reasoning": "..."
    })
    assert out is None
    assert err
