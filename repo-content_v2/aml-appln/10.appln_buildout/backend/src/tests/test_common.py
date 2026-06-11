"""Tests for pure-Python common helpers."""
from __future__ import annotations

from aml_app.common.behavioral_features import (
    compute_behavioral_metrics,
    country_risk,
    normalize_channel,
)
from aml_app.common.schemas import (
    BehavioralFinding,
    BehavioralMetrics,
    KYCProfile,
    SARJudgmentInput,
    SARJudgmentOutput,
    SemanticProfile,
    Transaction,
)
from aml_app.common.semantic_profile import (
    compute_semantic_profile,
    decision_target_from,
    derive_decision_target_calibrated,
)
from aml_app.common.typology_classifier import classify_typology


def test_schemas_roundtrip_minimal_sar_bundle():
    bundle = SARJudgmentInput(
        transactions=[Transaction(date="2026-04-01", amount=9500.0, currency="USD",
                                  counterparty="Cash deposit, Branch #14",
                                  channel="cash")],
        kyc_profile=KYCProfile(entity_id="CUST_1", entity_type="business",
                               expected_monthly_volume=50000,
                               business_purpose="Retail jewelry store",
                               risk_rating="medium",
                               incorporation_jurisdiction="US-NY"),
    )
    dumped = bundle.model_dump()
    reborn = SARJudgmentInput.model_validate(dumped)
    assert reborn.kyc_profile.entity_id == "CUST_1"
    assert reborn.task_type == "sar_judgment"


def test_normalize_channel_buckets():
    assert normalize_channel("ATM") == "cash"
    assert normalize_channel("check") == "cheque"
    assert normalize_channel("transfer") == "wire"
    assert normalize_channel("WIRE") == "wire"
    assert normalize_channel("") == "wire"


def test_country_risk_lookup():
    assert country_risk("US") < country_risk("KP")
    assert country_risk("US-NY") == country_risk("US")
    assert country_risk("UNKNOWN_COUNTRY_CODE") > 0.0


def test_behavioral_metrics_empty_bundle():
    metrics = compute_behavioral_metrics([], {"expected_monthly_volume": 0})
    assert metrics.tx_count == 0
    assert metrics.tx_total_usd == 0.0
    assert metrics.loop_detected is False


def test_behavioral_metrics_structuring_pattern():
    txs = [
        {"date": "2026-04-01", "amount": 9500.0, "currency": "USD",
         "counterparty": "Cash deposit, Branch #14", "channel": "cash"},
        {"date": "2026-04-02", "amount": 9700.0, "currency": "USD",
         "counterparty": "Cash deposit, Branch #21", "channel": "cash"},
        {"date": "2026-04-03", "amount": 9300.0, "currency": "USD",
         "counterparty": "Cash deposit, Branch #07", "channel": "cash"},
    ]
    kyc = KYCProfile(entity_id="CUST_2", entity_type="business",
                     expected_monthly_volume=10000,
                     business_purpose="Retail jewelry",
                     risk_rating="medium",
                     incorporation_jurisdiction="US-NY")
    m = compute_behavioral_metrics(txs, kyc)
    assert m.tx_count == 3
    assert m.tx_total_usd > 28000.0
    assert m.channel_mix.get("cash", 0.0) == 1.0
    assert m.vs_declared_volume_ratio > 0.0


def test_classify_typology_structuring():
    txs = [
        Transaction(date="2026-04-01", amount=9500.0, currency="USD",
                    counterparty="Cash deposit, Branch #14", channel="cash"),
        Transaction(date="2026-04-02", amount=9700.0, currency="USD",
                    counterparty="Cash deposit, Branch #21", channel="cash"),
        Transaction(date="2026-04-03", amount=9300.0, currency="USD",
                    counterparty="Cash deposit, Branch #07", channel="cash"),
    ]
    kyc = KYCProfile(entity_id="CUST_3", entity_type="business",
                     expected_monthly_volume=20000,
                     business_purpose="Convenience store",
                     risk_rating="medium",
                     incorporation_jurisdiction="US-NY")
    typ, desc = classify_typology(txs, kyc, [], None)
    assert typ in {"structuring", "smurfing"}
    assert "cash" in desc.lower() or "sub-$10K" in desc


def test_classify_typology_none_on_empty():
    kyc = KYCProfile(entity_id="X", entity_type="individual",
                     expected_monthly_volume=1000,
                     business_purpose="Wage earner",
                     risk_rating="low", incorporation_jurisdiction="US-CA")
    typ, _ = classify_typology([], kyc, [], None)
    assert typ == "none"


def test_semantic_profile_ctr_remap_to_layering():
    # Source says "structuring" but every channel is wire \u2192 should remap.
    txs = [Transaction(date="2026-04-01", amount=9000.0, currency="USD",
                       counterparty="X", channel="wire")]
    kyc = KYCProfile(entity_id="X", entity_type="business",
                     expected_monthly_volume=20000,
                     business_purpose="Trading",
                     risk_rating="medium",
                     incorporation_jurisdiction="US-CA")
    sp = compute_semantic_profile(txs, kyc, "structuring", [])
    assert sp.typology_inferred == "layering"
    assert sp.regulatory_frame == "layering_passthrough"


def test_semantic_profile_sanctions_override():
    txs = [Transaction(date="2026-04-01", amount=1000.0, currency="USD",
                       counterparty="X", channel="wire")]
    kyc = KYCProfile(entity_id="X", entity_type="business",
                     expected_monthly_volume=20000,
                     business_purpose="Trading",
                     risk_rating="medium",
                     incorporation_jurisdiction="US-CA")
    # Hit at 0.85 — under the tightened 0.90 override threshold — must NOT flip.
    sp = compute_semantic_profile(txs, kyc, "layering",
                                  [{"name": "ACME", "list": "OFAC", "match_score": 0.85}])
    assert sp.regulatory_frame == "layering_passthrough"

    # Hit at 0.95 — above the override threshold — must flip to sanctions.
    sp2 = compute_semantic_profile(txs, kyc, "layering",
                                   [{"name": "ACME", "list": "OFAC", "match_score": 0.95}])
    assert sp2.regulatory_frame == "sanctions"


def test_decision_target_rule():
    assert decision_target_from("none") == "not_suspicious"
    assert decision_target_from("structuring") == "suspicious"


# ---------------------------------------------------------------------------
# Calibrated decision_target derivation
# ---------------------------------------------------------------------------
def _benign_sem(frame="benign", typology="none"):
    return SemanticProfile(
        channel_mix={"wire": 1.0}, cash_present=False,
        regulatory_frame=frame, declared_volume_band="match",
        geo_risk="low", typology_inferred=typology,
    )


def test_calibrated_benign_low_risk_returns_not_suspicious():
    """Frame=benign + low risk + clean signals -> hard not_suspicious."""
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(),
        kyc={"risk_rating": "low"},
        sanctions_pep_hits=[],
        behavioral_metrics={"peel_chain_detected": False},
        auxiliary_findings={},
    )
    assert out == "not_suspicious"


def test_calibrated_layering_low_risk_clean_aux_returns_not_suspicious():
    """The exact pattern of our 22 worst FPs."""
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(frame="layering_passthrough", typology="layering"),
        kyc={"risk_rating": "low"},
        sanctions_pep_hits=[],
        behavioral_metrics={"peel_chain_detected": False,
                             "counterparty_concentration_top1": 0.3},
        auxiliary_findings={"behavioral": [{"red_flags": []}]},
    )
    assert out == "not_suspicious"


def test_calibrated_confident_sanctions_returns_suspicious():
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(frame="sanctions", typology="layering"),
        kyc={"risk_rating": "medium"},
        sanctions_pep_hits=[{"name": "ACME Logistics", "list": "OFAC",
                              "match_score": 0.95}],
        behavioral_metrics={},
        auxiliary_findings={},
    )
    assert out == "suspicious"


def test_calibrated_peel_chain_returns_suspicious():
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(frame="layering_passthrough", typology="layering"),
        kyc={"risk_rating": "medium"},
        sanctions_pep_hits=[],
        behavioral_metrics={"peel_chain_detected": True},
        auxiliary_findings={},
    )
    assert out == "suspicious"


def test_calibrated_ambiguous_returns_none():
    """High-risk customer in a mid-base-rate frame with aux signal -> omit."""
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(frame="layering_passthrough", typology="layering"),
        kyc={"risk_rating": "high"},
        sanctions_pep_hits=[],
        behavioral_metrics={"counterparty_concentration_top1": 0.5},
        auxiliary_findings={"behavioral": [{"red_flags": ["unusual cadence"]}]},
    )
    assert out is None


def test_calibrated_aux_flag_blocks_not_suspicious():
    """Even on a 'benign' frame, an aux red flag must veto not_suspicious."""
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(),
        kyc={"risk_rating": "low"},
        sanctions_pep_hits=[],
        behavioral_metrics={},
        auxiliary_findings={"numeric": [{"red_flags": ["arithmetic inconsistency"]}]},
    )
    assert out is None


def test_calibrated_high_risk_blocks_not_suspicious():
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(),
        kyc={"risk_rating": "high"},
        sanctions_pep_hits=[],
        behavioral_metrics={},
        auxiliary_findings={},
    )
    assert out is None


def test_calibrated_common_name_pep_is_filtered():
    """A PEP hit on 'Michael Brown' (common name) must NOT trigger suspicious."""
    out = derive_decision_target_calibrated(
        semantic_profile=_benign_sem(),
        kyc={"risk_rating": "low"},
        sanctions_pep_hits=[{"name": "Michael Brown", "list": "OpenSanctions",
                              "match_score": 0.95}],
        behavioral_metrics={},
        auxiliary_findings={},
    )
    # Common-name PEP filtered -> no structural flag, so we still get
    # clean signals (frame=benign, no risk, clean aux).
    assert out == "not_suspicious"


def test_sar_judgment_output_validation():
    SARJudgmentOutput.model_validate({"is_suspicious": True,
                                      "suspicious_activity_report": "..."})
    SARJudgmentOutput.model_validate({"is_suspicious": False,
                                      "suspicious_activity_report": ""})
