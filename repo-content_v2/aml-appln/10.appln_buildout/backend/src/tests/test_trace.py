"""CaseTrace round-trip + risk_score sanity."""
from __future__ import annotations

import json

from aml_app.utils.risk_score import risk_score
from aml_app.workflow.trace import CaseTrace, read_trace, write_trace


def test_case_trace_minimal(tmp_path):
    tr = CaseTrace(case_id="DEMO_TEST_001", entity_id="X",
                   sar_is_suspicious=True,
                   sar_narrative="Suspicious activity is identified...")
    path = write_trace(tr, tmp_path)
    assert path.exists()
    obj = json.loads(path.read_text())
    assert obj["case_id"] == "DEMO_TEST_001"
    assert obj["sar_is_suspicious"] is True

    re_read = read_trace("DEMO_TEST_001", tmp_path)
    assert re_read["sar_narrative"].startswith("Suspicious")


def test_case_trace_missing_id(tmp_path):
    assert read_trace("NOPE", tmp_path) is None


def test_risk_score_monotone():
    low = risk_score("low", n_sanctions_hits=0)
    med = risk_score("medium", n_sanctions_hits=0)
    hi = risk_score("high", n_sanctions_hits=0)
    eh = risk_score("enhanced", n_sanctions_hits=0)
    assert low < med < hi < eh


def test_risk_score_with_sanctions_and_geo():
    low_no = risk_score("low", n_sanctions_hits=0, country_risk_max=0.0)
    low_yes = risk_score("low", n_sanctions_hits=2, country_risk_max=0.7)
    assert low_yes > low_no
    assert 0 <= low_yes <= 100
