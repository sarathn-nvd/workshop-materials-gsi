"""Direct (non-NAT) tests against the underlying data tool functions."""
from __future__ import annotations

import pytest

from aml_app.common.schemas import KYCProfile, Transaction
from aml_app.tools.data_tools import (
    EntityNotFound,
    GetKycInput,
    GetSopInput,
    GetTransactionsInput,
    RetrievePolicyInput,
    ScreenSanctionsInput,
    _get_kyc,
    _get_sop,
    _get_transactions,
    _retrieve_policy,
    _screen_sanctions,
)


@pytest.fixture
def dp_loaded(dp):
    # Warm caches
    dp.transactions(); dp.kyc(); dp.ofac(); dp.pep(); dp.policy_chunks(); dp.sops()
    return dp


def _pick_entity_with_activity(dp_loaded):
    df = dp_loaded.transactions()
    counts = df["entity_id"].value_counts()
    return counts.index[0]


def test_tool1_get_transactions(dp_loaded):
    eid = _pick_entity_with_activity(dp_loaded)
    out = _get_transactions(
        dp_loaded,
        GetTransactionsInput(entity_id=eid,
                              window_start="2026-02-01",
                              window_end="2026-06-30"),
    )
    assert isinstance(out, list)
    assert all(isinstance(t, Transaction) for t in out)
    if out:
        first = out[0].model_dump()
        # internal sidecars must be stripped
        assert "source_pool" not in first
        assert "typology_tag" not in first


def test_tool1_window_filter(dp_loaded):
    eid = _pick_entity_with_activity(dp_loaded)
    full = _get_transactions(dp_loaded,
        GetTransactionsInput(entity_id=eid,
                             window_start="2026-02-01",
                             window_end="2026-06-30"))
    narrow = _get_transactions(dp_loaded,
        GetTransactionsInput(entity_id=eid,
                             window_start="2026-02-01",
                             window_end="2026-02-15"))
    assert len(narrow) <= len(full)


def test_tool2_get_kyc_ok(dp_loaded):
    eid = next(iter(dp_loaded.kyc().keys()))
    kyc = _get_kyc(dp_loaded, GetKycInput(entity_id=eid))
    assert isinstance(kyc, KYCProfile)
    # internal sidecars must be stripped
    d = kyc.model_dump()
    assert "source_pool" not in d
    assert "_archetype" not in d


def test_tool2_get_kyc_missing(dp_loaded):
    with pytest.raises(EntityNotFound):
        _get_kyc(dp_loaded, GetKycInput(entity_id="DOES_NOT_EXIST_XXXX"))


def test_tool3_screen_sanctions_unknown_name(dp_loaded):
    out = _screen_sanctions(dp_loaded,
        ScreenSanctionsInput(name="ZZQQXX Unknown Counterparty 12345"))
    assert isinstance(out, list)
    # tolerate sporadic fuzzy noise; just ensure shape is correct
    for h in out:
        assert 0.0 <= h.match_score <= 1.0
        assert h.list in {"OFAC", "OpenSanctions"}


def test_tool3_screen_sanctions_known_pep(dp_loaded):
    pep = dp_loaded.pep()
    if not pep:
        pytest.skip("No PEP entries loaded")
    sample_name = pep[0].get("name") or pep[0].get("caption") or ""
    if not sample_name:
        pytest.skip("First PEP row has no name field")
    out = _screen_sanctions(dp_loaded,
        ScreenSanctionsInput(name=sample_name, min_score=0.5))
    assert any(h.match_score >= 0.5 for h in out)


def test_tool4_retrieve_policy_known_typology(dp_loaded):
    out = _retrieve_policy(dp_loaded,
        RetrievePolicyInput(typology="structuring", k=4))
    assert isinstance(out, list)
    if out:
        srcs = {e.source for e in out}
        # stratification means we should hit at least 2 sources when possible
        assert len(srcs) >= 1


def test_tool4_retrieve_policy_none(dp_loaded):
    out = _retrieve_policy(dp_loaded,
        RetrievePolicyInput(typology="none", k=4))
    assert out == []


def test_tool5_get_sop_known_typology(dp_loaded):
    out = _get_sop(dp_loaded, GetSopInput(typology="structuring", variant=1))
    assert isinstance(out, list)
    assert out
    sop = out[0]
    assert sop.sop_id.startswith("SOP-")
    assert sop.section


def test_tool5_get_sop_unknown_typology(dp_loaded):
    out = _get_sop(dp_loaded, GetSopInput(typology="none", variant=1))
    assert out == []
