"""Tests for the v3 passage/prompt unification.

Validates two invariants that must hold for the trained model to see the
same inputs at SFT time and at production-agent inference time:

  1. RENDERER PARITY — `scripts.common.passage_render.render_bundle_passage`
     produces a passage shape byte-compatible with the backend's
     `aml_app.workflow.investigate_case._render_bundle_passage`.

  2. UNIFIED SYSTEM PROMPTS — Stage 6, Stage A2, Stage A3, Stage A3b, and
     the backend's `aml_app.skills.prompts` all reference the SAME aux
     system-prompt strings per skill.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SDG_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SDG_ROOT))

from scripts.common.passage_render import (   # noqa: E402
    render_bundle_passage,
    render_citation_passage,
)
from scripts.common.aux_prompts import (      # noqa: E402
    NUMERIC_SYSTEM, NUMERIC_USER_BUNDLE, NUMERIC_USER_PASSAGE,
    CITATION_SYSTEM, CITATION_USER_EXCERPT, CITATION_USER_CHUNK,
    STATUTORY_SYSTEM, STATUTORY_USER,
    BEHAVIORAL_SYSTEM, BEHAVIORAL_USER,
)


# ============================================================================
# Renderer shape contract
# ============================================================================
class TestRenderBundlePassage:
    def _sample(self):
        txs = [
            {"date": "2026-02-04", "channel": "ach", "amount": 6255.32,
             "currency": "USD", "counterparty": "TRX Holdings LLC", "notes": ""},
            {"date": "2026-02-05", "channel": "wire", "amount": 12000.0,
             "currency": "USD", "counterparty": "ACME Corp", "notes": "invoice"},
        ]
        kyc = {"entity_id": "SYN_test", "entity_type": "business",
               "expected_monthly_volume": 35486.0,
               "business_purpose": "B2B services",
               "risk_rating": "low",
               "incorporation_jurisdiction": "US-CA"}
        return txs, kyc

    def test_starts_with_transactions_block(self):
        txs, kyc = self._sample()
        out = render_bundle_passage(txs, kyc)
        assert out.startswith("[transactions]\n"), out[:80]

    def test_contains_kyc_block(self):
        txs, kyc = self._sample()
        out = render_bundle_passage(txs, kyc)
        assert "\n[kyc_profile]\n" in out, "missing [kyc_profile] block"

    def test_kyc_keys_in_canonical_order(self):
        txs, kyc = self._sample()
        out = render_bundle_passage(txs, kyc)
        kyc_section = out.split("[kyc_profile]", 1)[1]
        lines = [l.split(":", 1)[0].strip() for l in kyc_section.strip().splitlines()]
        assert lines == [
            "entity_id", "entity_type", "expected_monthly_volume",
            "business_purpose", "risk_rating", "incorporation_jurisdiction",
        ], lines

    def test_tx_columns_in_canonical_order(self):
        txs, kyc = self._sample()
        out = render_bundle_passage(txs, kyc)
        header = out.splitlines()[1]
        for col in ("date", "channel", "amount", "ccy", "counterparty", "notes"):
            assert col in header, f"missing column {col!r} in header: {header!r}"
        # `date` precedes `channel` precedes `amount`
        i_date = header.index("date")
        i_chan = header.index("channel")
        i_amt = header.index("amount")
        assert i_date < i_chan < i_amt, header

    def test_truncation_marker_when_over_cap(self):
        kyc = {"entity_id": "X", "entity_type": "business",
               "expected_monthly_volume": 0,
               "business_purpose": "",
               "risk_rating": "low",
               "incorporation_jurisdiction": "US"}
        txs = [{"date": "2026-01-01", "channel": "ach", "amount": 100.0,
                "currency": "USD", "counterparty": "a", "notes": ""}
               for _ in range(60)]
        out = render_bundle_passage(txs, kyc, max_transactions=20)
        assert "[... 40 more transactions not shown ...]" in out

    def test_empty_transactions_is_handled(self):
        out = render_bundle_passage([], {"entity_id": "X", "entity_type": "p",
                                          "expected_monthly_volume": 0,
                                          "business_purpose": "", "risk_rating": "low",
                                          "incorporation_jurisdiction": "US"})
        assert "(no transactions)" in out
        assert "[kyc_profile]" in out


# ============================================================================
# Renderer parity with backend
# ============================================================================
class TestBackendRendererParity:
    def test_byte_identical(self):
        """The backend's _render_bundle_passage and the SDG's
        render_bundle_passage must produce identical output for any
        canonical input. If this test ever fails, the trained model is
        seeing a different passage shape at inference than at training,
        and the v3 alignment is broken."""
        backend_path = (REPO_ROOT / "10.appln_buildout/backend/src/aml_app/"
                        "workflow/investigate_case.py")
        if not backend_path.exists():
            pytest.skip(f"backend not present at {backend_path}")
        backend_src = backend_path.read_text()
        # Extract the renderer block (_TX_FIELDS through end of _render_bundle_passage)
        m = re.search(
            r'(_TX_FIELDS\s*=.*?)\n# -------',
            backend_src, flags=re.DOTALL,
        )
        assert m, "could not locate backend renderer block"
        ns: dict = {}
        exec(m.group(1), ns)  # noqa: S102
        backend_render = ns["_render_bundle_passage"]

        for case in [
            ([{"date": "2026-02-04", "channel": "ach", "amount": 100.0,
               "currency": "USD", "counterparty": "X", "notes": ""}],
             {"entity_id": "E1", "entity_type": "business",
              "expected_monthly_volume": 1000.0,
              "business_purpose": "B2B", "risk_rating": "low",
              "incorporation_jurisdiction": "US"}),
            ([],
             {"entity_id": "E2", "entity_type": "individual",
              "expected_monthly_volume": 0,
              "business_purpose": "", "risk_rating": "high",
              "incorporation_jurisdiction": "US-NY"}),
            ([{"date": "2026-03-01", "channel": "wire", "amount": 1.5e6,
               "currency": "EUR", "counterparty": "Big Corp Ltd",
               "notes": "with extra long note text that should get truncated"}],
             {"entity_id": "E3", "entity_type": "business",
              "expected_monthly_volume": 500000,
              "business_purpose": "Import/export",
              "risk_rating": "medium", "incorporation_jurisdiction": "DE"}),
        ]:
            sdg_out = render_bundle_passage(*case)
            backend_out = backend_render(*case)
            assert sdg_out == backend_out, (
                f"renderer mismatch:\nSDG:\n{sdg_out!r}\nBACKEND:\n{backend_out!r}"
            )


# ============================================================================
# Unified system-prompt identity
# ============================================================================
class TestPromptUnification:
    def test_stage_6_numeric_is_unified(self):
        from scripts.non_auxiliary.stage_6_aux_findings.prompts import (
            NUMERIC_SYSTEM as s6
        )
        assert s6 is NUMERIC_SYSTEM

    def test_stage_6_citation_is_unified(self):
        from scripts.non_auxiliary.stage_6_aux_findings.prompts import (
            CITATION_SYSTEM as s6
        )
        assert s6 is CITATION_SYSTEM

    def test_stage_6_statutory_is_unified(self):
        from scripts.non_auxiliary.stage_6_aux_findings.prompts import (
            STATUTORY_SYSTEM as s6
        )
        assert s6 is STATUTORY_SYSTEM

    def test_stage_a2_finqa_is_unified_numeric(self):
        from scripts.auxiliary.stage_a2_generate.prompts import (
            FINQA_FIX1_SYSTEM as s
        )
        assert s is NUMERIC_SYSTEM

    def test_stage_a2_ffiec_is_unified_citation(self):
        from scripts.auxiliary.stage_a2_generate.prompts import (
            FFIEC_QA_SYSTEM as s
        )
        assert s is CITATION_SYSTEM

    def test_stage_a2_legalbench_is_unified_statutory(self):
        from scripts.auxiliary.stage_a2_generate.prompts import (
            LEGALBENCH_FIX3_SYSTEM as s
        )
        assert s is STATUTORY_SYSTEM

    def test_stage_a3_system_by_task_is_unified(self):
        from scripts.auxiliary.stage_a3_assemble.prompts import SYSTEM_BY_TASK
        assert SYSTEM_BY_TASK["auxiliary_numeric"] is NUMERIC_SYSTEM
        assert SYSTEM_BY_TASK["auxiliary_citation"] is CITATION_SYSTEM
        assert SYSTEM_BY_TASK["auxiliary_statutory"] is STATUTORY_SYSTEM

    def test_stage_a3b_behavioral_is_unified(self):
        from scripts.auxiliary.stage_a3b_behavioral.prompts import (
            AUX_BEHAVIORAL_SYSTEM
        )
        assert AUX_BEHAVIORAL_SYSTEM is BEHAVIORAL_SYSTEM


class TestBackendPromptParity:
    """The backend ships its own copy of the unified prompts (so it has no
    cross-repo Python import dependency). Strings must be byte-identical to
    the SDG canonical strings."""

    def test_backend_numeric_matches(self):
        backend_path = (REPO_ROOT / "10.appln_buildout/backend/src/aml_app/"
                        "skills/prompts.py")
        if not backend_path.exists():
            pytest.skip(f"backend not present at {backend_path}")
        src = backend_path.read_text()
        ns: dict = {}
        # Define an empty `reviewer_prompt` so exec succeeds
        exec(compile(src, str(backend_path), "exec"), ns)  # noqa: S102
        assert ns["AUX_NUMERIC_SYSTEM_PROMPT"] == NUMERIC_SYSTEM
        assert ns["AUX_CITATION_SYSTEM_PROMPT"] == CITATION_SYSTEM
        assert ns["AUX_STATUTORY_SYSTEM_PROMPT"] == STATUTORY_SYSTEM
        assert ns["AUX_BEHAVIORAL_SYSTEM_PROMPT"] == BEHAVIORAL_SYSTEM


# ============================================================================
# Output-shape contract embedded in the prompts
# ============================================================================
class TestUnifiedPromptOutputShape:
    def test_numeric_prompt_specifies_4_keys(self):
        # The unified prompt MUST tell the model to output the 4-key shape.
        assert "{question, answer, calculation, evidence}" in NUMERIC_SYSTEM

    def test_citation_prompt_specifies_3_keys(self):
        assert "{question, answer, evidence_span}" in CITATION_SYSTEM

    def test_statutory_prompt_specifies_4_keys(self):
        assert "{question, answer, label, reasoning}" in STATUTORY_SYSTEM

    def test_behavioral_prompt_specifies_4_keys(self):
        assert "{question, summary, metrics, evidence}" in BEHAVIORAL_SYSTEM

    def test_numeric_prompt_describes_both_input_forms(self):
        # Input form (a) — bundle
        assert "[transactions]" in NUMERIC_SYSTEM
        assert "[kyc_profile]" in NUMERIC_SYSTEM
        assert "transactions[i..j]" in NUMERIC_SYSTEM
        # Input form (b) — passage
        assert "passage" in NUMERIC_SYSTEM
        assert "Table" in NUMERIC_SYSTEM or "table" in NUMERIC_SYSTEM

    def test_citation_prompt_describes_both_input_forms(self):
        assert "[policy_excerpt]" in CITATION_SYSTEM
        # raw chunk form mentioned
        assert "raw regulatory chunk" in CITATION_SYSTEM


# ============================================================================
# Evidence-list coercion — gemma sometimes emits `evidence` as a JSON list
# of locators rather than a single string. The schemas accept either form
# and coerce to a single string. Without this, all such records hard-fail
# Pydantic validation in Stage A3 and get dropped (observed 22/22 FinQA
# numerics in the v3-unified smoke).
# ============================================================================
class TestEvidenceCoercion:
    def test_numeric_finding_accepts_evidence_list(self):
        from scripts.schemas import NumericFinding
        f = NumericFinding(
            answer="net revenue $94M",
            calculation="1. 2015=$5829. 2. 2014=$5735. 3. diff=$94.",
            evidence=["Table row '2015 net revenue' col 'amount'=$5829",
                      "Table row '2014 net revenue' col 'amount'=$5735"],
        )
        assert isinstance(f.evidence, str)
        assert "Table row '2015 net revenue'" in f.evidence
        assert "Table row '2014 net revenue'" in f.evidence
        assert "; " in f.evidence

    def test_numeric_finding_accepts_evidence_string(self):
        from scripts.schemas import NumericFinding
        f = NumericFinding(answer="x", calculation="y", evidence="transactions[0..3]")
        assert f.evidence == "transactions[0..3]"

    def test_numeric_finding_accepts_empty_evidence_list(self):
        from scripts.schemas import NumericFinding
        f = NumericFinding(answer="x", calculation="y", evidence=[])
        assert f.evidence == ""

    def test_citation_finding_accepts_evidence_span_list(self):
        from scripts.schemas import CitationFinding
        f = CitationFinding(
            answer="x",
            evidence_span=["span one", "span two"],
        )
        assert isinstance(f.evidence_span, str)
        assert "span one" in f.evidence_span
        assert "span two" in f.evidence_span

    def test_behavioral_finding_accepts_evidence_list(self):
        from scripts.schemas import BehavioralFinding, BehavioralMetrics
        f = BehavioralFinding(
            summary="x" * 100,
            metrics=BehavioralMetrics(
                tx_count=1, tx_total_usd=1.0,
                channel_mix={"wire": 1.0},
                velocity_24h_max=1, velocity_24h_avg_30d=1.0,
                unique_counterparties_7d=1, amount_z_score_max=0.0,
                country_risk_max=0.0, loop_detected=False,
                vs_declared_volume_ratio=1.0,
            ),
            evidence=["transactions[0..3]", "kyc_profile.expected_monthly_volume"],
        )
        assert isinstance(f.evidence, str)
        assert "transactions[0..3]" in f.evidence


# ============================================================================
# Citation-passage renderer
# ============================================================================
class TestRenderCitationPassage:
    def test_starts_with_policy_excerpt_marker(self):
        out = render_citation_passage({
            "source": "FFIEC BSA/AML Manual",
            "section": "Customer Identification Program",
            "url": "https://bsaaml.ffiec.gov/",
            "text": "A bank must implement a written CIP...",
        })
        assert out.startswith("[policy_excerpt]\n")
        assert "source: FFIEC BSA/AML Manual" in out
        assert "section: Customer Identification Program" in out
        assert "A bank must implement a written CIP..." in out
