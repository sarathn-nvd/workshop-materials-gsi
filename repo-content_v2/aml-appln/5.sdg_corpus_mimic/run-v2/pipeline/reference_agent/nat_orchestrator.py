"""NAT-equivalent runtime orchestrator (§2.1–2.5 of the strategy doc).

Processes the demo manifest case-by-case:

  for each case:
    1a. Tools 1+2+3 in parallel  -> transactions[], kyc_profile, sanctions_hits[]
    1b. typology_classifier      -> (typology, activity_descriptor)
    1c. Tools 4+5 in parallel    -> policy_excerpts[], sop_excerpts[]
    2.  4 aux calls in parallel  -> aux_runner.run_all_aux_calls(...)
    3.  guards + schema + judge  -> aux_gate.gate_responses(...)
    4.  final sar_judgment call  -> sar_caller.call_sar(...)
  emit one trace line.

Trace schema is documented inline below; eval.py joins traces against
demo/eval_keys.jsonl.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from pipeline.common.semantic_profile import compute_semantic_profile
from pipeline.common.typology_classifier import classify_typology
from pipeline.config import (
    DEMO_DIR,
    MANIFESTS_DIR,
)
from pipeline.reference_agent.aux_gate import GateDecision, gate_responses
from pipeline.reference_agent.aux_runner import AuxResponse, run_all_aux_calls
from pipeline.reference_agent.sar_caller import SARCallResult, call_sar
from pipeline.reference_agent.tool_clients import (
    EntityNotFound,
    get_kyc,
    get_transactions,
    retrieve,
    screen,
    sop,
)
from pipeline.schemas import (
    SARJudgmentInput,
)

logger = logging.getLogger("pipeline.reference_agent.nat_orchestrator")


# ============================================================================
# Trace record
# ============================================================================
@dataclass
class CaseTrace:
    case_id: str
    entity_id: str
    typology_hypothesis: str
    activity_descriptor: str
    n_transactions: int
    n_sanctions_hits: int
    n_policy_excerpts: int
    n_sop_excerpts: int
    aux_decisions: list[dict]                   # GateDecision dumps
    sar_is_suspicious: bool | None
    sar_narrative: str
    sar_parse_error: str | None
    wall_clock_ms: float
    started_at: str
    finished_at: str
    error: str | None = None


# ============================================================================
# Per-case execution
# ============================================================================
def _run_one_case(case: dict, *, enable_judge: bool = True) -> CaseTrace:
    case_id = case["case_id"]
    entity_id = case["entity_id"]
    win_start = case["investigation_window_start"]
    win_end = case["investigation_window_end"]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        # ---- window 1: Tools 1, 2, 3 in parallel -----------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            f_tx = pool.submit(get_transactions, entity_id, win_start, win_end)
            f_kyc = pool.submit(get_kyc, entity_id)
            f_tx_result = f_tx.result()
            kyc = f_kyc.result()
        # Tool 3 fan-out per unique counterparty
        unique_cps = list({t.counterparty for t in f_tx_result if t.counterparty})[:10]
        hits = []
        if unique_cps:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(screen, unique_cps))
            seen = set()
            for hit_list in results:
                for h in hit_list:
                    if (h.name, h.list) not in seen:
                        seen.add((h.name, h.list))
                        hits.append(h)

        # ---- window 1b: typology classifier (deterministic) ------------
        typology, descriptor = classify_typology(
            f_tx_result, kyc, hits, case.get("trigger_summary"),
        )

        # ---- window 1c: Tools 4, 5 in parallel -------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_pol = pool.submit(retrieve, typology, 3)
            f_sop = pool.submit(sop, typology)
            policy_excerpts = f_pol.result()
            sop_excerpts = f_sop.result()

        # ---- step 2: 4 aux calls in parallel ---------------------------
        aux_responses = run_all_aux_calls(
            typology=typology,
            transactions=f_tx_result,
            kyc_profile=kyc,
            sanctions_hits=hits,
            policy_excerpts=policy_excerpts,
            sop_excerpts=sop_excerpts,
        )

        # ---- step 3: guards + schema + judge ---------------------------
        af, decisions = gate_responses(
            aux_responses,
            transactions=f_tx_result,
            kyc_profile=kyc,
            policy_excerpts=policy_excerpts,
            typology=typology,
            enable_judge=enable_judge,
        )

        # ---- step 4: final sar_judgment --------------------------------
        # Compute the semantic profile hints the SFT model was trained
        # with (mirrors SFT pair_ground stage: every shipped record has
        # _regulatory_frame + _typology_inferred + _decision_target).
        sem = compute_semantic_profile(
            transactions=f_tx_result,
            kyc_profile=kyc,
            source_typology=typology,
            sanctions_pep_hits=hits,
        )
        # _decision_target derived from the classifier's typology guess:
        # `none` → not_suspicious; anything else → suspicious. This is the
        # runtime analogue of SFT's gold-label-derived target — the model
        # uses it as a prior and can override.
        decision_target = (
            "not_suspicious" if sem.typology_inferred == "none" else "suspicious"
        )

        bundle = SARJudgmentInput(
            transactions=f_tx_result,
            kyc_profile=kyc,
            sanctions_pep_hits=hits,
            policy_excerpts=policy_excerpts,
            sop_excerpts=sop_excerpts,
            auxiliary_findings=af if not af.is_empty() else None,
        )
        sar_result: SARCallResult = call_sar(
            bundle,
            regulatory_frame=sem.regulatory_frame,
            typology_inferred=sem.typology_inferred,
            decision_target=decision_target,
        )

        is_susp = sar_result.output.is_suspicious if sar_result.output else None
        narrative = sar_result.output.suspicious_activity_report if sar_result.output else ""

        return CaseTrace(
            case_id=case_id,
            entity_id=entity_id,
            typology_hypothesis=typology,
            activity_descriptor=descriptor,
            n_transactions=len(f_tx_result),
            n_sanctions_hits=len(hits),
            n_policy_excerpts=len(policy_excerpts),
            n_sop_excerpts=len(sop_excerpts),
            aux_decisions=[asdict(d) for d in decisions],
            sar_is_suspicious=is_susp,
            sar_narrative=narrative,
            sar_parse_error=sar_result.parse_error,
            wall_clock_ms=(time.time() - t0) * 1000.0,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except EntityNotFound as e:
        return CaseTrace(
            case_id=case_id, entity_id=entity_id,
            typology_hypothesis="ERROR",
            activity_descriptor="",
            n_transactions=0, n_sanctions_hits=0,
            n_policy_excerpts=0, n_sop_excerpts=0,
            aux_decisions=[],
            sar_is_suspicious=None, sar_narrative="", sar_parse_error=None,
            wall_clock_ms=(time.time() - t0) * 1000.0,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"EntityNotFound: {e}",
        )
    except Exception as e:
        logger.exception("case %s failed", case_id)
        return CaseTrace(
            case_id=case_id, entity_id=entity_id,
            typology_hypothesis="ERROR",
            activity_descriptor="",
            n_transactions=0, n_sanctions_hits=0,
            n_policy_excerpts=0, n_sop_excerpts=0,
            aux_decisions=[],
            sar_is_suspicious=None, sar_narrative="", sar_parse_error=None,
            wall_clock_ms=(time.time() - t0) * 1000.0,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(e).__name__}: {e}",
        )


# ============================================================================
# Batch runner
# ============================================================================
def run_batch(
    manifest_path: str | None = None,
    traces_path: str | None = None,
    *,
    case_concurrency: int = 4,
    enable_judge: bool = True,
    limit: int | None = None,
) -> int:
    manifest_path = manifest_path or str(DEMO_DIR / "manifest.jsonl")
    traces_path = traces_path or str(MANIFESTS_DIR / "agent_rollout_traces.jsonl")
    logger.info("Reading manifest: %s", manifest_path)
    cases: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if limit is not None:
        cases = cases[:limit]
    logger.info("%d cases to process (concurrency=%d, judge=%s)",
                len(cases), case_concurrency, enable_judge)

    traces: list[CaseTrace] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=case_concurrency) as pool:
        futures = {
            pool.submit(_run_one_case, c, enable_judge=enable_judge): c
            for c in cases
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            trace = fut.result()
            traces.append(trace)
            verdict = (
                "+" if trace.sar_is_suspicious is True
                else "-" if trace.sar_is_suspicious is False
                else "?"
            )
            logger.info(
                "%d/%d [%s] %s | typology=%s | %d tx, %d hits, %d pol, %d sop | %dms",
                i, len(cases), verdict, trace.case_id, trace.typology_hypothesis,
                trace.n_transactions, trace.n_sanctions_hits,
                trace.n_policy_excerpts, trace.n_sop_excerpts,
                int(trace.wall_clock_ms),
            )

    elapsed = time.time() - t0
    logger.info("Batch complete in %.1fs (%.1fs/case avg)",
                elapsed, elapsed / max(1, len(traces)))

    # Persist traces (sorted by case_id for stable output)
    traces.sort(key=lambda t: t.case_id)
    with open(traces_path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(asdict(t)) + "\n")
    logger.info("Wrote traces: %s (%d records)", traces_path, len(traces))
    return 0


# ============================================================================
# CLI
# ============================================================================
def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline.reference_agent.nat_orchestrator",
        description="Run NAT-equivalent batch over demo manifest.",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--traces")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-Judge step (debug / cost-saving).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of cases (debug).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    return run_batch(
        manifest_path=args.manifest,
        traces_path=args.traces,
        case_concurrency=args.concurrency,
        enable_judge=not args.no_judge,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
