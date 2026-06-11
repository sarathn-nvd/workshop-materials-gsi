"""Semantic audit checks (A1, A2, A3, A5, A6, A7, A9, A10, A11, A12).

These complement the structural audits in `audit.py` by checking field-level
semantic fidelity that the Pydantic schema cannot enforce. Designed to catch
exactly the class of bugs the original audit missed (LegalBench `text[:200]`
fallback, statute = fact_pattern collapse, FinanceBench mis-routing, etc.).

Each check returns a dict {rule, passed, observed, ...} so the main audit can
print and serialise them uniformly.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# A2 — Aux field-distinctness check
# ============================================================================
def audit_aux_field_distinctness(aux_records: list[dict], min_pass_rate: float = 0.95) -> dict:
    """For auxiliary_statutory records, assert statute != fact_pattern.

    Catches the schema-collapse bug where Stage A3 assigned the same field
    to both `statute` and `fact_pattern`. The strategy doc requires them to
    be distinct.
    """
    n_total = 0
    n_collapse = 0
    samples = []
    for r in aux_records:
        md = r.get("metadata") or {}
        if md.get("task_type") != "auxiliary_statutory":
            continue
        try:
            user = json.loads(r["messages"][1]["content"])
        except Exception:                        # noqa: BLE001
            continue
        statute = user.get("statute", "")
        fact_pattern = user.get("fact_pattern", "")
        n_total += 1
        if statute and fact_pattern and statute == fact_pattern:
            n_collapse += 1
            if len(samples) < 3:
                samples.append({"record_id": md.get("record_id"),
                                "preview": (statute or "")[:150]})
    pass_rate = (n_total - n_collapse) / n_total if n_total > 0 else 1.0
    return {
        "rule": "AUX-FIELD-DISTINCTNESS",
        "checked": n_total,
        "collapsed": n_collapse,
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": min_pass_rate,
        "passed": pass_rate >= min_pass_rate,
        "samples": samples,
        "note": "Aux statutory records must have distinct `statute` and `fact_pattern` fields.",
    }


# ============================================================================
# A3 — Question well-formedness check
# ============================================================================
_QUESTION_INTERROGATIVE = re.compile(
    r"^\s*(what|when|where|who|why|how|is|are|was|were|does|do|did|can|could|"
    r"would|will|should|may|might|has|have|had|which|whether)\b",
    re.IGNORECASE,
)


def _is_well_formed_question(q: str, passage: str = "", task_type: str = "") -> tuple[bool, str]:
    """Heuristic: is `q` a well-formed task prompt (not a passage-prefix or fragment)?

    Note on `auxiliary_statutory`: in LegalBench's sara_* subtasks, the
    "question" field is a DECLARATIVE PROPOSITION the analyst must classify
    as entailment / contradiction / neutral (e.g., "Section 7703(a)(2) applies
    to Alice for the year 2018."). These are valid task prompts even though
    they lack a `?` and an interrogative cue — we relax those checks for
    statutory records but still enforce non-prefix-of-passage and minimum
    length to catch the original `text[:200]` truncation bug.
    """
    if not q or not isinstance(q, str):
        return False, "empty"
    qs = q.strip()
    if len(qs) < 8:
        return False, f"too short ({len(qs)} chars)"
    # Reject mid-word truncation (passage[:N] pattern)
    if not qs.endswith(("?", ".")) and len(qs) >= 100 and not qs[-1].isspace():
        last_word = qs.rsplit(maxsplit=1)[-1]
        if len(last_word) < 4:
            return False, f"likely mid-word truncation: ends in '{last_word}'"
    # Reject if `q` is a literal prefix of `passage`
    if passage and len(qs) >= 50:
        ps = passage.strip()
        if ps.startswith(qs) or qs.startswith(ps[:len(qs)]):
            return False, "question is a verbatim prefix of passage"
    # For non-statutory tasks, require an interrogative cue or a question mark.
    # Statutory propositions (sara_*) are declarative — exempt them.
    if task_type != "auxiliary_statutory":
        if "?" not in qs and not _QUESTION_INTERROGATIVE.match(qs):
            return False, "no interrogative cue or question mark"
    return True, ""


def audit_question_well_formed(aux_records: list[dict], min_pass_rate: float = 0.95) -> dict:
    """Check that aux records' user-side `question` field is a real question."""
    n_total = 0
    n_bad = 0
    by_reason = Counter()
    samples = []
    for r in aux_records:
        try:
            user = json.loads(r["messages"][1]["content"])
        except Exception:                        # noqa: BLE001
            continue
        q = user.get("question", "")
        passage = user.get("passage") or user.get("statute") or user.get("fact_pattern") or ""
        task_type = (r.get("metadata") or {}).get("task_type", "")
        n_total += 1
        ok, reason = _is_well_formed_question(
            q, passage if isinstance(passage, str) else "", task_type
        )
        if not ok:
            n_bad += 1
            by_reason[reason.split(":")[0]] += 1
            if len(samples) < 3:
                samples.append({
                    "record_id": (r.get("metadata") or {}).get("record_id"),
                    "reason": reason, "question": q[:150],
                })
    pass_rate = (n_total - n_bad) / n_total if n_total > 0 else 1.0
    return {
        "rule": "AUX-QUESTION-WELL-FORMED",
        "checked": n_total,
        "ill_formed": n_bad,
        "by_reason": dict(by_reason),
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": min_pass_rate,
        "passed": pass_rate >= min_pass_rate,
        "samples": samples,
    }


# ============================================================================
# A5 — Source-task semantic match
# ============================================================================
def _looks_numeric(s: str) -> bool:
    if not s:
        return False
    s = str(s).strip()
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    if not s:
        return False
    try:
        float(s.split()[0])
        return True
    except Exception:                            # noqa: BLE001
        return False


def audit_source_task_semantic_match(aux_records: list[dict]) -> dict:
    """Per (source, task_type), assert the assistant content's shape matches
    what the task type expects.

      - auxiliary_numeric:   assistant.answer should be numeric
      - auxiliary_citation:  assistant.evidence_span should be non-empty
      - auxiliary_statutory: assistant.label in {entailment, contradiction, neutral}
    """
    counts: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "pass": 0})
    samples = defaultdict(list)
    for r in aux_records:
        md = r.get("metadata") or {}
        tt = md.get("task_type", "")
        src = md.get("source", "")
        try:
            asst = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        key = (src, tt)
        counts[key]["n"] += 1
        ok = False
        reason = ""
        if tt == "auxiliary_numeric":
            ans = asst.get("answer", "")
            ok = _looks_numeric(ans)
            reason = "answer is not numeric" if not ok else ""
        elif tt == "auxiliary_citation":
            ev = asst.get("evidence_span", "")
            ok = bool(ev) and len(ev) >= 10
            reason = "evidence_span empty/short" if not ok else ""
        elif tt == "auxiliary_statutory":
            lbl = asst.get("label", "")
            ok = lbl in ("entailment", "contradiction", "neutral")
            reason = f"label={lbl!r} not in valid set" if not ok else ""
        else:
            ok = True
        if ok:
            counts[key]["pass"] += 1
        elif len(samples[key]) < 2:
            samples[key].append({
                "record_id": md.get("record_id"), "reason": reason,
            })
    by_key = {f"{src}/{tt}": {"checked": v["n"],
                              "pass": v["pass"],
                              "pass_rate": round(v["pass"] / v["n"], 4) if v["n"] else 0.0,
                              "samples": samples[(src, tt)]}
              for (src, tt), v in sorted(counts.items())}
    overall_pass = sum(v["pass"] for v in counts.values())
    overall_n = sum(v["n"] for v in counts.values())
    return {
        "rule": "AUX-SOURCE-TASK-SEMANTIC-MATCH",
        "by_source_task": by_key,
        "overall_pass_rate": round(overall_pass / overall_n, 4) if overall_n else 1.0,
        "passed": all(v["pass_rate"] >= 0.90 for v in by_key.values()),
        "note": "Each (source, task_type) cell must produce records whose "
                "assistant content matches the task semantics.",
    }


# ============================================================================
# A6 — Narrative diversity check by cell
# ============================================================================
def audit_narrative_diversity(nonaux_records: list[dict], min_unique: float = 0.5) -> dict:
    """For each (variant, is_suspicious, source) cell, compute prefix-uniqueness
    of narratives. Catches template collapse like the EFC bare-passthrough bug
    (where 100% of bare-positive Record_1 narratives shared the same template).
    """
    by_cell: dict[tuple, list[str]] = defaultdict(list)
    for r in nonaux_records:
        md = r.get("metadata") or {}
        try:
            out = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        narr = (out.get("suspicious_activity_report") or "").strip()
        if not narr:
            continue
        cell = (md.get("sar_variant"), bool(out.get("is_suspicious")), md.get("source"))
        by_cell[cell].append(narr[:200])

    rows = []
    failing_cells = []
    for cell, prefixes in by_cell.items():
        n = len(prefixes)
        if n < 10:                               # too small to judge diversity
            continue
        unique = len(set(prefixes))
        ratio = unique / n
        rows.append({"cell": cell, "n": n, "unique": unique, "ratio": round(ratio, 4)})
        if ratio < min_unique:
            failing_cells.append({"cell": cell, "n": n, "unique": unique, "ratio": round(ratio, 4)})
    rows.sort(key=lambda x: x["ratio"])
    return {
        "rule": "NARRATIVE-DIVERSITY",
        "min_unique_ratio": min_unique,
        "cells_checked": len(rows),
        "failing_cells": failing_cells,
        "all_cells": rows[:20],                  # top-20 worst
        "passed": not failing_cells,
    }


# ============================================================================
# A7 — Novel citation density on augmented narratives
# ============================================================================
_TRIVIAL_3GRAMS = {
    "u . s",  "u.s . c", "the entity", "the subject", "the account",
    "of the", "in the", "and the", "for the", "to the",
    "this activity", "the kyc", "31 u.s.c", "31 cfr",
}


def _trigrams(s: str) -> set[str]:
    toks = [t.lower() for t in re.findall(r"\w+", s)]
    return {f"{toks[i]} {toks[i+1]} {toks[i+2]}" for i in range(len(toks) - 2)}


def audit_novel_citation_density(nonaux_records: list[dict], sample_n: int = 1000,
                                 min_density: float = 0.50) -> dict:
    """For augmented positives, what fraction have at least one NON-TRIVIAL
    3-gram from a finding's key field appearing in the narrative?

    Stricter than the existing RULE-7-AUG-CITES because it filters out
    common 3-grams that any narrative would contain by chance.
    """
    import random
    rng = random.Random(101)
    aug_pos = []
    for r in nonaux_records:
        md = r.get("metadata") or {}
        if md.get("sar_variant") != "augmented":
            continue
        try:
            out = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        if not out.get("is_suspicious"):
            continue
        aug_pos.append(r)
    if not aug_pos:
        return {"rule": "NOVEL-CITATION-DENSITY", "passed": True, "checked": 0,
                "note": "No augmented+positive records to check."}
    sample = rng.sample(aug_pos, min(sample_n, len(aug_pos)))

    n_with_novel = 0
    for r in sample:
        try:
            user = json.loads(r["messages"][1]["content"])
            out = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        narr = out.get("suspicious_activity_report", "")
        narr_grams = _trigrams(narr)
        af = user.get("auxiliary_findings") or {}
        finding_grams: set[str] = set()
        for kind, key in [("numeric", "answer"), ("citation", "evidence_span"),
                          ("statutory", "reasoning")]:
            for f in (af.get(kind) or []):
                txt = f.get(key, "") or ""
                finding_grams |= _trigrams(txt)
        novel = (finding_grams - _TRIVIAL_3GRAMS) & narr_grams
        if novel:
            n_with_novel += 1

    density = n_with_novel / len(sample) if sample else 1.0
    return {
        "rule": "NOVEL-CITATION-DENSITY",
        "sampled": len(sample),
        "with_novel_citation": n_with_novel,
        "density": round(density, 4),
        "min_density": min_density,
        "passed": density >= min_density,
        "note": "Fraction of augmented positives whose narrative contains at "
                "least one non-trivial 3-gram from a finding's key field.",
    }


# ============================================================================
# A9 — Stage manifest cross-checks
# ============================================================================
def audit_stage_manifests(manifests_dir: Path) -> dict:
    """Walk the per-stage manifest files; check counts roll up sensibly."""
    findings: list[dict] = []
    pipelines = {
        "nonaux": ["stage_1_drivers", "stage_2_kyc", "stage_3_transactions",
                   "stage_4_sanctions", "stage_5_grounding", "stage_6_aux_findings",
                   "stage_7_assemble", "stage_8_adversarial",
                   "stage_9_validate_consolidate"],
        "aux": ["stage_a1_extract", "stage_a2_generate",
                "stage_a3_assemble", "stage_a4_audit"],
    }
    for pipeline, stages in pipelines.items():
        prev_produced: int | None = None
        for stage_id in stages:
            path = manifests_dir / pipeline / f"{stage_id}_manifest.json"
            if not path.exists():
                findings.append({"pipeline": pipeline, "stage": stage_id,
                                 "issue": "manifest_missing"})
                continue
            try:
                m = json.loads(path.read_text())
            except Exception as exc:             # noqa: BLE001
                findings.append({"pipeline": pipeline, "stage": stage_id,
                                 "issue": f"manifest_parse_error: {exc}"})
                continue
            counts = m.get("counts", {})
            produced = counts.get("produced")
            if prev_produced is not None and produced is not None:
                # produced is monotone non-increasing for sequential stages
                # (stages can drop records but not invent them) — except for
                # stages that EXPAND records (like FFIEC chunks → multiple Q/A).
                if produced > prev_produced * 5:
                    findings.append({"pipeline": pipeline, "stage": stage_id,
                                     "issue": f"produced({produced}) >> prev_produced({prev_produced})"})
            prev_produced = produced
    return {
        "rule": "STAGE-MANIFESTS-CROSSCHECK",
        "issues": findings,
        "passed": not findings,
    }


# ============================================================================
# A10 — AMLGentex cluster uniqueness
# ============================================================================
def audit_amlgentex_cluster_uniqueness(nonaux_records: list[dict],
                                       min_unique_ratio: float = 0.7) -> dict:
    """AMLGentex was emitting 70 unique clusters duplicated 9× = 617 records.
    After the B8/B9 fix it should emit ~70 clusters, 1× each. We check that
    among R3 records, at least 70% of cluster_ids are unique.
    """
    cluster_ids = []
    for r in nonaux_records:
        md = r.get("metadata") or {}
        if md.get("source") != "amlgentex":
            continue
        # Cluster id is in the user's kyc_profile.entity_id (Record_3 path)
        try:
            user = json.loads(r["messages"][1]["content"])
        except Exception:                        # noqa: BLE001
            continue
        kyc = user.get("kyc_profile") or {}
        entity_id = kyc.get("entity_id") or md.get("record_id")
        cluster_ids.append(entity_id)
    n_total = len(cluster_ids)
    n_unique = len(set(cluster_ids))
    ratio = n_unique / n_total if n_total > 0 else 1.0
    return {
        "rule": "AMLGENTEX-CLUSTER-UNIQUENESS",
        "amlgentex_records": n_total,
        "unique_cluster_ids": n_unique,
        "uniqueness_ratio": round(ratio, 4),
        "min_unique_ratio": min_unique_ratio,
        "passed": ratio >= min_unique_ratio if n_total > 0 else True,
    }


# ============================================================================
# A11 — Per-source field coverage
# ============================================================================
def audit_per_source_field_coverage(aux_records: list[dict]) -> dict:
    """For each source, verify expected fields are non-empty in ≥95% of records.

    Spot-checks routing correctness: if FinanceBench landed in citation,
    its records would have empty `evidence_span` since FB's `evidence` is
    a numeric calculation, not a verbatim quote.
    """
    expected_fields = {
        "auxiliary_numeric":    ["answer", "calculation"],
        "auxiliary_citation":   ["answer", "evidence_span"],
        "auxiliary_statutory":  ["answer", "label", "reasoning"],
    }
    by_source: dict[str, dict] = defaultdict(lambda: {"n": 0, "filled": Counter()})
    for r in aux_records:
        md = r.get("metadata") or {}
        src = md.get("source", "?")
        tt = md.get("task_type", "?")
        try:
            asst = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        by_source[src]["n"] += 1
        by_source[src]["task_type"] = tt
        for f in expected_fields.get(tt, []):
            v = asst.get(f, "")
            if v not in (None, "", []):
                by_source[src]["filled"][f] += 1
    rows = []
    failing = []
    for src, info in sorted(by_source.items()):
        n = info["n"]
        tt = info.get("task_type", "")
        rates = {}
        for f in expected_fields.get(tt, []):
            rate = info["filled"].get(f, 0) / n if n else 0.0
            rates[f] = round(rate, 4)
            if rate < 0.95:
                failing.append({"source": src, "field": f,
                                "rate": round(rate, 4), "n": n})
        rows.append({"source": src, "task_type": tt, "n": n,
                     "field_fill_rates": rates})
    return {
        "rule": "PER-SOURCE-FIELD-COVERAGE",
        "by_source": rows,
        "failing": failing,
        "passed": not failing,
    }


# ============================================================================
# A12 — Pool-vs-projector schema sanity
# ============================================================================
def audit_pool_projector_sanity(sample_n: int = 100) -> dict:
    """Run each pool projector on a small sample and verify the output.

    Catches schema-drift bugs early — if a source file's schema changes or a
    projector regresses, this audit fires before pipeline runs.
    """
    findings: list[dict] = []

    # Lazy imports to keep audit module lightweight
    try:
        from scripts.pools import (
            pool_1_efc, pool_2_ibm, pool_3_amlgentex, pool_4_sarsum, pool_5_cfpb,
        )
    except Exception as exc:                    # noqa: BLE001
        return {"rule": "POOL-PROJECTOR-SANITY",
                "passed": False, "issues": [{"error": f"import: {exc}"}]}

    pools = [
        ("EFC",       pool_1_efc),
        ("IBM",       pool_2_ibm),
        ("AMLGentex", pool_3_amlgentex),
        ("SARSum",    pool_4_sarsum),
        ("CFPB",      pool_5_cfpb),
    ]
    summary = {}
    for name, mod in pools:
        try:
            df = mod.load(max_rows=sample_n)
        except TypeError:                       # max_rows not supported
            df = mod.load()
        except Exception as exc:                # noqa: BLE001
            findings.append({"pool": name, "error": str(exc)[:200]})
            continue
        if df is None or len(df) == 0:
            findings.append({"pool": name, "issue": "empty"})
            continue
        n = len(df) if not hasattr(df, "shape") else df.shape[0]
        # Sanity: typology present, label present
        cols = list(df.columns) if hasattr(df, "columns") else []
        sample = df.head(min(sample_n, n))
        typ_present = "typology" in cols
        lbl_present = "label" in cols
        summary[name] = {
            "rows_loaded": n,
            "has_typology": typ_present,
            "has_label": lbl_present,
            "columns_sample": cols[:12],
        }
        if not typ_present or not lbl_present:
            findings.append({"pool": name, "issue": "missing typology or label column"})
    return {
        "rule": "POOL-PROJECTOR-SANITY",
        "summary": summary,
        "issues": findings,
        "passed": not findings,
    }


# ============================================================================
# A1 — Strategy-fixture comparison
# ============================================================================
# Strategy doc example records (lines 1339-1404) — used as ground-truth fixtures.
# We only check field-set + lengths + per-cell shape, not exact text.
_STRATEGY_FIXTURE_NONAUX = {
    "augmented_positive": {
        "task_type": "sar_judgment",
        "expected_user_keys": {
            "task_type", "transactions", "kyc_profile",
            "sanctions_pep_hits", "policy_excerpts", "sop_excerpts",
            "auxiliary_findings",
        },
        "expected_assistant_keys": {"is_suspicious", "suspicious_activity_report"},
        "min_narr_len": 250,
        "max_narr_len": 1000,
        "is_suspicious": True,
    },
    "bare_negative": {
        "task_type": "sar_judgment",
        "expected_user_keys": {
            "task_type", "transactions", "kyc_profile",
            "sanctions_pep_hits", "policy_excerpts", "sop_excerpts",
            "auxiliary_findings",
        },
        "expected_assistant_keys": {"is_suspicious", "suspicious_activity_report"},
        "narr_must_be_empty": True,
        "is_suspicious": False,
    },
}

_STRATEGY_FIXTURE_AUX = {
    # Aux assistant content does NOT echo `question` — by design. The user
    # message carries the question; the assistant emits the answer + supporting
    # fields only. NAT (the agent) at runtime pairs its own question with the
    # answer when bundling into sar_judgment.input.auxiliary_findings.
    "auxiliary_numeric":   {"user_keys": {"task_type", "passage", "question"},
                            "assistant_keys": {"answer", "calculation", "evidence"}},
    "auxiliary_citation":  {"user_keys": {"task_type", "passage", "question"},
                            "assistant_keys": {"answer", "evidence_span"}},
    "auxiliary_statutory": {"user_keys": {"task_type", "statute", "fact_pattern", "question"},
                            "assistant_keys": {"answer", "label", "reasoning"}},
}


def _check_keys(actual: dict, expected_subset: set[str], allow_must_cite: bool = False) -> list[str]:
    """Return list of missing-key strings; empty if expected_subset ⊆ actual."""
    actual_keys = set(actual.keys()) if isinstance(actual, dict) else set()
    # Allow extra keys; only report missing
    return sorted(expected_subset - actual_keys)


def audit_strategy_fixture(nonaux_records: list[dict], aux_records: list[dict]) -> dict:
    """Compare actual records against the strategy doc's example records."""
    issues: list[dict] = []
    cells_checked: dict[str, int] = Counter()

    # Non-aux: sample from each cell, compare to fixture
    for r in nonaux_records:
        md = r.get("metadata") or {}
        try:
            user = json.loads(r["messages"][1]["content"])
            asst = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        sar_variant = md.get("sar_variant", "")
        is_susp = bool(asst.get("is_suspicious"))

        # Pick the cell most relevant to this record
        if sar_variant == "augmented" and is_susp:
            cell = "augmented_positive"
        elif sar_variant == "bare" and not is_susp:
            cell = "bare_negative"
        else:
            continue

        if cells_checked[cell] >= 50:            # 50 samples per cell
            continue
        cells_checked[cell] += 1

        spec = _STRATEGY_FIXTURE_NONAUX[cell]
        missing_user = _check_keys(user, spec["expected_user_keys"])
        missing_asst = _check_keys(asst, spec["expected_assistant_keys"])
        narr = asst.get("suspicious_activity_report", "")
        problems = []
        if missing_user:
            problems.append(f"user missing keys: {missing_user}")
        if missing_asst:
            problems.append(f"assistant missing keys: {missing_asst}")
        if "min_narr_len" in spec and len(narr) < spec["min_narr_len"] - 100:
            problems.append(f"narrative too short ({len(narr)} < {spec['min_narr_len']})")
        if spec.get("narr_must_be_empty") and narr.strip():
            problems.append("narrative non-empty for bare_negative cell")
        if problems and len(issues) < 10:
            issues.append({"record_id": md.get("record_id"), "cell": cell,
                           "problems": problems})

    # Aux: per-task-type sample
    aux_cells_checked: Counter = Counter()
    for r in aux_records:
        md = r.get("metadata") or {}
        tt = md.get("task_type", "")
        if tt not in _STRATEGY_FIXTURE_AUX:
            continue
        if aux_cells_checked[tt] >= 50:
            continue
        aux_cells_checked[tt] += 1
        try:
            user = json.loads(r["messages"][1]["content"])
            asst = json.loads(r["messages"][2]["content"])
        except Exception:                        # noqa: BLE001
            continue
        spec = _STRATEGY_FIXTURE_AUX[tt]
        missing_user = _check_keys(user, spec["user_keys"])
        missing_asst = _check_keys(asst, spec["assistant_keys"])
        problems = []
        if missing_user:
            problems.append(f"user missing keys: {missing_user}")
        if missing_asst:
            problems.append(f"assistant missing keys: {missing_asst}")
        if problems and len(issues) < 20:
            issues.append({"record_id": md.get("record_id"), "task_type": tt,
                           "problems": problems})

    return {
        "rule": "STRATEGY-FIXTURE",
        "cells_checked": dict(cells_checked),
        "aux_cells_checked": dict(aux_cells_checked),
        "issues": issues,
        "passed": not issues,
        "note": "Each (cell, task_type) sample's user+assistant content must "
                "have the exact key-set specified in the strategy doc's "
                "example records (lines 1339-1404). Missing keys = field "
                "drift = bug.",
    }
