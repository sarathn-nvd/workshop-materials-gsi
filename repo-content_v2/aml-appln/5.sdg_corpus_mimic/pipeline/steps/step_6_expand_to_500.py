"""Step 6 — Expand demo corpus from 200 to 500 cases (one-shot).

Append 300 new cases (DEMO_0201..DEMO_0500) on top of the existing
200-case manifest + eval_keys.

Cohort distribution of the 300 new cases:
  - 18 suspicious   (all remaining unused entities in suspicious_entities.parquet)
  - 10 near_miss    (all remaining unused entities in near_miss_entities.parquet)
  - 272 clean       (sampled from KYC entities not in either typology pool)

Per-typology effective scaling (positive + near-miss):
  - structuring          : 3 → 5  (+1 pos, +1 nm)
  - smurfing             : 3 → 4  (+1 pos)
  - layering             : 4 → 6  (+1 pos, +1 nm)
  - trade_based_ml       : 4 → 6  (+1 pos, +1 nm)
  - shell_company        : 4 → 6  (+1 pos, +1 nm)
  - human_trafficking    : 3 → 5  (+1 pos, +1 nm)
  - terrorist_financing  : 4 → 6  (+1 pos, +1 nm)
  - elder_exploitation   : 4 → 6  (+1 pos, +1 nm)
  - clean ("none")       : 140 → 412 (+272)

(Exact +N per typology depends on the deterministic shuffle below.)

The 200 existing cases are preserved byte-identically; only DEMO_0201..0500
are added. Run is idempotent: re-running will overwrite existing
DEMO_0201..0500 rows if they exist, but never touch DEMO_0001..0200.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# trigger_summary library — one per typology, sampled from existing 200 cases
# ---------------------------------------------------------------------------
TRIGGERS_BY_TYPOLOGY: dict[str, list[str]] = {
    "none": [
        "Counterparty country newly added to risk list",
        "New counterparty added to payee list",
        "Cash deposit above branch threshold",
        "Cross-border wire above $50K threshold",
        "Wire velocity flagged by monitoring rule",
        "Round-number wires through offshore entity",
    ],
    "structuring": [
        "Repeated currency deposits below CTR threshold",
        "Multiple sub-$10K cash deposits across branches",
        "High-frequency cash activity flagged by velocity rule",
    ],
    "smurfing": [
        "Fan-in pattern alert from velocity monitoring",
        "Multiple senders converging on a single account",
        "Aggregated daily cash inflows above expected",
    ],
    "layering": [
        "Rapid wire pass-through detected",
        "Counterparty graph cycle detected",
        "Unusually high wire velocity over short period",
    ],
    "trade_based_ml": [
        "Trade-finance wire flagged for invoice variance",
        "Cross-border wire above $50K threshold",
        "Large outbound wire from import-export entity",
    ],
    "shell_company": [
        "Holding-entity activity without operating expenses",
        "Beneficial ownership change recently filed",
        "Round-number wires through offshore entity",
    ],
    "human_trafficking": [
        "Multiple small cash withdrawals near transit hubs",
        "Corridor-jurisdiction wires from individual account",
    ],
    "terrorist_financing": [
        "Sanctions list adjacency on counterparty",
        "Fan-in inbound wires followed by corridor outbound",
    ],
    "elder_exploitation": [
        "New payee added to senior account",
        "Escalating wires from retiree account",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def derive_evidence_template(typology: str, label: bool, near_miss: bool,
                              existing_keys: list[dict]) -> dict:
    """Pick the expected_evidence template from the matching bucket in the
    current 200-case eval keys. Same (typology, label, near_miss) always
    produces the same template, so this is fully deterministic."""
    for k in existing_keys:
        if (k["expected_typology"] == typology and
            k["expected_label"] == label and
            k["near_miss"] == near_miss):
            return copy.deepcopy(k["expected_evidence"])
    raise ValueError(
        f"No template found for ({typology}, label={label}, near_miss={near_miss}). "
        "Current eval_keys must contain at least one example of every bucket."
    )


def alert_id(rng: random.Random) -> str:
    return f"ALT_{rng.randint(1_000_000, 9_999_999)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend-data",
                   default="/data/swami/gsi-training/10.appln_buildout/backend/data",
                   help="Backend data plane root (reads seeded_subpopulations + tool_2_kyc).")
    p.add_argument("--in-manifest",
                   default=None,
                   help="Input manifest.jsonl (defaults to <backend-data>/demo/manifest.jsonl).")
    p.add_argument("--in-keys",
                   default=None,
                   help="Input eval_keys.jsonl (defaults to <backend-data>/demo/eval_keys.jsonl).")
    p.add_argument("--out-dir",
                   default=None,
                   help="Output directory (defaults to <backend-data>/demo/). "
                        "Writes manifest.jsonl + eval_keys.jsonl + stratification_report.json.")
    p.add_argument("--target-total",
                   type=int, default=500,
                   help="Target total cases (default 500).")
    p.add_argument("--window-start", default="2026-02-01")
    p.add_argument("--window-end",   default="2026-04-30")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the deterministic shuffle / sample (default 42).")
    args = p.parse_args(argv)

    bd = Path(args.backend_data)
    in_mf  = Path(args.in_manifest) if args.in_manifest else bd / "demo" / "manifest.jsonl"
    in_ks  = Path(args.in_keys)     if args.in_keys     else bd / "demo" / "eval_keys.jsonl"
    out    = Path(args.out_dir)     if args.out_dir     else bd / "demo"

    existing_mf  = load_jsonl(in_mf)
    existing_ks  = load_jsonl(in_ks)
    n_existing   = len(existing_mf)
    if len(existing_ks) != n_existing:
        print(f"WARN: manifest has {n_existing} rows but eval_keys has {len(existing_ks)} — "
              "they should match", file=sys.stderr)

    n_to_add = args.target_total - n_existing
    if n_to_add <= 0:
        print(f"Target {args.target_total} ≤ current {n_existing}; nothing to do.")
        return 0
    print(f"Existing cases: {n_existing}, target: {args.target_total}, will add: {n_to_add}")

    # Load pools
    sus = pd.read_parquet(bd / "seeded_subpopulations" / "suspicious_entities.parquet")
    nm  = pd.read_parquet(bd / "seeded_subpopulations" / "near_miss_entities.parquet")
    kyc = pd.read_parquet(bd / "tool_2_kyc" / "entities.parquet")

    used_entities = {m["entity_id"] for m in existing_mf}
    sus_unused = sus[~sus["entity_id"].isin(used_entities)].copy()
    nm_unused  = nm[~nm["entity_id"].isin(used_entities)].copy()
    flagged    = set(sus["entity_id"]).union(nm["entity_id"])
    clean_unused = kyc[(~kyc["entity_id"].isin(flagged)) &
                        (~kyc["entity_id"].isin(used_entities))].copy()

    n_sus, n_nm, n_clean = len(sus_unused), len(nm_unused), len(clean_unused)
    print(f"Unused entities → sus={n_sus} nm={n_nm} clean={n_clean}")

    rng = random.Random(args.seed)

    # Deterministic shuffle of each pool
    sus_unused = sus_unused.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    nm_unused  = nm_unused.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    clean_unused = clean_unused.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Use ALL unused sus + nm, fill the rest with clean.
    take_sus   = min(n_sus, n_to_add)
    take_nm    = min(n_nm, n_to_add - take_sus)
    take_clean = max(0, n_to_add - take_sus - take_nm)
    if take_clean > n_clean:
        print(f"ERROR: not enough clean entities ({n_clean}) for {take_clean} needed.",
              file=sys.stderr)
        return 2
    print(f"Adding: sus={take_sus}  nm={take_nm}  clean={take_clean}")

    next_id = n_existing + 1  # DEMO_0201 etc.
    new_mf:   list[dict] = []
    new_keys: list[dict] = []

    def emit_one(entity_id: str, typology: str, label: bool, near_miss: bool):
        nonlocal next_id
        case_id = f"DEMO_{next_id:04d}"
        next_id += 1
        triggers = TRIGGERS_BY_TYPOLOGY.get(typology, TRIGGERS_BY_TYPOLOGY["none"])
        new_mf.append({
            "case_id":  case_id,
            "alert_id": alert_id(rng),
            "entity_id": entity_id,
            "investigation_window_start": args.window_start,
            "investigation_window_end":   args.window_end,
            "trigger_summary": rng.choice(triggers),
        })
        new_keys.append({
            "case_id":  case_id,
            "entity_id": entity_id,
            "expected_typology": typology,
            "expected_label":    label,
            "near_miss":         near_miss,
            "expected_evidence": derive_evidence_template(
                typology, label, near_miss, existing_ks),
        })

    # Emit suspicious (label=True, near_miss=False)
    for _, row in sus_unused.head(take_sus).iterrows():
        emit_one(row["entity_id"], row["typology"], True, False)

    # Emit near-miss (label=False, near_miss=True)
    for _, row in nm_unused.head(take_nm).iterrows():
        emit_one(row["entity_id"], row["typology"], False, True)

    # Emit clean (typology="none", label=False, near_miss=False)
    for _, row in clean_unused.head(take_clean).iterrows():
        emit_one(row["entity_id"], "none", False, False)

    # Combine + write
    all_mf  = existing_mf  + new_mf
    all_ks  = existing_ks  + new_keys

    # Sanity: no duplicate case_ids or entity_ids
    case_ids = [m["case_id"] for m in all_mf]
    ent_ids  = [m["entity_id"] for m in all_mf]
    dups_case = [k for k, v in Counter(case_ids).items() if v > 1]
    dups_ent  = [k for k, v in Counter(ent_ids).items()  if v > 1]
    if dups_case:
        print(f"ERROR: duplicate case_ids: {dups_case[:5]}", file=sys.stderr); return 3
    if dups_ent:
        print(f"ERROR: duplicate entity_ids: {dups_ent[:5]}", file=sys.stderr); return 3

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_mf, out / "manifest.jsonl")
    write_jsonl(all_ks, out / "eval_keys.jsonl")

    # Stratification report
    buckets = Counter()
    per_typ = Counter()
    for k in all_ks:
        if k["expected_label"]:
            buckets["suspicious"] += 1
        elif k["near_miss"]:
            buckets["near_miss"] += 1
        else:
            buckets["clean"] += 1
        per_typ[k["expected_typology"]] += 1

    report = {
        "n_total": len(all_ks),
        "buckets": dict(buckets),
        "per_typology": dict(per_typ),
        "window_start": args.window_start,
        "window_end":   args.window_end,
        "expansion_step": "step_6_expand_to_500",
        "added_in_this_run": n_to_add,
    }
    (out / "stratification_report.json").write_text(json.dumps(report, indent=2))

    print()
    print("=" * 60)
    print(f"  manifest.jsonl   : {len(all_mf)} rows")
    print(f"  eval_keys.jsonl  : {len(all_ks)} rows")
    print(f"  buckets          : {dict(buckets)}")
    print(f"  per_typology     : {dict(per_typ)}")
    print(f"  out_dir          : {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
