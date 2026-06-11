#!/usr/bin/env python3
"""Seed miniature AML CPT raw JSONL under data/raw/ (L1 + L2 sources).

Idempotent: safe to re-run. Used by nemo_curator.ipynb and checked into the repo
so the workshop runs without generating data inline.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

RAW_ROOT = Path(__file__).resolve().parent / "data" / "raw"

# Workshop corpus size. ~5x the original 120 so the 75-min hands-on has enough
# data to make each curation phase (and the GPU dedup) visibly do work.
TARGET_RECORDS = 600

FFIEC = [
    "Multiple cash deposits or withdrawals just below the $10,000 reporting threshold may indicate structuring under 31 U.S.C. § 5324(a)(3).",
    "Layering involves moving illicit funds through a series of transactions designed to disguise origin.",
    "Trade-based money laundering exploits international trade complexity to obscure fund movement.",
    "Smurfing breaks large transactions into smaller ones across accounts to evade reporting thresholds.",
    "Common structuring indicators: multiple cash deposits within $300 of $10,000 at one counterparty.",
]
FINCEN = [
    "FinCEN Advisory FIN-2020-A001: consider whether deposit patterns match declared monthly volume. Contact compliance@bank-example.com for guidance.",
    "SARs must be filed within 30 calendar days of detection. Call 1-800-555-0142 for guidance.",
    "PEP and sanctions screening should be re-run when KYC is refreshed or volume exceeds 3× declared monthly.",
    "Common-name false positives in OFAC SDN screening are a major source of analyst burnout.",
]
OFAC = [
    "OFAC SDN List entry: name match 0.87 against counterparty AC501879. Recommend disposition citing KYC consistency.",
    "Sanctions screening on SYN_6b558d20: 0 SDN hits, 1 PEP secondary-relative hit (0.32). Cleared per FFIEC § 5.4.",
    "OFAC SDN entry for John Smith — confirm jurisdictional context. Contact ofac-screening@bank-example.com.",
]
TX = [
    "Entity SYN_6b558d20: three cash deposits $9,793, $9,807, $9,791 over 2 days to AC501879 totaling $29,391.",
    "Wire $48,250 from SYN_6b558d20 to OFFS_993; KYC monthly $35,486; ratio 1.36× exceeds 1.20× review trigger.",
    "Entity SYN_3f99: five cash deposits $9,801–$9,820 over 5 days; inconsistent with declared card+wire profile.",
    "Routine payroll wire from ACME_PAYR to 47 counterparties; consistent with semi-monthly payroll cycle.",
]
EDGAR = [
    "Form 10-K discloses AML, OFAC, and BSA risks; financial crimes program supervised by the Chief Compliance Officer.",
    "Quarterly AML self-assessments follow FFIEC guidance; material findings reported to the Audit Committee.",
    "FY2025: 1,847 SARs filed and 12,300 OFAC alerts processed; independent testing found no material deficiencies.",
]
NOISE = [
    "click here click here click here click here",
    "!!! BUY NOW @@@@@@@@@@ LIMITED OFFER @@@@ click click click",
    "Page 1 of 5 · Page 2 of 5 · Page 3 of 5 · Page 4 of 5 · Page 5 of 5",
    "TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO",
]
NON_EN = [
    "Plusieurs dépôts en espèces juste au-dessous du seuil de déclaration peuvent indiquer une infraction.",
    "现金存款金额接近10,000美元上限可能构成分散存款罪。",
    "Mehrere Bareinzahlungen knapp unterhalb der Meldegrenze können auf Strukturierung hindeuten.",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.relative_to(RAW_ROOT.parent.parent)}  ({len(rows)} records)")


def build_rows() -> dict[str, list[dict]]:
    rows: list[dict] = []
    rid = 0

    def add(text: str, layer: str, source: str) -> None:
        nonlocal rid
        rid += 1
        rows.append({"id": f"{source}_{rid:04d}", "text": text, "layer": layer, "source": source})

    # Workshop-only edge cases (production INGEST + TEXT_CLEAN handles these too).
    rows.append({
        "id": "edgar_bytes_repr",
        "text": "b'Annual AML program disclosure per Form 10-K requirements.\\n'",
        "layer": "L1",
        "source": "edgar_10k",
    })
    rows.append({
        "id": "fincen_html",
        "text": "<p>FinCEN <b>Advisory</b> on <a href='#'>structuring</a> patterns near CTR thresholds.</p>",
        "layer": "L2",
        "source": "fincen_advisory",
    })
    rows.append({
        "id": "ffiec_boilerplate",
        "text": "Page 1 of 5\nFor more information visit www.ffiec.gov\n"
        + FFIEC[0],
        "layer": "L2",
        "source": "ffiec_bsa",
    })

    for t in EDGAR:
        add(t, "L1", "edgar_10k")
    for t in TX:
        add(t, "L1", "transactional")
    # Cross-source duplicate for phase 4 (XSOURCE_DEDUP): same statute text in two sources.
    shared_statute = (
        "31 U.S.C. § 5324(a)(3) prohibits structuring transactions to evade currency transaction reporting. "
        "Examiners should document patterns of sub-CTR deposits and counterparty concentration."
    )
    add(shared_statute, "L2", "ffiec_bsa")
    rows.append({
        "id": "fincen_advisory_xsrc",
        "text": shared_statute + " Additional FinCEN commentary on SAR filing timelines.",
        "layer": "L2",
        "source": "fincen_advisory",
    })

    for t in FFIEC:
        add(t, "L2", "ffiec_bsa")
    for t in FINCEN:
        add(t, "L2", "fincen_advisory")
    for t in OFAC:
        add(t, "L2", "ofac_sanctions")

    def _text(r: dict) -> str:
        return r.get("text") or r.get("content") or ""

    for src in [rows[0], rows[5], rows[10], rows[15]]:
        dup = {**src, "id": src["id"] + "_DUPE_EXACT"}
        if "text" not in dup and "content" in dup:
            dup["text"] = dup.pop("content")
        rows.append(dup)
    for src in [rows[0], rows[5], rows[12]]:
        t = _text(src)
        fuzzy = t.replace("  ", " ").replace(" the ", " THE ", 1) + "\n"
        rows.append({**src, "id": src["id"] + "_DUPE_FUZZY", "text": fuzzy, "content": None})
    for t in NOISE:
        add(t, "L2", "noise")
    for t in NON_EN:
        add(t, "L2", "multilingual")

    # Fill up to TARGET_RECORDS by rotating across all real sources/layers so the
    # corpus stays balanced (not one giant source). Each filler row is tagged with
    # a unique case number, so most survive dedup while a fraction collide.
    random.seed(42)
    pools = [
        ("L1", "edgar_10k", EDGAR),
        ("L1", "transactional", TX),
        ("L2", "ffiec_bsa", FFIEC),
        ("L2", "fincen_advisory", FINCEN),
        ("L2", "ofac_sanctions", OFAC),
    ]
    i = 0
    while len(rows) < TARGET_RECORDS:
        layer, source, pool = pools[i % len(pools)]
        base = random.choice(pool)
        add(base + f" (case {len(rows):04d})", layer, source)
        i += 1

    by_file: dict[str, list[dict]] = {}
    for r in rows:
        layer = "level_1" if r["layer"] == "L1" else "level_2"
        key = f"{layer}/{r['source']}.jsonl"
        by_file.setdefault(key, []).append(r)
    return by_file


def main() -> None:
    print(f"Writing raw corpus under {RAW_ROOT}")
    for rel, chunk in build_rows().items():
        _write_jsonl(RAW_ROOT / rel, chunk)
    print("Done.")


if __name__ == "__main__":
    main()
