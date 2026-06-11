"""Policy corpus index for Stage 5A.

Lists the 9 jsonl files (8 under cpt/level_2/ + 1 SFT FFIEC file) and a
mapping from file stem → canonical `source` enum value.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from scripts.config import POOLS

logger = logging.getLogger(__name__)


SOURCE_LABEL: dict[str, str] = {
    # cpt/level_2/*.jsonl
    "fincen_advisories": "FinCEN",
    "fincen_federal_register": "FinCEN",
    "fincen_sar_reviews": "FinCEN",
    "fincen_enforcement": "FinCEN",
    "fincen_files": "FinCEN",
    "fatf_publications": "FATF",
    "ofac_guidance": "OFAC",
    "courtlistener": "FFIEC",          # supplementary caselaw bucketed into FFIEC
    # sft/*
    "ffiec_manual": "FFIEC",
}


def list_files() -> list[Path]:
    """Return the 9 jsonl files comprising the policy corpus."""
    files: list[Path] = []
    cpt_dir = POOLS.policy_dir_cpt
    if cpt_dir.exists():
        for stem in (
            "fincen_advisories", "fincen_federal_register", "fincen_sar_reviews",
            "fincen_enforcement", "fincen_files", "fatf_publications",
            "ofac_guidance", "courtlistener",
        ):
            p = cpt_dir / f"{stem}.jsonl"
            if p.exists():
                files.append(p)
            else:
                logger.warning("policy_corpus: missing %s", p)
    if POOLS.policy_ffiec_sft.exists():
        files.append(POOLS.policy_ffiec_sft)
    else:
        logger.warning("policy_corpus: missing %s", POOLS.policy_ffiec_sft)
    return files


def iter_records(path: Path) -> Iterator[dict]:
    """Stream a policy jsonl one record at a time. Each record has `content`/`text` + metadata."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def source_for(path: Path) -> str:
    """Map a policy jsonl file to its canonical `source` enum."""
    return SOURCE_LABEL.get(path.stem, "FinCEN")
