"""OFAC + synthetic PEP unified loader for Stage 4.

Both files share the same columns (`targets.simple.csv` schema), so Stage 4
can union them transparently and filter by `schema` (Person / Company) and
`countries` per the strategy doc.
"""
from __future__ import annotations

import logging

import pandas as pd

from pipeline.config import POOLS

logger = logging.getLogger(__name__)


def load() -> pd.DataFrame:
    """Return concatenated OFAC + PEP names with a `pool` tag column.

    Columns (subset of the CSV schema): id, schema, name, aliases, countries,
    program_ids, dataset, plus `pool` ∈ {"OFAC", "PEP"}.
    """
    frames: list[pd.DataFrame] = []
    if POOLS.ofac_targets.exists():
        ofac = pd.read_csv(POOLS.ofac_targets)
        ofac["pool"] = "OFAC"
        frames.append(ofac)
    else:
        logger.warning("OFAC file missing: %s", POOLS.ofac_targets)

    if POOLS.pep_synthetic.exists():
        pep = pd.read_csv(POOLS.pep_synthetic)
        pep["pool"] = "PEP"
        frames.append(pep)
    else:
        logger.warning("PEP file missing: %s", POOLS.pep_synthetic)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Normalize column dtypes for safe filtering downstream
    for col in ("schema", "name", "countries", "program_ids", "pool"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    logger.info("ofac_pep.load → %d entries (OFAC + PEP combined)", len(df))
    return df


def filter_for_archetype(
    df: pd.DataFrame,
    *,
    is_business: bool,
    jurisdiction: str | None = None,
    pep_flavoured: bool = False,
) -> pd.DataFrame:
    """Filter the OFAC+PEP pool down to candidates matching the archetype profile."""
    if df.empty:
        return df
    sub = df.copy()
    # Business archetype → schema=Company; individual → schema=Person
    desired_schema = "Company" if is_business else "Person"
    if "schema" in sub.columns:
        sub = sub[sub["schema"] == desired_schema]
    if pep_flavoured:
        sub = sub[sub["pool"] == "PEP"]
    if jurisdiction:
        # Soft filter: prefer entries whose `countries` overlaps the record's jurisdiction
        cc = jurisdiction.split("-")[0].lower()  # "US-NY" → "us"
        if "countries" in sub.columns:
            preferred = sub[sub["countries"].str.contains(cc, case=False, na=False)]
            if len(preferred) >= 5:
                sub = preferred
    return sub
