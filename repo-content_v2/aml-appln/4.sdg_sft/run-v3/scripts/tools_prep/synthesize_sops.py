"""Stage 5B artifact builder — 24 SOP markdown variants (8 typologies × 3 variants).

Each variant has 6 sections (Investigation Steps, Escalation Criteria,
Documentation Requirements, Filing Decision, Tools and Systems, References)
synthesized via DataDesigner anchored on FFIEC + FinCEN passages.

Run once:
    python -m scripts.tools_prep.synthesize_sops
"""
from __future__ import annotations

import logging
from pathlib import Path

import data_designer.config as dd
import pandas as pd

from scripts.common.dd_helpers import (
    build_llm_text_column,
    build_local_seed_source,
    make_data_designer,
    make_model_config,
)
from scripts.common.io import write_parquet
from scripts.common.progress import configure_logging
from scripts.config import CONCURRENCY, INTERIM_DIR, TOOLS_SOP_DIR
from scripts.pools import policy_corpus
from scripts.tools_prep.build_policy_chunks import TYPOLOGY_KEYWORDS

logger = configure_logging("tools_prep.synthesize_sops")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SECTIONS = [
    "Investigation Steps",
    "Escalation Criteria",
    "Documentation Requirements",
    "Filing Decision",
    "Tools and Systems",
    "References",
]
VARIANTS_PER_TYPOLOGY = 3                 # strategy doc: 3 variants instead of 1
TYPOLOGIES = list(TYPOLOGY_KEYWORDS.keys())

SYSTEM_PROMPT = (
    "You are a senior AML/BSA compliance officer drafting an internal Standard "
    "Operating Procedure (SOP) for investigating a specific typology of "
    "suspicious activity. Your output is a single SOP section, written in clear "
    "imperative prose. Ground it in the supplied regulatory anchors. Output ONLY "
    "the section text — no headings, no preamble."
)

USER_TEMPLATE = (
    "Typology: {{ typology }}\n"
    "SOP Section: {{ section }}\n\n"
    "Regulatory anchors (FFIEC + FinCEN passages matching this typology):\n"
    "{{ anchor_text }}\n\n"
    "Write the {{ section }} section of SOP-{{ typology_upper }}-{{ variant_id }} "
    "(institution-internal procedure). Output 4-10 sentences in clear imperative "
    "voice (e.g., 'Pull all transactions for the entity in the prior 90 days.')."
)


def _gather_anchors(typology: str, max_chars_per_anchor: int = 1800, max_anchors: int = 3) -> str:
    """Pull 1-3 anchor passages from the policy corpus matching `typology`."""
    keywords = [kw.lower() for kw in TYPOLOGY_KEYWORDS[typology]]
    anchors: list[str] = []
    for path in policy_corpus.list_files():
        if "fincen" not in path.stem.lower() and "ffiec" not in path.stem.lower():
            continue
        for rec in policy_corpus.iter_records(path):
            text = rec.get("content") or rec.get("text") or ""
            if not isinstance(text, str) or len(text) < 200:
                continue
            low = text.lower()
            if any(kw in low for kw in keywords):
                anchors.append(f"[{path.stem}] {text[:max_chars_per_anchor].strip()}")
                if len(anchors) >= max_anchors:
                    return "\n\n---\n\n".join(anchors)
    return "\n\n---\n\n".join(anchors) if anchors else f"(no on-disk anchors for {typology})"


def _build_seed() -> Path:
    rows: list[dict] = []
    for typology in TYPOLOGIES:
        anchors = _gather_anchors(typology)
        typology_upper = typology.upper().replace("_", "-")
        for variant in range(1, VARIANTS_PER_TYPOLOGY + 1):
            for section in SECTIONS:
                rows.append({
                    "typology": typology,
                    "typology_upper": typology_upper,
                    "variant_id": f"{variant:02d}",
                    "section": section,
                    "anchor_text": anchors,
                })
    df = pd.DataFrame(rows)
    seed_path = INTERIM_DIR / "tools_prep_sop_seed.csv"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(seed_path, index=False)
    logger.info("SOP seed CSV: %d rows → %s", len(df), seed_path)
    return seed_path


def _write_markdown(generated_df: pd.DataFrame) -> list[Path]:
    written: list[Path] = []
    grouped = generated_df.groupby(["typology", "variant_id"])
    for (typology, variant_id), group in grouped:
        path = TOOLS_SOP_DIR / f"{typology}_v{variant_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            typology_upper = typology.upper().replace("_", "-")
            f.write(f"# SOP-{typology_upper}-{variant_id}\n\n")
            f.write(f"_Typology: `{typology}` (variant {variant_id})_\n\n")
            for section in SECTIONS:
                row = group[group["section"] == section]
                body = row["sop_text"].iloc[0] if len(row) else "(generation failed)"
                f.write(f"## {section}\n\n{body}\n\n")
        written.append(path)
        logger.info("Wrote %s", path)
    return written


def build(*, dry_run: bool = False) -> list[Path]:
    seed_path = _build_seed()
    n_rows = sum(1 for _ in seed_path.open()) - 1

    if dry_run:
        logger.info("[dry-run] would invoke DataDesigner for %d rows", n_rows)
        return []

    cb = dd.DataDesignerConfigBuilder(model_configs=[
        make_model_config(alias="sft-generator", max_parallel=CONCURRENCY.max_parallel_llm),
    ])
    cb.with_seed_dataset(build_local_seed_source(seed_path))
    cb.add_column(build_llm_text_column(
        name="sop_text",
        system_prompt=SYSTEM_PROMPT,
        prompt=USER_TEMPLATE,
    ))

    runtime = make_data_designer(TOOLS_SOP_DIR.parent / "_dd_artifacts_sop")
    result = runtime.create(cb, num_records=n_rows, dataset_name="tools_prep_sops_v1")
    df = result.load_dataset()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Re-attach the seed columns (typology, variant_id, section) for grouping
    seed_df = pd.read_csv(seed_path)
    if len(df) == len(seed_df):
        df = pd.concat([seed_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    return _write_markdown(df)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    build(dry_run=args.dry_run)
