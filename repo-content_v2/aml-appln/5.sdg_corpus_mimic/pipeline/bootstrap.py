"""Bootstrap — one-time copy of SFT scripts into pipeline/ (Step 1).

Reads pipeline/bootstrap_manifest.yaml and copies each listed file from
4.sdg_sft/scripts/ into pipeline/, applying one of three copy modes:

  copy_verbatim
      Straight shutil.copy2. Source SHA-256 recorded in the bootstrap report.

  rewrite_imports
      Copy, then rewrite `from scripts.<X>` -> `from pipeline.<X>` and
      `import scripts.<X>` -> `import pipeline.<X>`. The SFT root package is
      `scripts`; we re-root to `pipeline`.

  extract_typology_keywords
      Extract just the TYPOLOGY_KEYWORDS dict literal from
      tools_prep/build_policy_chunks.py via AST and emit a minimal module
      exposing it (clean import surface for runtime Tool 4 retrieval).

The bootstrap is idempotent: if a destination file already exists, it is
SHA-256-compared to what would have been written. A divergence surfaces
as a warning (not a fatal error) so local edits to copied files are
preserved by default. Pass --force to overwrite local edits.

Usage:
    python -m pipeline.bootstrap            # idempotent
    python -m pipeline.bootstrap --force    # overwrite local edits

A report of every copy decision is written to
pipeline/config.BOOTSTRAP_REPORT (manifests/bootstrap.json).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from pipeline.config import (
    BOOTSTRAP_MANIFEST,
    BOOTSTRAP_REPORT,
    PIPELINE_ROOT,
    SFT_SCRIPTS,
)

logger = logging.getLogger("pipeline.bootstrap")

CopyMode = Literal["copy_verbatim", "rewrite_imports", "extract_typology_keywords"]


# ============================================================================
# Per-file copy record (for the manifest report)
# ============================================================================
@dataclass
class CopyRecord:
    src: str
    dst: str
    mode: CopyMode
    status: Literal["created", "unchanged", "diverged_kept_local", "overwritten"]
    src_sha256: str
    dst_sha256_before: str | None
    dst_sha256_after: str
    bytes_written: int
    notes: str = ""


# ============================================================================
# Helpers
# ============================================================================
def _sha256(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Copy mode implementations — each returns the bytes to write
# ============================================================================
def _mode_copy_verbatim(src_bytes: bytes) -> bytes:
    return src_bytes


_IMPORT_REWRITES = [
    # `from scripts.X.Y import Z`  ->  `from pipeline.X.Y import Z`
    (re.compile(r"^(\s*)from\s+scripts\.", re.MULTILINE), r"\1from pipeline."),
    # `from scripts import X`  ->  `from pipeline import X`
    (re.compile(r"^(\s*)from\s+scripts\s+import\s+", re.MULTILINE),
     r"\1from pipeline import "),
    # `import scripts.X`  ->  `import pipeline.X`
    (re.compile(r"^(\s*)import\s+scripts\.", re.MULTILINE), r"\1import pipeline."),
    # `import scripts` (rare)  ->  `import pipeline`
    (re.compile(r"^(\s*)import\s+scripts(\s|$)", re.MULTILINE), r"\1import pipeline\2"),
]


def _mode_rewrite_imports(src_bytes: bytes) -> bytes:
    text = src_bytes.decode("utf-8")
    for pattern, repl in _IMPORT_REWRITES:
        text = pattern.sub(repl, text)
    return text.encode("utf-8")


def _mode_extract_typology_keywords(src_bytes: bytes) -> bytes:
    """Extract the TYPOLOGY_KEYWORDS dict literal from the source and emit a
    minimal module exposing only it. Uses AST to find the assignment so it
    survives reasonable refactors of the source file.
    """
    text = src_bytes.decode("utf-8")
    tree = ast.parse(text)
    target_node: ast.Assign | ast.AnnAssign | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TYPOLOGY_KEYWORDS":
                    target_node = node
                    break
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "TYPOLOGY_KEYWORDS"
                and node.value is not None
            ):
                target_node = node
        if target_node is not None:
            break
    if target_node is None:
        raise RuntimeError(
            "TYPOLOGY_KEYWORDS not found in source — cannot extract"
        )
    literal = ast.get_source_segment(text, target_node)
    out = (
        '"""Typology keyword dictionary — single source of truth for typology→keyword mapping.\n'
        "\n"
        "Auto-extracted by pipeline.bootstrap from\n"
        "4.sdg_sft/scripts/tools_prep/build_policy_chunks.py at bootstrap time.\n"
        "Re-run bootstrap to refresh.\n"
        '"""\n'
        "from __future__ import annotations\n"
        "\n"
        f"{literal}\n"
    )
    return out.encode("utf-8")


_MODES = {
    "copy_verbatim": _mode_copy_verbatim,
    "rewrite_imports": _mode_rewrite_imports,
    "extract_typology_keywords": _mode_extract_typology_keywords,
}


# ============================================================================
# Bootstrap driver
# ============================================================================
def _load_manifest(path: Path) -> tuple[Path, list[dict]]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    sft_root_rel = manifest["sft_root"]
    sft_root = (path.parent / sft_root_rel).resolve()
    if not sft_root.exists():
        raise FileNotFoundError(
            f"Manifest's sft_root resolves to {sft_root}, which does not exist. "
            f"Confirm 4.sdg_sft is checked out at the expected location."
        )
    return sft_root, manifest["copies"]


def _process_copy(
    entry: dict,
    sft_root: Path,
    *,
    force: bool,
) -> CopyRecord:
    src_rel: str = entry["src"]
    dst_rel: str = entry["dst"]
    mode: CopyMode = entry["mode"]
    if mode not in _MODES:
        raise ValueError(f"unknown copy mode: {mode}")

    src_path = sft_root / src_rel
    if not src_path.exists():
        raise FileNotFoundError(f"source missing: {src_path}")

    dst_path = PIPELINE_ROOT / dst_rel

    src_bytes = src_path.read_bytes()
    src_sha = _sha256(src_bytes)
    new_bytes = _MODES[mode](src_bytes)

    dst_existing = _read_bytes(dst_path)
    dst_sha_before = _sha256(dst_existing) if dst_existing else None
    new_sha = _sha256(new_bytes)

    if dst_existing and not force:
        if dst_sha_before == new_sha:
            return CopyRecord(
                src=src_rel,
                dst=dst_rel,
                mode=mode,
                status="unchanged",
                src_sha256=src_sha,
                dst_sha256_before=dst_sha_before,
                dst_sha256_after=new_sha,
                bytes_written=0,
                notes="destination already matches what would have been written",
            )
        # Local edits differ from what the manifest would produce.
        return CopyRecord(
            src=src_rel,
            dst=dst_rel,
            mode=mode,
            status="diverged_kept_local",
            src_sha256=src_sha,
            dst_sha256_before=dst_sha_before,
            dst_sha256_after=dst_sha_before,
            bytes_written=0,
            notes=(
                "local file differs from manifest-derived bytes; kept local. "
                "Pass --force to overwrite."
            ),
        )

    _ensure_parent(dst_path)
    dst_path.write_bytes(new_bytes)
    status: Literal["created", "overwritten"] = (
        "overwritten" if dst_existing else "created"
    )
    return CopyRecord(
        src=src_rel,
        dst=dst_rel,
        mode=mode,
        status=status,
        src_sha256=src_sha,
        dst_sha256_before=dst_sha_before,
        dst_sha256_after=new_sha,
        bytes_written=len(new_bytes),
    )


def run(*, force: bool = False) -> dict:
    if not SFT_SCRIPTS.exists():
        raise FileNotFoundError(
            f"{SFT_SCRIPTS} not found. Bootstrap requires the SFT scripts tree "
            f"to be present at this path; confirm 4.sdg_sft is checked out."
        )

    sft_root, copies = _load_manifest(BOOTSTRAP_MANIFEST)
    records: list[CopyRecord] = []
    for entry in copies:
        rec = _process_copy(entry, sft_root, force=force)
        records.append(rec)
        logger.info(
            "%-32s -> %-40s [%s] %s",
            rec.src, rec.dst, rec.mode, rec.status,
        )

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "sft_root": str(sft_root),
        "pipeline_root": str(PIPELINE_ROOT),
        "force": force,
        "n_total": len(records),
        "n_created": sum(1 for r in records if r.status == "created"),
        "n_overwritten": sum(1 for r in records if r.status == "overwritten"),
        "n_unchanged": sum(1 for r in records if r.status == "unchanged"),
        "n_diverged_kept_local": sum(
            1 for r in records if r.status == "diverged_kept_local"
        ),
        "records": [asdict(r) for r in records],
    }

    BOOTSTRAP_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with BOOTSTRAP_REPORT.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("bootstrap report written to %s", BOOTSTRAP_REPORT)
    return summary


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
        prog="pipeline.bootstrap",
        description="One-time copy of SFT scripts into pipeline/ (idempotent).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local edits to copied files.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging."
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    summary = run(force=args.force)
    diverged = summary["n_diverged_kept_local"]
    if diverged and not args.force:
        logger.warning(
            "%d file(s) diverged from manifest-derived bytes; kept local. "
            "Re-run with --force to overwrite.",
            diverged,
        )
    logger.info(
        "bootstrap complete: %d total | %d created | %d overwritten | "
        "%d unchanged | %d diverged",
        summary["n_total"], summary["n_created"], summary["n_overwritten"],
        summary["n_unchanged"], summary["n_diverged_kept_local"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
