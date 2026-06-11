#!/usr/bin/env python3
"""Build a self-contained backend deploy zip (no venv, portable data paths)."""
from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
OUT_ZIP = BACKEND.parent / "backend-deploy.zip"
STAGING = BACKEND.parent / ".backend-deploy-staging"

# Host-specific prefix baked into eval/benchmark artifacts during development.
HOST_PREFIX = "/data/swami/gsi-training/10.appln_buildout/backend/"
HOST_DATA_PREFIX = HOST_PREFIX + "data/"

EXCLUDE_DIRS = {"env", "__pycache__", ".pytest_cache", ".git"}
EXCLUDE_FILES = {"_nat_e2e.log"}  # dev-only log with host paths
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}


def should_skip_dir(name: str) -> bool:
    return name in EXCLUDE_DIRS


def copy_tree(src: Path, dst: Path) -> None:
    """Copy backend → staging; dereference symlinks so data is real files."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for root, dirs, files in os.walk(src, followlinks=False):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        if any(p in EXCLUDE_DIRS for p in rel.parts):
            dirs.clear()
            continue

        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        # Materialize symlinked subdirs (e.g. data/_shadow_*/traces_* → real copies).
        for name in list(dirs):
            if should_skip_dir(name):
                dirs.remove(name)
                continue
            src_d = root_p / name
            if src_d.is_symlink():
                dirs.remove(name)
                dst_d = out_dir / name
                target = src_d.resolve()
                if target.is_dir():
                    if dst_d.exists():
                        shutil.rmtree(dst_d)
                    shutil.copytree(target, dst_d, symlinks=False)
                else:
                    shutil.copy2(target, dst_d)

        for name in files:
            if name in EXCLUDE_FILES:
                continue
            sp = root_p / name
            if sp.suffix == ".pyc":
                continue
            dp = out_dir / name
            if sp.is_symlink():
                target = sp.resolve()
                if target.is_dir():
                    shutil.copytree(target, dp, symlinks=False, dirs_exist_ok=True)
                else:
                    shutil.copy2(target, dp)
            else:
                shutil.copy2(sp, dp)


def portable_text(text: str) -> str:
    text = text.replace(HOST_DATA_PREFIX, "data/")
    text = text.replace(HOST_PREFIX, "")
    return text


def rewrite_json_file(path: Path) -> bool:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    raw = json.dumps(doc)
    new_raw = portable_text(raw)
    if new_raw != raw:
        path.write_text(
            json.dumps(json.loads(new_raw), indent=2) + "\n",
            encoding="utf-8",
        )
        return True
    return False


def rewrite_text_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    new = portable_text(text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def rewrite_paths(staging: Path) -> int:
    changed = 0
    for path in staging.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            if rewrite_json_file(path):
                changed += 1
        elif path.suffix in TEXT_SUFFIXES:
            if rewrite_text_file(path):
                changed += 1
    return changed


def make_zip(staging: Path, out_zip: Path) -> tuple[int, int]:
    if out_zip.exists():
        out_zip.unlink()
    count = 0
    total = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                arc = "backend/" + path.relative_to(staging).as_posix()
                zf.write(path, arc)
                count += 1
                total += path.stat().st_size
    return count, total


def main() -> None:
    print(f"Staging from {BACKEND} …")
    copy_tree(BACKEND, STAGING)
    n = rewrite_paths(STAGING)
    print(f"Rewrote absolute paths in {n} text/json files")
    files, uncompressed = make_zip(STAGING, OUT_ZIP)
    zip_size = OUT_ZIP.stat().st_size
    print(f"Created {OUT_ZIP}")
    print(f"  files: {files:,}")
    print(f"  uncompressed: {uncompressed / 1024 / 1024:.2f} MiB")
    print(f"  zip size: {zip_size / 1024 / 1024:.2f} MiB")
    shutil.rmtree(STAGING)


if __name__ == "__main__":
    main()
