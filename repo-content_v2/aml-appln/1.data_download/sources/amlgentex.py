"""AMLGentex — SFT source (synthetic AML transaction generator).

The upstream repo (`aidotse/AMLGentex`) ships a generator framework, not
raw data. To keep Step 1's contract — *every entry under data/raw/<phase>/
must be actual data, not source code or build artefacts* — we split
AMLGentex into three locations:

1. **Out-of-tree source clone** at ``1.data_download/.amlgentex_repo/`` —
   the upstream Python package. Step 5 may re-invoke its generator from
   here with custom configs at larger scale.
2. **Out-of-tree generator venv** at ``1.data_download/.amlgentex_venv/`` —
   AMLGentex's pinned dependency set (numpy/scipy/sklearn/torch) installed
   by ``uv sync`` so its versions don't collide with our top-level venv.
3. **In-tree synthetic data** at ``data/raw/sft/amlgentex/synthetic/`` —
   the actual generator outputs (transactions, accounts, alert patterns,
   tx_log.parquet). This is the only thing that counts toward
   ``phase_stats.json`` for this source.

The synthetic baseline is small enough not to blow up Step 1 wall-clock
(~5 min cold, ~3 min warm) and gives downstream Step 5 a tangible
starting corpus. Step 5 / Step 7 may re-run the generator with different
scale parameters using the very same ``.amlgentex_repo/`` clone — that's
the contract we preserve.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from ._common import StatsTracker, dir_size, source_dir

logger = logging.getLogger(__name__)

AMLGENTEX_REPO = "https://github.com/aidotse/AMLGentex.git"

# Use the repo's bundled template as the baseline config. The template is
# small but exercises every layer of the pipeline (spatial graph, temporal
# simulation, alert-pattern injection, demographics-driven KYC). Step 5 can
# scale up later with its own custom configs.
TEMPLATE_RELATIVE_CONFIG = "experiments/template_experiment/config/data.yaml"

# Subprocess time budget for ``uv sync``. AMLGentex pulls in torch + a few
# heavy ML deps; cold install is ~3–6 min on a fast box. We give it 15 min
# so transient PyPI slowness doesn't fail the run.
UV_SYNC_TIMEOUT_S = 900
GENERATE_TIMEOUT_S = 1800   # 30 min cap on generator runtime

# AMLGentex source code + generator venv both live *outside* the data
# tree. They're build artefacts (the package source + its pinned deps),
# not data. Storing them here keeps ``data/raw/sft/amlgentex/`` showing
# only the actual synthetic outputs.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GENERATOR_REPO = _PROJECT_ROOT / ".amlgentex_repo"
_GENERATOR_VENV = _PROJECT_ROOT / ".amlgentex_venv"


def _clone_repo(repo_dir: Path) -> bool:
    """Clone AMLGentex if not already present. Returns True if cloned now."""
    if (repo_dir / ".git").exists():
        logger.info("amlgentex: repo already cloned at %s", repo_dir)
        return False
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    from git import Repo
    logger.info("amlgentex: cloning %s", AMLGENTEX_REPO)
    Repo.clone_from(AMLGENTEX_REPO, repo_dir, depth=1)
    return True


def _resolve_uv() -> str:
    """Return absolute path to ``uv`` (installed via requirements.txt)."""
    found = shutil.which("uv")
    if not found:
        raise RuntimeError(
            "amlgentex: 'uv' not found in PATH. Install with `pip install uv` "
            "or re-run setup.sh."
        )
    return found


def _generator_env(uv: str) -> dict[str, str]:
    """Env vars for ``uv`` so the generator venv lives outside the data tree."""
    return {
        **os.environ,
        # Point uv at our out-of-tree venv location. Honored by both
        # ``uv sync`` and ``uv run``.
        "UV_PROJECT_ENVIRONMENT": str(_GENERATOR_VENV),
    }


def _uv_sync(repo_dir: Path, uv: str) -> None:
    """Install AMLGentex's deps into the out-of-tree venv via ``uv sync``."""
    venv_marker = _GENERATOR_VENV / "pyvenv.cfg"
    if venv_marker.exists():
        logger.info("amlgentex: deps already synced (%s)", venv_marker)
        return
    logger.info(
        "amlgentex: installing pinned deps via 'uv sync' to %s (~3–6 min)…",
        _GENERATOR_VENV,
    )
    proc = subprocess.run(
        [uv, "sync", "--no-progress"],
        cwd=repo_dir,
        timeout=UV_SYNC_TIMEOUT_S,
        capture_output=True,
        text=True,
        env=_generator_env(uv),
    )
    if proc.returncode != 0:
        # Surface enough of the failure to debug, but don't dump 50 KB.
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"amlgentex: 'uv sync' failed (rc={proc.returncode})\n{tail}")
    logger.info("amlgentex: deps synced")


def _run_generator(repo_dir: Path, uv: str, log_path: Path) -> int:
    """Invoke ``scripts/generate.py`` with the bundled template config.

    Output paths in the template config are absolute (developer-machine
    paths) but ``load_data_config`` in AMLGentex auto-rewrites them
    relative to the config-file directory, so the artifacts land in
    ``experiments/template_experiment/{spatial,temporal}/``. We copy
    them out into ``data/raw/sft/amlgentex/synthetic/`` afterwards.

    Returns the subprocess return code.
    """
    config = repo_dir / TEMPLATE_RELATIVE_CONFIG
    if not config.exists():
        raise RuntimeError(
            f"amlgentex: template config missing at {config} — repo layout "
            "may have changed upstream"
        )
    logger.info("amlgentex: running generator (config=%s)…", config.name)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [uv, "run", "python", "scripts/generate.py", "--conf_file", str(config)],
            cwd=repo_dir,
            timeout=GENERATE_TIMEOUT_S,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=_generator_env(uv),
        )
    return proc.returncode


def _collect_artifacts(repo_dir: Path, dest: Path) -> tuple[int, int]:
    """Copy generator outputs out of the experiments/ tree into ``dest``.

    AMLGentex writes two output dirs per run:
      * ``experiments/<name>/spatial/``  — accounts.csv, transactions.csv,
        alert_models.csv, normal_models.csv, spatial graph artefacts
      * ``experiments/<name>/temporal/`` — tx_log.parquet (the temporal
        transaction stream — the headline artefact)
    We mirror both under ``dest`` so downstream code finds everything it
    needs in one place. Returns (files_copied, bytes_copied).
    """
    template_root = repo_dir / "experiments" / "template_experiment"
    files = 0
    nbytes = 0
    for sub in ("spatial", "temporal"):
        src = template_root / sub
        if not src.exists():
            continue
        target = dest / sub
        target.mkdir(parents=True, exist_ok=True)
        for p in src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(src)
                out_path = target / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out_path)
                files += 1
                nbytes += out_path.stat().st_size
    return files, nbytes


def download_amlgentex(tracker: StatsTracker) -> None:
    """Clone AMLGentex + run its generator; emit synthetic transaction data.

    Build artefacts (source clone, generator dep venv) live outside the
    data tree. Only the synthetic outputs land under ``data/raw/sft/...``.
    """
    with tracker.track(source="amlgentex", phase="sft") as stats:
        synthetic_dir = source_dir("sft", "amlgentex") / "synthetic"
        repo_dir = _GENERATOR_REPO

        cloned = _clone_repo(repo_dir)

        # Idempotent skip on the generator side: if a tx_log.parquet already
        # exists in synthetic/, treat the source as fully populated.
        existing_tx_log = synthetic_dir / "temporal" / "tx_log.parquet"
        if existing_tx_log.exists() and existing_tx_log.stat().st_size > 0:
            logger.info(
                "amlgentex: synthetic data already present at %s — skipping generator",
                existing_tx_log,
            )
        else:
            uv = _resolve_uv()
            _uv_sync(repo_dir, uv)

            log_path = synthetic_dir.parent / "generate.log"
            synthetic_dir.parent.mkdir(parents=True, exist_ok=True)
            rc = _run_generator(repo_dir, uv, log_path)
            if rc != 0:
                # Don't fail the whole task — the repo is still usable for
                # later steps. Surface enough info to debug.
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8")[-2000:]
                except OSError:
                    pass
                logger.warning(
                    "amlgentex: generator returned rc=%d; see %s for details. tail:\n%s",
                    rc, log_path, tail,
                )
                stats.notes["generator_rc"] = rc
            files, nbytes = _collect_artifacts(repo_dir, synthetic_dir)
            stats.notes["synthetic_files_copied"] = files
            stats.notes["synthetic_bytes_copied"] = nbytes

        # Stats reflect only the in-tree data, not the out-of-tree build
        # artefacts (repo clone, generator venv).
        stats.files_written = sum(
            1 for _ in synthetic_dir.rglob("*") if _.is_file()
        ) if synthetic_dir.exists() else 0
        stats.bytes_written = dir_size(synthetic_dir) if synthetic_dir.exists() else 0
        stats.notes["repo_cloned"] = cloned
        stats.notes["repo_url"] = AMLGENTEX_REPO
        stats.notes["repo_location"] = str(_GENERATOR_REPO.relative_to(_PROJECT_ROOT))
        stats.notes["venv_location"] = str(_GENERATOR_VENV.relative_to(_PROJECT_ROOT))


TASKS_SFT = {"amlgentex": download_amlgentex}
ALL_TASKS = dict(TASKS_SFT)
