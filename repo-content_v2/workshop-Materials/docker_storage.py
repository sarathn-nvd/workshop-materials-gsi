"""Ensure Docker and workshop caches use a spacious data volume.

Call :func:`ensure_docker_storage` from the first code cell of any notebook
that runs ``docker pull`` / ``docker run``. Idempotent — safe to re-run.

Logs to ``/data/logs/docker_storage.log`` and stdout.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.environ.get("DOCKER_DATA_ROOT", "/data/docker"))
LOG_DIR = Path(os.environ.get("WORKSHOP_LOG_DIR", "/data/logs"))
CACHE_ROOT = Path(os.environ.get("WORKSHOP_CACHE_ROOT", "/data/cache"))
DEFAULT_WORK_ROOT = Path(os.environ.get("WORKSHOP_WORK_ROOT", "/data/workshop"))
DAEMON_JSON = Path("/etc/docker/daemon.json")
LEGACY_DOCKER_ROOT = Path("/var/lib/docker")

_LOG = logging.getLogger("docker_storage")


def _setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "docker_storage.log"
    if not _LOG.handlers:
        _LOG.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        _LOG.addHandler(fh)
        _LOG.addHandler(sh)
    return log_file


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    _LOG.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _sudo(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["sudo", "-n", *cmd], check=check)


def _sudo_path_exists(path: Path) -> bool:
    """Existence check that works on root-owned paths (e.g. docker data-root)."""
    return _sudo(["test", "-e", str(path)], check=False).returncode == 0


def _sudo_dir_nonempty(path: Path) -> bool:
    return (
        _sudo(
            ["find", str(path), "-mindepth", "1", "-print", "-quit"],
            check=False,
        ).returncode
        == 0
    )


def _target_has_docker_data(target: Path) -> bool:
    """True when target already holds docker storage (marker or daemon layout)."""
    marker = target / ".migration_complete"
    if _sudo_path_exists(marker) and _sudo_dir_nonempty(target):
        return True
    return any(
        _sudo_path_exists(target / name)
        for name in ("overlay2", "containers", "image")
    )


def _disk_stats(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / 2**30, 1),
        "used_gb": round(usage.used / 2**30, 1),
        "free_gb": round(usage.free / 2**30, 1),
        "pct_used": round(100 * usage.used / usage.total, 1) if usage.total else 0,
    }


def _read_daemon_config() -> dict:
    if not DAEMON_JSON.exists():
        return {}
    try:
        return json.loads(DAEMON_JSON.read_text())
    except json.JSONDecodeError:
        _LOG.warning("could not parse %s", DAEMON_JSON)
        return {}


def _docker_root_from_info() -> str | None:
    try:
        cp = _run(["docker", "info", "--format", "{{.DockerRootDir}}"], check=False)
    except FileNotFoundError:
        return None
    if cp.returncode != 0:
        return None
    root = (cp.stdout or "").strip()
    return root or None


def _docker_running() -> bool:
    cp = _run(["docker", "info"], check=False)
    return cp.returncode == 0


def _stop_docker() -> None:
    _LOG.info("stopping docker ...")
    _sudo(["systemctl", "stop", "docker", "docker.socket", "containerd"], check=False)


def _start_docker() -> None:
    _LOG.info("starting docker ...")
    _sudo(["systemctl", "start", "docker"])
    for _ in range(30):
        if _docker_running():
            _LOG.info("docker is running")
            return
        time.sleep(1)
    raise RuntimeError("docker failed to start — see /data/logs/docker_storage.log")


def _set_daemon_data_root(data_root: Path) -> None:
    cfg = _read_daemon_config()
    cfg["data-root"] = str(data_root)
    tmp = CACHE_ROOT / "tmp" / "daemon.json.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cfg, indent=4) + "\n")
    _sudo(["cp", str(tmp), str(DAEMON_JSON)])
    tmp.unlink(missing_ok=True)
    _LOG.info("set docker data-root -> %s", data_root)


def _migrate_docker_data(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if not LEGACY_DOCKER_ROOT.exists():
        _LOG.info("no legacy docker dir at %s — nothing to migrate", LEGACY_DOCKER_ROOT)
        return

    marker = target / ".migration_complete"
    if _target_has_docker_data(target):
        _LOG.info("docker data already present at %s", target)
        if not _sudo_path_exists(marker):
            _sudo(["touch", str(marker)])
    else:
        _LOG.info("migrating %s -> %s (rsync) ...", LEGACY_DOCKER_ROOT, target)
        _sudo(
            [
                "rsync",
                "-aHAXx",
                f"{LEGACY_DOCKER_ROOT}/",
                f"{target}/",
            ]
        )
        _sudo(["touch", str(marker)])
        _LOG.info("rsync complete")

    backup = LEGACY_DOCKER_ROOT.with_suffix(".docker.bak")
    if LEGACY_DOCKER_ROOT.exists():
        _LOG.info("removing legacy docker dir %s to free root disk", LEGACY_DOCKER_ROOT)
        _sudo(["rm", "-rf", str(LEGACY_DOCKER_ROOT)])


def _ensure_user_caches() -> None:
    """Point temp and download caches at the spacious volume."""
    dirs = {
        "TMPDIR": CACHE_ROOT / "tmp",
        "DOCKER_TMPDIR": CACHE_ROOT / "tmp",
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "UV_CACHE_DIR": CACHE_ROOT / "uv",
        "HF_HOME": CACHE_ROOT / "hf",
        "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        _LOG.info("env %s=%s", name, path)


def ensure_docker_storage(
    data_root: Path | str | None = None,
    *,
    min_free_gb: float = 20.0,
) -> Path:
    """Ensure Docker storage and caches live on a volume with enough free space."""
    log_file = _setup_logging()
    data_root = Path(data_root or DEFAULT_DATA_ROOT)
    _LOG.info("=== ensure_docker_storage (log: %s) ===", log_file)

    root_stats = _disk_stats(Path("/"))
    data_stats = _disk_stats(data_root.parent)
    _LOG.info(
        "disk /: %.1fG used / %.1fG (%.1f%%)",
        root_stats["used_gb"],
        root_stats["total_gb"],
        root_stats["pct_used"],
    )
    _LOG.info(
        "disk %s: %.1fG used / %.1fG (%.1fG free)",
        data_root.parent,
        data_stats["used_gb"],
        data_stats["total_gb"],
        data_stats["free_gb"],
    )

    _ensure_user_caches()

    cfg_root = _read_daemon_config().get("data-root")
    live_root = _docker_root_from_info()
    current_root = Path(live_root or cfg_root or LEGACY_DOCKER_ROOT)
    _LOG.info("docker data-root (config): %s", cfg_root)
    _LOG.info("docker data-root (live):   %s", live_root)

    need_migrate = (
        str(current_root) != str(data_root)
        or (root_stats["pct_used"] >= 90 and str(current_root).startswith("/var"))
        or (LEGACY_DOCKER_ROOT.exists() and data_stats["free_gb"] >= min_free_gb)
    )

    if need_migrate and data_stats["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"Target volume {data_root.parent} has only {data_stats['free_gb']}G free; "
            f"need at least {min_free_gb}G for docker migration."
        )

    if need_migrate:
        _LOG.info("migrating docker storage to %s ...", data_root)
        _stop_docker()
        _migrate_docker_data(data_root)
        _set_daemon_data_root(data_root)
        _start_docker()
    elif not _docker_running():
        _start_docker()

    final_root = _docker_root_from_info() or str(data_root)
    final_stats = _disk_stats(Path(final_root).parent)
    _LOG.info("docker ready — data-root=%s, free=%sG", final_root, final_stats["free_gb"])
    print(f"docker storage ok: {final_root} ({final_stats['free_gb']}G free on volume)")
    return Path(final_root)


def workshop_work_dir(module_name: str) -> Path:
    """Writable training work dir on the spacious volume (checkpoints, caches)."""
    work_dir = DEFAULT_WORK_ROOT / module_name / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def docker_workspace_volumes(nb_dir: Path, work_dir: Path) -> list[str]:
    """Split bind-mounts: small inputs from notebook dir, large outputs on /data."""
    nb_dir = nb_dir.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    return [
        "-v",
        f"{nb_dir / 'data'}:/workspace/data:ro",
        "-v",
        f"{nb_dir / 'recipes'}:/workspace/recipes",
        "-v",
        f"{work_dir}:/workspace/work",
    ]


if __name__ == "__main__":
    ensure_docker_storage()
