"""Ensure Docker and workshop caches use a spacious data volume.

Call :func:`ensure_docker_storage` from the first code cell of any notebook
that runs ``docker pull`` / ``docker run``. Idempotent — safe to re-run.

Auto-detects whether the spacious volume is mounted at ``/data`` or
``/ephemeral`` (override with ``WORKSHOP_STORAGE_ROOT``).

Logs to ``<storage>/logs/docker_storage.log`` and stdout.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

DAEMON_JSON = Path("/etc/docker/daemon.json")
LEGACY_DOCKER_ROOT = Path("/var/lib/docker")
LEGACY_CONTAINERD_ROOT = Path("/var/lib/containerd")

_STORAGE_ROOT: Path | None = None
_LOG = logging.getLogger("docker_storage")


def workshop_storage_root() -> Path:
    """Return the spacious volume mount (``/data`` or ``/ephemeral``)."""
    global _STORAGE_ROOT
    if _STORAGE_ROOT is not None:
        return _STORAGE_ROOT

    if env := os.environ.get("WORKSHOP_STORAGE_ROOT", "").strip():
        _STORAGE_ROOT = Path(env)
        return _STORAGE_ROOT

    root_dev = os.stat("/").st_dev
    candidates: list[tuple[Path, int, bool]] = []
    for name in ("/data", "/ephemeral"):
        path = Path(name)
        if not path.is_dir():
            continue
        try:
            free = shutil.disk_usage(path).free
            separate = os.stat(path).st_dev != root_dev
            candidates.append((path, free, separate))
        except OSError:
            continue

    if not candidates:
        _STORAGE_ROOT = Path("/data")
    else:
        pool = [c for c in candidates if c[2]] or candidates
        _STORAGE_ROOT = max(pool, key=lambda c: c[1])[0]

    os.environ.setdefault("WORKSHOP_STORAGE_ROOT", str(_STORAGE_ROOT))
    return _STORAGE_ROOT


def docker_data_root() -> Path:
    return Path(os.environ.get("DOCKER_DATA_ROOT", workshop_storage_root() / "docker"))


def workshop_log_dir() -> Path:
    return Path(os.environ.get("WORKSHOP_LOG_DIR", workshop_storage_root() / "logs"))


def workshop_cache_root() -> Path:
    return Path(os.environ.get("WORKSHOP_CACHE_ROOT", workshop_storage_root() / "cache"))


def workshop_work_root() -> Path:
    return Path(os.environ.get("WORKSHOP_WORK_ROOT", workshop_storage_root() / "workshop"))


def _setup_logging() -> Path:
    log_dir = workshop_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "docker_storage.log"
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
    _LOG.info("stopping docker and containerd ...")
    _sudo(["systemctl", "stop", "docker"], check=False)
    _sudo(["systemctl", "stop", "containerd"], check=False)


def _start_docker() -> None:
    _LOG.info("starting containerd then docker ...")
    _sudo(["systemctl", "start", "containerd"])
    _sudo(["systemctl", "start", "docker"])
    for _ in range(30):
        if _docker_running():
            _LOG.info("docker is running")
            return
        time.sleep(1)
    log_file = workshop_log_dir() / "docker_storage.log"
    raise RuntimeError(f"docker failed to start — see {log_file}")


def _set_daemon_data_root(data_root: Path) -> None:
    cfg = _read_daemon_config()
    cfg["data-root"] = str(data_root)
    tmp = workshop_cache_root() / "tmp" / "daemon.json.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cfg, indent=4) + "\n")
    _sudo(["cp", str(tmp), str(DAEMON_JSON)])
    tmp.unlink(missing_ok=True)
    _LOG.info("set docker data-root -> %s", data_root)


def _configure_nvidia_runtime() -> None:
    _LOG.info("configuring NVIDIA container runtime for docker ...")
    _sudo(
        ["nvidia-ctk", "runtime", "configure", "--runtime=docker"],
        check=False,
    )
    _sudo(
        [
            "nvidia-ctk",
            "config",
            "--in-place",
            "--set",
            "nvidia-container-runtime.mode=legacy",
        ],
        check=False,
    )


def _containerd_target(storage_root: Path) -> Path:
    return storage_root / "containerd-data"


def _containerd_needs_migration(storage_root: Path) -> bool:
    target = _containerd_target(storage_root)
    if not LEGACY_CONTAINERD_ROOT.exists():
        return True
    if LEGACY_CONTAINERD_ROOT.is_symlink():
        try:
            return LEGACY_CONTAINERD_ROOT.resolve() != target.resolve()
        except OSError:
            return True
    # Real directory on root disk — migrate to spacious volume.
    return True


def _ensure_containerd_storage(storage_root: Path) -> None:
    target = _containerd_target(storage_root)
    if not _containerd_needs_migration(storage_root):
        _LOG.info("containerd already on %s", target)
        return

    _LOG.info("relocating containerd data to %s ...", target)
    _sudo(["mkdir", "-p", str(target)])

    if LEGACY_CONTAINERD_ROOT.exists() and not LEGACY_CONTAINERD_ROOT.is_symlink():
        if _sudo_dir_nonempty(LEGACY_CONTAINERD_ROOT):
            _LOG.info("migrating %s -> %s (rsync) ...", LEGACY_CONTAINERD_ROOT, target)
            _sudo(
                [
                    "rsync",
                    "-aHAXx",
                    f"{LEGACY_CONTAINERD_ROOT}/",
                    f"{target}/",
                ]
            )
        _sudo(["rm", "-rf", str(LEGACY_CONTAINERD_ROOT)])

    if not LEGACY_CONTAINERD_ROOT.exists():
        _sudo(["ln", "-s", str(target), str(LEGACY_CONTAINERD_ROOT)])
        _LOG.info("symlinked %s -> %s", LEGACY_CONTAINERD_ROOT, target)


def _migrate_docker_data(target: Path) -> None:
    _sudo(["mkdir", "-p", str(target)])
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

    if LEGACY_DOCKER_ROOT.exists():
        _LOG.info("removing legacy docker dir %s to free root disk", LEGACY_DOCKER_ROOT)
        _sudo(["rm", "-rf", str(LEGACY_DOCKER_ROOT)])


def _ensure_user_caches() -> None:
    """Point temp and download caches at the spacious volume."""
    cache_root = workshop_cache_root()
    dirs = {
        "TMPDIR": cache_root / "tmp",
        "DOCKER_TMPDIR": cache_root / "tmp",
        "PIP_CACHE_DIR": cache_root / "pip",
        "UV_CACHE_DIR": cache_root / "uv",
        "HF_HOME": cache_root / "hf",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "LOCAL_NIM_CACHE": cache_root / "nim",
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
    storage_root = workshop_storage_root()
    log_file = _setup_logging()
    data_root = Path(data_root or docker_data_root())
    _LOG.info("=== ensure_docker_storage (storage=%s, log: %s) ===", storage_root, log_file)

    root_stats = _disk_stats(Path("/"))
    data_stats = _disk_stats(storage_root)
    _LOG.info(
        "disk /: %.1fG used / %.1fG (%.1f%%)",
        root_stats["used_gb"],
        root_stats["total_gb"],
        root_stats["pct_used"],
    )
    _LOG.info(
        "disk %s: %.1fG used / %.1fG (%.1fG free)",
        storage_root,
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
        or _containerd_needs_migration(storage_root)
    )

    if need_migrate and data_stats["free_gb"] < min_free_gb:
        raise RuntimeError(
            f"Target volume {storage_root} has only {data_stats['free_gb']}G free; "
            f"need at least {min_free_gb}G for docker migration."
        )

    if need_migrate:
        _LOG.info("migrating docker/containerd storage to %s ...", storage_root)
        _stop_docker()
        _migrate_docker_data(data_root)
        _ensure_containerd_storage(storage_root)
        _set_daemon_data_root(data_root)
        _configure_nvidia_runtime()
        _start_docker()
    elif not _docker_running():
        _start_docker()

    final_root = _docker_root_from_info() or str(data_root)
    final_stats = _disk_stats(Path(final_root).parent)
    _LOG.info("docker ready — data-root=%s, free=%sG", final_root, final_stats["free_gb"])
    print(
        f"docker storage ok: {final_root} "
        f"({final_stats['free_gb']}G free on {storage_root})"
    )
    return Path(final_root)


def workshop_work_dir(module_name: str) -> Path:
    """Writable training work dir on the spacious volume (checkpoints, caches)."""
    work_dir = workshop_work_root() / module_name / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def workshop_module_name(nb_dir: Path | None = None) -> str:
    """Notebook folder name, e.g. ``M7-model_training``."""
    return (nb_dir or Path.cwd()).resolve().name


def workshop_nim_cache_dir() -> Path:
    """Host path for NIM container cache bind-mounts."""
    path = workshop_cache_root() / "nim"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_NIM_CACHE"] = str(path)
    return path


def workshop_download_dir(subpath: str = "") -> Path:
    """Writable download area on the spacious volume (e.g. HF checkpoints)."""
    base = workshop_storage_root() / "downloads"
    path = base / subpath if subpath else base
    path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_work_dir(nb_dir: Path) -> Path:
    """Repo-local ``work/`` kept for read-only fallback after migration."""
    return nb_dir.resolve() / "work"


def setup_workshop_paths(
    nb_dir: Path | None = None,
    *,
    module_name: str | None = None,
    min_free_gb: float = 20.0,
) -> tuple[Path, Path]:
    """Ensure spacious-volume storage and return ``(notebook_dir, work_dir)``."""
    ensure_docker_storage(min_free_gb=min_free_gb)
    nb_dir = (nb_dir or Path.cwd()).resolve()
    work_dir = workshop_work_dir(module_name or workshop_module_name(nb_dir))
    workshop_nim_cache_dir()
    return nb_dir, work_dir


def glob_work_paths(
    work_dir: Path,
    nb_dir: Path,
    pattern: str,
) -> list[Path]:
    """Glob on spacious-volume work dir, then fall back to legacy repo ``work/``."""
    hits = sorted(work_dir.glob(pattern))
    if hits:
        return hits
    return sorted(legacy_work_dir(nb_dir).glob(pattern))


def docker_passwd_workaround_env() -> list[str]:
    """`-e` flags so PyTorch works when ``docker run -u <host-uid>`` has no /etc/passwd entry."""
    return [
        "-e",
        "USER=workshop",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
    ]


def docker_workspace_volumes(nb_dir: Path, work_dir: Path) -> list[str]:
    """Split bind-mounts: small inputs from notebook dir, large outputs on spacious volume."""
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
