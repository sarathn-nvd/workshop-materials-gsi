"""Per-notebook uv virtualenv bootstrap for workshop Jupyter notebooks.

Each notebook gets its own ``.venv`` under its working directory. The first code
cell calls :func:`bootstrap_notebook_env` once per session; missing packages are
installed with ``uv pip`` (no pip module required in the venv).
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


_UV_INSTALL_CMD = "curl -LsSf https://astral.sh/uv/install.sh | sh"
_UV_BIN_DIRS = (
    Path.home() / ".local" / "bin",
    Path.home() / ".cargo" / "bin",
)


def _ensure_uv_on_path() -> None:
    path_parts = os.environ.get("PATH", "").split(":")
    for d in _UV_BIN_DIRS:
        if d.is_dir():
            d_str = str(d)
            if d_str not in path_parts:
                os.environ["PATH"] = f"{d_str}:{os.environ.get('PATH', '')}"


def _install_uv() -> None:
    print("uv not found — installing via astral.sh ...")
    subprocess.check_call(["sh", "-c", _UV_INSTALL_CMD])
    _ensure_uv_on_path()


def _uv() -> str:
    _ensure_uv_on_path()
    uv = shutil.which("uv")
    if uv is None:
        _install_uv()
        uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "uv install completed but the binary is still not on PATH. "
            f"Try manually: {_UV_INSTALL_CMD}"
        )
    return uv


def bootstrap_notebook_env(nb_dir: Path | None = None) -> Path:
    """Create or reuse a notebook-local ``.venv`` and wire it into this kernel."""
    nb_dir = (nb_dir or Path.cwd()).resolve()
    venv_dir = nb_dir / ".venv"
    py = venv_dir / "bin" / "python"

    if not py.exists():
        print(f"creating uv venv: {venv_dir}")
        subprocess.check_call(
            [
                _uv(),
                "venv",
                str(venv_dir),
                "--python",
                f"{sys.version_info.major}.{sys.version_info.minor}",
                "--allow-existing",
            ]
        )

    site = (
        venv_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site.is_dir():
        site_str = str(site)
        if site_str not in sys.path:
            sys.path.insert(0, site_str)

    bin_dir = str(venv_dir / "bin")
    path_parts = os.environ.get("PATH", "").split(":")
    if bin_dir not in path_parts:
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    sys.executable = str(py)
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    print(f"notebook env: {venv_dir} (python {py})")
    return py


def _importable(mod: str) -> bool:
    """Check importability in the notebook venv, not the Jupyter kernel env."""
    probe = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec({mod!r}) else 1)"
    )
    return (
        subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
        ).returncode
        == 0
    )


def ensure(
    mod: str,
    packages: list[str],
    *,
    extra_index: str | None = None,
    quiet: bool = False,
) -> None:
    """Install *packages* with uv when *mod* is not importable."""
    if _importable(mod):
        print(f"ok: {mod}")
        return

    print(f"installing: {mod} ...")
    cmd = [_uv(), "pip", "install", "--python", sys.executable, *packages]
    if extra_index:
        cmd.extend(["--extra-index-url", extra_index])
    if quiet:
        cmd.append("-q")
    subprocess.check_call(cmd)
    importlib.invalidate_caches()
