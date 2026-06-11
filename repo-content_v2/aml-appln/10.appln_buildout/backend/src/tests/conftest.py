"""Shared test fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Anchor every test on the backend's local data plane.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "data"
os.environ.setdefault("NAT_AML_DATA_DIR", str(DATA_DIR))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def dp(data_dir):
    from aml_app.utils.data_loader import get_data_plane, reset_data_plane
    reset_data_plane()
    return get_data_plane(str(data_dir))


@pytest.fixture(autouse=True)
def _isolate_traces_dir(tmp_path, monkeypatch, data_dir):
    """Redirect traces / dispositions writes to a per-test tmpdir so the
    real ./data/traces stays clean."""
    # Build a shadow data dir whose write-only subfolders point at tmp.
    shadow = tmp_path / "data"
    shadow.mkdir()
    # Symlink the read-only subfolders so loaders see them.
    for sub in ("tool_1_transactions", "tool_2_kyc", "tool_3_sanctions",
                "tool_4_policy", "tool_5_sop", "demo", "seeded_subpopulations",
                "seed_traces"):
        src = data_dir / sub
        if src.exists():
            (shadow / sub).symlink_to(src)
    # Force the data loader to point at the shadow.
    from aml_app.utils import data_loader
    data_loader.reset_data_plane()
    data_loader.get_data_plane(str(shadow))
    monkeypatch.setenv("NAT_AML_DATA_DIR", str(shadow))
    yield
    data_loader.reset_data_plane()
