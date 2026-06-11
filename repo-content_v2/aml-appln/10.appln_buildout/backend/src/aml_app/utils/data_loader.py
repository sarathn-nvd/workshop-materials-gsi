"""Process-singleton loaders for the local data plane.

Loads Parquet / CSV / JSONL / Markdown artifacts from `./data/` once per
process and keeps them in memory. Idempotent. Used by every data tool
and by the analytics / system routes.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("aml_app.utils.data_loader")


class DataPlane:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).resolve()
        self._lock = threading.Lock()
        self._tx_df: pd.DataFrame | None = None
        self._kyc_idx: dict[str, dict] | None = None
        self._ofac: list[dict] | None = None
        self._pep: list[dict] | None = None
        self._policy_df: pd.DataFrame | None = None
        self._sops: dict[str, list[tuple[str, str]]] | None = None
        self._manifest: list[dict] | None = None
        self._eval_keys: dict[str, dict] | None = None
        self._stratification: dict | None = None
        self._suspicious: pd.DataFrame | None = None
        self._near_miss: pd.DataFrame | None = None
        self._seed_traces: list[dict] | None = None
        self._tx_schema: dict | None = None
        self._kyc_schema: dict | None = None
        self._tx_stats: dict | None = None
        self._kyc_stats: dict | None = None

    # ------------------------------------------------------------------
    # Tool 1 — transactions
    # ------------------------------------------------------------------
    def transactions(self) -> pd.DataFrame:
        with self._lock:
            if self._tx_df is None:
                path = self.data_dir / "tool_1_transactions" / "transactions.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"Missing: {path}")
                df = pd.read_parquet(path)
                df["_date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
                self._tx_df = df
                logger.info("Loaded transactions: %d rows", len(df))
            return self._tx_df

    # ------------------------------------------------------------------
    # Tool 2 — KYC
    # ------------------------------------------------------------------
    def kyc(self) -> dict[str, dict]:
        with self._lock:
            if self._kyc_idx is None:
                path = self.data_dir / "tool_2_kyc" / "entities.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"Missing: {path}")
                df = pd.read_parquet(path)
                self._kyc_idx = {str(r["entity_id"]): r.to_dict() for _, r in df.iterrows()}
                logger.info("Loaded KYC: %d entities", len(self._kyc_idx))
            return self._kyc_idx

    # ------------------------------------------------------------------
    # Tool 3 — sanctions / PEP
    # ------------------------------------------------------------------
    def ofac(self) -> list[dict]:
        with self._lock:
            if self._ofac is None:
                self._ofac = self._load_csv(self.data_dir / "tool_3_sanctions" / "ofac.csv")
                logger.info("Loaded OFAC: %d rows", len(self._ofac))
            return self._ofac

    def pep(self) -> list[dict]:
        with self._lock:
            if self._pep is None:
                self._pep = self._load_csv(self.data_dir / "tool_3_sanctions" / "pep.csv")
                logger.info("Loaded PEP: %d rows", len(self._pep))
            return self._pep

    @staticmethod
    def _load_csv(path: Path) -> list[dict]:
        if not path.exists():
            logger.warning("Sanctions file missing: %s", path)
            return []
        with path.open("r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # ------------------------------------------------------------------
    # Tool 4 — policy chunks
    # ------------------------------------------------------------------
    def policy_chunks(self) -> pd.DataFrame:
        with self._lock:
            if self._policy_df is None:
                path = self.data_dir / "tool_4_policy" / "policy_chunks.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"Missing: {path}")
                self._policy_df = pd.read_parquet(path)
                logger.info("Loaded policy chunks: %d rows", len(self._policy_df))
            return self._policy_df

    # ------------------------------------------------------------------
    # Tool 5 — SOPs
    # ------------------------------------------------------------------
    def sops(self) -> dict[str, list[tuple[str, str]]]:
        """Map of sop_id (e.g. SOP-STRUCTURING-01) → list of (section, body)."""
        with self._lock:
            if self._sops is None:
                self._sops = {}
                sop_dir = self.data_dir / "tool_5_sop"
                if not sop_dir.exists():
                    logger.warning("SOPs dir missing: %s", sop_dir)
                    return self._sops
                for md in sop_dir.glob("*.md"):
                    m = re.match(r"^(?P<typ>[a-z_]+)_v(?P<n>\d+)\.md$", md.name)
                    if not m:
                        continue
                    sop_id = (
                        f"SOP-{m.group('typ').upper().replace('_', '-')}-"
                        f"{int(m.group('n')):02d}"
                    )
                    self._sops[sop_id] = _split_md_sections(md.read_text(encoding="utf-8"))
                logger.info("Loaded SOPs: %d typology files", len(self._sops))
            return self._sops

    # ------------------------------------------------------------------
    # Demo / eval / seeded
    # ------------------------------------------------------------------
    def manifest(self) -> list[dict]:
        with self._lock:
            if self._manifest is None:
                path = self.data_dir / "demo" / "manifest.jsonl"
                with path.open("r", encoding="utf-8") as fh:
                    self._manifest = [json.loads(l) for l in fh if l.strip()]
            return self._manifest

    def eval_keys(self) -> dict[str, dict]:
        with self._lock:
            if self._eval_keys is None:
                path = self.data_dir / "demo" / "eval_keys.jsonl"
                self._eval_keys = {}
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        self._eval_keys[obj["case_id"]] = obj
            return self._eval_keys

    def stratification(self) -> dict:
        with self._lock:
            if self._stratification is None:
                path = self.data_dir / "demo" / "stratification_report.json"
                self._stratification = json.load(path.open("r", encoding="utf-8"))
            return self._stratification

    def suspicious_entities(self) -> pd.DataFrame:
        with self._lock:
            if self._suspicious is None:
                path = self.data_dir / "seeded_subpopulations" / "suspicious_entities.parquet"
                self._suspicious = pd.read_parquet(path)
            return self._suspicious

    def near_miss_entities(self) -> pd.DataFrame:
        with self._lock:
            if self._near_miss is None:
                path = self.data_dir / "seeded_subpopulations" / "near_miss_entities.parquet"
                self._near_miss = pd.read_parquet(path)
            return self._near_miss

    def seed_traces(self) -> list[dict]:
        with self._lock:
            if self._seed_traces is None:
                path = self.data_dir / "seed_traces" / "agent_rollout_traces.jsonl"
                self._seed_traces = []
                if path.exists():
                    with path.open("r", encoding="utf-8") as fh:
                        self._seed_traces = [json.loads(l) for l in fh if l.strip()]
            return self._seed_traces

    # ------------------------------------------------------------------
    # Schemas / stats (for /api/system/config)
    # ------------------------------------------------------------------
    def tx_schema(self) -> dict:
        with self._lock:
            if self._tx_schema is None:
                self._tx_schema = json.load(
                    (self.data_dir / "tool_1_transactions" / "schema.json").open()
                )
            return self._tx_schema

    def kyc_schema(self) -> dict:
        with self._lock:
            if self._kyc_schema is None:
                self._kyc_schema = json.load(
                    (self.data_dir / "tool_2_kyc" / "schema.json").open()
                )
            return self._kyc_schema

    def tx_stats(self) -> dict:
        with self._lock:
            if self._tx_stats is None:
                self._tx_stats = json.load(
                    (self.data_dir / "tool_1_transactions" / "stats.json").open()
                )
            return self._tx_stats

    def kyc_stats(self) -> dict:
        with self._lock:
            if self._kyc_stats is None:
                self._kyc_stats = json.load(
                    (self.data_dir / "tool_2_kyc" / "stats.json").open()
                )
            return self._kyc_stats

    # ------------------------------------------------------------------
    # Write dirs
    # ------------------------------------------------------------------
    @property
    def traces_dir(self) -> Path:
        d = self.data_dir / "traces"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def dispositions_dir(self) -> Path:
        d = self.data_dir / "dispositions"
        d.mkdir(parents=True, exist_ok=True)
        return d


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_md_sections(txt: str) -> list[tuple[str, str]]:
    parts = _SECTION_RE.split(txt)
    out: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        out.append((title, body))
    return out


# ---------------------------------------------------------------------------
# Module-level cached instance (one per process)
# ---------------------------------------------------------------------------
_singleton: DataPlane | None = None
_singleton_lock = threading.Lock()


def get_data_plane(data_dir: str | Path | None = None) -> DataPlane:
    """Return a process-wide DataPlane. First call sets the data_dir."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            if data_dir is None:
                raise RuntimeError(
                    "DataPlane not initialized; first call must supply data_dir."
                )
            _singleton = DataPlane(data_dir)
        return _singleton


def reset_data_plane() -> None:
    """Test helper — drop the singleton."""
    global _singleton
    with _singleton_lock:
        _singleton = None
