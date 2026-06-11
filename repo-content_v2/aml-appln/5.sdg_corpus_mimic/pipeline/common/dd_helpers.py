"""DataDesigner factories pinned at the local NIM endpoint.

Every LLM-using stage builds its config through these helpers so the model
endpoint, model id, temperature/timeout/concurrency live in exactly one place.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import data_designer.config as dd
from data_designer.config.models import (
    ChatCompletionInferenceParams,
    ModelConfig,
    ModelProvider,
)
from data_designer.interface import DataDesigner

from pipeline.config import (
    LLM_API_KEY,
    LLM_ENDPOINT,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER_NAME,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    LLM_TOP_P,
)

logger = logging.getLogger(__name__)


def make_local_provider() -> ModelProvider:
    """OpenAI-compatible local NIM provider."""
    return ModelProvider(
        name=LLM_PROVIDER_NAME,
        endpoint=LLM_ENDPOINT,
        provider_type="openai",
        api_key=LLM_API_KEY,
    )


def make_model_config(
    *,
    alias: str = "sft-generator",
    max_parallel: int = 32,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> ModelConfig:
    """Single-model config; every LLM stage builds one of these."""
    return ModelConfig(
        alias=alias,
        model=LLM_MODEL,
        provider=LLM_PROVIDER_NAME,
        inference_parameters=ChatCompletionInferenceParams(
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            top_p=top_p if top_p is not None else LLM_TOP_P,
            max_tokens=max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
            timeout=timeout if timeout is not None else LLM_TIMEOUT_SEC,
            max_parallel_requests=max_parallel,
        ),
    )


def make_data_designer(artifact_path: Path) -> DataDesigner:
    """Construct a DataDesigner runtime that writes artifacts under `artifact_path`."""
    artifact_path = Path(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)
    return DataDesigner(
        artifact_path=str(artifact_path),
        model_providers=[make_local_provider()],
    )


def build_local_seed_source(seed_path: Path) -> dd.LocalFileSeedSource:
    """Wrap a local file as a DataDesigner LocalFileSeedSource for `with_seed_dataset`."""
    return dd.LocalFileSeedSource(path=str(seed_path))


def build_llm_text_column(
    *,
    name: str,
    system_prompt: str,
    prompt: str,
    model_alias: str = "sft-generator",
) -> dd.LLMTextColumnConfig:
    """A single-column LLM generation step."""
    return dd.LLMTextColumnConfig(
        name=name,
        model_alias=model_alias,
        system_prompt=system_prompt,
        prompt=prompt,
    )


def run_dd_pass(
    *,
    seed_df: "pd.DataFrame",
    system_prompt: str,
    user_template: str,
    output_column: str,
    dataset_name: str,
    artifact_path: Path,
    max_parallel: int,
    max_tokens: int = 2048,
    timeout: int = 300,
    temperature: float = 0.7,
) -> "pd.DataFrame":
    """One-shot DataDesigner pass for a single LLM column.

    Writes seed_df to a temp CSV, runs DD, returns the joined dataset (seed +
    generated column). The column index is preserved so the caller can re-attach
    the LLM output to the input row by position.
    """
    import pandas as pd

    artifact_path = Path(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Stage seed CSV — DataDesigner reads via DuckDB in strict-mode. Two
    # things are needed for that to work on free-form text columns
    # (Stage 7 pair-and-ground packs SARSum narratives and bundle JSON
    # into the seed, all of which routinely contain commas / quotes /
    # newlines):
    #   1. QUOTE_ALL so every value is wrapped in double-quotes.
    #   2. Strip embedded \r\n inside string values (DuckDB strict-mode
    #      treats unescaped newlines inside quoted fields as row breaks).
    #
    # The LLM sees the post-strip text. Newlines inside the prompt are
    # cosmetic — replacing them with spaces does not change semantics.
    import csv as _csv
    seed_clean = seed_df.copy()
    for col in seed_clean.columns:
        if seed_clean[col].dtype == object:
            seed_clean[col] = (
                seed_clean[col]
                .astype(str)
                .str.replace("\r\n", " ", regex=False)
                .str.replace("\n", " ", regex=False)
                .str.replace("\r", " ", regex=False)
            )
    seed_csv = artifact_path / f"{dataset_name}_seed.csv"
    seed_clean.to_csv(seed_csv, index=False, quoting=_csv.QUOTE_ALL)
    n_rows = len(seed_df)
    if n_rows == 0:
        return pd.DataFrame()

    cb = dd.DataDesignerConfigBuilder(model_configs=[
        make_model_config(
            alias="sft-generator",
            max_parallel=max_parallel,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
        ),
    ])
    cb.with_seed_dataset(build_local_seed_source(seed_csv))
    cb.add_column(build_llm_text_column(
        name=output_column,
        system_prompt=system_prompt,
        prompt=user_template,
    ))

    runtime = make_data_designer(artifact_path=artifact_path)
    logger.info("DD pass start: dataset=%s rows=%d max_parallel=%d", dataset_name, n_rows, max_parallel)
    result = runtime.create(cb, num_records=n_rows, dataset_name=dataset_name)
    df_out = result.load_dataset()
    if not isinstance(df_out, pd.DataFrame):
        df_out = pd.DataFrame(df_out)
    logger.info("DD pass done: dataset=%s produced=%d", dataset_name, len(df_out))
    return df_out


# ============================================================================
# JSON-output parsing helper (DD outputs are strings; many of our prompts
# request strict JSON — this strips ```json fences and parses safely)
# ============================================================================
def safe_json_loads(s: str) -> dict | list | None:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    import json
    s = (s or "").strip()
    for fence in ("```json", "```JSON", "```"):
        if s.startswith(fence):
            s = s[len(fence):].strip()
            if s.endswith("```"):
                s = s[:-3].strip()
            break
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None
