"""Minimal OpenAI-compatible LLM client.

Wraps openai.OpenAI to talk to the local vLLM / NIM endpoint defined in
pipeline.config. Handles retries with exponential backoff and exposes a
uniform `chat()` for the runtime aux / judge / SAR callers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from pipeline.config import (
    LLM_API_KEY,
    LLM_ENDPOINT,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    LLM_TOP_P,
)

logger = logging.getLogger("pipeline.reference_agent.llm_client")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


_client_singleton: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI(
            base_url=LLM_ENDPOINT,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SEC,
        )
    return _client_singleton


def chat(
    *,
    system: str,
    user: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    model: str | None = None,
    retries: int = 3,
) -> LLMResponse:
    """One chat completion call. Synchronous.

    Wraps the OpenAI-compatible vLLM/NIM endpoint at config.LLM_ENDPOINT.
    Retries up to `retries` times on transient API errors with exponential
    backoff.
    """
    client = _get_client()
    use_model = model or LLM_MODEL
    use_temp = temperature if temperature is not None else LLM_TEMPERATURE
    use_max = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    use_top = top_p if top_p is not None else LLM_TOP_P

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=use_temp,
                max_tokens=use_max,
                top_p=use_top,
            )
            elapsed_ms = (time.time() - t0) * 1000.0
            txt = resp.choices[0].message.content or ""
            usage = resp.usage
            return LLMResponse(
                text=txt,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=elapsed_ms,
                model=use_model,
            )
        except (APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            wait_s = 2 ** attempt
            logger.warning(
                "LLM call failed (attempt %d/%d): %s; retrying in %ds",
                attempt + 1, retries + 1, e, wait_s,
            )
            time.sleep(wait_s)
    assert last_err is not None
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")
