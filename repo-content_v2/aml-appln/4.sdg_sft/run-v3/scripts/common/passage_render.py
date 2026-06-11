"""Canonical bundle-passage renderer.

This is the SINGLE source of truth for the `[transactions] / [kyc_profile]`
passage shape that:

  - Stage 6 (inline aux numeric findings) passes to the LLM in training,
  - Stage A3b (standalone aux_behavioral training records) passes to the LLM,
  - the production backend agent (`aml_app.workflow.investigate_case`)
    passes to the trained model at inference time.

Keeping all three sites on this one function eliminates training/serving
distribution drift on the aux-skill input. The intentionally-restricted
6-field schema (`date, channel, amount, currency, counterparty, notes`) is
exactly the field set the production transactions tool returns; richer
source-pool fields (sender/receiver account, country pair) carried by
EFC/IBM bundles are NOT included here because the trained model will never
see them in production — they're encoded into the precomputed metrics
block separately for behavioral findings.

Use:
    from scripts.common.passage_render import render_bundle_passage
    passage = render_bundle_passage(transactions, kyc_profile)
"""
from __future__ import annotations

from typing import Iterable, Mapping


# Runtime-available transaction fields, in column order.
_TX_FIELDS = ("date", "channel", "amount", "currency", "counterparty", "notes")

# KYC fields, in the order they appear in the rendered block. Exactly the
# six fields the production tool_2_kyc lookup returns.
_KYC_FIELDS = (
    "entity_id",
    "entity_type",
    "expected_monthly_volume",
    "business_purpose",
    "risk_rating",
    "incorporation_jurisdiction",
)

# Column widths chosen so a typical row fits ~110 chars (well under model
# context concerns) AND the columns line up visually for the LLM.
_W_DATE = 10
_W_CHAN = 6
_W_AMT  = 14
_W_CCY  = 4
_W_CP   = 32


def _fmt_tx_row(t: Mapping) -> str:
    """Render a single transaction as a fixed-width row.

    Format:
        2026-02-04 | ach    |       6,255.32 | USD  | TRX Holdings LLC               | note...
    """
    date = str(t.get("date") or "").ljust(_W_DATE)[:_W_DATE]
    chan = str(t.get("channel") or "").ljust(_W_CHAN)[:_W_CHAN]
    try:
        amt_val = float(t.get("amount") or 0.0)
    except (TypeError, ValueError):
        amt_val = 0.0
    amt = f"{amt_val:>{_W_AMT},.2f}"
    ccy = str(t.get("currency") or "USD").ljust(_W_CCY)[:_W_CCY]
    cp  = str(t.get("counterparty") or "").ljust(_W_CP)[:_W_CP]
    note_raw = str(t.get("notes") or "").strip()
    note = f" | {note_raw[:40]}" if note_raw else ""
    return f"{date} | {chan} | {amt} | {ccy} | {cp}{note}"


def render_bundle_passage(
    transactions: Iterable[Mapping],
    kyc_profile: Mapping,
    *,
    max_transactions: int = 50,
) -> str:
    """Render a (transactions, kyc) bundle as the canonical aux-skill passage.

    Args:
        transactions: iterable of transaction dicts. Recognised keys:
            date, channel, amount, currency, counterparty, notes.
            Unknown keys are ignored. Missing keys render as blank/zero.
        kyc_profile: KYC dict. Recognised keys are the six listed in
            `_KYC_FIELDS`. Unknown keys are ignored.
        max_transactions: hard cap on the number of rows rendered (default 50).
            Higher counts are surfaced via the trailing summary line.

    Returns:
        A passage string in this exact shape:

            [transactions]
            <header row>
            <tx row 1>
            ...
            <tx row N>
            [... K more transactions not shown ...]   # only if truncated

            [kyc_profile]
            entity_id: ...
            entity_type: ...
            expected_monthly_volume: ...
            business_purpose: ...
            risk_rating: ...
            incorporation_jurisdiction: ...
    """
    tx_list = list(transactions or [])
    shown = tx_list[: max_transactions]

    lines: list[str] = ["[transactions]"]
    header = (
        f"{'date'.ljust(_W_DATE)} | "
        f"{'channel'.ljust(_W_CHAN)} | "
        f"{'amount'.rjust(_W_AMT)} | "
        f"{'ccy'.ljust(_W_CCY)} | "
        f"{'counterparty'.ljust(_W_CP)} | notes"
    )
    lines.append(header)
    if not shown:
        lines.append("(no transactions)")
    for t in shown:
        lines.append(_fmt_tx_row(t))
    if len(tx_list) > len(shown):
        lines.append(
            f"[... {len(tx_list) - len(shown)} more transactions not shown ...]"
        )

    lines.append("")
    lines.append("[kyc_profile]")
    for k in _KYC_FIELDS:
        lines.append(f"{k}: {kyc_profile.get(k, '')}")
    return "\n".join(lines)


def render_citation_passage(excerpt: Mapping) -> str:
    """Render a single policy excerpt as the canonical citation-skill passage.

    Both Stage 6 (citation finding) and the backend's citation aux call
    feed this shape. Stage A2 (FFIEC) keeps its raw chunk passage — the
    citation prompt explicitly accepts either form because there's no clean
    way to merge an FFIEC raw chunk with a structured `policy_excerpt`.
    """
    section = str(excerpt.get("section") or "")
    source = str(excerpt.get("source") or "")
    url = str(excerpt.get("url") or "")
    text = str(excerpt.get("text") or "")
    lines = ["[policy_excerpt]", f"source: {source}", f"section: {section}"]
    if url:
        lines.append(f"url: {url}")
    lines.append("")
    lines.append(text)
    return "\n".join(lines)


__all__ = ["render_bundle_passage", "render_citation_passage"]
