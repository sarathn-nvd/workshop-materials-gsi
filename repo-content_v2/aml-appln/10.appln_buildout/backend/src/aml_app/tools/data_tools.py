"""5 data tools registered as a NAT function group.

- get_transactions(entity_id, window_start, window_end) -> list[Transaction]
- get_kyc(entity_id) -> KYCProfile
- screen_sanctions(name, country=None, min_score=0.55) -> list[SanctionsHit]
- retrieve_policy(typology, k=4) -> list[PolicyExcerpt]
- get_sop(typology, variant=1, section=None) -> list[SOPExcerpt]

Internal-column stripping at the response boundary is enforced here.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

from aml_app.common.schemas import (
    KYCProfile,
    PolicyExcerpt,
    SanctionsHit,
    SOPExcerpt,
    Transaction,
    Typology,
)
from aml_app.utils.data_loader import DataPlane, get_data_plane

logger = logging.getLogger("aml_app.tools.data_tools")


# ---------------------------------------------------------------------------
# Per-tool input schemas (typed wrappers — surfaced to the LLM as the tool's
# JSON schema)
# ---------------------------------------------------------------------------
class GetTransactionsInput(BaseModel):
    entity_id: str = Field(..., description="The entity identifier (e.g. SYN_22da3c8f).")
    window_start: str = Field(..., description="Window start date (YYYY-MM-DD).")
    window_end: str = Field(..., description="Window end date (YYYY-MM-DD).")


class GetKycInput(BaseModel):
    entity_id: str = Field(..., description="The entity identifier.")


class ScreenSanctionsInput(BaseModel):
    name: str = Field(..., description="Counterparty name to screen.")
    country: Optional[str] = Field(default=None, description="Optional ISO country code.")
    # 0.55 was the reference-agent default and produced ~9.8 spurious hits per
    # benign business counterparty (generic words like Holdings/Services/LLC
    # fuzzy-match against OFAC's commercial-name pool). 0.85 keeps real matches
    # (sanctioned exact-name + close variants) while dropping the noise floor.
    min_score: float = Field(default=0.85, ge=0.0, le=1.0,
                             description="Minimum fuzzy match score to retain (0\u20131).")


class RetrievePolicyInput(BaseModel):
    typology: str = Field(..., description="One of the 8 canonical typologies, or 'none'.")
    k: int = Field(default=4, ge=1, le=20, description="Number of excerpts to return.")


class GetSopInput(BaseModel):
    typology: str = Field(..., description="One of the 8 canonical typologies, or 'none'.")
    variant: int = Field(default=1, ge=1, le=9, description="SOP variant number.")
    section: Optional[str] = Field(default=None, description="Optional section title.")


# ---------------------------------------------------------------------------
# Function group config
# ---------------------------------------------------------------------------
class AmlDataToolsConfig(FunctionGroupBaseConfig, name="aml_data_tools"):
    data_dir: str = Field(
        ...,
        description="Path to the local data plane (./data).",
    )


# ---------------------------------------------------------------------------
# Per-tool implementations (pure-Python, side-effect-free reads)
# ---------------------------------------------------------------------------
class EntityNotFound(KeyError):
    pass


def _get_transactions(dp: DataPlane, args: GetTransactionsInput) -> list[Transaction]:
    df = dp.transactions()
    ws = pd.Timestamp(args.window_start)
    we = pd.Timestamp(args.window_end)
    mask = (
        (df["entity_id"] == args.entity_id)
        & (df["_date_parsed"] >= ws)
        & (df["_date_parsed"] <= we)
    )
    rows = df.loc[mask].to_dict("records")
    out: list[Transaction] = []
    for r in rows:
        out.append(Transaction(
            date=str(r["date"]),
            amount=float(r["amount"]),
            currency=str(r["currency"]),
            counterparty=str(r["counterparty"]),
            channel=str(r["channel"]),
            notes=str(r.get("notes", "") or ""),
        ))
    return out


def _get_kyc(dp: DataPlane, args: GetKycInput) -> KYCProfile:
    idx = dp.kyc()
    row = idx.get(args.entity_id)
    if row is None:
        raise EntityNotFound(args.entity_id)
    keep = {k: row[k] for k in KYCProfile.model_fields if k in row}
    return KYCProfile.model_validate(keep)


def _screen_sanctions(dp: DataPlane, args: ScreenSanctionsInput) -> list[SanctionsHit]:
    if not args.name:
        return []
    ofac = dp.ofac()
    pep = dp.pep()
    hits: list[SanctionsHit] = []

    def _score(entry: dict, list_tag: str) -> SanctionsHit | None:
        cand = (entry.get("caption") or entry.get("name") or entry.get("Name") or "").strip()
        if not cand:
            return None
        score = fuzz.token_set_ratio(args.name, cand) / 100.0
        if args.country and entry.get("countries"):
            if args.country in str(entry["countries"]):
                score = min(1.0, score * 1.05)
        if score < args.min_score:
            return None
        return SanctionsHit(name=cand, list=list_tag, match_score=round(score, 3))

    for e in ofac:
        h = _score(e, "OFAC")
        if h is not None:
            hits.append(h)
    for e in pep:
        h = _score(e, "OpenSanctions")
        if h is not None:
            hits.append(h)
    hits.sort(key=lambda x: -x.match_score)
    return hits[:5]


_TYPE_TO_SOURCES_PRIORITY = ["FinCEN", "FFIEC", "FATF", "OFAC"]


def _retrieve_policy(dp: DataPlane, args: RetrievePolicyInput) -> list[PolicyExcerpt]:
    if args.typology == "none":
        return []
    df = dp.policy_chunks()
    if df.empty or "typology_tags" not in df.columns:
        return []

    typ = args.typology

    def _has_typology(tags) -> bool:
        if tags is None:
            return False
        try:
            return typ in list(tags)
        except TypeError:
            return False

    matching = df[df["typology_tags"].apply(_has_typology)]
    if matching.empty:
        return []

    out: list[PolicyExcerpt] = []
    by_src = {s: matching[matching["source"] == s] for s in _TYPE_TO_SOURCES_PRIORITY}
    src_order = _TYPE_TO_SOURCES_PRIORITY * ((args.k // 4) + 1)
    for s in src_order:
        if len(out) >= args.k:
            break
        bucket = by_src.get(s)
        if bucket is None or bucket.empty:
            continue
        picked_idx = bucket.index[len(out) % len(bucket)]
        row = bucket.loc[picked_idx]
        out.append(PolicyExcerpt(
            source=str(row.get("source", "FinCEN")),
            section=str(row.get("section", ""))[:200],
            url=row.get("url") if pd.notna(row.get("url")) else None,
            text=str(row.get("text", ""))[:1500],
        ))
        by_src[s] = bucket.drop(picked_idx)
    return out[:args.k]


_DEFAULT_SECTION_PRIORITY = [
    "Investigation Steps",
    "Escalation Criteria",
    "Documentation Requirements",
    "Filing Decision",
    "Tools and Systems",
    "References",
]


def _get_sop(dp: DataPlane, args: GetSopInput) -> list[SOPExcerpt]:
    if args.typology == "none":
        return []
    cache = dp.sops()
    typ_token = args.typology.upper().replace("_", "-")
    sop_id = f"SOP-{typ_token}-{args.variant:02d}"
    sections = cache.get(sop_id, [])
    if not sections:
        return []
    if args.section:
        for title, body in sections:
            if args.section.lower() in title.lower():
                return [SOPExcerpt(sop_id=sop_id, section=title, text=body[:1500])]
        return []
    for pri in _DEFAULT_SECTION_PRIORITY:
        for title, body in sections:
            if title.lower() == pri.lower():
                return [SOPExcerpt(sop_id=sop_id, section=title, text=body[:1500])]
    title, body = sections[0]
    return [SOPExcerpt(sop_id=sop_id, section=title, text=body[:1500])]


# ---------------------------------------------------------------------------
# Function-group registration
# ---------------------------------------------------------------------------
@register_function_group(config_type=AmlDataToolsConfig)
async def aml_data_tools(
    config: AmlDataToolsConfig, builder: Builder
) -> AsyncGenerator[FunctionGroup, None]:
    dp = get_data_plane(config.data_dir)
    # Pre-warm so first request doesn't pay the load cost.
    dp.transactions(); dp.kyc(); dp.ofac(); dp.pep(); dp.policy_chunks(); dp.sops()

    group = FunctionGroup(config=config)

    async def get_transactions(args: GetTransactionsInput) -> list[dict]:
        """Returns money-movement events for an entity over a window."""
        rows = _get_transactions(dp, args)
        return [r.model_dump() for r in rows]

    async def get_kyc(args: GetKycInput) -> dict:
        """Returns the canonical KYC / CRM profile for one entity."""
        try:
            return _get_kyc(dp, args).model_dump()
        except EntityNotFound:
            return {"error": f"entity not found: {args.entity_id}"}

    async def screen_sanctions(args: ScreenSanctionsInput) -> list[dict]:
        """Fuzzy-match a counterparty name against the OFAC + PEP pools."""
        return [h.model_dump() for h in _screen_sanctions(dp, args)]

    async def retrieve_policy(args: RetrievePolicyInput) -> list[dict]:
        """Stratified top-k retrieval over the typology-tagged policy corpus."""
        return [p.model_dump() for p in _retrieve_policy(dp, args)]

    async def get_sop(args: GetSopInput) -> list[dict]:
        """Return one institution-internal SOP excerpt for the given typology."""
        return [s.model_dump() for s in _get_sop(dp, args)]

    group.add_function("get_transactions", get_transactions,
                       input_schema=GetTransactionsInput,
                       description="Get transactions for an entity over a date window.")
    group.add_function("get_kyc", get_kyc,
                       input_schema=GetKycInput,
                       description="Get the KYC profile for an entity.")
    group.add_function("screen_sanctions", screen_sanctions,
                       input_schema=ScreenSanctionsInput,
                       description="Fuzzy-match a counterparty name against OFAC + PEP pools.")
    group.add_function("retrieve_policy", retrieve_policy,
                       input_schema=RetrievePolicyInput,
                       description="Stratified top-k retrieval of regulatory excerpts for a typology.")
    group.add_function("get_sop", get_sop,
                       input_schema=GetSopInput,
                       description="Return one institution-internal SOP excerpt for a typology.")

    yield group
