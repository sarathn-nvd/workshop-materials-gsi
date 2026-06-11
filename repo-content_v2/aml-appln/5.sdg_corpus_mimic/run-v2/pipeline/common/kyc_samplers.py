"""Per-archetype KYC sampler tables (lifted from SDG_STRATEGY_SFT §Stage 2).

Used when synthesizing KYC for clean-filler entities (Step 3). Tables are
trimmed/simplified from the SFT versions; we only need plausible random
values for entities that have no SFT-bundle anchor.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


# ============================================================================
# Archetype catalog (16 archetypes per SDG_STRATEGY_SFT §1 Stage 1)
# ============================================================================
ARCHETYPES: list[str] = [
    "individual_wage_earner",
    "individual_small_business_owner",
    "individual_retiree_65+",
    "retail_business_jewelry",
    "retail_business_restaurant",
    "retail_business_laundromat",
    "retail_business_convenience",
    "retail_services_non_cash",
    "import_export_firm",
    "broker_dealer",
    "money_services_business",
    "shell_holding_offshore",
    "shell_holding_domestic",
    "professional_services_gatekeeper",
    "crypto_exchange_vasp",
    "ngo_charity",
]


@dataclass(frozen=True)
class ArchetypeProfile:
    archetype: str
    entity_type: str   # 'individual' or 'business'
    jurisdictions: list[tuple[str, float]]   # (code, weight) — weights sum ~ 1.0
    volume_band: tuple[int, int]              # (lo, hi) USD/month
    business_purpose_template: str            # for entity_type == 'business'
    individual_role_template: str = ""        # for entity_type == 'individual'


# Per-archetype profiles. Jurisdictions truncated to top-3 per archetype
# (the SFT had longer tails; for clean-filler this is sufficient).
PROFILES: dict[str, ArchetypeProfile] = {
    "individual_wage_earner": ArchetypeProfile(
        "individual_wage_earner", "individual",
        [("US-CA", 0.4), ("US-NY", 0.3), ("US-TX", 0.3)],
        (2_000, 15_000),
        "",
        "Salaried wage earner; primary income from {industry}; expected monthly throughput ~${vol:,}.",
    ),
    "individual_small_business_owner": ArchetypeProfile(
        "individual_small_business_owner", "individual",
        [("US-CA", 0.4), ("US-TX", 0.4), ("US-FL", 0.2)],
        (10_000, 80_000),
        "",
        "Self-employed proprietor; small-business cash flow; expected monthly throughput ~${vol:,}.",
    ),
    "individual_retiree_65+": ArchetypeProfile(
        "individual_retiree_65+", "individual",
        [("US-FL", 0.5), ("US-CA", 0.3), ("US-AZ", 0.2)],
        (1_000, 8_000),
        "",
        "Retired individual on fixed pension income; expected monthly throughput ~${vol:,}.",
    ),
    "retail_business_jewelry": ArchetypeProfile(
        "retail_business_jewelry", "business",
        [("US-NY", 0.4), ("US-CA", 0.3), ("US-FL", 0.3)],
        (30_000, 200_000),
        "Retail jewelry store; mixed cash and card receipts; declared monthly volume ~${vol:,}.",
    ),
    "retail_business_restaurant": ArchetypeProfile(
        "retail_business_restaurant", "business",
        [("US-CA", 0.4), ("US-NY", 0.4), ("US-TX", 0.2)],
        (40_000, 400_000),
        "Restaurant / hospitality establishment; daily cash receipts; declared monthly volume ~${vol:,}.",
    ),
    "retail_business_laundromat": ArchetypeProfile(
        "retail_business_laundromat", "business",
        [("US-CA", 0.4), ("US-TX", 0.4), ("US-FL", 0.2)],
        (20_000, 250_000),
        "Coin-operated laundry / car wash; high cash throughput; declared monthly volume ~${vol:,}.",
    ),
    "retail_business_convenience": ArchetypeProfile(
        "retail_business_convenience", "business",
        [("US-CA", 0.4), ("US-TX", 0.4), ("US-NY", 0.2)],
        (50_000, 500_000),
        "Convenience / grocery store; multi-shift cash operation; declared monthly volume ~${vol:,}.",
    ),
    "retail_services_non_cash": ArchetypeProfile(
        "retail_services_non_cash", "business",
        [("US-CA", 0.5), ("US-NY", 0.3), ("US-WA", 0.2)],
        (10_000, 1_000_000),
        "Professional / B2B services firm; primarily card and wire receipts; declared monthly volume ~${vol:,}.",
    ),
    "import_export_firm": ArchetypeProfile(
        "import_export_firm", "business",
        [("US-CA", 0.4), ("US-NY", 0.3), ("Hong Kong", 0.3)],
        (100_000, 10_000_000),
        "Import-export trading firm; trade-finance flows; declared monthly volume ~${vol:,}.",
    ),
    "broker_dealer": ArchetypeProfile(
        "broker_dealer", "business",
        [("US-NY", 0.5), ("US-IL", 0.3), ("UK", 0.2)],
        (500_000, 50_000_000),
        "Securities broker / dealer firm; intermediary capital flows; declared monthly volume ~${vol:,}.",
    ),
    "money_services_business": ArchetypeProfile(
        "money_services_business", "business",
        [("US-CA", 0.4), ("US-TX", 0.4), ("US-NY", 0.2)],
        (200_000, 5_000_000),
        "Money services business (currency exchange / transmitter); declared monthly volume ~${vol:,}.",
    ),
    "shell_holding_offshore": ArchetypeProfile(
        "shell_holding_offshore", "business",
        [("BVI", 0.5), ("Cayman", 0.3), ("Panama", 0.2)],
        (50_000, 5_000_000),
        "Offshore holding company; primary purpose is asset holding; declared monthly volume ~${vol:,}.",
    ),
    "shell_holding_domestic": ArchetypeProfile(
        "shell_holding_domestic", "business",
        [("Delaware-US", 0.5), ("Nevada-US", 0.3), ("Wyoming-US", 0.2)],
        (50_000, 5_000_000),
        "Domestic holding LLC; opaque-ownership vehicle; declared monthly volume ~${vol:,}.",
    ),
    "professional_services_gatekeeper": ArchetypeProfile(
        "professional_services_gatekeeper", "business",
        [("UK", 0.4), ("US-NY", 0.4), ("Switzerland", 0.2)],
        (50_000, 500_000),
        "Legal / accounting gatekeeper firm; trust and escrow services; declared monthly volume ~${vol:,}.",
    ),
    "crypto_exchange_vasp": ArchetypeProfile(
        "crypto_exchange_vasp", "business",
        [("Malta", 0.4), ("Singapore", 0.4), ("Estonia", 0.2)],
        (1_000_000, 100_000_000),
        "Virtual asset service provider (crypto exchange); declared monthly volume ~${vol:,}.",
    ),
    "ngo_charity": ArchetypeProfile(
        "ngo_charity", "business",
        [("US-DC", 0.4), ("UK", 0.3), ("Switzerland", 0.3)],
        (5_000, 200_000),
        "Non-governmental organization / charity; donation-funded operations; declared monthly volume ~${vol:,}.",
    ),
}


# Heuristic mapping from EFC entity_type → archetype (clean-pool top-up).
EFC_ENTITY_TYPE_TO_ARCHETYPE = {
    "PERSON": "individual_wage_earner",
    "INDIVIDUAL": "individual_wage_earner",
    "ORGANIZATION": "retail_services_non_cash",
    "BUSINESS": "retail_services_non_cash",
    "ACCOUNT": "retail_services_non_cash",
}


# ============================================================================
# Per-typology compatible archetypes (mirrors SFT RULE-3-COMPAT-*)
# ============================================================================
TYPOLOGY_COMPATIBLE_ARCHETYPES: dict[str, list[str]] = {
    "structuring": [
        "individual_wage_earner", "individual_small_business_owner",
        "retail_business_jewelry", "retail_business_restaurant",
        "retail_business_laundromat", "retail_business_convenience",
        "money_services_business",
    ],
    "smurfing": [
        "individual_wage_earner", "individual_small_business_owner",
        "retail_business_jewelry", "retail_business_restaurant",
        "retail_business_laundromat", "retail_business_convenience",
        "money_services_business",
    ],
    "layering": [
        "broker_dealer", "money_services_business", "crypto_exchange_vasp",
        "shell_holding_offshore", "shell_holding_domestic",
        "import_export_firm", "individual_wage_earner",
        "individual_small_business_owner",
    ],
    "trade_based_ml": [
        "import_export_firm", "broker_dealer", "retail_business_convenience",
    ],
    "shell_company": [
        "shell_holding_offshore", "shell_holding_domestic",
        "professional_services_gatekeeper",
    ],
    "human_trafficking": [
        "individual_wage_earner", "individual_small_business_owner",
    ],
    "terrorist_financing": [
        "individual_wage_earner", "ngo_charity",
    ],
    "elder_exploitation": [
        "individual_retiree_65+",
    ],
}


def sample_kyc_for_typology(
    typology: str,
    rng: random.Random,
    *,
    is_positive: bool = True,
) -> dict:
    """Pick a typology-compatible archetype then sample KYC.

    For positives we bias `expected_monthly_volume` LOW (so observed
    activity will exceed declared volume — the canonical suspicious
    signal); the volume sub-range choice happens inside
    sample_kyc_for_archetype via the `label_positive` flag.
    """
    archetypes = TYPOLOGY_COMPATIBLE_ARCHETYPES.get(typology)
    if not archetypes:
        archetypes = ["retail_services_non_cash"]
    archetype = rng.choice(archetypes)
    kyc = sample_kyc_for_archetype(archetype, rng, label_positive=is_positive)
    kyc["_archetype"] = archetype
    return kyc


# ============================================================================
# Risk-rating distribution (per SFT Stage 2 §risk_rating table)
# ============================================================================
_RISK_RATING_WEIGHTS_CLEAN = [
    ("low",      0.50),
    ("medium",   0.30),
    ("high",     0.15),
    ("enhanced", 0.05),
]
_RISK_RATING_WEIGHTS_POSITIVE = [
    ("low",      0.10),    # rare under-rating failure mode
    ("medium",   0.55),
    ("high",     0.25),
    ("enhanced", 0.10),
]


# ============================================================================
# Sampler API
# ============================================================================
def sample_kyc_for_archetype(
    archetype: str,
    rng: random.Random,
    *,
    label_positive: bool = False,
) -> dict:
    """Return a KYCProfile-shaped dict.

    For clean filler we pass label_positive=False; the risk-rating sampler
    skews toward 'low'. Volume bands are sampled log-uniform inside the
    archetype band.
    """
    prof = PROFILES[archetype]

    # Jurisdiction (weighted choice over the listed entries)
    pop, weights = zip(*prof.jurisdictions)
    jurisdiction = rng.choices(pop, weights=weights, k=1)[0]

    # Volume band — log-uniform. For positives, bias to LOW half of band so
    # observed activity will exceed declared volume — the canonical
    # suspicious signal.
    lo, hi = prof.volume_band
    import math
    if label_positive:
        log_lo = math.log(lo)
        log_hi = math.log(lo + (hi - lo) * 0.3)
    else:
        log_lo = math.log(lo)
        log_hi = math.log(hi)
    monthly_volume = int(round(math.exp(rng.uniform(log_lo, log_hi))))

    # Risk rating — different table for positive vs negative
    weights = _RISK_RATING_WEIGHTS_POSITIVE if label_positive else _RISK_RATING_WEIGHTS_CLEAN
    rr_pop, rr_w = zip(*weights)
    risk_rating = rng.choices(rr_pop, weights=rr_w, k=1)[0]

    # business_purpose (template)
    if prof.entity_type == "individual":
        industry = "service-industry employer"
        bp = prof.individual_role_template.format(industry=industry, vol=monthly_volume)
    else:
        bp = prof.business_purpose_template.format(vol=monthly_volume)

    return {
        "entity_type": prof.entity_type,
        "expected_monthly_volume": float(monthly_volume),
        "business_purpose": bp,
        "risk_rating": risk_rating,
        "incorporation_jurisdiction": jurisdiction,
    }


def archetype_for_efc_row(entity_type_raw: str) -> str:
    et = (entity_type_raw or "").strip().upper()
    return EFC_ENTITY_TYPE_TO_ARCHETYPE.get(et, "retail_services_non_cash")
