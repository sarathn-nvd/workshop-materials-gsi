"""Sanctions / PEP noise pool — common-name false-positive seeds.

Stage 4 uses this pool to inject *noise hits* into a small subset of
benign-by-construction records. The result: those records get
`_regulatory_frame=sanctions` (via `compute_semantic_profile`'s sanctions
override at score ≥ 0.5) but `label=False`. This breaks the v2 pattern
where the sanctions frame was 99% positive, training the model to
distinguish a high-score-but-noisy PEP/OFAC hit from an actionable one.

Three noise categories, weighted roughly equally:

1. **Common-name PEPs.** First-name + Last-name pairs drawn from the most
   common US given names + common surnames pool. These mimic OpenSanctions
   PEP-list noise where "Michael Brown" matches a real foreign PEP but is
   indistinguishable from the millions of unrelated Michael Browns.

2. **Common-suffix corporate collisions.** "Global Holdings LLC",
   "International Trading Corp", "Pacific Logistics Inc" — names whose
   suffix matches OFAC entries but whose prefix is generic.

3. **Generic single-name matches.** Surname-only matches that frequently
   trigger fuzzy-match false positives.

All noise records are flagged in metadata as `_noise=true` so audits can
distinguish them from real sanctions hits.
"""
from __future__ import annotations

import random


# ============================================================================
# Common US given names (top ~60 each, drawn from SSA + Census frequency)
# ============================================================================
_COMMON_FIRST_NAMES = (
    # Male
    "Michael", "James", "John", "Robert", "David", "William", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth",
    "George", "Joshua", "Kevin", "Brian", "Edward", "Ronald", "Timothy",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
    "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Frank",
    "Benjamin", "Gregory", "Samuel", "Raymond", "Patrick", "Alexander",
    "Jack", "Dennis", "Tyler", "Aaron", "Henry", "Adam", "Douglas",
    "Nathan", "Peter", "Zachary",
    # Female
    "Mary", "Jennifer", "Linda", "Patricia", "Barbara", "Elizabeth",
    "Susan", "Margaret", "Lisa", "Nancy", "Sandra", "Betty", "Dorothy",
    "Sarah", "Karen", "Helen", "Donna", "Carol", "Ruth", "Sharon",
    "Michelle", "Laura", "Emily", "Kimberly", "Deborah", "Amanda",
    "Stephanie", "Rebecca", "Virginia", "Pamela", "Martha", "Debra",
    "Amy", "Catherine", "Janet", "Christine", "Anna", "Melissa", "Diane",
    "Olivia", "Cynthia", "Maria",
)

# Common US surnames (US Census top 100)
_COMMON_LAST_NAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
    "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan",
    "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos",
    "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks", "Chavez",
    "Wood", "James", "Bennett", "Gray",
)

# Generic corporate suffixes (and a prefix pool) — the OFAC list has many
# entries like "* Global Holdings LLC" so common suffixes match indiscriminately.
_GENERIC_CORP_PREFIXES = (
    "Global", "International", "Pacific", "Atlantic", "Eastern", "Western",
    "Continental", "United", "American", "National", "Federal", "Capital",
    "Coastal", "Mountain", "Heritage", "Premier", "Apex", "Pioneer",
    "Sterling", "Imperial", "Strategic", "Diversified", "Allied",
)
_GENERIC_CORP_NOUNS = (
    "Holdings", "Group", "Trading", "Logistics", "Industries", "Enterprises",
    "Consulting", "Services", "Partners", "Capital", "Ventures", "Solutions",
    "Resources", "Commerce", "Exports", "Imports", "Distributors",
)
_CORP_SUFFIXES = ("LLC", "Corp", "Inc", "Ltd", "Holdings", "Group")

# Single-name matches that frequently fuzzy-match real OFAC entries.
_GENERIC_SURNAMES_ONLY = (
    "Hassan", "Khan", "Ahmed", "Ibrahim", "Ali", "Mohamed", "Rodriguez",
    "Garcia", "Petrov", "Volkov", "Singh", "Patel", "Lee", "Park",
    "Chen", "Wang", "Li", "Yamamoto",
)

# Plausible PEP lists / dataset labels (so the noise looks realistic)
_NOISE_LIST_OPTIONS = ("OpenSanctions", "PEP", "OFAC")


# ============================================================================
# Public API
# ============================================================================
def make_common_name_pep(rng: random.Random) -> dict:
    """Generate one common-name PEP noise hit."""
    name = f"{rng.choice(_COMMON_FIRST_NAMES)} {rng.choice(_COMMON_LAST_NAMES)}"
    return {
        "name": name,
        "list": rng.choice(("OpenSanctions", "PEP")),
        # Score band per strategy doc §5.2 Stage 4: 0.85–0.94 — high enough
        # to trigger compute_semantic_profile's sanctions override (≥ 0.5)
        # but low enough that an analyst should hold for confirmation.
        "match_score": round(rng.uniform(0.85, 0.94), 2),
        "_noise": True,
        "_noise_category": "common_name_pep",
    }


def make_corporate_collision(rng: random.Random) -> dict:
    """Generate one corporate-suffix-collision OFAC noise hit."""
    name = (
        f"{rng.choice(_GENERIC_CORP_PREFIXES)} "
        f"{rng.choice(_GENERIC_CORP_NOUNS)} "
        f"{rng.choice(_CORP_SUFFIXES)}"
    )
    return {
        "name": name,
        "list": "OFAC",
        "match_score": round(rng.uniform(0.85, 0.92), 2),
        "_noise": True,
        "_noise_category": "corp_collision",
    }


def make_generic_surname_match(rng: random.Random) -> dict:
    """Generate one single-name surname-match noise hit."""
    return {
        "name": rng.choice(_GENERIC_SURNAMES_ONLY),
        "list": rng.choice(("PEP", "OpenSanctions")),
        "match_score": round(rng.uniform(0.85, 0.91), 2),
        "_noise": True,
        "_noise_category": "generic_surname",
    }


def make_noise_hit(rng: random.Random) -> dict:
    """Pick one noise category uniformly and produce a noise hit."""
    category = rng.choices(
        ("common_name_pep", "corp_collision", "generic_surname"),
        weights=[0.45, 0.35, 0.20],
    )[0]
    if category == "common_name_pep":
        return make_common_name_pep(rng)
    if category == "corp_collision":
        return make_corporate_collision(rng)
    return make_generic_surname_match(rng)
