"""Source-pool loaders. Each module exposes a `load()` function that returns
a pandas DataFrame projected to a stable schema. Stage 1 + Stage A1 import
from here.
"""
from scripts.pools import (
    ofac_pep,
    policy_corpus,
    pool_1_efc,
    pool_2_ibm,
    pool_3_amlgentex,
    pool_4_sarsum,
    pool_5_cfpb,
)

__all__ = [
    "pool_1_efc", "pool_2_ibm", "pool_3_amlgentex", "pool_4_sarsum", "pool_5_cfpb",
    "ofac_pep", "policy_corpus",
]
