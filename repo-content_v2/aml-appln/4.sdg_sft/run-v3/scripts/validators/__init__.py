"""All RULE-N-* implementations live here. Two surfaces:

- `rules_per_record`: per-record rules called by individual stages and by the
  Stage 9 / Stage A4 final sweep.
- `rules_corpus`: aggregate rules computed across the produced corpus.
"""
from scripts.validators import rules_corpus, rules_per_record

__all__ = ["rules_corpus", "rules_per_record"]
