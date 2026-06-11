"""SFT synthetic-data generation pipeline.

End-to-end implementation of `../SDG_STRATEGY_SFT.md`. The package exposes:

- `scripts.main`              — CLI entrypoint
- `scripts.config`            — paths + endpoints + concurrency
- `scripts.schemas`           — Pydantic single-source-of-truth
- `scripts.common`            — shared infrastructure
- `scripts.validators`        — RULE-N-* enforcement
- `scripts.pools`             — source-pool loaders
- `scripts.tools_prep`        — one-time artifact builders
- `scripts.non_auxiliary`     — non-auxiliary pipeline (Stages 1-9)
- `scripts.auxiliary`         — auxiliary pipeline (Stages A1-A4)
- `scripts.combine_corpus`    — downstream merge placeholder
"""

__version__ = "0.1.0"
