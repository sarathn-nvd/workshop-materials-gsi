# pipeline/reference_agent — Test / Reference Implementation ONLY

**This directory is NOT part of the production-mimic data deliverable.**

## What this is

A working, end-to-end agentic pipeline implemented as Python modules. It exists for one reason: to drive the production-mimic data through a realistic per-case loop so the data layer can be tested against a model endpoint while it's being built.

It's a **reference implementation** of what the future production agentic pipeline must do. It is NOT the deliverable.

## What this contains

| File | Role |
|---|---|
| `tool_clients.py` | In-process Python readers for Tools 1–5 (Parquet / CSV / file lookups) |
| `llm_client.py` | OpenAI-compatible LLM client wrapper |
| `prompts.py` | System prompts for `sar_judgment` + the 4 auxiliary task types + reviewer prompts |
| `aux_runner.py` | Fires the 4 auxiliary task calls in parallel to the model endpoint |
| `aux_gate.py` | Per-response gating (input-availability guard + schema validity + LLM-as-Judge) |
| `sar_caller.py` | Assembles the final `sar_judgment` bundle and invokes the model endpoint |
| `nat_orchestrator.py` | Per-case batch loop wiring all of the above |

## What this is NOT

- It is not architecturally optimized for production
- It is not the agentic pipeline that will be deployed
- It is not a contract for what the production agent must use as its libraries

## When to use this

| Use case | Use this? |
|---|---|
| Testing whether the production-mimic data is consumable end-to-end | ✅ Yes |
| Validating that a candidate model checkpoint produces sensible SAR outputs over realistic bundles | ✅ Yes |
| Scoring a checkpoint against `demo/eval_keys.jsonl` with `pipeline/eval.py` | ✅ Yes |
| As the production agent | ❌ No — build the production agent separately, following the canonical consumption contract in [`AGENT_USAGE_GUIDE.md`](../../AGENT_USAGE_GUIDE.md) |
| As a Python library dependency for any data-generation step | ❌ No — the data-gen pipeline (steps 1–8) does NOT import from this directory |

## Isolation

The data-generation pipeline (`pipeline/orchestrator.py` + `pipeline/steps/`) is fully isolated from this directory. It has zero imports from `pipeline.reference_agent.*`. You can delete this directory entirely and the data-generation pipeline still works.

## How to run (test / sanity-check only)

```bash
# Drive the test agent over the demo manifest
python -m pipeline.reference_agent.nat_orchestrator --concurrency 8

# Score against eval_keys
python -m pipeline.eval
```

Output:
- `data/final/prod_mimic/manifests/agent_rollout_traces.jsonl` — per-case reasoning chain
- `data/final/prod_mimic/manifests/agent_rollout_eval.json` — scored metrics
