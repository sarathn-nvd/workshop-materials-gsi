"""NAT entry point \u2014 importing this module triggers every @register_function /
@register_function_group decorator in the package, so the components show up
in `nat info components` and in the workflow_builder.
"""
from __future__ import annotations

# Pure-Python helpers (no NAT registration) \u2014 importing for side effects
# of module init only.
from aml_app.common import (  # noqa: F401
    schemas,
    behavioral_features,
    semantic_profile,
    typology_classifier,
)

# Leaf tools
from aml_app.tools import data_tools, hints  # noqa: F401

# Skill calls
from aml_app.skills import aux_call, sar_caller  # noqa: F401

# Gating
from aml_app.gating import aux_gate  # noqa: F401

# Workflow orchestrator (deterministic Python wiring of the leaves)
from aml_app.workflow import investigate_case  # noqa: F401

# API handlers
from aml_app.api import (  # noqa: F401
    alerts,
    analytics,
    entities,
    eval_comparison,
    misc,
    network,
    skills,
)
