#!/usr/bin/env bash
# Pre-load the bundled baseline rollout into ./data/traces/ so the
# analytics dashboard has content on first load.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
BACKEND="$(cd "$SRC/.." && pwd)"

if [[ -f "$BACKEND/env/bin/activate" ]]; then
  source "$BACKEND/env/bin/activate"
fi
export NAT_AML_DATA_DIR="${NAT_AML_DATA_DIR:-$BACKEND/data}"

python - <<'PY'
import asyncio
from aml_app.api.misc import demo_seed_traces, DemoSeedTracesConfig, Empty

async def main():
    cm = demo_seed_traces(DemoSeedTracesConfig(), builder=None)
    fi = await cm.__aenter__()
    fn = fi.single_fn
    out = await fn(Empty())
    print(out)

asyncio.run(main())
PY
