#!/usr/bin/env bash
# Both agents are scripted, so this runs without starting a vLLM server.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
exec "$REPO/.venv/bin/python" -m habitat_llm.examples.coordination_demo \
    --config-name baselines/allocation_gate "$@"
