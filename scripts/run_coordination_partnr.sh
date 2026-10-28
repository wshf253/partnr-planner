#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
if [[ -z "${PARTNR_LLM_BASE_URL:-}" ]]; then
    export PARTNR_EXPERIMENT=partner_coordination
    exec bash "$REPO/scripts/run_qwen35_partnr.sh" "$@"
fi
exec "$REPO/.venv/bin/python" -m habitat_llm.examples.coordination_demo "$@"
