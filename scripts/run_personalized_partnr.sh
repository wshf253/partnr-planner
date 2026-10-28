#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# Set PARTNR_LLM_BASE_URL to reuse a server; otherwise start the existing Qwen service.
if [[ -z "${PARTNR_LLM_BASE_URL:-}" ]]; then
    export PARTNR_EXPERIMENT=personalized_takeover
    exec bash "$REPO/scripts/run_qwen35_partnr.sh" "$@"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
exec "$REPO/.venv/bin/python" -m habitat_llm.examples.personalization_demo "$@"
