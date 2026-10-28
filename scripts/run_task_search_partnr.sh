#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO/scripts/run_coordination_partnr.sh" \
  --config-name baselines/partner_task_search "$@"
