#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_ENV="$HOME/work/llm-serving/.venv-qwen35"
MODEL="/lab/ckpt/jsj412/Qwen3.5-27B"

cd "$REPO"

test -x "$SERVE_ENV/bin/vllm"
test -x "$REPO/.venv/bin/python"
test -f "$MODEL/config.json"

export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Choose a currently unused local port.
PORT="$("$SERVE_ENV/bin/python" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
export PARTNR_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

RUN_DIR="$REPO/outputs/qwen35/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"
echo "Logs and results: $RUN_DIR"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$SERVE_ENV/bin/vllm" serve "$MODEL" \
    --served-model-name qwen35-27b \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --max-model-len 16384 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.75 \
    --enforce-eager \
    > "$RUN_DIR/vllm.log" 2>&1 &

SERVER_PID=$!

echo "Waiting for vLLM. Startup log: $RUN_DIR/vllm.log"

READY=0
for ((i=0; i<180; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -n 100 "$RUN_DIR/vllm.log"
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 2 \
        "http://127.0.0.1:${PORT}/health" >/dev/null; then
        READY=1
        break
    fi
    sleep 5
done

if [[ "$READY" != 1 ]]; then
    echo "vLLM startup timed out."
    tail -n 100 "$RUN_DIR/vllm.log"
    exit 1
fi

# Exercise the actual adapter in the PARTNR environment.
"$REPO/.venv/bin/python" - <<'PY'
from habitat_llm.llm import instantiate_llm

llm = instantiate_llm("local_chat")
answer = llm.generate("Reply with exactly: READY", max_length=64)
print("Local LLM response:", answer)
assert answer.strip(), "Empty response"
PY

# Opt-in controlled personalization experiment, sharing this server lifecycle.
if [[ "${PARTNR_EXPERIMENT:-baseline}" == "personalized_takeover" ]]; then
    "$REPO/.venv/bin/python" -m habitat_llm.examples.personalization_demo \
        hydra.run.dir="$RUN_DIR/personalization" "$@" \
        2>&1 | tee "$RUN_DIR/partnr.log"
    exit 0
fi

if [[ "${PARTNR_EXPERIMENT:-baseline}" == "partner_coordination" ]]; then
    "$REPO/.venv/bin/python" -m habitat_llm.examples.coordination_demo \
        hydra.run.dir="$RUN_DIR/coordination" "$@" \
        2>&1 | tee "$RUN_DIR/partnr.log"
    exit 0
fi

# No arguments: run the first dataset episode.
# Pass ++episode_indices=null to run the full dataset.
if [[ "$#" -eq 0 ]]; then
    set -- '++episode_indices=[0]'
fi

"$REPO/.venv/bin/python" -m habitat_llm.examples.planner_demo \
    --config-name baselines/centralized_zero_shot_react_summary.yaml \
    llm@evaluation.planner.plan_config.llm=local_chat \
    evaluation.planner.plan_config.constrained_generation=False \
    habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    num_proc=1 \
    num_runs_per_episode=1 \
    evaluation.save_video=False \
    hydra.run.dir="$RUN_DIR/planner" \
    "$@" \
    2>&1 | tee "$RUN_DIR/partnr.log"
