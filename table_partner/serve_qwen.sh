#!/usr/bin/env bash
set -eu
checkpoint="${QWEN_CHECKPOINT:-/lab/ckpt/jsj412/Qwen3.8-27B}"
model="${QWEN_MODEL:-qwen3.8-27B}"
if [[ ! -r "$checkpoint/config.json" ]]; then
    echo "Cannot read $checkpoint/config.json" >&2
    exit 1
fi
exec vllm serve "$checkpoint" --served-model-name "$model" \
    --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8010}" "$@"
