#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen3-4B-Thinking-2507}"
export OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/veriseek_rl_only}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-veriseek_rl_only_qwen3}"

bash "${SCRIPT_DIR}/train_veriseek_grpo.sh" "$@"
