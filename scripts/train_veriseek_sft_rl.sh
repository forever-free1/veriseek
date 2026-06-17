#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_MODEL_ID="Qwen/Qwen3-4B-Thinking-2507"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${SFT_CHECKPOINT_PATH:-}}"
if [[ -z "${SFT_MODEL_PATH}" ]]; then
    cat >&2 <<'EOF'
SFT_MODEL_PATH must point to a completed VeriSeek-SFT checkpoint.
Run scripts/train_veriseek_sft.sh first with SFT-compatible data, then pass:
  SFT_MODEL_PATH=/path/to/veriseek_sft_checkpoint bash scripts/train_veriseek_sft_rl.sh
EOF
    exit 2
fi

echo "VeriSeek SFT+RL starts from an SFT checkpoint originally trained from ${BASE_MODEL_ID}."
export MODEL_PATH="${SFT_MODEL_PATH}"
export OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/veriseek_sft_rl}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-veriseek_sft_rl_qwen3}"

bash "${SCRIPT_DIR}/train_veriseek_grpo.sh" "$@"
