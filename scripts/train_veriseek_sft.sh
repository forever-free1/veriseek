#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_DIR="${ROOT_DIR}/SFT"

export MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen3-4B-Thinking-2507}"
export TRAIN_FILES="${SFT_TRAIN_FILES:-${TRAIN_FILES:-}}"
export VAL_FILES="${SFT_VAL_FILES:-${VAL_FILES:-${TRAIN_FILES}}}"
export PROJECT_NAME="${PROJECT_NAME:-VeriSeek}"
export EXP_NAME="${EXP_NAME:-veriseek_sft_qwen3}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}"
export CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/outputs/veriseek_sft}"
export NUM_GPUS="${NUM_GPUS:-${GPU_NUM:-1}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${MAX_STEPS:-5}}"
export SAVE_FREQ="${SAVE_FREQ:-5}"

if [[ -z "${TRAIN_FILES}" ]]; then
    cat >&2 <<'EOF'
SFT_TRAIN_FILES or TRAIN_FILES must point to an SFT-compatible parquet file.
The upstream SFT trainer does not consume VeriSeek RL rows directly.
Expected options:
  - multiturn SFT columns configured by SFT/train_sft.sh, or
  - single-turn prompt/response columns via PROMPT_KEY and RESPONSE_KEY.
EOF
    exit 2
fi

cd "${SFT_DIR}"
bash train_sft.sh
