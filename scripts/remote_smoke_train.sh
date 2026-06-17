#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen3-4B-Thinking-2507}"
export TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/data/processed/scifact/train.parquet}"
export VAL_FILE="${VAL_FILE:-${ROOT_DIR}/data/processed/scifact/dev.parquet}"
export OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/veriseek_smoke_qwen3_1gpu}"
export GPU_NUM="${GPU_NUM:-1}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export MAX_STEPS="${MAX_STEPS:-5}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export AGENT_GRPO_N="${AGENT_GRPO_N:-1}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1280}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
export MODEL_DTYPE="${MODEL_DTYPE:-bf16}"

bash "${SCRIPT_DIR}/train_veriseek_grpo.sh" "$@"
