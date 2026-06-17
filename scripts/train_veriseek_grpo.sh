#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RL_DIR="${ROOT_DIR}/RL"
BASE_MODEL_ID="Qwen/Qwen3-4B-Thinking-2507"
LOCAL_MODEL_PATH="${ROOT_DIR}/models/Qwen3-4B-Thinking-2507"

if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
elif [ -f "${RL_DIR}/.env" ]; then
    set -a
    source "${RL_DIR}/.env"
    set +a
fi

export GRPC_PYTHON_BUILD_WITH_CYTHON=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export PYTHONFAULTHANDLER=1
export TORCH_DISABLE_ADDR2LINE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ENABLE_MONITORING=0
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export PYTHONPATH="${ROOT_DIR}/runtime_patches:${PYTHONPATH:-}"

PROJECT_NAME="${PROJECT_NAME:-VeriSeek}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-veriseek_scifact_smoke}"
MODEL_PATH="${MODEL_PATH:-${LOCAL_MODEL_PATH}}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/data/processed/scifact/train.parquet}"
VAL_FILE="${VAL_FILE:-${ROOT_DIR}/data/processed/scifact/dev.parquet}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/veriseek_smoke}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-${OUTPUT}/eval_log}"
TOTAL_TRAINING_STEPS="${MAX_STEPS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
AGENT_GRPO_N="${AGENT_GRPO_N:-1}"

USE_INFO_GAIN=false
ADV_ESTIMATOR=grpo
if [ "${TRAIN_REWARD_TYPE:-f1}" != "f1" ]; then
    echo "VeriSeek uses deterministic evidence reward only; set TRAIN_REWARD_TYPE=f1 or leave it unset." >&2
    exit 2
fi
TRAIN_REWARD_TYPE="f1"
MASK_TOOL_RESPONSE="${MASK_TOOL_RESPONSE:-true}"
USE_ASYNC_ROLLOUT="${USE_ASYNC_ROLLOUT:-false}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1280}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
ULYSSES_SP_SIZE="${ULYSSES_SP_SIZE:-1}"
ASYNC_PROMPT_PAD="${ASYNC_PROMPT_PAD:-1024}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SAVE_FREQ="${SAVE_FREQ:-5}"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NUM="${GPU_NUM:-$(nvidia-smi -L | wc -l)}"
else
    GPU_NUM="${GPU_NUM:-1}"
fi

mkdir -p "${OUTPUT}" "${EVAL_LOG_PATH}"

if [ "${USE_ASYNC_ROLLOUT}" = "true" ]; then
    _max_seq=$((ASYNC_PROMPT_PAD + MAX_MODEL_LEN))
else
    _max_seq=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
fi
PPO_MAX_TOKEN_LEN=$(( (_max_seq + ULYSSES_SP_SIZE - 1) / ULYSSES_SP_SIZE + 1000 ))

cd "${RL_DIR}"

echo "VeriSeek base model: ${BASE_MODEL_ID}"
echo "Using MODEL_PATH=${MODEL_PATH}"

HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
IGPO_ROLLOUT_IG=0 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_info_gain="${USE_INFO_GAIN}" \
    algorithm.use_format_penalty=false \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.actor.mask_tool_response="${MASK_TOOL_RESPONSE}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.nccl_timeout=7200 \
    critic.optim.lr=1e-5 \
    critic.model.path="${MODEL_PATH}" \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.gamma=0.95 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${GPU_NUM}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.validation_data_dir="${EVAL_LOG_PATH}" \
    trainer.default_local_dir="${OUTPUT}" \
    agent_grpo.n="${AGENT_GRPO_N}" \
    max_turns=1 \
    reward_model.train_reward_type="${TRAIN_REWARD_TYPE}" \
    +reward_model.valid_reward_type=f1 \
    reward_model.reward_manager=naive_batch \
    +reward_model.reward_kwargs.deepthink_disabled=true \
    data.return_raw_chat=true \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    $([ "${USE_ASYNC_ROLLOUT}" = "true" ] && echo "actor_rollout_ref.rollout.mode=async" || true) \
    $([ "${USE_ASYNC_ROLLOUT}" = "true" ] && echo "actor_rollout_ref.rollout.prompt_length=${ASYNC_PROMPT_PAD}" || true) \
    $([ "${USE_ASYNC_ROLLOUT}" = "true" ] && echo "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN}" || true) \
    $([ "${USE_ASYNC_ROLLOUT}" = "true" ] && echo "actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/dr_agent_loop.yaml" || true) \
    ${EXTRA_HYDRA_ARGS:-} \
    "$@" \
    > >(stdbuf -oL tee "${OUTPUT}/training.log") 2>&1
