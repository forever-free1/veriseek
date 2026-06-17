# VeriSeek Smoke Training Runbook

This runbook describes a 5-step GPU smoke training job with the only public/default base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Local default path:

```text
models/Qwen3-4B-Thinking-2507
```

The smoke run verifies data loading, rollout, deterministic evidence reward routing, GRPO advantage computation, logging, and checkpoint saving. It does not replace full experiments.

## 1. Prepare Local Model

```bash
bash scripts/local_prepare_assets.sh
```

Equivalent direct command:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B-Thinking-2507",
    local_dir="models/Qwen3-4B-Thinking-2507",
    local_dir_use_symlinks=False,
)
PY
```

## 2. Prepare Tiny SciFact Data

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact \
  --max_train 20 \
  --max_dev 5 \
  --max_test 5
```

Expected files:

```text
data/processed/scifact/train.parquet
data/processed/scifact/dev.parquet
data/processed/scifact/test.parquet
```

Each row must include `prompt`, `reward_model.ground_truth`, and `data_source=scifact_evidence`.

## 3. Required Environment Variables

`MODEL_PATH`

: Default `models/Qwen3-4B-Thinking-2507`.

`TRAIN_FILE`

: Path to `train.parquet`.

`VAL_FILE`

: Path to `dev.parquet`.

`MAX_STEPS`

: Use `5` for smoke training.

`TRAIN_BATCH_SIZE`

: Use `2` for the first smoke run.

Recommended output directories:

```text
outputs/qwen3_base_eval/
outputs/veriseek_sft/
outputs/veriseek_rl_only/
outputs/veriseek_sft_rl/
```

Optional variables:

- `OUTPUT`: checkpoint and log directory
- `GPU_NUM`: number of GPUs to request from the trainer
- `TENSOR_MODEL_PARALLEL_SIZE`: default `1`
- `PPO_MINI_BATCH_SIZE`: default equals `TRAIN_BATCH_SIZE`
- `AGENT_GRPO_N`: default `1`
- `MAX_MODEL_LEN`: default `2048`
- `MAX_PROMPT_LEN`: default `1280`
- `MAX_RESPONSE_LEN`: default `512`
- `MAX_NUM_BATCHED_TOKENS`: default `2048`
- `ROLLOUT_MAX_NUM_SEQS`: default `2`
- `ROLLOUT_GPU_MEMORY_UTILIZATION`: default `0.45`
- `MODEL_DTYPE`: default `bf16`
- `SAVE_FREQ`: default `5`

The script sets `algorithm.use_info_gain=false`, `algorithm.adv_estimator=grpo`, and forces `TRAIN_REWARD_TYPE=f1` so the deterministic evidence reward route is used instead of an LLM judge.

## 4. One-GPU RL-Only Smoke Command

```bash
MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_smoke_qwen3_1gpu \
GPU_NUM=1 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
MAX_STEPS=5 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
AGENT_GRPO_N=1 \
MAX_PROMPT_LEN=1280 \
MAX_RESPONSE_LEN=512 \
MAX_MODEL_LEN=2048 \
bash scripts/train_veriseek_grpo.sh
```

Equivalent wrapper:

```bash
bash scripts/remote_smoke_train.sh
```

## 5. Two-GPU RL-Only Smoke Command

```bash
MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_smoke_qwen3_2gpu \
GPU_NUM=2 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
MAX_STEPS=5 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
AGENT_GRPO_N=1 \
MAX_PROMPT_LEN=1280 \
MAX_RESPONSE_LEN=512 \
MAX_MODEL_LEN=2048 \
bash scripts/train_veriseek_grpo.sh
```

Use `TENSOR_MODEL_PARALLEL_SIZE=1` for the first 2xA800 run because Qwen3-4B fits on one A800 and this keeps the rollout path simple. If a longer context or larger rollout batch requires tensor parallel rollout, try `TENSOR_MODEL_PARALLEL_SIZE=2`.

## 6. Two-GPU 50-Step Readiness Command

```bash
MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_rl_only_50step_2gpu \
GPU_NUM=2 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
MAX_STEPS=50 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
AGENT_GRPO_N=1 \
MAX_PROMPT_LEN=1280 \
MAX_RESPONSE_LEN=512 \
MAX_MODEL_LEN=2048 \
SAVE_FREQ=50 \
bash scripts/train_veriseek_grpo.sh
```

## 7. SFT Smoke Command

SFT uses the upstream SFT trainer. It requires SFT-compatible data, not the VeriSeek RL parquet rows.

```bash
MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
SFT_TRAIN_FILES=/path/to/sft_train.parquet \
SFT_VAL_FILES=/path/to/sft_val.parquet \
CKPT_DIR=$PWD/outputs/veriseek_sft \
GPU_NUM=1 \
MAX_STEPS=5 \
TRAIN_BATCH_SIZE=2 \
bash scripts/train_veriseek_sft.sh
```

If `SFT_TRAIN_FILES` is missing, the script exits with an explicit schema note.

## 8. SFT+RL Smoke Command

Run this only after an SFT checkpoint exists.

```bash
SFT_MODEL_PATH=$PWD/outputs/veriseek_sft \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_sft_rl \
GPU_NUM=1 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
MAX_STEPS=5 \
TRAIN_BATCH_SIZE=2 \
bash scripts/train_veriseek_sft_rl.sh
```

## 9. Expected Logs

The RL scripts write:

```text
$OUTPUT/training.log
$OUTPUT/eval_log/
```

Useful log signals:

- `VeriSeek base model: Qwen/Qwen3-4B-Thinking-2507`
- `algorithm.use_info_gain=false`
- `algorithm.adv_estimator=grpo`
- `reward_model.reward_manager=naive_batch`
- `data.train_files=...train.parquet`
- `data.val_files=...dev.parquet`
- reward prints with `data_source scifact_evidence`
- `training/global_step` increases to `5`

## 10. Verify Checkpoint Save

The default smoke setting uses `SAVE_FREQ=5`, so a successful 5-step run should save a checkpoint under `$OUTPUT`.

```bash
find "$OUTPUT" -maxdepth 4 -type d | sort
grep -n "save_checkpoint\|Saving" "$OUTPUT/training.log" || true
grep -n "training/global_step" "$OUTPUT/training.log" | tail
```

## 11. Common Failure Modes

Missing model files

: Run `bash scripts/local_prepare_assets.sh` or set `MODEL_PATH` to a valid local Qwen3 checkpoint.

Missing SFT data

: `scripts/train_veriseek_sft.sh` requires SFT-compatible data. The RL parquet rows are not SFT-ready.

Missing full RL dependencies

: Install the upstream RL dependencies in `RL/requirements.txt`, plus the CUDA-specific torch/vLLM stack required by your machine.

CUDA OOM

: Lower `MAX_MODEL_LEN`, `MAX_PROMPT_LEN`, `MAX_RESPONSE_LEN`, `ROLLOUT_MAX_NUM_SEQS`, or `ROLLOUT_GPU_MEMORY_UTILIZATION`. On 2 GPUs, keep `TENSOR_MODEL_PARALLEL_SIZE=1` first; use `2` only if rollout tensor parallelism is needed.

Dataset path error

: Re-run the SciFact preparation command and pass absolute `TRAIN_FILE` and `VAL_FILE` paths.

No checkpoint after 5 steps

: Confirm `SAVE_FREQ=5`, inspect `$OUTPUT/training.log`, and check whether the run reached `training/global_step=5`.

## 12. Preflight Without Full Training

```bash
bash scripts/debug_veriseek_pipeline.sh
```

This prepares a tiny SciFact split, inspects parquet schema, runs synthetic reward checks, runs lightweight tests, and checks shell syntax. It does not start RL training.
