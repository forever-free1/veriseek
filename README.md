# VeriSeek: Evidence-Seeking Scientific QA With Qwen3-4B

[中文说明](README_CN.md)

VeriSeek is a minimal, reproducible project for training and evaluating evidence-grounded scientific QA agents. The public/default base model is:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Local default path:

```text
models/Qwen3-4B-Thinking-2507
```

Starting from the same compact reasoning model, VeriSeek compares whether scientific evidence grounding is better learned by supervised imitation, evidence-aware reward optimization, or a two-stage SFT+RL pipeline.

## Method Overview

![VeriSeek method overview](assets/veriseek_method_overview.svg)

## Main Result

On SciFact dev, the best VeriSeek checkpoint is the gated n=4 SFT+RL run at `global_step_200`. It improves over the SFT baseline on both label accuracy and evidence F1:

![VeriSeek SciFact benchmark](assets/veriseek_scifact_benchmark.svg)

| Model | Training Path | SciFact Acc | SciFact Evidence F1 | Format Success | Unsupported Rate |
|---|---|---:|---:|---:|---:|
| VeriSeek-SFT | SFT | 0.780 | 0.376 | 0.993 | 0.467 |
| Old VeriSeek-SFT-RL | SFT+RL, n=1 reward | 0.747 | 0.331 | 0.993 | 0.567 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 50 | 0.783 | 0.389 | 0.993 | 0.437 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 100 | 0.790 | 0.400 | 0.990 | 0.420 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 150 | 0.790 | 0.405 | 0.990 | 0.417 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 200 | **0.793** | **0.406** | 0.990 | **0.413** |

Compared with SFT, the step-200 gated n=4 checkpoint gives:

```text
SciFact Acc:         +0.013
SciFact Evidence F1: +0.029
Unsupported Rate:    -0.053
```

The old SFT+RL run is kept as a negative control: its training reward rose, but the model became too conservative and over-predicted `NOT_ENOUGH_INFO`. The final gated n=4 run fixes that failure mode by making format a hard gate, capping weak-evidence answers, and using four sampled responses per prompt for a real GRPO comparison signal.

## Training Paths

The project compares four evaluation conditions:

- `Qwen3-4B Base`: untrained base model evaluation when possible.
- `VeriSeek-SFT`: supervised fine-tuning from `Qwen/Qwen3-4B-Thinking-2507`.
- `VeriSeek-RL`: evidence-aware RL directly from `Qwen/Qwen3-4B-Thinking-2507`.
- `VeriSeek-SFT-RL`: evidence-aware RL initialized from the VeriSeek-SFT checkpoint.

SFT teaches the output format and task behavior. RL-only tests whether deterministic evidence reward can induce evidence-seeking behavior without imitation. SFT+RL tests whether imitation followed by reward optimization gives the best trade-off.

## Repository Map

```text
data/                         dataset converters for SciFact, QASPER, LitQA2
eval/                         deterministic evaluation scripts and metrics
RL/verl/utils/reward_score/   VeriSeek evidence reward implementation
scripts/train_veriseek_*.sh   SFT, RL-only, and SFT+RL entrypoints
scripts/eval_gated_checkpoints.sh
scripts/plot_veriseek_results.py
docs/                         reward, data, smoke-training, and benchmark notes
assets/                       benchmark figure and source data
```

The MVP avoids trainer rewrites, rollout changes, search/visit protocol changes, PDF parsing, figure/table parsing, multimodal inputs, embedding similarity, and LLM-as-a-judge rewards.

## Reward Objective

The first reward version was:

```text
R = 0.45 * R_answer
  + 0.35 * R_evidence
  + 0.15 * R_format
  + 0.05 * R_conciseness
```

SciFact now uses a gated deterministic variant after the first SFT+RL run exposed a conservative `NOT_ENOUGH_INFO` failure mode:

- invalid format receives zero reward;
- for `SUPPORTS` and `REFUTES`, weak evidence caps the total score;
- predicting `NOT_ENOUGH_INFO` for answerable claims receives only a small reward;
- `NOT_ENOUGH_INFO` gold examples are rewarded for a correct label plus empty or concise evidence;
- QASPER keeps the weighted answer/evidence/format/conciseness reward.

See [docs/reward_design.md](docs/reward_design.md).

## Environment

The successful full run used:

```text
Python 3.11
PyTorch 2.9.1
CUDA 12.8
2 x NVIDIA A800 80GB
```

The 5-step smoke job can run on one GPU. The 200-step gated n=4 run was executed on two A800 GPUs.

## Download The Base Model

```bash
bash scripts/local_prepare_assets.sh
```

This downloads:

```text
Qwen/Qwen3-4B-Thinking-2507 -> models/Qwen3-4B-Thinking-2507
```

## Prepare SciFact Data

Tiny debug split:

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact \
  --max_train 20 \
  --max_dev 5 \
  --max_test 5
```

Full SciFact split:

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact
```

Rows are written as parquet records compatible with the existing training stack:

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {"ground_truth": "{\"answer\": \"SUPPORTS\", \"evidence\": [\"...\"]}"},
  "data_source": "scifact_evidence"
}
```

## Run A 5-Step RL Smoke Job

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

## Reproduce The Gated n=4 SFT+RL Run

First prepare an SFT checkpoint. The wrapper uses the upstream SFT trainer and requires SFT-compatible data:

```bash
MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
TRAIN_FILES=/path/to/sft_train.parquet \
VAL_FILES=/path/to/sft_dev.parquet \
CKPT_DIR=$PWD/outputs/veriseek_sft \
NUM_GPUS=2 \
TRAIN_BATCH_SIZE=2 \
TOTAL_TRAINING_STEPS=400 \
SAVE_FREQ=50 \
bash scripts/train_veriseek_sft.sh
```

Then run evidence-aware SFT+RL from the SFT checkpoint:

```bash
SFT_MODEL_PATH=$PWD/outputs/veriseek_sft_hf \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_sft_rl_gated_n4 \
EXPERIMENT_NAME=veriseek_sft_rl_gated_n4 \
MAX_STEPS=200 \
SAVE_FREQ=50 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
GPU_NUM=2 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
MAX_PROMPT_LEN=1664 \
MAX_RESPONSE_LEN=512 \
MAX_MODEL_LEN=2560 \
MAX_NUM_BATCHED_TOKENS=4096 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.55 \
ROLLOUT_MAX_NUM_SEQS=8 \
AGENT_GRPO_N=4 \
bash scripts/train_veriseek_sft_rl.sh
```

Expected checkpoints:

```text
outputs/veriseek_sft_rl_gated_n4/global_step_50
outputs/veriseek_sft_rl_gated_n4/global_step_100
outputs/veriseek_sft_rl_gated_n4/global_step_150
outputs/veriseek_sft_rl_gated_n4/global_step_200
```

## Evaluate Checkpoints

```bash
RUN_DIR=$PWD/outputs/veriseek_sft_rl_gated_n4 \
BENCH_DIR=$PWD/outputs/benchmarks/gated_n4 \
TMP_PREFIX=$PWD/outputs/tmp_gated_n4_step \
STEPS="50 100 150 200" \
bash scripts/eval_gated_checkpoints.sh
```

This merges each FSDP checkpoint into a temporary Hugging Face directory, generates SciFact dev predictions, computes strict and relaxed metrics, writes component diagnostics, and removes the temporary model directory after each step.

## Recreate The Figure

```bash
python scripts/plot_veriseek_results.py \
  --source assets/veriseek_scifact_benchmark_source.tsv \
  --output_prefix assets/veriseek_scifact_benchmark
```

Outputs:

```text
assets/veriseek_scifact_benchmark.svg
assets/veriseek_scifact_benchmark.pdf
assets/veriseek_scifact_benchmark.png
assets/veriseek_scifact_benchmark.tiff
assets/veriseek_method_overview.svg
assets/veriseek_method_overview.pdf
assets/veriseek_method_overview.png
```

## Evaluate A Prediction File

```bash
python eval/eval_scifact.py \
  --pred_path outputs/scifact_predictions.jsonl \
  --mode both
```

Prediction JSONL records should include `prediction` and either `ground_truth` or `reward_model.ground_truth`.

## Documentation

- [Reward design](docs/reward_design.md)
- [Data format](docs/data_format.md)
- [Reproduction notes](docs/reproduction.md)
- [Smoke training runbook](docs/smoke_training.md)
- [SciFact benchmark report](docs/benchmark_report.md)
