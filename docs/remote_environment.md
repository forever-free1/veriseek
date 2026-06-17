# Remote Environment Setup

This project ships the model files locally under:

```text
models/Qwen3-4B-Thinking-2507
```

After uploading the packed repository to a GPU machine, install the runtime on the remote machine instead of pre-building it locally. CUDA, torch, flash-attn, and vLLM wheels must match the rented GPU image.

## Minimal CPU Preflight Dependencies

```bash
python -m pip install -U pip
python -m pip install pandas pyarrow requests datasets huggingface_hub
```

Run:

```bash
bash scripts/debug_veriseek_pipeline.sh
```

## RL Training Dependencies

From the repository root on the GPU machine:

```bash
cd RL
python -m pip install -r requirements.txt
```

Install CUDA-specific packages according to the GPU image before or alongside the RL requirements:

```bash
# Example only. Match these to the rented machine CUDA version.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install vllm
python -m pip install flash-attn --no-build-isolation
```

## Smoke Training

```bash
cd /path/to/veriseek

python data/prepare_scifact.py \
  --output_dir data/processed/scifact \
  --max_train 20 \
  --max_dev 5 \
  --max_test 5

MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_smoke_qwen3_1gpu \
GPU_NUM=1 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
MAX_STEPS=5 \
TRAIN_BATCH_SIZE=2 \
bash scripts/train_veriseek_grpo.sh
```

If using two GPUs:

```bash
MODEL_PATH=$PWD/models/Qwen3-4B-Thinking-2507 \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_smoke_qwen3_2gpu \
GPU_NUM=2 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
MAX_STEPS=5 \
TRAIN_BATCH_SIZE=2 \
bash scripts/train_veriseek_grpo.sh
```
