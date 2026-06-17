# VeriSeek SFT Scaffold

This directory contains the supervised fine-tuning scaffold used by VeriSeek. The top-level wrapper is:

```bash
bash scripts/train_veriseek_sft.sh
```

The SFT stage starts from the public/default base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

and writes a checkpoint that can be used as the initialization for evidence-aware RL:

```bash
SFT_MODEL_PATH=$PWD/outputs/veriseek_sft_hf \
bash scripts/train_veriseek_sft_rl.sh
```

## Data

The upstream SFT trainer expects SFT-compatible data, not the RL reward rows directly. Pass the data through environment variables:

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

Common modes:

- multi-turn SFT with a `messages` column;
- single-turn SFT with `MULTITURN=False`, `PROMPT_KEY`, and `RESPONSE_KEY`.

See the top-level `README.md` for the full VeriSeek reproduction flow.
