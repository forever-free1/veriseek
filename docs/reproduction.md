# VeriSeek Reproduction Notes

VeriSeek uses `Qwen/Qwen3-4B-Thinking-2507` as the only public/default base model.

## Prepare Local Model

```bash
bash scripts/local_prepare_assets.sh
```

This downloads the model to:

```text
models/Qwen3-4B-Thinking-2507
```

## Run Lightweight Tests

```bash
python tests/test_evidence_reward.py -v
python tests/test_data_and_eval.py -v
python tests/test_model_defaults.py -v
```

## Prepare Tiny SciFact

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact \
  --max_train 20 \
  --max_dev 5 \
  --max_test 5
```

## RL-Only Smoke Train

```bash
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

## Training Path Wrappers

```bash
bash scripts/train_veriseek_sft.sh
bash scripts/train_veriseek_rl_only.sh
bash scripts/train_veriseek_sft_rl.sh
```

`train_veriseek_sft.sh` wraps the upstream SFT trainer and requires SFT-compatible data. `train_veriseek_rl_only.sh` starts evidence-aware RL from the Qwen3 base model. `train_veriseek_sft_rl.sh` starts evidence-aware RL from a completed SFT checkpoint.

Recommended output directories:

```text
outputs/qwen3_base_eval/
outputs/veriseek_sft/
outputs/veriseek_rl_only/
outputs/veriseek_sft_rl/
```

## Evaluate Predictions

Prediction JSONL records should include `prediction` and either `ground_truth` or `reward_model.ground_truth`.

```bash
python eval/eval_scifact.py --pred_path outputs/scifact_predictions.jsonl
python eval/eval_qasper.py --pred_path outputs/qasper_predictions.jsonl
python eval/eval_litqa2.py --pred_path outputs/litqa2_predictions.jsonl
```
