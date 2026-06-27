# VeriSeek SciFact Benchmark Report

Date: 2026-06-17

This report records the public SciFact dev benchmark for the VeriSeek MVP. All training paths start from the same public base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

## Evaluation Setup

Dataset: SciFact dev, 300 examples.

Generation: deterministic Transformers decoding with `do_sample=False` and `max_new_tokens=192`.

Strict VeriSeek output protocol:

```text
<answer>
SUPPORTS / REFUTES / NOT_ENOUGH_INFO
</answer>

<evidence>
[1] evidence sentence
</evidence>
```

Metrics:

- `label_accuracy`: accuracy over `SUPPORTS`, `REFUTES`, and `NOT_ENOUGH_INFO`;
- `evidence_f1`: token-level F1 between predicted and gold evidence text;
- `format_success_rate`: whether the output contains parseable answer/evidence blocks;
- `unsupported_answer_rate`: fraction of parsed answers predicted as `NOT_ENOUGH_INFO`.

Base and RL-only are reported as prefix-constrained label diagnostics because they do not reliably follow the strict VeriSeek XML evidence protocol.

## Main Result

**VeriSeek SFT+RL** gives the strongest public SciFact dev result among the four evaluated training paths:

![VeriSeek SciFact benchmark](../assets/veriseek_scifact_benchmark.png)

| Model | Training Path | SciFact Answer Acc | SciFact Evidence F1 | Evaluation Note |
|---|---|---:|---:|---|
| Qwen3-4B Base | none | 0.553 | n/a | prefix-constrained label diagnostic |
| VeriSeek RL-only | RL-only | 0.563 | n/a | prefix-constrained label diagnostic |
| VeriSeek SFT | SFT | 0.780 | 0.376 | strict XML evidence evaluation |
| VeriSeek SFT+RL | SFT+RL | **0.793** | **0.406** | strict XML evidence evaluation |

Relative to SFT:

```text
SciFact Answer Acc: +0.013
SciFact Evidence F1: +0.029
Unsupported Rate: -0.053
```

The key result is that SFT+RL improves evidence F1 while also improving answer accuracy. This is the behavior VeriSeek is designed to test: reward optimization should refine evidence grounding rather than merely change label priors.

## Reproduction Commands

Prepare SciFact:

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact
```

Run VeriSeek SFT+RL from an SFT checkpoint:

```bash
SFT_MODEL_PATH=$PWD/outputs/veriseek_sft_hf \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_sft_rl \
EXPERIMENT_NAME=veriseek_sft_rl \
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

Evaluate the public checkpoint:

```bash
RUN_DIR=$PWD/outputs/veriseek_sft_rl \
BENCH_DIR=$PWD/outputs/benchmarks/veriseek_sft_rl \
TMP_PREFIX=$PWD/outputs/tmp_veriseek_sft_rl \
STEPS="200" \
bash scripts/eval_gated_checkpoints.sh
```

Regenerate the figure:

```bash
python scripts/plot_veriseek_results.py \
  --source assets/veriseek_scifact_benchmark_source.tsv \
  --output_prefix assets/veriseek_scifact_benchmark
```

## Checked-In Figure Source

```text
assets/veriseek_scifact_benchmark_source.tsv
```

## Conclusion

The current VeriSeek MVP result is positive but intentionally modest. SFT is essential for learning the output protocol, while the SFT+RL stage provides a small but consistent improvement in both answer accuracy and evidence F1 on SciFact dev.
