# VeriSeek SciFact Benchmark Report

Date: 2026-06-17

This report records the SciFact dev benchmark used for the VeriSeek MVP. All main experiments start from the same public base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

## Evaluation Setup

Dataset: SciFact dev, 300 examples.

Generation: deterministic Transformers decoding with `do_sample=False` and `max_new_tokens=192`.

Strict output protocol:

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

## Main Result

The final gated n=4 SFT+RL checkpoint improves over the SFT baseline on both SciFact label accuracy and evidence F1:

![VeriSeek SciFact benchmark](../assets/veriseek_scifact_benchmark.svg)

| Model | Training Path | SciFact Acc | Evidence F1 | Format Success | Unsupported Rate |
|---|---|---:|---:|---:|---:|
| VeriSeek-SFT | SFT | 0.780 | 0.376 | 0.993 | 0.467 |
| Old VeriSeek-SFT-RL | SFT+RL, n=1 reward | 0.747 | 0.331 | 0.993 | 0.567 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 50 | 0.783 | 0.389 | 0.993 | 0.437 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 100 | 0.790 | 0.400 | 0.990 | 0.420 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 150 | 0.790 | 0.405 | 0.990 | 0.417 |
| VeriSeek-SFT-RL | gated SFT+RL, n=4, step 200 | **0.793** | **0.406** | 0.990 | **0.413** |

Relative to SFT:

```text
SciFact Acc:         +0.013
SciFact Evidence F1: +0.029
Unsupported Rate:    -0.053
```

## Negative Control

The old SFT+RL run is retained as a negative control. It increased training reward but reduced held-out SciFact dev quality:

| Model | SciFact Acc | Evidence F1 | Unsupported Rate |
|---|---:|---:|---:|
| VeriSeek-SFT | 0.780 | 0.376 | 0.467 |
| Old VeriSeek-SFT-RL | 0.747 | 0.331 | 0.567 |

The old run over-predicted `NOT_ENOUGH_INFO`, becoming more conservative without improving evidence quality. This motivated the gated reward:

- invalid format receives zero reward;
- weak evidence caps score for answerable claims;
- `NOT_ENOUGH_INFO` on answerable claims receives only a small reward;
- `AGENT_GRPO_N=4` provides a true group comparison signal.

## Reproduction Commands

Prepare SciFact:

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact
```

Run gated n=4 SFT+RL from an SFT checkpoint:

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

Evaluate checkpoints:

```bash
RUN_DIR=$PWD/outputs/veriseek_sft_rl_gated_n4 \
BENCH_DIR=$PWD/outputs/benchmarks/gated_n4 \
TMP_PREFIX=$PWD/outputs/tmp_gated_n4_step \
STEPS="50 100 150 200" \
bash scripts/eval_gated_checkpoints.sh
```

Regenerate the figure:

```bash
python scripts/plot_veriseek_results.py \
  --source assets/veriseek_scifact_benchmark_source.tsv \
  --output_prefix assets/veriseek_scifact_benchmark
```

## Artifacts

Remote benchmark artifacts from the successful run:

```text
/hy-tmp/veriseek/outputs/benchmarks/gated_n4/summary.tsv
/hy-tmp/veriseek/outputs/benchmarks/gated_n4/eval.log
/hy-tmp/veriseek/outputs/benchmarks/gated_n4/*_metrics.json
/hy-tmp/veriseek/outputs/benchmarks/gated_n4/*_components.json
/hy-tmp/veriseek/outputs/benchmarks/gated_n4/*_step_*.jsonl
```

Checked-in figure source:

```text
assets/veriseek_scifact_benchmark_source.tsv
```

## Conclusion

The current VeriSeek MVP result is positive but modest. SFT remains essential for learning the output protocol. Naive SFT+RL can reward conservative unsupported answers. Gated reward plus n=4 group sampling produces the best SciFact dev result in this run, improving both answer accuracy and evidence F1 while reducing unsupported predictions.
