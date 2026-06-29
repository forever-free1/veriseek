# VeriSeek：基于 Qwen3-4B 的科学问答证据寻址训练

[English README](README.md)

VeriSeek 是一个小型、可复现的科学问答 evidence grounding 项目。它要求模型不仅给出答案，还要给出可检查的证据。项目唯一公开默认 base model 是：

```text
Qwen/Qwen3-4B-Thinking-2507
```

本地默认路径是：

```text
models/Qwen3-4B-Thinking-2507
```

核心问题是：从同一个 compact reasoning model 出发，科学证据 grounding 更适合通过监督模仿学到，还是通过 evidence-aware reward optimization 学到，或者需要 SFT+RL 两阶段结合？

## 方法概览

![VeriSeek method overview](assets/veriseek_method_overview.svg)

VeriSeek 刻意保持训练栈简单：

- 不重写 trainer；
- 不重设计 rollout；
- 不修改 search/visit tool protocol；
- 不加入 PDF、figure、table 或 multimodal parsing；
- 不使用 embedding similarity reward；
- 不使用 LLM-as-a-judge reward。

模型通过一个确定性的 answer/evidence 协议训练和评测：

```text
<answer>
SUPPORTS / REFUTES / NOT_ENOUGH_INFO
</answer>

<evidence>
[1] evidence sentence
</evidence>
```

## 主要结果

在 SciFact dev（`n = 300`）上，最终公开模型 **VeriSeek SFT+RL** 在四条训练路径中取得了最好的答案准确率和证据 grounding 指标。

![VeriSeek SciFact benchmark](assets/veriseek_scifact_benchmark.png)

| Model | Training Path | SciFact Answer Acc | SciFact Evidence F1 | Evaluation Note |
|---|---|---:|---:|---|
| Qwen3-4B Base | none | 0.553 | n/a | prefix-constrained label diagnostic |
| VeriSeek RL-only | RL-only | 0.563 | n/a | prefix-constrained label diagnostic |
| VeriSeek SFT | SFT | 0.780 | 0.376 | strict XML evidence evaluation |
| VeriSeek SFT+RL | SFT+RL | **0.793** | **0.406** | strict XML evidence evaluation |

相对 SFT baseline：

```text
SciFact Answer Acc: +0.013
SciFact Evidence F1: +0.029
Unsupported Rate: -0.053
```

Base 和 RL-only 主要作为标签空间诊断对照。它们不能稳定遵循 VeriSeek 的 XML evidence 协议，因此 evidence F1 不作为严格可比的 grounding 分数报告。

## 训练路径

VeriSeek 从同一个 base model 比较四种条件：

- `Qwen3-4B Base`：未训练 base model；
- `VeriSeek RL-only`：直接从 `Qwen/Qwen3-4B-Thinking-2507` 做 evidence-aware RL；
- `VeriSeek SFT`：从 `Qwen/Qwen3-4B-Thinking-2507` 做监督微调；
- `VeriSeek SFT+RL`：先做 VeriSeek SFT，再从 SFT checkpoint 做 evidence-aware RL。

SFT 负责教会模型输出协议和任务行为。RL-only 测试 evidence reward 本身是否足以诱导证据寻址行为。SFT+RL 测试“先模仿，再用奖励优化”是否能得到最好的折中。

## 奖励设计

最终 SciFact reward 是确定性的 evidence-aware reward：

```text
if format is invalid:
    R = 0

if gold is NOT_ENOUGH_INFO:
    R = 0.80 * R_answer + 0.20 * R_empty_or_concise_evidence

if gold is SUPPORTS or REFUTES and prediction is NOT_ENOUGH_INFO:
    R = 0.05

otherwise:
    R = 0.35 * R_answer + 0.55 * R_evidence + 0.10 * R_conciseness
    if R_evidence < 0.20:
        R = min(R, 0.25)
```

这个 reward 只使用标签正确性、token-level evidence overlap、输出格式和证据简洁性。详见 [docs/reward_design.md](docs/reward_design.md)。

## RL 训练的最痛点：冷启动与奖励稀疏

RL-only 路径（VeriSeek RL-only）相比未训练的 base model 仅提升 +0.01，本质上是零有效增益。两个紧密耦合的问题共同导致了这一结果。

### 冷启动

Qwen3-4B-Thinking base model 从未见过 `<answer>`/`<evidence>` 这套 XML 协议。没有 SFT 预热，模型完全不知道输出应当遵循这一结构。在 RL rollout 中，模型生成的是自由形式的推理文本，几乎不可能偶然命中所需的 XML 格式。

直接证据：RL-only 的 format success rate 为 **0.0**。模型在整个训练和评测过程中，从未产生过一个符合 VeriSeek 协议的有效输出。

### 奖励稀疏

SciFact reward 使用硬格式门槛：格式无效 → R 无条件为 0。没有部分得分，没有格式距离奖励，没有任何塑形信号。当所有 rollout 都返回 R=0 时，奖励地貌是一片平地。

### 为什么 GRPO 有效增益为零

GRPO 通过在组内比较采样响应来计算优势（advantage）。当一个 prompt 的所有响应都得到 R=0 时，每个 token 的组内相对优势 A 全为零。策略梯度 ∇ log π(a|s) · A 恒等于零——GRPO 没有任何信号来区分好坏行为。模型仅在随机噪声中漂移，无法产生有意义的格式学习或准确率提升。

Base（0.553）和 RL-only（0.563）的答案准确率均通过 prefix-constrained label extraction 测量，而非通过 VeriSeek XML 协议。两个模型都不能稳定遵循 evidence 协议，因此它们的 evidence F1 不作为严格可比的 grounding 分数报告。

### SFT+RL 如何规避这两个问题

SFT 通过行为克隆将 format success rate 提升至 ~0.99，为 RL 提供了密集的奖励地貌，使证据质量和答案正确性能够提供可区分的信号。RL 阶段则进一步优化证据选择的精细度，并减少无根据的猜测：

```text
Format success rate:  0.0  → 0.993  (SFT 之后)
Unsupported rate:     0.467 → 0.413  (RL 之后)
Evidence F1:          0.376 → 0.406  (RL 之后)
```

## 仓库结构

```text
data/                         SciFact、QASPER、LitQA2 数据转换器
eval/                         确定性评测脚本与指标
RL/verl/utils/reward_score/   VeriSeek evidence reward 实现
scripts/train_veriseek_*.sh   SFT、RL-only、SFT+RL 训练入口
scripts/eval_gated_checkpoints.sh
scripts/plot_veriseek_results.py
docs/                         reward、data、smoke training、benchmark 文档
assets/                       benchmark 图和源数据表
```

## 环境

成功的完整实验使用：

```text
Python 3.11
PyTorch 2.9.1
CUDA 12.8
2 x NVIDIA A800 80GB
```

5-step smoke job 可以在单卡上运行。可复现的 VeriSeek SFT+RL 训练使用 2 张 A800。

## 下载 base model

```bash
bash scripts/local_prepare_assets.sh
```

该脚本下载：

```text
Qwen/Qwen3-4B-Thinking-2507 -> models/Qwen3-4B-Thinking-2507
```

## 准备 SciFact 数据

小型 debug split：

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact \
  --max_train 20 \
  --max_dev 5 \
  --max_test 5
```

完整 SciFact split：

```bash
python data/prepare_scifact.py \
  --output_dir data/processed/scifact
```

每行 parquet 数据兼容现有训练栈：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {"ground_truth": "{\"answer\": \"SUPPORTS\", \"evidence\": [\"...\"]}"},
  "data_source": "scifact_evidence"
}
```

## 运行 5-step RL smoke job

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

等价 wrapper：

```bash
bash scripts/remote_smoke_train.sh
```

## 复现 VeriSeek SFT+RL

先准备 SFT checkpoint。该 wrapper 使用上游 SFT trainer，需要 SFT-compatible 数据：

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

然后从 SFT checkpoint 启动 evidence-aware SFT+RL：

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

公开结果对应：

```text
outputs/veriseek_sft_rl/global_step_200
```

## 评测公开 checkpoint

```bash
RUN_DIR=$PWD/outputs/veriseek_sft_rl \
BENCH_DIR=$PWD/outputs/benchmarks/veriseek_sft_rl \
TMP_PREFIX=$PWD/outputs/tmp_veriseek_sft_rl \
STEPS="200" \
bash scripts/eval_gated_checkpoints.sh
```

该脚本会把 FSDP checkpoint 临时 merge 成 Hugging Face 目录，生成 SciFact dev 预测，计算 strict/relaxed 指标和 component 诊断，并在评测结束后删除临时模型目录。

## 重画图

```bash
python scripts/plot_veriseek_results.py \
  --source assets/veriseek_scifact_benchmark_source.tsv \
  --output_prefix assets/veriseek_scifact_benchmark

python scripts/plot_veriseek_method.py
```

输出：

```text
assets/veriseek_scifact_benchmark.png
assets/veriseek_method_overview.svg
assets/veriseek_method_overview.pdf
assets/veriseek_method_overview.png
```

## 评测预测文件

```bash
python eval/eval_scifact.py \
  --pred_path outputs/scifact_predictions.jsonl \
  --mode both
```

预测 JSONL 每行应包含 `prediction`，并包含 `ground_truth` 或 `reward_model.ground_truth`。

## 文档

- [Reward design](docs/reward_design.md)
- [Data format](docs/data_format.md)
- [Reproduction notes](docs/reproduction.md)
- [Smoke training runbook](docs/smoke_training.md)
- [SciFact benchmark report](docs/benchmark_report.md)
