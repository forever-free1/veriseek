# VeriSeek：基于 Qwen3-4B 的科学问答证据寻址训练

[English README](README.md) | [掘金博客：从 SFT 到 GRPO——把 Qwen3-4B 训练成会找证据的科学问答模型](https://juejin.cn/post/7657129096330854451)

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

RL-only 路径最终只比 base model 高了 +0.01。我们当时看到这个数字的第一反应是：哪里出了问题？

### 表象之下的信号问题

ArenaRL 在开放 agent 训练中的工作给了我们一个有用的概念——**歧视性崩溃**。在 LLM-as-judge 场景中，随着策略优化、输出越来越相似，逐点打分法会把奖励压缩进一个窄带（0-1 分制下的 0.8-0.9）。组内方差缩小到与裁判模型自身的采样噪声无法区分，奖励信号被噪声淹没，GRPO 的 advantage 计算拿不到真实差异。

我们的 RL-only 实验碰上了同一个模式的更硬版本。每个 rollout 都精确地得到 R=0。每一组、每一步，方差为零。信噪比从一开始就不存在。

### 格式门槛如何制造了平坦地貌

SciFact reward 使用硬格式门槛：格式无效 → R=0（见[奖励设计](#奖励设计)）。我们刻意保持了这种设计——部分得分或编辑距离塑形会引入它们自己的问题，比如给"几乎合法但下游解析仍会失败的 XML"打正分。当时我们没有充分意识到的是，这个门槛与冷启动策略之间的相互作用。

Qwen3-4B-Thinking base model 从未见过 `<answer>`/`<evidence>` 标签。它的 rollout 是连贯的思维链——文本合理，结构错误。**Format success rate: 0.0，持续了整个训练过程。** 每个 rollout，每一步，R=0。

### GRPO 为什么无从下手

GRPO 在每组响应内做归一化：A_i = (R_i - μ_group) / σ_group。ArenaRL 的核心洞察——组内相对比较比绝对打分更鲁棒——正是这种设计存在的理由。组内相对优势是 GRPO 对噪声或校准偏差奖励的回答。

但这个机制有一个它无法绕过的依赖：组内必须有奖励分布。当 N 个响应全部 R=0：

- μ_group = 0，σ_group = 0
- A_i = 0/0 → 被截断为 0
- ∇ log π(a|s) · A ≡ 0，对每个 token 成立

策略梯度恒为零。模型在整个训练过程中仅随采样噪声漂移。没有格式习得，没有准确率提升，没有证据 grounding。

### SFT 预热改变了什么

SFT 在 RL 介入之前把输出协议交给了模型。仅靠监督样本，format success rate 跳到了 0.993。格式一旦可靠，奖励函数在几乎每个 rollout 上都能真正执行——而一旦执行，证据质量和答案正确性就开始在组内产生真实的方差：

```text
Format success rate:  0.0  → 0.993  (SFT 之后)
Unsupported rate:     0.467 → 0.413  (RL 之后)
Evidence F1:          0.376 → 0.406  (RL 之后)
```

RL 阶段随后做了它该做的事：区分好的证据选择和更好的证据选择，压低无根据猜测的比例。但这一切都建立在奖励地貌先有了可导航的地形之后。

### 回头看

RL-only 的结果不说明 GRPO 脆弱，也不说明冷启动有多难。它说明的是一个简单的约束：让 GRPO 对奖励噪声鲁棒的组内相对机制，恰好有一个硬地板——零方差进，零梯度出。硬格式门槛保持了奖励的纯净，再来一次我们仍然会这样设计。但它同时意味着，策略需要在强化学习可以开始之前先达到一个最低可行的格式成功率，而达成这一点最干净的方式，是监督模仿。

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
