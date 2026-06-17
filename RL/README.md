# VeriSeek RL Scaffold

This directory contains the RL training scaffold used by VeriSeek. It is based on the upstream `verl`-style agent training stack, but VeriSeek keeps the changes deliberately small:

- deterministic evidence reward in `verl/utils/reward_score/evidence_reward.py`;
- reward routing for `scifact_evidence` and `qasper_evidence`;
- GRPO training through the existing `verl.trainer.main_ppo` entrypoint;
- information-gain reward disabled by the top-level VeriSeek training scripts.

VeriSeek does not rewrite the trainer, rollout logic, or search/visit tool protocol.

## Recommended Entrypoint

Use the repository-level wrapper instead of invoking files in this directory directly:

```bash
bash scripts/train_veriseek_grpo.sh
```

For the main gated n=4 SFT+RL run:

```bash
SFT_MODEL_PATH=$PWD/outputs/veriseek_sft_hf \
TRAIN_FILE=$PWD/data/processed/scifact/train.parquet \
VAL_FILE=$PWD/data/processed/scifact/dev.parquet \
OUTPUT=$PWD/outputs/veriseek_sft_rl_gated_n4 \
MAX_STEPS=200 \
SAVE_FREQ=50 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
GPU_NUM=2 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
AGENT_GRPO_N=4 \
bash scripts/train_veriseek_sft_rl.sh
```

See the top-level `README.md` and `docs/smoke_training.md` for full reproducible commands.

## Notes

The original long-horizon information-gain components are left in place for compatibility with the training scaffold, but VeriSeek's public experiments use deterministic evidence reward only:

```bash
USE_INFO_GAIN=false
IGPO_ROLLOUT_IG=0
TRAIN_REWARD_TYPE=f1
```

The reward remains deterministic and does not use external embedding models or LLM-as-a-judge scoring.
