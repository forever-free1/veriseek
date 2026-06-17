#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/hy-tmp/veriseek}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/veriseek_sft_rl_gated_v2}"
BENCH_DIR="${BENCH_DIR:-${ROOT_DIR}/outputs/benchmarks/gated_v2}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/processed/scifact/dev.parquet}"
STEPS="${STEPS:-50 100 150 200 250 300 350 400}"
TMP_PREFIX="${TMP_PREFIX:-${ROOT_DIR}/outputs/tmp_gated_v2_step}"

mkdir -p "${BENCH_DIR}"

for step in ${STEPS}; do
    ckpt="${RUN_DIR}/global_step_${step}/actor"
    tmp="${TMP_PREFIX}_${step}_hf"
    pred="${BENCH_DIR}/scifact_dev_gated_v2_step_${step}.jsonl"
    metrics="${BENCH_DIR}/scifact_dev_gated_v2_step_${step}_metrics.json"
    components="${BENCH_DIR}/scifact_dev_gated_v2_step_${step}_components.json"

    echo "===== step ${step} ====="
    if [[ ! -d "${ckpt}" ]]; then
        echo "Missing checkpoint actor dir: ${ckpt}" >&2
        exit 2
    fi

    rm -rf "${tmp}"
    df -h /hy-tmp | tail -1

    (
        cd "${ROOT_DIR}/RL"
        PYTHONPATH="${PWD}" python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${ckpt}" \
            --target_dir "${tmp}"
    )

    (
        cd "${ROOT_DIR}"
        CUDA_VISIBLE_DEVICES=0 python scripts/run_scifact_benchmark_generate.py \
            --model_path "${tmp}" \
            --data_path "${DATA_PATH}" \
            --out_jsonl "${pred}" \
            --batch_size 8 \
            --max_new_tokens 192
        python eval/eval_scifact.py --pred_path "${pred}" --mode both | tee "${metrics}"
        python eval/analyze_reward_components.py --pred_path "${pred}" --task scifact | tee "${components}"
    )

    rm -rf "${tmp}"
    df -h /hy-tmp | tail -1
    echo "===== done step ${step} ====="
done
