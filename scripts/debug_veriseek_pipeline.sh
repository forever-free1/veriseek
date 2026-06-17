#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/processed/scifact_debug}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export DATA_DIR

echo "[0/5] Checking Python parquet dependencies"
"${PYTHON_BIN}" - <<'PY'
try:
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "debug_veriseek_pipeline.sh requires pandas and pyarrow to write/read parquet files. "
        "Install them in this Python environment before running the preflight."
    ) from exc
PY

echo "[1/5] Preparing tiny SciFact split in ${DATA_DIR}"
cd "${ROOT_DIR}"
"${PYTHON_BIN}" data/prepare_scifact.py \
  --output_dir "${DATA_DIR}" \
  --max_train 20 \
  --max_dev 5 \
  --max_test 5

echo "[2/5] Inspecting generated parquet files"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd

data_dir = Path(os.environ.get("DATA_DIR", "data/processed/scifact_debug"))
required = {"prompt", "reward_model", "data_source"}
for split in ["train", "dev", "test"]:
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise SystemExit(f"missing parquet file: {path}")
    df = pd.read_parquet(path)
    print(f"\n{path}")
    print("columns:", list(df.columns))
    if df.empty:
        raise SystemExit(f"{path} is empty")
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
    row = df.iloc[0].to_dict()
    print("example row:", row)
    if not row["prompt"]:
        raise SystemExit(f"{path} has empty prompt")
    if not isinstance(row["reward_model"], dict) or "ground_truth" not in row["reward_model"]:
        raise SystemExit(f"{path} reward_model must contain ground_truth")
    if not isinstance(row["reward_model"]["ground_truth"], str):
        raise SystemExit(f"{path} ground_truth must be a JSON string")
    parsed = json.loads(row["reward_model"]["ground_truth"])
    if "answer" not in parsed or "evidence" not in parsed:
        raise SystemExit(f"{path} ground_truth JSON must contain answer and evidence")
    if row["data_source"] != "scifact_evidence":
        raise SystemExit(f"{path} data_source expected scifact_evidence, got {row['data_source']}")
PY

echo "[3/5] Running synthetic reward checks"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import json
from pathlib import Path

path = Path("RL/verl/utils/reward_score/evidence_reward.py")
spec = importlib.util.spec_from_file_location("evidence_reward", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

gold = json.dumps({
    "answer": "SUPPORTS",
    "evidence": ["Drug A reduced inflammation."]
})
correct = "<answer>SUPPORTS</answer><evidence>[1] Drug A reduced inflammation.</evidence>"
incorrect = "<answer>REFUTES</answer><evidence>[1] Unrelated sentence.</evidence>"

for name, prediction in [("correct", correct), ("incorrect", incorrect)]:
    components = mod.compute_components(prediction, gold, "scifact_evidence")
    score = mod.compute_score(prediction, gold, "scifact_evidence")
    print(name, "components=", components, "score=", score)

if mod.compute_score(correct, gold, "scifact_evidence") <= mod.compute_score(incorrect, gold, "scifact_evidence"):
    raise SystemExit("correct prediction should score higher than incorrect prediction")
PY

echo "[4/5] Running lightweight unit tests"
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" tests/test_evidence_reward.py -v
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" tests/test_data_and_eval.py -v

echo "[5/5] Checking shell syntax"
bash -n scripts/train_veriseek_grpo.sh

echo "VeriSeek preflight completed successfully."
