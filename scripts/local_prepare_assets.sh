#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="Qwen/Qwen3-4B-Thinking-2507"
LOCAL_DIR="${LOCAL_DIR:-${ROOT_DIR}/models/Qwen3-4B-Thinking-2507}"

mkdir -p "${LOCAL_DIR}"

python3 - <<PY
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

model_id = "${MODEL_ID}"
local_dir = Path("${LOCAL_DIR}")
snapshot_download(
    repo_id=model_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
)
print(f"Downloaded {model_id} to {local_dir}")
PY
