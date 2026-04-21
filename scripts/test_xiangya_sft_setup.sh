#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_ID="${EXPERIMENT_ID:-xiangya_sft_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sft数据}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/${EXPERIMENT_ID}/xiangya_sft_split.json}"

cd "${PROJECT_ROOT}"

echo "[test] checking experiment setup"

python scripts/check_qwen_server.py --timeout 5 >/dev/null 2>&1 || true

python - <<PY
import json
from pathlib import Path

from dataio.case_loader import load_case_by_index, resolve_registered_dataset_loader

data_root = Path("${DATA_ROOT}")
split_json = Path("${SPLIT_JSON}")

spec = resolve_registered_dataset_loader(data_root)
assert spec is not None, "xiangya_sft loader not registered"
assert spec.dataset_name == "xiangya_sft", spec.dataset_name

case = load_case_by_index(0, data_root)
assert case.dataset_name == "xiangya_sft", case.dataset_name
assert case.label_space_id == "xiangya_sft_grouped", case.label_space_id
assert Path(case.image_path).exists(), case.image_path

payload = json.loads(split_json.read_text(encoding="utf-8"))
assert payload["dataset_name"] == "xiangya_sft", payload["dataset_name"]
assert payload["train"], "empty train split"
assert payload["test"], "empty test split"

print("xiangya_sft setup ok")
PY

echo "[ok] Xiangya SFT setup test passed."
