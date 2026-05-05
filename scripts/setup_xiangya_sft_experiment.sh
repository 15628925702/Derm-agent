#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_ID="${EXPERIMENT_ID:-xiangya_sft_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sft数据}"
BASE_POLICY_CONFIG="${BASE_POLICY_CONFIG:-${PROJECT_ROOT}/state/policy/versions/heuristic_with_penalty.json}"
ASSET_ROOT="${ASSET_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/${EXPERIMENT_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dataset_adaptation/${EXPERIMENT_ID}}"
SPLIT_JSON="${SPLIT_JSON:-${OUTPUT_ROOT}/xiangya_sft_split.json}"
SPLIT_ID="${SPLIT_ID:-xiangya_sft_grouped_v1}"

mkdir -p "${OUTPUT_ROOT}"

echo "[info] project root     : ${PROJECT_ROOT}"
echo "[info] experiment id    : ${EXPERIMENT_ID}"
echo "[info] data root        : ${DATA_ROOT}"
echo "[info] asset root       : ${ASSET_ROOT}"
echo "[info] output root      : ${OUTPUT_ROOT}"
echo "[info] split json       : ${SPLIT_JSON}"
echo "[info] split id         : ${SPLIT_ID}"
echo "[info] base policy      : ${BASE_POLICY_CONFIG}"

cd "${PROJECT_ROOT}"

python scripts/manage_dataset_experiment_assets.py init \
  --experiment-id "${EXPERIMENT_ID}" \
  --base-policy-config "${BASE_POLICY_CONFIG}"

python - <<PY
from pathlib import Path
from configs.dataset_splits import write_fixed_split_json

write_fixed_split_json(
    Path("${SPLIT_JSON}"),
    split_id="${SPLIT_ID}",
    data_root=Path("${DATA_ROOT}").parent,
)
print("${SPLIT_JSON}")
PY

echo "[ok] Xiangya SFT experiment initialized."
echo "[ok] env file  : ${ASSET_ROOT}/experiment.env"
echo "[ok] split json: ${SPLIT_JSON}"
