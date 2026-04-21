#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_ID="${EXPERIMENT_ID:-xiangya_sft_v1}"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/${EXPERIMENT_ID}/split_states}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
TARGET_SPLITS="${TARGET_SPLITS:-val,test}"

echo "[info] project root      : ${PROJECT_ROOT}"
echo "[info] experiment id     : ${EXPERIMENT_ID}"
echo "[info] split state root  : ${SPLIT_STATE_ROOT}"
echo "[info] source split      : ${SOURCE_SPLIT}"
echo "[info] target splits     : ${TARGET_SPLITS}"

cd "${PROJECT_ROOT}"

python scripts/manage_dataset_experiment_assets.py promote-state \
  --split-state-root "${SPLIT_STATE_ROOT}" \
  --source-split "${SOURCE_SPLIT}" \
  --target-splits "${TARGET_SPLITS}"

echo "[ok] Xiangya SFT state promoted."
