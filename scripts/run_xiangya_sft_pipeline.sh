#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_ID="${EXPERIMENT_ID:-xiangya_sft_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sft数据}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-12}"
COMPARE_LIMIT="${COMPARE_LIMIT:-12}"
COMPARE_SPLIT="${COMPARE_SPLIT:-test}"
PROMOTE_AFTER_BOOTSTRAP="${PROMOTE_AFTER_BOOTSTRAP:-1}"

echo "[info] project root            : ${PROJECT_ROOT}"
echo "[info] experiment id           : ${EXPERIMENT_ID}"
echo "[info] data root               : ${DATA_ROOT}"
echo "[info] bootstrap count         : ${BOOTSTRAP_COUNT}"
echo "[info] compare limit           : ${COMPARE_LIMIT}"
echo "[info] compare split           : ${COMPARE_SPLIT}"
echo "[info] promote after bootstrap : ${PROMOTE_AFTER_BOOTSTRAP}"

cd "${PROJECT_ROOT}"

echo "[step 1/4] setup experiment assets"
EXPERIMENT_ID="${EXPERIMENT_ID}" DATA_ROOT="${DATA_ROOT}" \
  bash "${PROJECT_ROOT}/scripts/setup_xiangya_sft_experiment.sh"

echo "[step 2/4] bootstrap train state"
EXPERIMENT_ID="${EXPERIMENT_ID}" DATA_ROOT="${DATA_ROOT}" COUNT="${BOOTSTRAP_COUNT}" \
  bash "${PROJECT_ROOT}/scripts/bootstrap_xiangya_sft_train_cases.sh"

if [[ "${PROMOTE_AFTER_BOOTSTRAP}" == "1" ]]; then
  echo "[step 3/4] promote train state to val/test"
  EXPERIMENT_ID="${EXPERIMENT_ID}" \
    bash "${PROJECT_ROOT}/scripts/promote_xiangya_sft_state.sh"
else
  echo "[step 3/4] promote skipped"
fi

echo "[step 4/4] run frozen compare"
EXPERIMENT_ID="${EXPERIMENT_ID}" DATA_ROOT="${DATA_ROOT}" LIMIT="${COMPARE_LIMIT}" DATA_SPLIT="${COMPARE_SPLIT}" \
  bash "${PROJECT_ROOT}/scripts/run_compare_xiangya_sft.sh"

echo "[ok] Xiangya SFT pipeline completed."
