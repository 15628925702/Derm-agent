#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_INDEX="${START_INDEX:-0}"
COUNT="${COUNT:-24}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/ham10000}"
POLICY_ROOT="${POLICY_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/ham10000_v1/policy}"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/ham10000_v1/split_states}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/dataset_adaptation/ham10000_v1/bootstrap}"
RUN_MODE="${RUN_MODE:-ham10000_train_bootstrap}"
CLIENT_BASE_URL="${CLIENT_BASE_URL:-http://127.0.0.1:8000/v1}"
CLIENT_API_KEY="${CLIENT_API_KEY:-EMPTY}"
CLIENT_MODEL="${CLIENT_MODEL:-Qwen2.5-VL-7B-Instruct}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

mkdir -p "${OUTPUT_DIR}" "${POLICY_ROOT}" "${SPLIT_STATE_ROOT}"
LOG_PATH="${OUTPUT_DIR}/${RUN_MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[info] project root      : ${PROJECT_ROOT}" | tee -a "${LOG_PATH}"
echo "[info] data root         : ${DATA_ROOT}" | tee -a "${LOG_PATH}"
echo "[info] output dir        : ${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
echo "[info] policy root       : ${POLICY_ROOT}" | tee -a "${LOG_PATH}"
echo "[info] split state root  : ${SPLIT_STATE_ROOT}" | tee -a "${LOG_PATH}"
echo "[info] client base url   : ${CLIENT_BASE_URL}" | tee -a "${LOG_PATH}"
echo "[info] client model      : ${CLIENT_MODEL}" | tee -a "${LOG_PATH}"
echo "[info] start index       : ${START_INDEX}" | tee -a "${LOG_PATH}"
echo "[info] count             : ${COUNT}" | tee -a "${LOG_PATH}"
echo "[info] stop on error     : ${STOP_ON_ERROR}" | tee -a "${LOG_PATH}"

cd "${PROJECT_ROOT}"

DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" \
DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
python - <<'PY'
from agent.experiment_state import ensure_split_state_paths
for name in ("train", "val", "test"):
    ensure_split_state_paths(data_split=name)
PY

end_index=$((START_INDEX + COUNT - 1))
for i in $(seq "${START_INDEX}" "${end_index}"); do
  current=$((i - START_INDEX + 1))
  echo "[progress] case ${current}/${COUNT} (global_index=${i})" | tee -a "${LOG_PATH}"
  cmd=(
    python scripts/debug_single_case.py
    --case-index "${i}"
    --data-root "${DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --enable-writeback
    --data-split train
    --run-mode "${RUN_MODE}"
    --client-base-url "${CLIENT_BASE_URL}"
    --client-api-key "${CLIENT_API_KEY}"
    --client-model "${CLIENT_MODEL}"
  )
  if [[ -n "${CLIENT_TIMEOUT}" ]]; then
    cmd+=(--client-timeout "${CLIENT_TIMEOUT}")
  fi
  if [[ -n "${CLIENT_MAX_RETRIES}" ]]; then
    cmd+=(--client-max-retries "${CLIENT_MAX_RETRIES}")
  fi

  if ! DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" "${cmd[@]}" >> "${LOG_PATH}" 2>&1; then
    echo "[error] failed at case-index=${i}. see ${LOG_PATH}" | tee -a "${LOG_PATH}"
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
  fi
done

echo "[ok] HAM10000 bootstrap completed. log: ${LOG_PATH}" | tee -a "${LOG_PATH}"
