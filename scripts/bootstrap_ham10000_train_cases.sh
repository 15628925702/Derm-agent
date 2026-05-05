#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_INDEX="${START_INDEX:-0}"
COUNT="${COUNT:-24}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/ham10000}"
POLICY_ROOT="${POLICY_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/ham10000_v2/policy}"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/ham10000_v2/split_states}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/dataset_adaptation/ham10000_v2/bootstrap}"
RUN_MODE="${RUN_MODE:-ham10000_train_bootstrap}"
DATA_SPLIT="${DATA_SPLIT:-train}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/ham10000_v2/ham10000_split.json}"
SPLIT_ID="${SPLIT_ID:-ham10000_balanced_v1}"
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
echo "[info] data split        : ${DATA_SPLIT}" | tee -a "${LOG_PATH}"
echo "[info] split json        : ${SPLIT_JSON}" | tee -a "${LOG_PATH}"
echo "[info] split id          : ${SPLIT_ID}" | tee -a "${LOG_PATH}"
echo "[info] client base url   : ${CLIENT_BASE_URL}" | tee -a "${LOG_PATH}"
echo "[info] client model      : ${CLIENT_MODEL}" | tee -a "${LOG_PATH}"
echo "[info] split offset      : ${START_INDEX}" | tee -a "${LOG_PATH}"
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

mapfile -t CASE_INDICES < <(
  DATA_ROOT="${DATA_ROOT}" \
  SPLIT_JSON="${SPLIT_JSON}" \
  SPLIT_ID="${SPLIT_ID}" \
  DATA_SPLIT="${DATA_SPLIT}" \
  START_INDEX="${START_INDEX}" \
  COUNT="${COUNT}" \
  python - <<'PY'
import json
import os
from pathlib import Path

from configs.dataset_splits import build_fixed_split_payload

data_root = Path(os.environ["DATA_ROOT"])
split_json = Path(os.environ["SPLIT_JSON"])
split_id = os.environ["SPLIT_ID"]
data_split = os.environ["DATA_SPLIT"]
start_index = max(0, int(os.environ["START_INDEX"]))
count = max(0, int(os.environ["COUNT"]))

if split_json.exists():
    payload = json.loads(split_json.read_text(encoding="utf-8"))
else:
    payload = build_fixed_split_payload(split_id=split_id, data_root=data_root)

indices = payload.get(f"{data_split}_case_indices")
if not indices:
    raw_range = payload.get(f"{data_split}_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ValueError(f"Split payload missing `{data_split}_range` and `{data_split}_case_indices`.")
    start, end = int(raw_range[0]), int(raw_range[1])
    indices = list(range(start, end + 1))

selected = indices[start_index : start_index + count]
for item in selected:
    print(int(item))
PY
)

if [[ "${#CASE_INDICES[@]}" -eq 0 ]]; then
  echo "[error] no case indices resolved for split=${DATA_SPLIT} offset=${START_INDEX} count=${COUNT}" | tee -a "${LOG_PATH}"
  exit 1
fi

resolved_count="${#CASE_INDICES[@]}"
echo "[info] resolved cases    : ${resolved_count}" | tee -a "${LOG_PATH}"

for idx in "${!CASE_INDICES[@]}"; do
  i="${CASE_INDICES[$idx]}"
  current=$((idx + 1))
  echo "[progress] case ${current}/${resolved_count} (metadata_index=${i})" | tee -a "${LOG_PATH}"
  cmd=(
    python scripts/debug_single_case.py
    --case-index "${i}"
    --data-root "${DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --enable-writeback
    --data-split "${DATA_SPLIT}"
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
