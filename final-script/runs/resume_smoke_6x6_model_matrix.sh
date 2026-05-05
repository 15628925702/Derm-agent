#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/DermAgent"
RUN_GROUP="${RUN_GROUP:-smoke_6x6_20260504T080311Z}"
RUN_ROOT="${PROJECT_ROOT}/outputs/model_runs/${RUN_GROUP}"
STATE_ROOT="${PROJECT_ROOT}/state/model_runs/${RUN_GROUP}"
SUMMARY_PATH="${RUN_ROOT}/summary.tsv"

mkdir -p "${RUN_ROOT}" "${STATE_ROOT}"

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  printf 'model\tdataset\tstatus\tlog\n' > "${SUMMARY_PATH}"
fi

MODELS=(medgemma skinvl llama hulumed dermatollama)
DATASETS=(pad20 isic2019 scin sd198 xiangya_sft ham10000)

start_model() {
  local model_key="$1"
  local server_log="${RUN_ROOT}/${model_key}_server.log"
  : > "${server_log}"

  echo "[runner] starting ${model_key} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${server_log}"

  case "${model_key}" in
    medgemma)
      FORCE_RESTART=1 WAIT_SECONDS=300 CHECK_INTERVAL_SECONDS=5 \
      STOP_WAIT_SECONDS=10 GPU_RELEASE_WAIT_SECONDS=20 \
      GPU_MEMORY_UTILIZATION=0.84 MAX_MODEL_LEN=12288 MAX_NUM_SEQS=2 \
        bash "${PROJECT_ROOT}/scripts/switch_model_server.sh" medgemma "${PROJECT_ROOT}" >> "${server_log}" 2>&1
      ;;
    skinvl)
      FORCE_RESTART=1 WAIT_SECONDS=300 CHECK_INTERVAL_SECONDS=5 \
      STOP_WAIT_SECONDS=10 GPU_RELEASE_WAIT_SECONDS=20 \
      MAX_NEW_TOKENS_DEFAULT=256 \
        bash "${PROJECT_ROOT}/scripts/switch_model_server.sh" skinvl "${PROJECT_ROOT}" >> "${server_log}" 2>&1
      ;;
    llama)
      FORCE_RESTART=1 WAIT_SECONDS=300 CHECK_INTERVAL_SECONDS=5 \
      STOP_WAIT_SECONDS=10 GPU_RELEASE_WAIT_SECONDS=20 \
      GPU_MEMORY_UTILIZATION=0.84 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 \
        bash "${PROJECT_ROOT}/scripts/switch_model_server.sh" llama "${PROJECT_ROOT}" >> "${server_log}" 2>&1
      ;;
    hulumed)
      FORCE_RESTART=1 WAIT_SECONDS=300 CHECK_INTERVAL_SECONDS=5 \
      STOP_WAIT_SECONDS=10 GPU_RELEASE_WAIT_SECONDS=20 \
      GPU_MEMORY_UTILIZATION=0.84 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 \
        bash "${PROJECT_ROOT}/scripts/switch_model_server.sh" hulumed "${PROJECT_ROOT}" >> "${server_log}" 2>&1
      ;;
    dermatollama)
      FORCE_RESTART=1 WAIT_SECONDS=300 CHECK_INTERVAL_SECONDS=5 \
      STOP_WAIT_SECONDS=10 GPU_RELEASE_WAIT_SECONDS=20 \
      GPU_MEMORY_UTILIZATION=0.84 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 \
        bash "${PROJECT_ROOT}/scripts/switch_model_server.sh" dermatollama "${PROJECT_ROOT}" >> "${server_log}" 2>&1
      ;;
    *)
      echo "[error] unsupported model: ${model_key}" | tee -a "${server_log}"
      return 2
      ;;
  esac

  echo "[runner] ${model_key} START_OK" | tee -a "${server_log}"
}

stop_models() {
  local log_path="$1"
  STOP_WAIT_SECONDS=10 GPU_RELEASE_WAIT_SECONDS=20 WAIT_SECONDS=30 \
    bash "${PROJECT_ROOT}/scripts/switch_model_server.sh" stop "${PROJECT_ROOT}" >> "${log_path}" 2>&1 || true
}

has_summary_row() {
  local model_key="$1"
  local dataset_id="$2"
  awk -F '\t' -v m="${model_key}" -v d="${dataset_id}" \
    'NR > 1 && $1 == m && $2 == d { found = 1 } END { exit found ? 0 : 1 }' \
    "${SUMMARY_PATH}"
}

append_summary() {
  local model_key="$1"
  local dataset_id="$2"
  local status="$3"
  local log_path="$4"

  if has_summary_row "${model_key}" "${dataset_id}"; then
    return 0
  fi

  printf '%s\t%s\t%s\t%s\n' "${model_key}" "${dataset_id}" "${status}" "${log_path}" >> "${SUMMARY_PATH}"
}

run_dataset() {
  local model_key="$1"
  local dataset_id="$2"
  local dataset_log="${RUN_ROOT}/${model_key}_${dataset_id}.log"

  if has_summary_row "${model_key}" "${dataset_id}"; then
    echo "[runner] skip existing ${model_key}/${dataset_id}"
    return 0
  fi

  rm -rf "${RUN_ROOT:?}/${model_key}/${dataset_id}" "${STATE_ROOT:?}/${model_key}/${dataset_id}"
  mkdir -p "${RUN_ROOT}/${model_key}" "${STATE_ROOT}/${model_key}"
  : > "${dataset_log}"

  if MODEL_KEY="${model_key}" \
    DATASET_ID="${dataset_id}" \
    RUN_GROUP="${RUN_GROUP}" \
    MODEL_STATE_ROOT="${STATE_ROOT}/${model_key}" \
    MODEL_OUTPUT_ROOT="${RUN_ROOT}/${model_key}" \
    BOOTSTRAP_COUNT=1 \
    COMPARE_LIMIT=1 \
    STOP_ON_ERROR=1 \
    SERVER_CHECK_TIMEOUT=240 \
    SERVER_WAIT_RETRIES=3 \
    SERVER_WAIT_INTERVAL_SECONDS=5 \
    SERVER_CHAT_CHECK=1 \
    SERVER_CHAT_CHECK_RETRIES=1 \
    SERVER_CHAT_CHECK_INTERVAL_SECONDS=5 \
      bash "${PROJECT_ROOT}/final-script/runs/run_model_dataset_final_round.sh" >> "${dataset_log}" 2>&1; then
    append_summary "${model_key}" "${dataset_id}" "OK" "${dataset_log}"
    echo "[runner] ${model_key}/${dataset_id} OK"
    return 0
  fi

  append_summary "${model_key}" "${dataset_id}" "FAILED" "${dataset_log}"
  echo "[runner] ${model_key}/${dataset_id} FAILED"
  tail -120 "${dataset_log}" || true
  return 1
}

model_complete() {
  local model_key="$1"
  local dataset_id
  for dataset_id in "${DATASETS[@]}"; do
    if ! has_summary_row "${model_key}" "${dataset_id}"; then
      return 1
    fi
  done
  return 0
}

cd "${PROJECT_ROOT}"

for model_key in "${MODELS[@]}"; do
  if model_complete "${model_key}"; then
    echo "[runner] model ${model_key} already complete"
    continue
  fi

  server_log="${RUN_ROOT}/${model_key}_server.log"
  if ! start_model "${model_key}"; then
    echo "[runner] ${model_key} START_FAILED"
    for dataset_id in "${DATASETS[@]}"; do
      append_summary "${model_key}" "${dataset_id}" "START_FAILED" "${server_log}"
    done
    stop_models "${server_log}"
    continue
  fi

  for dataset_id in "${DATASETS[@]}"; do
    run_dataset "${model_key}" "${dataset_id}" || true
  done

  stop_models "${server_log}"
  echo "[runner] stopped ${model_key}"
done

echo "[runner] final summary:"
cat "${SUMMARY_PATH}"
