#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/data/gh/DermAgent"
PY="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python"
ENV_BIN="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin"
FINAL_ROOT="${PROJECT_ROOT}/paper_data/final_3x3_delta_pilot_20260507"
LOG_DIR="${FINAL_ROOT}/run_logs"
SERVICE_LOG_DIR="${FINAL_ROOT}/service_logs"
COMPARE_DIR="${FINAL_ROOT}/compare_reports/medgemma/scin_rerun_j02b"
STATE_ROOT="${FINAL_ROOT}/state/medgemma/scin"
CASE_EXPORT_DIR="${PROJECT_ROOT}/paper_data/case_level_exports/final_3x3_delta_pilot_20260507__medgemma__scin__eval300_rerun_j02b"
RUN_NAME="final_3x3_delta_pilot_20260507__medgemma__scin__eval300_rerun_j02b"
JOB_LOG="${LOG_DIR}/j02b_medgemma_scin_rerun.log"
SERVICE_LOG="${SERVICE_LOG_DIR}/j02b_medgemma_scin_gpu1_port8101.log"
PID_FILE="${FINAL_ROOT}/state/j02b_medgemma_scin_port8101.pid"

mkdir -p "${LOG_DIR}" "${SERVICE_LOG_DIR}" "${COMPARE_DIR}" "${CASE_EXPORT_DIR}"

export PATH="${ENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="EMPTY"
export OPENAI_TIMEOUT=120
export DERMAGENT_REPO_ROOT="${PROJECT_ROOT}"

{
  echo "[j02b] started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[j02b] compare_dir=${COMPARE_DIR}"
  echo "[j02b] case_export_dir=${CASE_EXPORT_DIR}"

  CUDA_VISIBLE_DEVICES=1 \
  PORT=8101 \
  LOG_FILE="${SERVICE_LOG}" \
  PID_FILE="${PID_FILE}" \
  SERVED_MODEL_NAME="medgemma-4b-it" \
  OPENAI_API_KEY="EMPTY" \
  FORCE_RESTART=1 \
  WAIT_SECONDS=900 \
  bash "${PROJECT_ROOT}/scripts/start_medgemma_server.sh" "${PROJECT_ROOT}"

  echo "[j02b] service ready; frozen compare starting"

  DERMAGENT_SCIN_LABEL_SPACE_ID=scin_grouped \
  DERMAGENT_POLICY_ROOT="${STATE_ROOT}/policy" \
  DERMAGENT_SPLIT_STATE_ROOT="${STATE_ROOT}/split_states" \
  OPENAI_BASE_URL="http://127.0.0.1:8101/v1" \
  OPENAI_API_KEY="EMPTY" \
  OPENAI_MODEL="medgemma-4b-it" \
  OPENAI_TIMEOUT=120 \
  "${PY}" "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
    --data-root "${PROJECT_ROOT}/data/scin" \
    --limit 300 \
    --case-offset 0 \
    --data-split test \
    --split-json "${FINAL_ROOT}/splits/scin_final_3x3_delta_pilot_split.json" \
    --output-dir "${COMPARE_DIR}" \
    --policy-label "${RUN_NAME} frozen eval rerun" \
    --export-paper-case-data \
    --paper-case-data-dir "${CASE_EXPORT_DIR}" \
    --client-timeout 120 \
    --client-max-retries 1

  echo "[j02b] compare completed"
} >> "${JOB_LOG}" 2>&1

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
fi

echo "[j02b] finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOB_LOG}"
