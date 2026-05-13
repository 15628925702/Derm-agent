#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/data/gh/DermAgent"
PY="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python"
ENV_BIN="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin"
FINAL_ROOT="${PROJECT_ROOT}/paper_data/final_6x5_continuous_20260507"
ACCEL_ROOT="${FINAL_ROOT}/medgemma_ham10000_accel_20260508"
LOG_DIR="${ACCEL_ROOT}/run_logs"
SERVICE_LOG_DIR="${ACCEL_ROOT}/service_logs"
COMPARE_ROOT="${ACCEL_ROOT}/compare_reports"
CASE_EXPORT_ROOT="${PROJECT_ROOT}/paper_data/case_level_exports"
START_MEDGEMMA="${PROJECT_ROOT}/scripts/start_medgemma_server.sh"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/merge_compare_partial_and_shards.py"

export PATH="${ENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="EMPTY"

mkdir -p "${LOG_DIR}" "${SERVICE_LOG_DIR}" "${COMPARE_ROOT}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

stop_pid() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 3
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi
}

start_medgemma_replica() {
  local gpu="$1" port="$2" tag="$3"
  local log_file="${SERVICE_LOG_DIR}/${tag}_gpu${gpu}_port${port}.log"
  local pid_file="${ACCEL_ROOT}/${tag}_gpu${gpu}_port${port}.pid"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PORT="${port}" \
  LOG_FILE="${log_file}" \
  PID_FILE="${pid_file}" \
  SERVED_MODEL_NAME="medgemma-4b-it" \
  OPENAI_API_KEY="EMPTY" \
  FORCE_RESTART=1 \
  WAIT_SECONDS=900 \
  bash "${START_MEDGEMMA}" "${PROJECT_ROOT}"
}

run_shard() {
  local tag="$1" gpu="$2" port="$3" offset="$4" limit="$5"
  DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" \
  DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
  OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
  OPENAI_API_KEY="EMPTY" \
  OPENAI_MODEL="medgemma-4b-it" \
  OPENAI_TIMEOUT=120 \
  OPENAI_MAX_RETRIES=1 \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY}" "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
    --data-root "${DATA_ROOT}" \
    --limit "${limit}" \
    --case-offset "${offset}" \
    --data-split test \
    --split-json "${SPLIT_JSON}" \
    --output-dir "${COMPARE_ROOT}/medgemma/${tag}" \
    --policy-label "${tag} frozen eval shard" \
    --export-paper-case-data \
    --paper-case-data-dir "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__medgemma__ham10000__eval300_${tag}" \
    --client-timeout 120 \
    --client-max-retries 1 \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

DATA_ROOT="${PROJECT_ROOT}/data/ham10000"
SPLIT_JSON="${FINAL_ROOT}/splits/ham10000_final_6x5_continuous_split.json"
POLICY_ROOT="${FINAL_ROOT}/state/medgemma/ham10000/policy"
SPLIT_STATE_ROOT="${FINAL_ROOT}/state/medgemma/ham10000/split_states"
PARTIAL_ROOT="${FINAL_ROOT}/compare_reports/medgemma/ham10000/compare_agent_vs_qwen_20260507T184405Z"
FINAL_DATASET_DIR="${FINAL_ROOT}/compare_reports/medgemma/ham10000"
FINAL_REPORT_TOP="${FINAL_DATASET_DIR}/compare_agent_vs_qwen_20260507T184405Z.json"
FINAL_REPORT_INNER="${PARTIAL_ROOT}/compare_agent_vs_qwen_20260507T184405Z.json"
FINAL_EXPORT_DIR="${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__medgemma__ham10000__eval300"

log "starting 8 medgemma replicas"
start_medgemma_replica 0 8140 mgm_a0
start_medgemma_replica 1 8141 mgm_a1
start_medgemma_replica 2 8142 mgm_a2
start_medgemma_replica 3 8143 mgm_a3
start_medgemma_replica 4 8144 mgm_a4
start_medgemma_replica 5 8145 mgm_a5
start_medgemma_replica 6 8146 mgm_a6
start_medgemma_replica 7 8147 mgm_a7

log "launching 8 ham10000 shards"
run_shard "ham10000_shard_000_38" 0 8140 0 38 &
P1=$!
run_shard "ham10000_shard_038_38" 1 8141 38 38 &
P2=$!
run_shard "ham10000_shard_076_38" 2 8142 76 38 &
P3=$!
run_shard "ham10000_shard_114_38" 3 8143 114 38 &
P4=$!
run_shard "ham10000_shard_152_37" 4 8144 152 37 &
P5=$!
run_shard "ham10000_shard_189_37" 5 8145 189 37 &
P6=$!
run_shard "ham10000_shard_226_37" 6 8146 226 37 &
P7=$!
run_shard "ham10000_shard_263_37" 7 8147 263 37 &
P8=$!

wait "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}"
log "all shards completed, merging report"

"${PY}" "${MERGE_SCRIPT}" \
  --partial-run-root "${PARTIAL_ROOT}" \
  --ignore-partial-records \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_000_38"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_038_38"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_076_38"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_114_38"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_152_37"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_189_37"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_226_37"/compare_agent_vs_qwen_*.json \
  --shard-report "${COMPARE_ROOT}/medgemma/ham10000_shard_263_37"/compare_agent_vs_qwen_*.json \
  --output-report "${FINAL_REPORT_INNER}" \
  --paper-case-data-dir "${FINAL_EXPORT_DIR}" \
  --export-stem "case_level_compare_export_20260507T184405Z"

cp "${FINAL_REPORT_INNER}" "${FINAL_REPORT_TOP}"
log "merged compare report ready: ${FINAL_REPORT_TOP}"

for pid_file in "${ACCEL_ROOT}"/*.pid; do
  [[ -f "${pid_file}" ]] || continue
  stop_pid "$(cat "${pid_file}" 2>/dev/null || true)"
done

log "medgemma ham10000 acceleration done"
