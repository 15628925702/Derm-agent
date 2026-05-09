#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/data/gh/DermAgent"
PY="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python"
ENV_BIN="/home/zhongnan/miniconda3/envs/dermagent-6x6/bin"
FINAL_ROOT="${PROJECT_ROOT}/paper_data/final_6x5_continuous_20260507"
ACCEL_ROOT="${FINAL_ROOT}/llama_tail_accel_20260508"
LOG_DIR="${ACCEL_ROOT}/run_logs"
SERVICE_LOG_DIR="${ACCEL_ROOT}/service_logs"
COMPARE_ROOT="${ACCEL_ROOT}/compare_reports"
CASE_EXPORT_ROOT="${PROJECT_ROOT}/paper_data/case_level_exports"
START_LLAMA="${PROJECT_ROOT}/scripts/start_llama_server.sh"
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

start_llama_replica() {
  local gpu="$1" port="$2" tag="$3"
  local log_file="${SERVICE_LOG_DIR}/${tag}_gpu${gpu}_port${port}.log"
  local pid_file="${ACCEL_ROOT}/${tag}_gpu${gpu}_port${port}.pid"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PORT="${port}" \
  LOG_FILE="${log_file}" \
  PID_FILE="${pid_file}" \
  SERVED_MODEL_NAME="Llama-3.2-11B-Vision-Instruct" \
  OPENAI_API_KEY="EMPTY" \
  FORCE_RESTART=1 \
  WAIT_SECONDS=900 \
  bash "${START_LLAMA}" "${PROJECT_ROOT}"
}

run_shard() {
  local tag="$1" gpu="$2" port="$3" data_root="$4" split_json="$5" policy_root="$6" split_state_root="$7" offset="$8" limit="$9" output_dir="${10}" export_dir="${11}"
  DERMAGENT_POLICY_ROOT="${policy_root}" \
  DERMAGENT_SPLIT_STATE_ROOT="${split_state_root}" \
  OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
  OPENAI_API_KEY="EMPTY" \
  OPENAI_MODEL="Llama-3.2-11B-Vision-Instruct" \
  OPENAI_TIMEOUT=120 \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY}" "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
    --data-root "${data_root}" \
    --limit "${limit}" \
    --case-offset "${offset}" \
    --data-split test \
    --split-json "${split_json}" \
    --output-dir "${output_dir}" \
    --policy-label "${tag} frozen eval shard" \
    --export-paper-case-data \
    --paper-case-data-dir "${export_dir}" \
    --client-timeout 120 \
    --client-max-retries 1 \
    > "${LOG_DIR}/${tag}.log" 2>&1
}

log "collecting current compare pids"
HAM_COMPARE_PID="$(ps -eo pid,cmd | grep 'compare_agent_vs_qwen.py' | grep 'compare_reports/llama/ham10000' | awk '{print $1}' | head -1 || true)"
SD_COMPARE_PID="$(ps -eo pid,cmd | grep 'compare_agent_vs_qwen.py' | grep 'compare_reports/llama/sd198' | awk '{print $1}' | head -1 || true)"
HAM_WRAPPER_PID="$(ps -o ppid= -p "${HAM_COMPARE_PID:-0}" 2>/dev/null | awk '{print $1}')"
SD_WRAPPER_PID="$(ps -o ppid= -p "${SD_COMPARE_PID:-0}" 2>/dev/null | awk '{print $1}')"
log "stopping compare pid ham10000=${HAM_COMPARE_PID:-none} wrapper=${HAM_WRAPPER_PID:-none} sd198=${SD_COMPARE_PID:-none} wrapper=${SD_WRAPPER_PID:-none}"
stop_pid "${HAM_COMPARE_PID:-}"
stop_pid "${SD_COMPARE_PID:-}"
stop_pid "${HAM_WRAPPER_PID:-}"
stop_pid "${SD_WRAPPER_PID:-}"

HAM_PARTIAL_ROOT="${FINAL_ROOT}/compare_reports/llama/ham10000/compare_agent_vs_qwen_20260507T212226Z"
SD_PARTIAL_ROOT="${FINAL_ROOT}/compare_reports/llama/sd198/compare_agent_vs_qwen_20260507T215522Z"
HAM_SPLIT_JSON="${FINAL_ROOT}/splits/ham10000_final_6x5_continuous_split.json"
SD_SPLIT_JSON="${FINAL_ROOT}/splits/sd198_final_6x5_continuous_split.json"
HAM_POLICY_ROOT="${FINAL_ROOT}/state/llama/ham10000/policy"
HAM_SPLIT_STATE_ROOT="${FINAL_ROOT}/state/llama/ham10000/split_states"
SD_POLICY_ROOT="${FINAL_ROOT}/state/llama/sd198/policy"
SD_SPLIT_STATE_ROOT="${FINAL_ROOT}/state/llama/sd198/split_states"

log "starting 6 extra llama replicas on free gpus"
start_llama_replica 0 8130 ham_a0
start_llama_replica 1 8131 ham_a1
start_llama_replica 2 8132 ham_a2
start_llama_replica 3 8133 sd_a0
start_llama_replica 5 8135 sd_a1
start_llama_replica 7 8137 ham_a3

log "launching 8 shards"
run_shard "ham_shard_186_23" 0 8130 "${PROJECT_ROOT}/data/ham10000" "${HAM_SPLIT_JSON}" "${HAM_POLICY_ROOT}" "${HAM_SPLIT_STATE_ROOT}" 186 23 "${COMPARE_ROOT}/llama/ham10000_shard_186_23" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__ham10000__eval300_shard_186_23" &
P1=$!
run_shard "ham_shard_209_23" 1 8131 "${PROJECT_ROOT}/data/ham10000" "${HAM_SPLIT_JSON}" "${HAM_POLICY_ROOT}" "${HAM_SPLIT_STATE_ROOT}" 209 23 "${COMPARE_ROOT}/llama/ham10000_shard_209_23" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__ham10000__eval300_shard_209_23" &
P2=$!
run_shard "ham_shard_232_23" 2 8132 "${PROJECT_ROOT}/data/ham10000" "${HAM_SPLIT_JSON}" "${HAM_POLICY_ROOT}" "${HAM_SPLIT_STATE_ROOT}" 232 23 "${COMPARE_ROOT}/llama/ham10000_shard_232_23" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__ham10000__eval300_shard_232_23" &
P3=$!
run_shard "sd_shard_235_22" 3 8133 "${PROJECT_ROOT}/data/sd198" "${SD_SPLIT_JSON}" "${SD_POLICY_ROOT}" "${SD_SPLIT_STATE_ROOT}" 235 22 "${COMPARE_ROOT}/llama/sd198_shard_235_22" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__sd198__eval300_shard_235_22" &
P4=$!
run_shard "ham_shard_255_23" 4 8124 "${PROJECT_ROOT}/data/ham10000" "${HAM_SPLIT_JSON}" "${HAM_POLICY_ROOT}" "${HAM_SPLIT_STATE_ROOT}" 255 23 "${COMPARE_ROOT}/llama/ham10000_shard_255_23" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__ham10000__eval300_shard_255_23" &
P5=$!
run_shard "sd_shard_257_22" 5 8135 "${PROJECT_ROOT}/data/sd198" "${SD_SPLIT_JSON}" "${SD_POLICY_ROOT}" "${SD_SPLIT_STATE_ROOT}" 257 22 "${COMPARE_ROOT}/llama/sd198_shard_257_22" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__sd198__eval300_shard_257_22" &
P6=$!
run_shard "ham_shard_278_22" 6 8126 "${PROJECT_ROOT}/data/ham10000" "${HAM_SPLIT_JSON}" "${HAM_POLICY_ROOT}" "${HAM_SPLIT_STATE_ROOT}" 278 22 "${COMPARE_ROOT}/llama/ham10000_shard_278_22" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__ham10000__eval300_shard_278_22" &
P7=$!
run_shard "sd_shard_279_21" 7 8137 "${PROJECT_ROOT}/data/sd198" "${SD_SPLIT_JSON}" "${SD_POLICY_ROOT}" "${SD_SPLIT_STATE_ROOT}" 279 21 "${COMPARE_ROOT}/llama/sd198_shard_279_21" "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__sd198__eval300_shard_279_21" &
P8=$!

wait "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}"
log "all shards completed, merging reports"

HAM_SHARDS=(
  "${COMPARE_ROOT}/llama/ham10000_shard_186_23"/compare_agent_vs_qwen_*.json
  "${COMPARE_ROOT}/llama/ham10000_shard_209_23"/compare_agent_vs_qwen_*.json
  "${COMPARE_ROOT}/llama/ham10000_shard_232_23"/compare_agent_vs_qwen_*.json
  "${COMPARE_ROOT}/llama/ham10000_shard_255_23"/compare_agent_vs_qwen_*.json
  "${COMPARE_ROOT}/llama/ham10000_shard_278_22"/compare_agent_vs_qwen_*.json
)
SD_SHARDS=(
  "${COMPARE_ROOT}/llama/sd198_shard_235_22"/compare_agent_vs_qwen_*.json
  "${COMPARE_ROOT}/llama/sd198_shard_257_22"/compare_agent_vs_qwen_*.json
  "${COMPARE_ROOT}/llama/sd198_shard_279_21"/compare_agent_vs_qwen_*.json
)

"${PY}" "${MERGE_SCRIPT}" \
  --partial-run-root "${HAM_PARTIAL_ROOT}" \
  --shard-report "${HAM_SHARDS[0]}" \
  --shard-report "${HAM_SHARDS[1]}" \
  --shard-report "${HAM_SHARDS[2]}" \
  --shard-report "${HAM_SHARDS[3]}" \
  --shard-report "${HAM_SHARDS[4]}" \
  --output-report "${HAM_PARTIAL_ROOT}/compare_agent_vs_qwen_20260507T212226Z.json" \
  --paper-case-data-dir "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__ham10000__eval300" \
  --export-stem "case_level_compare_export_20260507T212226Z"

"${PY}" "${MERGE_SCRIPT}" \
  --partial-run-root "${SD_PARTIAL_ROOT}" \
  --shard-report "${SD_SHARDS[0]}" \
  --shard-report "${SD_SHARDS[1]}" \
  --shard-report "${SD_SHARDS[2]}" \
  --output-report "${SD_PARTIAL_ROOT}/compare_agent_vs_qwen_20260507T215522Z.json" \
  --paper-case-data-dir "${CASE_EXPORT_ROOT}/final_6x5_continuous_20260507__llama__sd198__eval300" \
  --export-stem "case_level_compare_export_20260507T215522Z"

log "merged compare reports ready"
