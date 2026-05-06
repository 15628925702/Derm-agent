#!/usr/bin/env bash
#
# Dynamic 8-GPU 6x6 run for the 64-bootstrap / 100-compare medium experiment.
# Phase A uses GPU4 for a HuluMed replica. Phase B reclaims GPU2/GPU3 from
# Qwen/MedGemma and runs SkinVL in two dataset shards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dermagent-6x6}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-64}"
COMPARE_COUNT="${COMPARE_COUNT:-100}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
COMPARE_SEED="${COMPARE_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/6x6_64_100_8gpu_dynamic_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/service_logs" "${PROJECT_ROOT}/state"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [dynamic] $*" | tee -a "${OUTPUT_ROOT}/dynamic_runner.log"
}

activate_env() {
  # shellcheck disable=SC1091
  source /home/zhongnan/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
}

port_pids() {
  local port="$1"
  ss -ltnp 2>/dev/null \
    | awk -v port=":${port}" '$4 ~ port {print $NF}' \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u
}

stop_port() {
  local port="$1"
  local pids
  pids="$(port_pids "${port}" || true)"
  if [[ -z "${pids}" ]]; then
    log "No listener on port ${port}"
    return 0
  fi
  log "Stopping port ${port}: ${pids}"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 5
  pids="$(port_pids "${port}" || true)"
  if [[ -n "${pids}" ]]; then
    log "Force stopping port ${port}: ${pids}"
    kill -KILL ${pids} 2>/dev/null || true
  fi
}

kill_gpu_compute_processes() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -u || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  log "Stopping remaining compute process(es) on GPU${gpu}: ${pids}"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 5
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -u || true)"
  if [[ -n "${pids}" ]]; then
    log "Force stopping remaining compute process(es) on GPU${gpu}: ${pids}"
    kill -KILL ${pids} 2>/dev/null || true
  fi
}

wait_models_endpoint() {
  local port="$1"
  local label="$2"
  local attempts="${3:-60}"
  for _ in $(seq 1 "${attempts}"); do
    if curl -s -m 5 -H "Authorization: Bearer EMPTY" "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      log "${label} ready on port ${port}"
      return 0
    fi
    sleep 5
  done
  log "ERROR: ${label} did not become ready on port ${port}"
  return 1
}

ensure_existing_endpoint() {
  local port="$1"
  local label="$2"
  wait_models_endpoint "${port}" "${label}" 12
}

start_hulumed_replica() {
  if curl -s -m 5 -H "Authorization: Bearer EMPTY" "http://127.0.0.1:8023/v1/models" >/dev/null; then
    log "HuluMed replica already running on port 8023"
    return 0
  fi

  log "Reclaiming GPU4 from SkinVL main service for HuluMed replica"
  stop_port 8011
  kill_gpu_compute_processes 4

  log "Starting HuluMed replica on GPU4 / port 8023"
  CUDA_VISIBLE_DEVICES=4 \
  CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
  PORT=8023 \
  HOST=127.0.0.1 \
  OPENAI_API_KEY=EMPTY \
  FORCE_RESTART=1 \
  LOG_FILE="${OUTPUT_ROOT}/service_logs/hulumed_replica_8023.log" \
  PID_FILE="${PROJECT_ROOT}/state/hulumed_replica_8023.pid" \
  bash "${SCRIPT_DIR}/start_hulumed_server.sh" "${PROJECT_ROOT}" \
    | tee -a "${OUTPUT_ROOT}/dynamic_runner.log" \
    || {
      log "WARN: HuluMed replica failed to start; continuing without hulumed_replica lane"
      return 1
    }
}

start_skinvl_shard() {
  local gpu="$1"
  local port="$2"
  local name="$3"

  log "Starting ${name} on GPU${gpu} / port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
  PORT="${port}" \
  HOST=127.0.0.1 \
  OPENAI_API_KEY=EMPTY \
  FORCE_RESTART=1 \
  LOG_FILE="${OUTPUT_ROOT}/service_logs/${name}_${port}.log" \
  PID_FILE="${PROJECT_ROOT}/state/${name}_${port}.pid" \
  bash "${SCRIPT_DIR}/start_skinvl_server.sh" "${PROJECT_ROOT}" \
    | tee -a "${OUTPUT_ROOT}/dynamic_runner.log" \
    || {
      log "WARN: ${name} failed to start on port ${port}"
      return 1
    }
}

phase_a() {
  log "Phase A: running Qwen, MedGemma, Llama x2, DermatoLlama x2, HuluMed x2"
  export BOOTSTRAP_COUNT COMPARE_COUNT BOOTSTRAP_SEED COMPARE_SEED OUTPUT_ROOT
  export USE_EXISTING_SERVERS=1
  export MAX_PARALLEL_MODELS=8

  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/run_6x6_matrix_parallel.sh"

  MODEL_DATASETS[llama]="ham10000 pad20 xiangya"
  MODEL_DATASETS[llama_replica]="sd198 scin isic2019"
  MODEL_DATASETS[dermatollama]="ham10000 sd198 isic2019"
  MODEL_DATASETS[dermatollama_replica]="pad20 scin xiangya"
  MODEL_DATASETS[hulumed]="pad20 ham10000 xiangya"
  MODEL_DATASETS[hulumed_replica]="isic2019 scin sd198"

  MODELS=("${PHASE_A_MODELS[@]}")
  if ((${#MODELS[@]} == 0)); then
    log "WARN: no Phase A models are available; skipping Phase A"
    return 0
  fi
  DATASETS=(ham10000 isic2019 pad20 scin sd198 xiangya)
  main
}

phase_b() {
  log "Phase B: reclaiming GPU2/GPU3 and running SkinVL in two shards"
  stop_port 8000
  stop_port 8010
  kill_gpu_compute_processes 2
  kill_gpu_compute_processes 3

  local skinvl_models=()

  start_skinvl_shard 2 8025 skinvl_heavy &
  local pid_a=$!
  start_skinvl_shard 3 8026 skinvl_light &
  local pid_b=$!
  if wait "${pid_a}"; then
    if wait_models_endpoint 8025 skinvl_heavy 12; then
      skinvl_models+=(skinvl_heavy)
    fi
  else
    log "WARN: skinvl_heavy startup failed; skipping heavy SkinVL shard"
  fi
  if wait "${pid_b}"; then
    if wait_models_endpoint 8026 skinvl_light 12; then
      skinvl_models+=(skinvl_light)
    fi
  else
    log "WARN: skinvl_light startup failed; skipping light SkinVL shard"
  fi

  export USE_EXISTING_SERVERS=1
  export MAX_PARALLEL_MODELS=2

  if ((${#skinvl_models[@]} == 0)); then
    log "WARN: no SkinVL shard is available; skipping Phase B"
    return 0
  fi

  MODEL_DATASETS=()
  MODEL_DATASETS[skinvl_heavy]="pad20 ham10000 isic2019 scin"
  MODEL_DATASETS[skinvl_light]="sd198 xiangya"
  MODELS=("${skinvl_models[@]}")
  DATASETS=(ham10000 isic2019 pad20 scin sd198 xiangya)
  main
}

main_dynamic() {
  activate_env
  log "Output root: ${OUTPUT_ROOT}"
  log "Counts: bootstrap=${BOOTSTRAP_COUNT}, compare=${COMPARE_COUNT}"

  PHASE_A_MODELS=()
  if ensure_existing_endpoint 8000 qwen; then PHASE_A_MODELS+=(qwen); else log "WARN: skipping qwen"; fi
  if ensure_existing_endpoint 8010 medgemma; then PHASE_A_MODELS+=(medgemma); else log "WARN: skipping medgemma"; fi
  if ensure_existing_endpoint 8012 llama; then PHASE_A_MODELS+=(llama); else log "WARN: skipping llama"; fi
  if ensure_existing_endpoint 8013 hulumed; then PHASE_A_MODELS+=(hulumed); else log "WARN: skipping hulumed"; fi
  if ensure_existing_endpoint 8014 dermatollama; then PHASE_A_MODELS+=(dermatollama); else log "WARN: skipping dermatollama"; fi
  if ensure_existing_endpoint 8022 llama_replica; then PHASE_A_MODELS+=(llama_replica); else log "WARN: skipping llama_replica"; fi
  if ensure_existing_endpoint 8024 dermatollama_replica; then PHASE_A_MODELS+=(dermatollama_replica); else log "WARN: skipping dermatollama_replica"; fi
  if start_hulumed_replica && ensure_existing_endpoint 8023 hulumed_replica; then
    PHASE_A_MODELS+=(hulumed_replica)
  else
    log "WARN: skipping hulumed_replica"
  fi

  phase_a
  phase_b

  log "Dynamic 8-GPU run complete: ${OUTPUT_ROOT}/summary.tsv"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main_dynamic "$@"
fi
