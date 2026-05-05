#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${2:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

HOST="${HOST:-127.0.0.1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
STOP_WAIT_SECONDS="${STOP_WAIT_SECONDS:-30}"
GPU_RELEASE_WAIT_SECONDS="${GPU_RELEASE_WAIT_SECONDS:-60}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-2}"
STOP_ONLY="${STOP_ONLY:-0}"
FORCE_RESTART="${FORCE_RESTART:-1}"
KILL_ALL_VLLM="${KILL_ALL_VLLM:-1}"

usage() {
  cat <<'EOF'
Usage:
  switch_model_server.sh <qwen|medgemma|skinvl|llama|hulumed|dermatollama|stop> [project_root]

Environment:
  STOP_ONLY=1                  stop current local model services without starting a target
  KILL_ALL_VLLM=1              stop all local vLLM serve and orphaned EngineCore processes
  FORCE_RESTART=1              passed to target start script
  STOP_WAIT_SECONDS=30         graceful stop wait window
  GPU_RELEASE_WAIT_SECONDS=60  wait window for vLLM/SkinVL GPU processes to exit
EOF
}

TARGET="${1:-}"
if [[ -z "${TARGET}" || "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
  usage
  exit 0
fi

case "${TARGET}" in
  qwen|medgemma|skinvl|llama|hulumed|dermatollama|stop) ;;
  *)
    echo "[error] unknown target: ${TARGET}"
    usage
    exit 1
    ;;
esac

declare -A MODEL_PORTS=(
  [qwen]=8000
  [medgemma]=8010
  [skinvl]=8011
  [llama]=8012
  [hulumed]=8013
  [dermatollama]=8014
)

declare -A MODEL_SCRIPTS=(
  [qwen]="${SCRIPT_DIR}/start_qwen_server.sh"
  [medgemma]="${SCRIPT_DIR}/start_medgemma_server.sh"
  [skinvl]="${SCRIPT_DIR}/start_skinvl_server.sh"
  [llama]="${SCRIPT_DIR}/start_llama_server.sh"
  [hulumed]="${SCRIPT_DIR}/start_hulumed_server.sh"
  [dermatollama]="${SCRIPT_DIR}/start_dermatollama_server.sh"
)

declare -A PID_FILES=(
  [qwen]="${PROJECT_ROOT}/state/qwen_server.pid"
  [medgemma]="${PROJECT_ROOT}/state/medgemma_server.pid"
  [skinvl]="${PROJECT_ROOT}/state/skinvl_server.pid"
  [llama]="${PROJECT_ROOT}/state/llama_server.pid"
  [hulumed]="${PROJECT_ROOT}/state/hulumed_server.pid"
  [dermatollama]="${PROJECT_ROOT}/state/dermatollama_server.pid"
)

collect_descendants() {
  local pid="$1"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    collect_descendants "${child}"
  done
  echo "${pid}"
}

is_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

pid_is_model_like() {
  local pid="$1"
  local args comm
  if ! is_alive "${pid}"; then
    return 1
  fi
  comm="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
  args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [[ "${comm} ${args}" =~ VLLM::EngineCore|vllm[[:space:]]+serve|serve_skinvl_openai.py|serve_transformers_openai.py|/root/MM-Skin|SkinVL-MM ]]
}

kill_pid_tree() {
  local root_pid="$1"
  local label="$2"
  if ! is_alive "${root_pid}"; then
    return 0
  fi

  local pids
  pids="$(collect_descendants "${root_pid}" | awk '!seen[$0]++' | tr '\n' ' ')"
  if [[ -z "${pids// }" ]]; then
    return 0
  fi

  echo "[info] stopping ${label}: ${pids}"
  kill -TERM ${pids} 2>/dev/null || true

  local deadline=$((SECONDS + STOP_WAIT_SECONDS))
  local pid
  while (( SECONDS < deadline )); do
    local any_alive=0
    for pid in ${pids}; do
      if is_alive "${pid}"; then
        any_alive=1
        break
      fi
    done
    if [[ "${any_alive}" == "0" ]]; then
      return 0
    fi
    sleep 1
  done

  local still_alive=()
  for pid in ${pids}; do
    if is_alive "${pid}"; then
      still_alive+=("${pid}")
    fi
  done
  if [[ "${#still_alive[@]}" -gt 0 ]]; then
    echo "[warn] force killing ${label}: ${still_alive[*]}"
    kill -KILL "${still_alive[@]}" 2>/dev/null || true
  fi
}

stop_pid_files() {
  local name pid_file pid
  for name in "${!PID_FILES[@]}"; do
    pid_file="${PID_FILES[$name]}"
    if [[ ! -f "${pid_file}" ]]; then
      continue
    fi
    pid="$(tr -dc '0-9' < "${pid_file}" || true)"
    if [[ -n "${pid}" ]]; then
      if pid_is_model_like "${pid}"; then
        kill_pid_tree "${pid}" "${name} pid-file process"
      else
        echo "[warn] removing stale ${name} pid file; pid ${pid} is not a known model process."
      fi
    fi
    rm -f "${pid_file}"
  done
}

pids_listening_on_port() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    return 0
  fi
  ss -ltnp "sport = :${port}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u \
    | tr '\n' ' '
}

stop_known_ports() {
  local name port pids pid
  for name in "${!MODEL_PORTS[@]}"; do
    port="${MODEL_PORTS[$name]}"
    pids="$(pids_listening_on_port "${port}")"
    if [[ -z "${pids// }" ]]; then
      continue
    fi
    echo "[warn] port ${port} (${name}) has listener pid(s): ${pids}"
    for pid in ${pids}; do
      kill_pid_tree "${pid}" "port ${port} listener"
    done
  done
}

stop_process_patterns() {
  local pids pid

  if [[ "${KILL_ALL_VLLM}" == "1" ]]; then
    pids="$(pgrep -f "vllm serve" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "[warn] found vLLM serve process(es): ${pids}"
      for pid in ${pids}; do
        if [[ "${pid}" != "$$" ]]; then
          kill_pid_tree "${pid}" "vLLM serve process"
        fi
      done
    fi
  fi

  pids="$(pgrep -f "serve_skinvl_openai.py|serve_transformers_openai.py|/root/MM-Skin|SkinVL-MM" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "[warn] found SkinVL process(es): ${pids}"
    for pid in ${pids}; do
      if [[ "${pid}" != "$$" ]]; then
        kill_pid_tree "${pid}" "direct model process"
      fi
    done
  fi

  pids="$(pgrep -f "VLLM::EngineCore|VLLM::EngineCore_DP" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "[warn] found orphaned vLLM EngineCore process(es): ${pids}"
    for pid in ${pids}; do
      if [[ "${pid}" != "$$" ]]; then
        kill_pid_tree "${pid}" "orphaned vLLM EngineCore"
      fi
    done
  fi
}

stop_gpu_residuals() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local gpu_pids pid args comm
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -z "${gpu_pids// }" ]]; then
    return 0
  fi

  for pid in ${gpu_pids}; do
    if ! is_alive "${pid}"; then
      continue
    fi
    comm="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${comm} ${args}" =~ VLLM::EngineCore|vllm|serve_skinvl_openai.py|serve_transformers_openai.py|SkinVL-MM|/root/MM-Skin ]]; then
      echo "[warn] found GPU residual process pid=${pid}: ${comm} ${args}"
      kill_pid_tree "${pid}" "GPU residual model process"
    fi
  done
}

wait_for_model_gpu_release() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local deadline=$((SECONDS + GPU_RELEASE_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    local residual_found=0
    local gpu_pids pid args comm
    gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ' || true)"
    for pid in ${gpu_pids}; do
      if ! is_alive "${pid}"; then
        continue
      fi
      comm="$(ps -p "${pid}" -o comm= 2>/dev/null || true)"
      args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
      if [[ "${comm} ${args}" =~ VLLM::EngineCore|vllm|serve_skinvl_openai.py|serve_transformers_openai.py|SkinVL-MM|/root/MM-Skin ]]; then
        residual_found=1
        break
      fi
    done
    if [[ "${residual_found}" == "0" ]]; then
      return 0
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  echo "[warn] GPU still has model-like residual process(es) after ${GPU_RELEASE_WAIT_SECONDS}s:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

stop_all_model_services() {
  echo "[info] stopping local model services before switch"
  stop_pid_files
  stop_known_ports
  stop_process_patterns
  stop_gpu_residuals
  wait_for_model_gpu_release
}

start_target() {
  local target="$1"
  local script="${MODEL_SCRIPTS[$target]}"
  local port="${MODEL_PORTS[$target]}"

  if [[ ! -f "${script}" ]]; then
    echo "[error] target script not found: ${script}"
    exit 1
  fi

  echo "[info] starting ${target} on ${HOST}:${port}"
  export FORCE_RESTART
  export HOST
  export OPENAI_API_KEY
  bash "${script}" "${PROJECT_ROOT}"
}

stop_all_model_services

if [[ "${TARGET}" == "stop" || "${STOP_ONLY}" == "1" ]]; then
  echo "[ok] stopped local model services."
  exit 0
fi

start_target "${TARGET}"
