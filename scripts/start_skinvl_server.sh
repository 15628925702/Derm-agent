#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/state"
mkdir -p "${PROJECT_ROOT}/.tmp"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dermagent-6x6}"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MODEL_ROOTS=("${DERMAGENT_MODELS_ROOT:-${WORKSPACE_ROOT}/models}" "/data/gh/models" "/models" "/root/models")
PREFERRED_MODEL_PATHS=(
  "${DERMAGENT_MODELS_ROOT:-${WORKSPACE_ROOT}/models}/SkinVL-MM"
  "/data/gh/models/SkinVL-MM"
  "/models/SkinVL-MM"
  "/root/models/SkinVL-MM"
  "/models/skinvl-mm"
  "/root/models/skinvl-mm"
)
MODEL_PATH="${MODEL_PATH:-${PREFERRED_MODEL_PATHS[0]}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/skinvl_server.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/state/skinvl_server.pid}"
FORCE_RESTART="${FORCE_RESTART:-1}"

MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT:-256}"
DEVICE="${DEVICE:-cuda}"
CONV_MODE="${CONV_MODE:-mistral_instruct}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-5}"

resolve_python_bin() {
  local conda_base env_bin
  conda_base="$(conda info --base 2>/dev/null || true)"
  env_bin="${conda_base}/envs/${CONDA_ENV_NAME}/bin/python"
  if [[ -x "${env_bin}" ]]; then
    echo "${env_bin}"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo ""
}

detect_model_path() {
  if [[ -d "${MODEL_PATH}" ]]; then
    echo "${MODEL_PATH}"
    return 0
  fi

  local preferred_path
  for preferred_path in "${PREFERRED_MODEL_PATHS[@]}"; do
    if [[ -d "${preferred_path}" ]]; then
      echo "${preferred_path}"
      return 0
    fi
  done

  local model_root
  for model_root in "${MODEL_ROOTS[@]}"; do
    if [[ -d "${model_root}" ]]; then
      local first_match
      first_match="$(find "${model_root}" -maxdepth 1 -mindepth 1 -type d \( -iname 'SkinVL-MM' -o -iname 'skinvl-mm' -o -iname 'skinvl*' \) | sort | head -n 1 || true)"
      if [[ -n "${first_match}" ]]; then
        echo "${first_match}"
        return 0
      fi
    fi
  done

  echo "${MODEL_PATH}"
}

service_ready() {
  local service_url
  service_url="http://${HOST}:${PORT}/v1/models"
  curl -s "${service_url}" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/tmp/skinvl_models.json 2>/dev/null
}

stop_stale_processes() {
  local existing_pids
  existing_pids="$(pgrep -f "vllm serve .*--port ${PORT}" || true)"
  if [[ -n "${existing_pids}" ]]; then
    echo "[warn] found stale vLLM process(es): ${existing_pids}"
    pkill -f "vllm serve .*--port ${PORT}" || true
    sleep 3
  fi

  if pgrep -f "vllm serve .*--port ${PORT}" >/dev/null 2>&1; then
    echo "[error] stale vLLM process is still alive after kill attempt."
    exit 1
  fi
}

RESOLVED_MODEL_PATH="$(detect_model_path)"
if [[ ! -d "${RESOLVED_MODEL_PATH}" ]]; then
  echo "[error] model directory was not found."
  echo "[error] preferred paths: ${PREFERRED_MODEL_PATHS[*]}"
  echo "[error] current MODEL_PATH: ${MODEL_PATH}"
  echo "[error] please export MODEL_PATH=/path/to/your/model"
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[error] python executable not found."
  echo "[error] activate conda env '${CONDA_ENV_NAME}' or install python in it."
  exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-SkinVL-MM}"
SERVICE_URL="http://${HOST}:${PORT}/v1/models"
export LIBRARY_PATH="$(dirname "$(dirname "${PYTHON_BIN}")")/lib:${LIBRARY_PATH:-}"

if service_ready && [[ "${FORCE_RESTART}" != "1" ]]; then
  echo "[ok] SkinVL service is already ready."
  cat /tmp/skinvl_models.json
  exit 0
fi

stop_stale_processes

echo "Project root      : ${PROJECT_ROOT}"
echo "Model path        : ${RESOLVED_MODEL_PATH}"
echo "Served model name : ${SERVED_MODEL_NAME}"
echo "Base URL          : http://${HOST}:${PORT}/v1"
echo "API key           : ${OPENAI_API_KEY}"
echo "Log file          : ${LOG_FILE}"
echo "PID file          : ${PID_FILE}"
echo "Force restart     : ${FORCE_RESTART}"
echo "Device            : ${DEVICE}"
echo "Conv mode         : ${CONV_MODE}"
echo "Max new tokens    : ${MAX_NEW_TOKENS_DEFAULT}"

launch_args=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/serve_skinvl_openai.py"
  --model-path "${RESOLVED_MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${OPENAI_API_KEY}"
  --device "${DEVICE}"
  --conv-mode "${CONV_MODE}"
  --max-new-tokens-default "${MAX_NEW_TOKENS_DEFAULT}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] command: ${launch_args[*]}"
  exit 0
fi

if command -v setsid >/dev/null 2>&1; then
  nohup setsid "${launch_args[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup "${launch_args[@]}" > "${LOG_FILE}" 2>&1 &
fi

echo $! > "${PID_FILE}"

echo "[info] waiting for SkinVL service on ${SERVICE_URL}"
echo "[info] startup pid: $(cat "${PID_FILE}")"

attempts=$((WAIT_SECONDS / CHECK_INTERVAL_SECONDS))
for _ in $(seq 1 "${attempts}"); do
  if service_ready; then
    echo "[ok] SkinVL service is ready."
    cat /tmp/skinvl_models.json
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[error] SkinVL service did not become ready within ${WAIT_SECONDS} seconds."
echo "[error] check log: ${LOG_FILE}"
tail -n 80 "${LOG_FILE}" || true
exit 1
