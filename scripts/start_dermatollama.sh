#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/state"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/models/Llama-3.2-11B-Vision-Instruct}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/root/models/DermatoLlama-full}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8006}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dermatollama_server.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/state/dermatollama_server.pid}"
FORCE_RESTART="${FORCE_RESTART:-1}"

MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT:-768}"
DEVICE="${DEVICE:-cuda}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-5}"

resolve_python_bin() {
  local env_bin
  env_bin="/root/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
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

service_ready() {
  local service_url
  service_url="http://${HOST}:${PORT}/v1/models"
  curl -s "${service_url}" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/tmp/dermatollama_models.json 2>/dev/null
}

stop_stale_processes() {
  local existing_pids
  existing_pids="$(pgrep -f "transformers_vlm_server.py.*--port ${PORT}" || true)"
  if [[ -n "${existing_pids}" ]]; then
    echo "[warn] found stale process(es): ${existing_pids}"
    pkill -f "transformers_vlm_server.py.*--port ${PORT}" || true
    sleep 3
  fi

  if pgrep -f "transformers_vlm_server.py.*--port ${PORT}" >/dev/null 2>&1; then
    echo "[error] stale process is still alive after kill attempt."
    exit 1
  fi
}

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "[error] base model directory not found: ${BASE_MODEL_PATH}"
  exit 1
fi

if [[ ! -d "${LORA_ADAPTER_PATH}" ]]; then
  echo "[error] LoRA adapter directory not found: ${LORA_ADAPTER_PATH}"
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[error] python executable not found."
  echo "[error] activate conda env '${CONDA_ENV_NAME}' or install python in it."
  exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dermatollama-full}"
SERVICE_URL="http://${HOST}:${PORT}/v1/models"

if service_ready && [[ "${FORCE_RESTART}" != "1" ]]; then
  echo "[ok] DermatoLlama-full service is already ready."
  cat /tmp/dermatollama_models.json
  exit 0
fi

stop_stale_processes

echo "Project root      : ${PROJECT_ROOT}"
echo "Base model path   : ${BASE_MODEL_PATH}"
echo "LoRA adapter path : ${LORA_ADAPTER_PATH}"
echo "Served model name : ${SERVED_MODEL_NAME}"
echo "Base URL          : http://${HOST}:${PORT}/v1"
echo "API key           : ${OPENAI_API_KEY}"
echo "Log file          : ${LOG_FILE}"
echo "PID file          : ${PID_FILE}"
echo "Force restart     : ${FORCE_RESTART}"
echo "Device            : ${DEVICE}"
echo "Max new tokens    : ${MAX_NEW_TOKENS_DEFAULT}"

launch_args=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/integrations/transformers_vlm_server.py"
  --model-path "${BASE_MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --model-type "mllama"
  --lora-adapter-path "${LORA_ADAPTER_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${OPENAI_API_KEY}"
  --device "${DEVICE}"
  --max-new-tokens-default "${MAX_NEW_TOKENS_DEFAULT}"
)

if command -v setsid >/dev/null 2>&1; then
  nohup setsid "${launch_args[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup "${launch_args[@]}" > "${LOG_FILE}" 2>&1 &
fi

echo $! > "${PID_FILE}"

echo "[info] waiting for DermatoLlama-full service on ${SERVICE_URL}"
echo "[info] startup pid: $(cat "${PID_FILE}")"

attempts=$((WAIT_SECONDS / CHECK_INTERVAL_SECONDS))
for _ in $(seq 1 "${attempts}"); do
  if service_ready; then
    echo "[ok] DermatoLlama-full service is ready."
    cat /tmp/dermatollama_models.json
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[error] DermatoLlama-full service did not become ready within ${WAIT_SECONDS} seconds."
echo "[error] check log: ${LOG_FILE}"
tail -n 80 "${LOG_FILE}" || true
exit 1
