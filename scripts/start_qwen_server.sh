#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/state"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-derm-qwen}"
MODEL_ROOTS=("/models" "/root/models")
PREFERRED_MODEL_PATHS=(
  "/models/Qwen2.5-VL-7B-Instruct"
  "/root/models/Qwen2.5-VL-7B-Instruct"
)
MODEL_PATH="${MODEL_PATH:-${PREFERRED_MODEL_PATHS[0]}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen_server.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/state/qwen_server.pid}"
FORCE_RESTART="${FORCE_RESTART:-0}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":0,\"cudagraph_mode\":0}}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-5}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MM_LIMIT_IMAGE="${MM_LIMIT_IMAGE:-1}"
MM_LIMIT_VIDEO="${MM_LIMIT_VIDEO:-0}"

resolve_vllm_bin() {
  if command -v vllm >/dev/null 2>&1; then
    command -v vllm
    return 0
  fi

  local env_bin
  env_bin="/root/miniconda3/envs/${CONDA_ENV_NAME}/bin/vllm"
  if [[ -x "${env_bin}" ]]; then
    echo "${env_bin}"
    return 0
  fi

  echo ""
}

resolve_default_max_num_seqs() {
  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    echo "${MAX_NUM_SEQS}"
    return 0
  fi
  if [[ "${MAX_MODEL_LEN}" -gt 12288 ]]; then
    echo "1"
    return 0
  fi
  echo "1"
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
      first_match="$(find "${model_root}" -maxdepth 1 -mindepth 1 -type d -name 'Qwen*' | sort | head -n 1 || true)"
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
  curl -s "${service_url}" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/tmp/qwen_models.json 2>/dev/null
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

VLLM_BIN="$(resolve_vllm_bin)"
if [[ -z "${VLLM_BIN}" ]]; then
  echo "[error] vllm executable not found."
  echo "[error] activate conda env '${CONDA_ENV_NAME}' or install vllm in it."
  exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${RESOLVED_MODEL_PATH}")}"
SERVICE_URL="http://${HOST}:${PORT}/v1/models"
MAX_NUM_SEQS="$(resolve_default_max_num_seqs)"

if service_ready && [[ "${FORCE_RESTART}" != "1" ]]; then
  echo "[ok] Qwen service is already ready."
  echo "[info] desired max model len : ${MAX_MODEL_LEN}"
  echo "[info] desired max num seqs  : ${MAX_NUM_SEQS}"
  echo "[info] if the running service still shows a lower max_model_len, restart with FORCE_RESTART=1 to apply the new context limit."
  cat /tmp/qwen_models.json
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
echo "GPU mem util      : ${GPU_MEMORY_UTILIZATION}"
echo "Max model len     : ${MAX_MODEL_LEN}"
echo "Max num seqs      : ${MAX_NUM_SEQS}"
echo "MM limit          : image=${MM_LIMIT_IMAGE} video=${MM_LIMIT_VIDEO}"
echo "CPU offload (GB)  : ${CPU_OFFLOAD_GB}"
echo "Enforce eager     : ${ENFORCE_EAGER}"
echo "Compilation config: ${COMPILATION_CONFIG}"
echo "CUDA alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "[info] defaults are tuned for stability on heavy multimodal cases."
echo "[info] if you need longer context later, raise MAX_MODEL_LEN and GPU_MEMORY_UTILIZATION gradually."

export PYTORCH_CUDA_ALLOC_CONF

vllm_args=(
  serve "${RESOLVED_MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${OPENAI_API_KEY}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --compilation-config "${COMPILATION_CONFIG}"
  --limit-mm-per-prompt.image "${MM_LIMIT_IMAGE}"
  --limit-mm-per-prompt.video "${MM_LIMIT_VIDEO}"
  --skip-mm-profiling
)

if [[ "${CPU_OFFLOAD_GB}" != "0" ]]; then
  vllm_args+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  vllm_args+=(--enforce-eager)
fi

nohup "${VLLM_BIN}" "${vllm_args[@]}" > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[info] waiting for Qwen service on ${SERVICE_URL}"
echo "[info] startup pid: $(cat "${PID_FILE}")"

attempts=$((WAIT_SECONDS / CHECK_INTERVAL_SECONDS))
for _ in $(seq 1 "${attempts}"); do
  if service_ready; then
    echo "[ok] Qwen service is ready."
    cat /tmp/qwen_models.json
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[error] Qwen service did not become ready within ${WAIT_SECONDS} seconds."
echo "[error] check log: ${LOG_FILE}"
tail -n 80 "${LOG_FILE}" || true
exit 1
