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
  "${DERMAGENT_MODELS_ROOT:-${WORKSPACE_ROOT}/models}/Llama-3.2-11B-Vision-Instruct"
  "/data/gh/models/Llama-3.2-11B-Vision-Instruct"
  "/models/Llama-3.2-11B-Vision-Instruct"
  "/root/models/Llama-3.2-11B-Vision-Instruct"
)
MODEL_PATH="${MODEL_PATH:-${PREFERRED_MODEL_PATHS[0]}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8012}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/llama_server.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/state/llama_server.pid}"
FORCE_RESTART="${FORCE_RESTART:-1}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MODEL_IMPL="${MODEL_IMPL:-transformers}"
DTYPE="${DTYPE:-bfloat16}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":0,\"cudagraph_mode\":0}}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-5}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MM_LIMIT_IMAGE="${MM_LIMIT_IMAGE:-1}"
MM_LIMIT_VIDEO="${MM_LIMIT_VIDEO:-0}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-1}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT:-256}"

resolve_vllm_bin() {
  if command -v vllm >/dev/null 2>&1; then
    command -v vllm
    return 0
  fi

  local conda_base env_bin
  conda_base="$(conda info --base 2>/dev/null || true)"
  env_bin="${conda_base}/envs/${CONDA_ENV_NAME}/bin/vllm"
  if [[ -x "${env_bin}" ]]; then
    echo "${env_bin}"
    return 0
  fi

  echo ""
}

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
      first_match="$(find "${model_root}" -maxdepth 1 -mindepth 1 -type d -iname 'Llama-3.2-11B-Vision-Instruct' | sort | head -n 1 || true)"
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
  curl -s "${service_url}" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/tmp/llama_models.json 2>/dev/null
}

stop_stale_processes() {
  local existing_pids
  existing_pids="$(pgrep -f "(vllm serve .*--port ${PORT}|serve_transformers_openai.py .*--port ${PORT})" || true)"
  if [[ -n "${existing_pids}" ]]; then
    echo "[warn] found stale model process(es) on port ${PORT}: ${existing_pids}"
    pkill -f "(vllm serve .*--port ${PORT}|serve_transformers_openai.py .*--port ${PORT})" || true
    sleep 3
  fi

  if pgrep -f "(vllm serve .*--port ${PORT}|serve_transformers_openai.py .*--port ${PORT})" >/dev/null 2>&1; then
    echo "[error] stale model process is still alive after kill attempt."
    exit 1
  fi
}

check_model_compatibility() {
  "${PYTHON_BIN}" - "${RESOLVED_MODEL_PATH}" <<'PY'
import sys
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

model_path = sys.argv[1]
cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
architectures = getattr(cfg, "architectures", None) or []
if "MllamaForConditionalGeneration" not in architectures:
    raise SystemExit(
        f"[error] expected MllamaForConditionalGeneration, got architectures={architectures}"
    )
processor = AutoProcessor.from_pretrained(model_path)
if not hasattr(processor, "apply_chat_template"):
    raise SystemExit("[error] loaded processor does not expose apply_chat_template.")
if AutoModelForImageTextToText is None:
    raise SystemExit("[error] installed transformers is missing AutoModelForImageTextToText.")
print("[ok] Llama 3.2 Vision config and direct Transformers serving check passed.")
PY
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

check_model_compatibility

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${RESOLVED_MODEL_PATH}")}"
SERVICE_URL="http://${HOST}:${PORT}/v1/models"

if service_ready && [[ "${FORCE_RESTART}" != "1" ]]; then
  echo "[ok] Llama service is already ready."
  cat /tmp/llama_models.json
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
echo "MM cache (GB)     : ${MM_PROCESSOR_CACHE_GB}"
echo "CPU offload (GB)  : ${CPU_OFFLOAD_GB}"
echo "Server impl       : direct Transformers OpenAI-compatible API"
echo "Dtype             : ${DTYPE}"
echo "Device            : ${DEVICE}"
echo "Max new tokens    : ${MAX_NEW_TOKENS_DEFAULT}"
echo "Enforce eager     : ${ENFORCE_EAGER}"
echo "Compilation config: ${COMPILATION_CONFIG}"
echo "CUDA alloc conf   : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "[info] Llama 3.2 Vision is served directly because this vLLM version removed native Mllama support and its generic multimodal backend does not match Mllama processor fields."

export PYTORCH_CUDA_ALLOC_CONF
export LIBRARY_PATH="$(dirname "$(dirname "${PYTHON_BIN}")")/lib:${LIBRARY_PATH:-}"

launch_args=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/serve_transformers_openai.py"
  --model-path "${RESOLVED_MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --backend "mllama"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${OPENAI_API_KEY}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
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

echo "[info] waiting for Llama service on ${SERVICE_URL}"
echo "[info] startup pid: $(cat "${PID_FILE}")"

attempts=$((WAIT_SECONDS / CHECK_INTERVAL_SECONDS))
for _ in $(seq 1 "${attempts}"); do
  if service_ready; then
    echo "[ok] Llama service is ready."
    cat /tmp/llama_models.json
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[error] Llama service did not become ready within ${WAIT_SECONDS} seconds."
echo "[error] check log: ${LOG_FILE}"
tail -n 80 "${LOG_FILE}" || true
exit 1
