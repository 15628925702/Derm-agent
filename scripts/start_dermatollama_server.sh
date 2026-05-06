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
PREFERRED_BASE_MODEL_PATHS=(
  "${DERMAGENT_MODELS_ROOT:-${WORKSPACE_ROOT}/models}/Llama-3.2-11B-Vision-Instruct"
  "/data/gh/models/Llama-3.2-11B-Vision-Instruct"
  "/models/Llama-3.2-11B-Vision-Instruct"
  "/root/models/Llama-3.2-11B-Vision-Instruct"
)
PREFERRED_ADAPTER_PATHS=(
  "${DERMAGENT_MODELS_ROOT:-${WORKSPACE_ROOT}/models}/DermatoLlama-full"
  "/data/gh/models/DermatoLlama-full"
  "/models/DermatoLlama-full"
  "/root/models/DermatoLlama-full"
)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PREFERRED_BASE_MODEL_PATHS[0]}}"
ADAPTER_PATH="${ADAPTER_PATH:-${PREFERRED_ADAPTER_PATHS[0]}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8014}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dermatollama_server.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/state/dermatollama_server.pid}"
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

BASE_SERVED_MODEL_NAME="${BASE_SERVED_MODEL_NAME:-Llama-3.2-11B-Vision-Instruct}"
LORA_MODULE_NAME="${LORA_MODULE_NAME:-DermatoLlama-full}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"

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

detect_base_model_path() {
  if [[ -d "${BASE_MODEL_PATH}" ]]; then
    echo "${BASE_MODEL_PATH}"
    return 0
  fi

  local preferred_path
  for preferred_path in "${PREFERRED_BASE_MODEL_PATHS[@]}"; do
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

  echo "${BASE_MODEL_PATH}"
}

detect_adapter_path() {
  if [[ -d "${ADAPTER_PATH}" ]]; then
    echo "${ADAPTER_PATH}"
    return 0
  fi

  local preferred_path
  for preferred_path in "${PREFERRED_ADAPTER_PATHS[@]}"; do
    if [[ -d "${preferred_path}" ]]; then
      echo "${preferred_path}"
      return 0
    fi
  done

  local model_root
  for model_root in "${MODEL_ROOTS[@]}"; do
    if [[ -d "${model_root}" ]]; then
      local first_match
      first_match="$(find "${model_root}" -maxdepth 1 -mindepth 1 -type d -iname 'DermatoLlama-full' | sort | head -n 1 || true)"
      if [[ -n "${first_match}" ]]; then
        echo "${first_match}"
        return 0
      fi
    fi
  done

  echo "${ADAPTER_PATH}"
}

service_ready() {
  local service_url
  service_url="http://${HOST}:${PORT}/v1/models"
  curl -s "${service_url}" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/tmp/dermatollama_models.json 2>/dev/null
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

check_peft_dependency() {
  local peft_status
  peft_status="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("peft")
if spec is None:
    print("missing")
else:
    import peft
    print(getattr(peft, "__version__", "installed"))
PY
)"

  if [[ "${peft_status}" == "missing" ]]; then
    echo "[error] peft is not installed in conda env '${CONDA_ENV_NAME}'."
    echo "[error] DermatoLlama-full is a PEFT LoRA adapter, not a standalone full model."
    echo "[error] install a compatible peft package before launching this adapter, for example peft==0.16.0."
    exit 1
  else
    echo "[ok] peft dependency detected: ${peft_status}"
  fi
}

check_model_and_adapter_compatibility() {
  "${PYTHON_BIN}" - "${RESOLVED_BASE_MODEL_PATH}" "${RESOLVED_ADAPTER_PATH}" "${MAX_LORA_RANK}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoConfig

base_path = Path(sys.argv[1])
adapter_path = Path(sys.argv[2])
max_lora_rank = int(sys.argv[3])

base_cfg = AutoConfig.from_pretrained(str(base_path), trust_remote_code=True)
base_architectures = getattr(base_cfg, "architectures", None) or []
if "MllamaForConditionalGeneration" not in base_architectures:
    raise SystemExit(
        f"[error] expected MllamaForConditionalGeneration base model, got architectures={base_architectures}"
    )
adapter_config_path = adapter_path / "adapter_config.json"
if not adapter_config_path.exists():
    raise SystemExit(f"[error] adapter_config.json not found: {adapter_config_path}")

with adapter_config_path.open() as f:
    adapter_cfg = json.load(f)

if adapter_cfg.get("peft_type") != "LORA":
    raise SystemExit(f"[error] expected LORA adapter, got peft_type={adapter_cfg.get('peft_type')!r}")

rank = int(adapter_cfg.get("r", 0))
if rank <= 0:
    raise SystemExit(f"[error] invalid LoRA rank in adapter_config.json: r={rank}")
if rank > max_lora_rank:
    raise SystemExit(
        f"[error] adapter rank r={rank} exceeds MAX_LORA_RANK={max_lora_rank}; raise MAX_LORA_RANK before launching."
    )

target_modules = adapter_cfg.get("target_modules") or []
if not target_modules:
    raise SystemExit("[error] adapter_config.json has no target_modules.")

base_name = str(adapter_cfg.get("base_model_name_or_path", ""))
if "Llama-3.2-11B-Vision-Instruct" not in base_name:
    print(f"[warn] adapter base_model_name_or_path is {base_name!r}; expected Llama-3.2-11B-Vision-Instruct.")

print("[ok] DermatoLlama base model, adapter config, and rank checks passed.")
PY
}

RESOLVED_BASE_MODEL_PATH="$(detect_base_model_path)"
if [[ ! -d "${RESOLVED_BASE_MODEL_PATH}" ]]; then
  echo "[error] base model directory was not found."
  echo "[error] preferred paths: ${PREFERRED_BASE_MODEL_PATHS[*]}"
  echo "[error] current BASE_MODEL_PATH: ${BASE_MODEL_PATH}"
  echo "[error] please export BASE_MODEL_PATH=/path/to/Llama-3.2-11B-Vision-Instruct"
  exit 1
fi

RESOLVED_ADAPTER_PATH="$(detect_adapter_path)"
if [[ ! -d "${RESOLVED_ADAPTER_PATH}" ]]; then
  echo "[error] adapter directory was not found."
  echo "[error] preferred paths: ${PREFERRED_ADAPTER_PATHS[*]}"
  echo "[error] current ADAPTER_PATH: ${ADAPTER_PATH}"
  echo "[error] please export ADAPTER_PATH=/path/to/DermatoLlama-full"
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[error] python executable not found."
  echo "[error] activate conda env '${CONDA_ENV_NAME}' or install python in it."
  exit 1
fi

check_peft_dependency
check_model_and_adapter_compatibility

SERVICE_URL="http://${HOST}:${PORT}/v1/models"

if service_ready && [[ "${FORCE_RESTART}" != "1" ]]; then
  echo "[ok] DermatoLlama service is already ready."
  cat /tmp/dermatollama_models.json
  exit 0
fi

stop_stale_processes

echo "Project root       : ${PROJECT_ROOT}"
echo "Base model path    : ${RESOLVED_BASE_MODEL_PATH}"
echo "Adapter path       : ${RESOLVED_ADAPTER_PATH}"
echo "Base model name    : ${BASE_SERVED_MODEL_NAME}"
echo "LoRA model name    : ${LORA_MODULE_NAME}"
echo "Base URL           : http://${HOST}:${PORT}/v1"
echo "API key            : ${OPENAI_API_KEY}"
echo "Log file           : ${LOG_FILE}"
echo "PID file           : ${PID_FILE}"
echo "Force restart      : ${FORCE_RESTART}"
echo "GPU mem util       : ${GPU_MEMORY_UTILIZATION}"
echo "Max model len      : ${MAX_MODEL_LEN}"
echo "Max num seqs       : ${MAX_NUM_SEQS}"
echo "MM limit           : image=${MM_LIMIT_IMAGE} video=${MM_LIMIT_VIDEO}"
echo "MM cache (GB)      : ${MM_PROCESSOR_CACHE_GB}"
echo "CPU offload (GB)   : ${CPU_OFFLOAD_GB}"
echo "Server impl        : direct Transformers OpenAI-compatible API"
echo "Dtype              : ${DTYPE}"
echo "Device             : ${DEVICE}"
echo "Max new tokens     : ${MAX_NEW_TOKENS_DEFAULT}"
echo "Max LoRA rank      : ${MAX_LORA_RANK}"
echo "Max LoRAs          : ${MAX_LORAS}"
echo "Enforce eager      : ${ENFORCE_EAGER}"
echo "Compilation config : ${COMPILATION_CONFIG}"
echo "CUDA alloc conf    : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "LoRA mount mode    : direct PEFT adapter"
echo "[info] send requests with model='${LORA_MODULE_NAME}' to use the DermatoLlama adapter."
echo "[info] DermatoLlama is served directly because this vLLM version removed native Mllama support."

export PYTORCH_CUDA_ALLOC_CONF
export LIBRARY_PATH="$(dirname "$(dirname "${PYTHON_BIN}")")/lib:${LIBRARY_PATH:-}"

launch_args=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/serve_transformers_openai.py"
  --model-path "${RESOLVED_BASE_MODEL_PATH}"
  --served-model-name "${LORA_MODULE_NAME}"
  --backend "mllama_lora"
  --adapter-path "${RESOLVED_ADAPTER_PATH}"
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

echo "[info] waiting for DermatoLlama service on ${SERVICE_URL}"
echo "[info] startup pid: $(cat "${PID_FILE}")"

attempts=$((WAIT_SECONDS / CHECK_INTERVAL_SECONDS))
for _ in $(seq 1 "${attempts}"); do
  if service_ready; then
    echo "[ok] DermatoLlama service is ready."
    cat /tmp/dermatollama_models.json
    exit 0
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[error] DermatoLlama service did not become ready within ${WAIT_SECONDS} seconds."
echo "[error] check log: ${LOG_FILE}"
tail -n 80 "${LOG_FILE}" || true
exit 1
