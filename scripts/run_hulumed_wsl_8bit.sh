#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/0-newResearch/temp/Derm-agent"
MODEL="/mnt/g/0-newResearch/dermagent/models/Hulu-Med-4B"
VENV="/home/guo/venvs/dermagent-wsl"
LOG="${ROOT}/logs/hulumed_wsl_8bit.log"
GPU_MAX_MEMORY_GB="${GPU_MAX_MEMORY_GB:-5.0}"
CPU_MAX_MEMORY_GB="${CPU_MAX_MEMORY_GB:-32}"
MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT:-256}"

mkdir -p "${ROOT}/logs" "${ROOT}/.offload/hulumed"
source "${VENV}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_ENABLE_PARALLEL_LOADING=FALSE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

cd "${ROOT}"
python scripts/serve_transformers_openai.py \
  --model-path "${MODEL}" \
  --served-model-name "Hulu-Med-4B" \
  --backend hulumed \
  --host 127.0.0.1 \
  --port 8013 \
  --api-key EMPTY \
  --device cuda \
  --dtype float16 \
  --max-new-tokens-default "${MAX_NEW_TOKENS_DEFAULT}" \
  --device-map auto \
  --gpu-max-memory-gb "${GPU_MAX_MEMORY_GB}" \
  --cpu-max-memory-gb "${CPU_MAX_MEMORY_GB}" \
  --offload-folder "${ROOT}/.offload/hulumed" \
  --load-in-8bit \
  2>&1 | tee "${LOG}"
