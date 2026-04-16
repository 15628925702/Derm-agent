#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_REPO_ID="${HF_REPO_ID:-}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
LOCAL_PATH="${LOCAL_PATH:-${PROJECT_ROOT}/final-score/final_runs}"
REPO_TYPE="${REPO_TYPE:-dataset}"
REVISION="${REVISION:-main}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRIVATE="${PRIVATE:-1}"

if [[ -z "${HF_REPO_ID}" ]]; then
  echo "[error] HF_REPO_ID is required, for example: username/dermagent-final-runs" >&2
  exit 1
fi

if [[ ! -d "${LOCAL_PATH}" ]]; then
  echo "[error] local path does not exist: ${LOCAL_PATH}" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "[error] hf CLI is not installed or not on PATH" >&2
  exit 1
fi

token_args=()
if [[ -n "${HF_TOKEN}" ]]; then
  token_args=(--token "${HF_TOKEN}")
elif ! hf auth whoami >/dev/null 2>&1; then
  echo "[error] Hugging Face is not authenticated. Set HF_TOKEN or run: hf auth login" >&2
  exit 1
fi

private_flag="--no-private"
if [[ "${PRIVATE}" == "1" ]]; then
  private_flag="--private"
fi

echo "[info] repo id        : ${HF_REPO_ID}"
echo "[info] repo type      : ${REPO_TYPE}"
echo "[info] local path     : ${LOCAL_PATH}"
echo "[info] revision       : ${REVISION}"
echo "[info] num workers    : ${NUM_WORKERS}"
echo "[info] private create : ${PRIVATE}"

cmd=(
  hf upload-large-folder
  "${HF_REPO_ID}"
  "${LOCAL_PATH}"
  --repo-type "${REPO_TYPE}"
  --revision "${REVISION}"
  --num-workers "${NUM_WORKERS}"
  "${private_flag}"
  "${token_args[@]}"
)

echo "[run] ${cmd[*]}"
"${cmd[@]}"

echo "[done] uploaded ${LOCAL_PATH} to https://huggingface.co/datasets/${HF_REPO_ID}/tree/${REVISION}"
