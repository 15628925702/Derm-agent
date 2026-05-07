#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export PORT="${PORT:-8200}"
export HOST="${HOST:-127.0.0.1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
export LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/qwen_physician_summary_server.log}"
export PID_FILE="${PID_FILE:-${PROJECT_ROOT}/state/qwen_physician_summary_server.pid}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

exec bash "${PROJECT_ROOT}/scripts/start_qwen_server.sh" "${PROJECT_ROOT}"
