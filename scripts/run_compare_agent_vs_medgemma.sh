#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/pad_ufes_20_split.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8010/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-medgemma-4b-it}"
LIMIT="${LIMIT:-6}"
DATA_SPLIT="${DATA_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/comparison}"

cd "${PROJECT_ROOT}"

python scripts/compare_agent_vs_qwen.py \
  --data-root "${DATA_ROOT}" \
  --split-json "${SPLIT_JSON}" \
  --agent-base-url "${BASE_URL}" \
  --agent-api-key "${API_KEY}" \
  --agent-model "${MODEL_NAME}" \
  --baseline-base-url "${BASE_URL}" \
  --baseline-api-key "${API_KEY}" \
  --baseline-model "${MODEL_NAME}" \
  --baseline-label "Direct MedGemma" \
  --baseline-description "Direct MedGemma baseline with no agent evidence package." \
  --limit "${LIMIT}" \
  --data-split "${DATA_SPLIT}" \
  --phase full \
  --output-dir "${OUTPUT_DIR}"
