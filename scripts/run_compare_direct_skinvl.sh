#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/pad_ufes_20_split.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8011/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-SkinVL-MM}"
LIMIT="${LIMIT:-6}"
DATA_SPLIT="${DATA_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/comparison}"

cd "${PROJECT_ROOT}"

env \
  OPENAI_BASE_URL="${BASE_URL}" \
  OPENAI_API_KEY="${API_KEY}" \
  OPENAI_MODEL="${MODEL_NAME}" \
  DERMAGENT_POLICY_ROOT="${PROJECT_ROOT}/state/policy_skinvl" \
  DERMAGENT_SPLIT_STATE_ROOT="${PROJECT_ROOT}/state_skinvl/split_states" \
  python scripts/compare_agent_vs_qwen.py \
    --data-root "${DATA_ROOT}" \
    --split-json "${SPLIT_JSON}" \
    --limit "${LIMIT}" \
    --data-split "${DATA_SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --policy-config "${PROJECT_ROOT}/state/policy_skinvl/current_stable_policy.json" \
    --policy-label "Direct SkinVL"
