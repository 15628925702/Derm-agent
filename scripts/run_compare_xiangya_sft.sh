#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_ID="${EXPERIMENT_ID:-xiangya_sft_v1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sft数据}"
POLICY_ROOT="${POLICY_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/${EXPERIMENT_ID}/policy}"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/${EXPERIMENT_ID}/split_states}"
POLICY_CONFIG="${POLICY_CONFIG:-${POLICY_ROOT}/current_stable_policy.json}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/${EXPERIMENT_ID}/xiangya_sft_split.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/results_only/xiangya_sft_compare}"
LIMIT="${LIMIT:-12}"
CASE_OFFSET="${CASE_OFFSET:-0}"
DATA_SPLIT="${DATA_SPLIT:-test}"
POLICY_LABEL="${POLICY_LABEL:-xiangya_sft heuristic stable policy}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-}"

mkdir -p "${OUTPUT_DIR}"

echo "[info] project root      : ${PROJECT_ROOT}"
echo "[info] experiment id     : ${EXPERIMENT_ID}"
echo "[info] data root         : ${DATA_ROOT}"
echo "[info] policy root       : ${POLICY_ROOT}"
echo "[info] split state root  : ${SPLIT_STATE_ROOT}"
echo "[info] split json        : ${SPLIT_JSON}"
echo "[info] output dir        : ${OUTPUT_DIR}"
echo "[info] limit             : ${LIMIT}"
echo "[info] case offset       : ${CASE_OFFSET}"
echo "[info] data split        : ${DATA_SPLIT}"

cd "${PROJECT_ROOT}"

cmd=(
  python scripts/compare_agent_vs_qwen.py
  --data-root "${DATA_ROOT}"
  --limit "${LIMIT}"
  --case-offset "${CASE_OFFSET}"
  --data-split "${DATA_SPLIT}"
  --split-json "${SPLIT_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --policy-config "${POLICY_CONFIG}"
  --policy-label "${POLICY_LABEL}"
)

if [[ -n "${CLIENT_TIMEOUT}" ]]; then
  cmd+=(--client-timeout "${CLIENT_TIMEOUT}")
fi
if [[ -n "${CLIENT_MAX_RETRIES}" ]]; then
  cmd+=(--client-max-retries "${CLIENT_MAX_RETRIES}")
fi

env \
  DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" \
  DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
  "${cmd[@]}"
