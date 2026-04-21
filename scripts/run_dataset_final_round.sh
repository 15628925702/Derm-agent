#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ID="${DATASET_ID:?DATASET_ID is required}"
DATASET_LABEL="${DATASET_LABEL:-$DATASET_ID}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
POLICY_ROOT="${POLICY_ROOT:?POLICY_ROOT is required}"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:?SPLIT_STATE_ROOT is required}"
COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:?COMPARE_SPLIT_JSON is required}"
COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:?COMPARE_POLICY_CONFIG is required}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:?BOOTSTRAP_SCRIPT is required}"

SETUP_SCRIPT="${SETUP_SCRIPT:-}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-24}"
BOOTSTRAP_START_INDEX="${BOOTSTRAP_START_INDEX:-0}"
BOOTSTRAP_OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/final_round/${DATASET_ID}/bootstrap}"
RUN_MODE="${RUN_MODE:-${DATASET_ID}_final_bootstrap}"
PROMOTE_STATE="${PROMOTE_STATE:-1}"
PROMOTE_SOURCE_SPLIT="${PROMOTE_SOURCE_SPLIT:-train}"
PROMOTE_TARGET_SPLITS="${PROMOTE_TARGET_SPLITS:-val,test}"
COMPARE_LIMIT="${COMPARE_LIMIT:?COMPARE_LIMIT is required}"
COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
COMPARE_DATA_SPLIT="${COMPARE_DATA_SPLIT:-test}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/final_round/${DATASET_ID}/compare}"
COMPARE_POLICY_LABEL="${COMPARE_POLICY_LABEL:-${DATASET_ID} final-round policy}"
DRY_RUN="${DRY_RUN:-0}"
SERVER_CHECK_TIMEOUT="${SERVER_CHECK_TIMEOUT:-15}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

ROUND_ROOT="${PROJECT_ROOT}/outputs/final_round/${DATASET_ID}"
SETUP_MARKER="${ROUND_ROOT}/markers/setup.done"
BOOTSTRAP_MARKER="${ROUND_ROOT}/markers/bootstrap_${BOOTSTRAP_START_INDEX}_${BOOTSTRAP_COUNT}.done"
PROMOTE_MARKER="${ROUND_ROOT}/markers/promote_${PROMOTE_SOURCE_SPLIT}_to_$(echo "${PROMOTE_TARGET_SPLITS}" | tr ',' '_').done"
COMPARE_MARKER="${ROUND_ROOT}/markers/compare_${COMPARE_DATA_SPLIT}_${COMPARE_CASE_OFFSET}_${COMPARE_LIMIT}.done"

mkdir -p "${ROUND_ROOT}/markers" "${BOOTSTRAP_OUTPUT_DIR}" "${COMPARE_OUTPUT_DIR}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[error] missing ${label}: ${path}" >&2
    exit 1
  fi
}

echo "[final-round] dataset        : ${DATASET_LABEL}"
echo "[final-round] dataset id     : ${DATASET_ID}"
echo "[final-round] data root      : ${DATA_ROOT}"
echo "[final-round] policy root    : ${POLICY_ROOT}"
echo "[final-round] split root     : ${SPLIT_STATE_ROOT}"
echo "[final-round] split json     : ${COMPARE_SPLIT_JSON}"
echo "[final-round] compare policy : ${COMPARE_POLICY_CONFIG}"
echo "[final-round] bootstrap cnt  : ${BOOTSTRAP_COUNT}"
echo "[final-round] compare limit  : ${COMPARE_LIMIT}"
echo "[final-round] compare offset : ${COMPARE_CASE_OFFSET}"
echo "[final-round] compare split  : ${COMPARE_DATA_SPLIT}"
echo "[final-round] bootstrap dir  : ${BOOTSTRAP_OUTPUT_DIR}"
echo "[final-round] compare dir    : ${COMPARE_OUTPUT_DIR}"
echo "[final-round] dry run        : ${DRY_RUN}"

require_file "${BOOTSTRAP_SCRIPT}" "bootstrap script"
require_file "${DATA_ROOT}" "data root"

if [[ -n "${SETUP_SCRIPT}" ]]; then
  require_file "${SETUP_SCRIPT}" "setup script"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  python "${PROJECT_ROOT}/scripts/check_qwen_server.py" --timeout "${SERVER_CHECK_TIMEOUT}" >/dev/null
fi

run_stage() {
  local marker="$1"
  local label="$2"
  shift 2
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${label} already completed (${marker})"
    return 0
  fi
  echo "[run] ${label}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
  touch "${marker}"
}

if [[ -n "${SETUP_SCRIPT}" ]]; then
  run_stage "${SETUP_MARKER}" "setup" bash "${SETUP_SCRIPT}"
fi

require_file "${COMPARE_SPLIT_JSON}" "compare split json"
require_file "${COMPARE_POLICY_CONFIG}" "compare policy"

run_stage \
  "${BOOTSTRAP_MARKER}" \
  "bootstrap" \
  env \
    DATA_ROOT="${DATA_ROOT}" \
    POLICY_ROOT="${POLICY_ROOT}" \
    SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
    OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR}" \
    RUN_MODE="${RUN_MODE}" \
    DATA_SPLIT="train" \
    SPLIT_JSON="${COMPARE_SPLIT_JSON}" \
    START_INDEX="${BOOTSTRAP_START_INDEX}" \
    COUNT="${BOOTSTRAP_COUNT}" \
    STOP_ON_ERROR="${STOP_ON_ERROR}" \
    bash "${BOOTSTRAP_SCRIPT}"

if [[ "${PROMOTE_STATE}" == "1" ]]; then
  run_stage \
    "${PROMOTE_MARKER}" \
    "promote-state" \
    python "${PROJECT_ROOT}/scripts/manage_dataset_experiment_assets.py" promote-state \
      --split-state-root "${SPLIT_STATE_ROOT}" \
      --source-split "${PROMOTE_SOURCE_SPLIT}" \
      --target-splits "${PROMOTE_TARGET_SPLITS}"
fi

run_stage \
  "${COMPARE_MARKER}" \
  "compare" \
  env \
    DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" \
    DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
    python "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
      --data-root "${DATA_ROOT}" \
      --limit "${COMPARE_LIMIT}" \
      --case-offset "${COMPARE_CASE_OFFSET}" \
      --data-split "${COMPARE_DATA_SPLIT}" \
      --split-json "${COMPARE_SPLIT_JSON}" \
      --output-dir "${COMPARE_OUTPUT_DIR}" \
      --policy-config "${COMPARE_POLICY_CONFIG}" \
      --policy-label "${COMPARE_POLICY_LABEL}"

echo "[ok] final round completed for ${DATASET_LABEL}"
