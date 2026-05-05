#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-48}"
BOOTSTRAP_START_INDEX="${BOOTSTRAP_START_INDEX:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-${BOOTSTRAP_COUNT}}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
EVAL_CASE_OFFSET="${EVAL_CASE_OFFSET:-0}"

POLICY_ROOT="${POLICY_ROOT:-${PROJECT_ROOT}/state/policy_medgemma}"
SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state_medgemma/split_states}"
BOOTSTRAP_OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/medgemma_bootstrap_overnight}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/comparison}"
CHECKPOINT_OUT_DIR="${CHECKPOINT_OUT_DIR:-${PROJECT_ROOT}/outputs/checkpoints}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-medgemma_overnight_${TIMESTAMP}}"
BOOTSTRAP_RUN_MODE="${BOOTSTRAP_RUN_MODE:-medgemma_train_bootstrap_${BOOTSTRAP_COUNT}}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PAD_DATA_ROOT="${PAD_DATA_ROOT:-${PROJECT_ROOT}/data/pad_ufes_20}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/pad_ufes_20_split.json}"

CLIENT_BASE_URL="${CLIENT_BASE_URL:-http://127.0.0.1:8010/v1}"
CLIENT_API_KEY="${CLIENT_API_KEY:-EMPTY}"
CLIENT_MODEL="${CLIENT_MODEL:-medgemma-4b-it}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-}"

MEDGEMMA_MAX_MODEL_LEN="${MEDGEMMA_MAX_MODEL_LEN:-8192}"
MEDGEMMA_GPU_MEMORY_UTILIZATION="${MEDGEMMA_GPU_MEMORY_UTILIZATION:-0.85}"
MEDGEMMA_MAX_NUM_SEQS="${MEDGEMMA_MAX_NUM_SEQS:-1}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/overnight_logs}"
mkdir -p "${LOG_DIR}" "${BOOTSTRAP_OUTPUT_DIR}" "${POLICY_ROOT}" "${SPLIT_STATE_ROOT}"
LOG_PATH="${LOG_DIR}/run_medgemma_overnight_${TIMESTAMP}.log"

log() {
  echo "$1" | tee -a "${LOG_PATH}"
}

run_logged() {
  log "[cmd] $*"
  "$@" 2>&1 | tee -a "${LOG_PATH}"
}

log "[info] project root              : ${PROJECT_ROOT}"
log "[info] policy root               : ${POLICY_ROOT}"
log "[info] split state root          : ${SPLIT_STATE_ROOT}"
log "[info] bootstrap output dir      : ${BOOTSTRAP_OUTPUT_DIR}"
log "[info] compare output dir        : ${COMPARE_OUTPUT_DIR}"
log "[info] train run id              : ${TRAIN_RUN_ID}"
log "[info] bootstrap count           : ${BOOTSTRAP_COUNT}"
log "[info] train limit               : ${TRAIN_LIMIT}"
log "[info] train epochs              : ${TRAIN_EPOCHS}"
log "[info] eval limit                : ${EVAL_LIMIT}"
log "[info] eval case offset          : ${EVAL_CASE_OFFSET}"
log "[info] client base url           : ${CLIENT_BASE_URL}"
log "[info] client model              : ${CLIENT_MODEL}"
log "[info] split json                : ${SPLIT_JSON}"
log "[info] log path                  : ${LOG_PATH}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${SPLIT_JSON}" ]]; then
  log "[info] split json missing; generating deterministic PAD-UFES-20 split json"
  run_logged python - <<'PY'
from pathlib import Path
from configs.dataset_splits import write_fixed_split_json
write_fixed_split_json(
    Path('/root/DermAgent/outputs/pad_ufes_20_split.json'),
    data_root=Path('/root/DermAgent/data'),
)
print('/root/DermAgent/outputs/pad_ufes_20_split.json')
PY
fi

log "[info] starting MedGemma service with conservative overnight settings"
run_logged env \
  FORCE_RESTART=1 \
  MAX_MODEL_LEN="${MEDGEMMA_MAX_MODEL_LEN}" \
  GPU_MEMORY_UTILIZATION="${MEDGEMMA_GPU_MEMORY_UTILIZATION}" \
  MAX_NUM_SEQS="${MEDGEMMA_MAX_NUM_SEQS}" \
  bash "${PROJECT_ROOT}/scripts/start_medgemma_server.sh" "${PROJECT_ROOT}"

log "[step] bootstrap MedGemma train cases"
run_logged env \
  COUNT="${BOOTSTRAP_COUNT}" \
  START_INDEX="${BOOTSTRAP_START_INDEX}" \
  POLICY_ROOT="${POLICY_ROOT}" \
  SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
  OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR}" \
  RUN_MODE="${BOOTSTRAP_RUN_MODE}" \
  DATA_ROOT="${PAD_DATA_ROOT}" \
  CLIENT_BASE_URL="${CLIENT_BASE_URL}" \
  CLIENT_API_KEY="${CLIENT_API_KEY}" \
  CLIENT_MODEL="${CLIENT_MODEL}" \
  CLIENT_TIMEOUT="${CLIENT_TIMEOUT}" \
  CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES}" \
  bash "${PROJECT_ROOT}/scripts/bootstrap_medgemma_train_cases.sh"

log "[step] train learned MedGemma components"
train_cmd=(
  python scripts/train_learned_components.py
  --stages 0,1,2,3,4
  --run-id "${TRAIN_RUN_ID}"
  --records-root "${BOOTSTRAP_OUTPUT_DIR}"
  --data-root "${DATA_ROOT}"
  --split-json "${SPLIT_JSON}"
  --training-split train
  --stage3-data-split val
  --limit "${TRAIN_LIMIT}"
  --epochs "${TRAIN_EPOCHS}"
  --checkpoint-out-dir "${CHECKPOINT_OUT_DIR}"
  --client-base-url "${CLIENT_BASE_URL}"
  --client-api-key "${CLIENT_API_KEY}"
  --client-model "${CLIENT_MODEL}"
)
if [[ -n "${CLIENT_TIMEOUT}" ]]; then
  train_cmd+=(--client-timeout "${CLIENT_TIMEOUT}")
fi
if [[ -n "${CLIENT_MAX_RETRIES}" ]]; then
  train_cmd+=(--client-max-retries "${CLIENT_MAX_RETRIES}")
fi
run_logged env \
  DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" \
  DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
  "${train_cmd[@]}"

CANDIDATE_POLICY_PATH="${PROJECT_ROOT}/outputs/train_runs/${TRAIN_RUN_ID}/stage3_policy_candidate_evaluation/candidate_policy.json"

log "[step] run test comparison with learned MedGemma policy"
compare_cmd=(
  python scripts/compare_agent_vs_qwen.py
  --data-root "${DATA_ROOT}"
  --split-json "${SPLIT_JSON}"
  --agent-base-url "${CLIENT_BASE_URL}"
  --agent-api-key "${CLIENT_API_KEY}"
  --agent-model "${CLIENT_MODEL}"
  --baseline-base-url "${CLIENT_BASE_URL}"
  --baseline-api-key "${CLIENT_API_KEY}"
  --baseline-model "${CLIENT_MODEL}"
  --baseline-label "Direct MedGemma"
  --baseline-description "Direct MedGemma baseline with no agent evidence package."
  --policy-config "${CANDIDATE_POLICY_PATH}"
  --limit "${EVAL_LIMIT}"
  --case-offset "${EVAL_CASE_OFFSET}"
  --data-split test
  --phase full
  --output-dir "${COMPARE_OUTPUT_DIR}"
)
run_logged env \
  DERMAGENT_POLICY_ROOT="${POLICY_ROOT}" \
  DERMAGENT_SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT}" \
  "${compare_cmd[@]}"

log "[ok] overnight MedGemma pipeline completed"
log "[ok] train run root              : ${PROJECT_ROOT}/outputs/train_runs/${TRAIN_RUN_ID}"
log "[ok] candidate policy            : ${CANDIDATE_POLICY_PATH}"
log "[ok] bootstrap output dir        : ${BOOTSTRAP_OUTPUT_DIR}"
log "[ok] comparison output dir       : ${COMPARE_OUTPUT_DIR}"
log "[ok] log path                    : ${LOG_PATH}"
