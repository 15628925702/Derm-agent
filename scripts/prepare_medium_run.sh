#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-medium_signal_no_ablation_v1}"
DRY_RUN="${DRY_RUN:-1}"
PRUNE_LEGACY_STATE="${PRUNE_LEGACY_STATE:-1}"

log() {
  printf '%s\n' "$1"
}

remove_path() {
  local target="$1"
  if [[ ! -e "${target}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] remove ${target}"
    return 0
  fi
  rm -rf "${target}"
  log "[removed] ${target}"
}

log "[info] project root: ${PROJECT_ROOT}"
log "[info] run id      : ${RUN_ID}"
log "[info] dry run     : ${DRY_RUN}"
log "[info] prune legacy: ${PRUNE_LEGACY_STATE}"
log "[info] preserving split-aware state under ${PROJECT_ROOT}/state/split_states/{train,val,test}"
log "[info] preserving stable policy under ${PROJECT_ROOT}/state/policy/current_stable_policy.json"
log "[info] preserving latest pre-medium readiness report under ${PROJECT_ROOT}/outputs/analysis/pre_medium_smoke_eval_check_v2"

ROOT_OUTPUT_FILES=(
  "${PROJECT_ROOT}/outputs/case_execution_records.jsonl"
)

ROOT_OUTPUT_DIRS=(
  "${PROJECT_ROOT}/outputs/controller_training_smoke"
  "${PROJECT_ROOT}/outputs/controller_training_data_repair_manual"
  "${PROJECT_ROOT}/outputs/controller_training_data_repair_verify"
  "${PROJECT_ROOT}/outputs/controller_training_step3_check_v3"
  "${PROJECT_ROOT}/outputs/repair_eval_check"
  "${PROJECT_ROOT}/outputs/analysis/pre_medium_smoke_eval_check"
  "${PROJECT_ROOT}/outputs/skill_helpfulness_step3_check"
  "${PROJECT_ROOT}/outputs/skill_helpfulness_step3_check_v2"
  "${PROJECT_ROOT}/outputs/skill_helpfulness_step3_check_v3"
  "${PROJECT_ROOT}/outputs/skill_helpfulness_step3_check_v4"
  "${PROJECT_ROOT}/outputs/skill_helpfulness_step3_check_v5"
  "${PROJECT_ROOT}/outputs/skill_helpfulness_step3_check_v6"
  "${PROJECT_ROOT}/outputs/smoke_cycles/${RUN_ID}"
)

LEGACY_STATE_FILES=(
  "${PROJECT_ROOT}/state/experience_bank.json"
  "${PROJECT_ROOT}/state/experience_bank.jsonl"
)

for path in "${ROOT_OUTPUT_FILES[@]}"; do
  remove_path "${path}"
done

for path in "${ROOT_OUTPUT_DIRS[@]}"; do
  remove_path "${path}"
done

shopt -s nullglob
for path in "${PROJECT_ROOT}"/outputs/PAT_*; do
  remove_path "${path}"
done
shopt -u nullglob

if [[ "${PRUNE_LEGACY_STATE}" == "1" ]]; then
  for path in "${LEGACY_STATE_FILES[@]}"; do
    remove_path "${path}"
  done
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "[done] dry-run complete. Re-run with DRY_RUN=0 to actually clean."
else
  log "[done] selective cleanup complete."
fi
