#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DERMAGENT_SCIN_LABEL_SPACE_ID="${DERMAGENT_SCIN_LABEL_SPACE_ID:-scin_grouped}"
export DATASET_ID="scin"
export DATASET_LABEL="SCIN grouped"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/scin/official_mirror}"
export POLICY_ROOT="${POLICY_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/scin_v1/policy}"
export SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/scin_v1/split_states}"
export COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/scin_v1/scin_split.json}"
export COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/scin_v1/policy/current_stable_policy.json}"
export BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_scin_train_cases.sh}"
export BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-24}"
export BOOTSTRAP_OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/final_round/scin/bootstrap}"
export RUN_MODE="${RUN_MODE:-scin_final_bootstrap}"
export COMPARE_LIMIT="${COMPARE_LIMIT:-40}"
export COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-4278}"
export COMPARE_DATA_SPLIT="${COMPARE_DATA_SPLIT:-test}"
export COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/final_round/scin/compare}"
export COMPARE_POLICY_LABEL="${COMPARE_POLICY_LABEL:-SCIN final-round policy}"

bash "${PROJECT_ROOT}/scripts/run_dataset_final_round.sh"
