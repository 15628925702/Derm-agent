#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATASET_ID="isic2019"
export DATASET_LABEL="ISIC2019"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/isic2019}"
export POLICY_ROOT="${POLICY_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/isic2019_v1/policy}"
export SPLIT_STATE_ROOT="${SPLIT_STATE_ROOT:-${PROJECT_ROOT}/state/dataset_adaptation/isic2019_v1/split_states}"
export COMPARE_SPLIT_JSON="${COMPARE_SPLIT_JSON:-${PROJECT_ROOT}/outputs/dataset_adaptation/isic2019_v1/isic2019_split.json}"
export COMPARE_POLICY_CONFIG="${COMPARE_POLICY_CONFIG:-${PROJECT_ROOT}/state/dataset_adaptation/isic2019_v1/policy/current_stable_policy.json}"
export BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${PROJECT_ROOT}/scripts/bootstrap_isic2019_train_cases.sh}"
export BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-24}"
export BOOTSTRAP_OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/final_round/isic2019/bootstrap}"
export RUN_MODE="${RUN_MODE:-isic2019_final_bootstrap}"
export COMPARE_LIMIT="${COMPARE_LIMIT:-40}"
export COMPARE_CASE_OFFSET="${COMPARE_CASE_OFFSET:-0}"
export COMPARE_DATA_SPLIT="${COMPARE_DATA_SPLIT:-test}"
export COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/final_round/isic2019/compare}"
export COMPARE_POLICY_LABEL="${COMPARE_POLICY_LABEL:-ISIC2019 final-round policy}"

bash "${PROJECT_ROOT}/scripts/run_dataset_final_round.sh"
