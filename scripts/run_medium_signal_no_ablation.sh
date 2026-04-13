#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-medium_signal_no_ablation_v1}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-180}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-3}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-1}"
RESUME_SEED_CASE_INDEX="${RESUME_SEED_CASE_INDEX:-}"

cd "${PROJECT_ROOT}"

command=(
  python "${PROJECT_ROOT}/scripts/run_smoke_training_cycle.py"
  --profile medium_signal_no_ablation_v1
  --run-id "${RUN_ID}"
  --client-timeout "${CLIENT_TIMEOUT}"
  --client-max-retries "${CLIENT_MAX_RETRIES}"
)

if [[ "${CLEAN_OUTPUTS}" == "1" ]]; then
  command+=(--clean-outputs)
fi

if [[ -n "${RESUME_SEED_CASE_INDEX}" ]]; then
  command+=(--resume-seed-case-index "${RESUME_SEED_CASE_INDEX}")
fi

command+=("$@")

"${command[@]}"
