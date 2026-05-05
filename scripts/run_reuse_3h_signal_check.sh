#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-reuse3h_signal_check_v1}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-180}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-3}"
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"
BASE_SEEDED_CASES="${BASE_SEEDED_CASES:-120}"

cd "${PROJECT_ROOT}"

command=(
  python "${PROJECT_ROOT}/scripts/run_smoke_training_cycle.py"
  --profile reuse3h_incremental_no_ablation_v1
  --run-id "${RUN_ID}"
  --client-timeout "${CLIENT_TIMEOUT}"
  --client-max-retries "${CLIENT_MAX_RETRIES}"
  --resume-seed-case-index "${BASE_SEEDED_CASES}"
)

if [[ "${CLEAN_OUTPUTS}" == "1" ]]; then
  command+=(--clean-outputs)
fi

command+=("$@")

"${command[@]}"
