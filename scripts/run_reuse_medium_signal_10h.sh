#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-reuse_medium_signal_10h_$(date -u +%Y%m%dT%H%M%SZ)}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-180}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-3}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-reuse3h_signal_check_20260328T141307Z}"

# This profile is designed to reuse prior accumulated experience rather than
# deleting /outputs and starting from zero.
CLEAN_OUTPUTS="${CLEAN_OUTPUTS:-0}"

# Reuse the earlier seeded train cases and only add new ones on top.
BASE_SEEDED_CASES="${BASE_SEEDED_CASES:-136}"
SOURCE_TRAIN_RUN_ROOT="${PROJECT_ROOT}/outputs/smoke_cycles/${SOURCE_RUN_ID}/train_runs/${SOURCE_RUN_ID}"
CONTROLLER_CKPT="${CONTROLLER_CKPT:-${SOURCE_TRAIN_RUN_ROOT}/stage1_supervised_controller_training/checkpoints/controller_planner_scorer__candidate__${SOURCE_RUN_ID}.pt}"
RETRIEVAL_CKPT="${RETRIEVAL_CKPT:-${SOURCE_TRAIN_RUN_ROOT}/stage2_retrieval_scorer_training/checkpoints/retrieval_reranker__candidate__${SOURCE_RUN_ID}.pt}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${CONTROLLER_CKPT}" ]]; then
  echo "[error] controller checkpoint not found: ${CONTROLLER_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${RETRIEVAL_CKPT}" ]]; then
  echo "[error] retrieval checkpoint not found: ${RETRIEVAL_CKPT}" >&2
  exit 1
fi

command=(
  python "${PROJECT_ROOT}/scripts/run_smoke_training_cycle.py"
  --profile reuse_medium_signal_10h_v1
  --run-id "${RUN_ID}"
  --client-timeout "${CLIENT_TIMEOUT}"
  --client-max-retries "${CLIENT_MAX_RETRIES}"
  --resume-seed-case-index "${BASE_SEEDED_CASES}"
  --controller-checkpoint-in "${CONTROLLER_CKPT}"
  --retrieval-checkpoint-in "${RETRIEVAL_CKPT}"
)

if [[ "${CLEAN_OUTPUTS}" == "1" ]]; then
  command+=(--clean-outputs)
fi

command+=("$@")

"${command[@]}"
