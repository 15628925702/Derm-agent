#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_RUN_ID="${SOURCE_RUN_ID:-reuse3h_signal_check_20260328T141307Z}"
RUN_ID="${RUN_ID:-stage34_probe_30m_$(date -u +%Y%m%dT%H%M%SZ)}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-180}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-2}"

# Keep this genuinely short: stage3 evaluates stable + candidate, so LIMIT=4
# already means 8 full agent rollouts before the tiny compare/eval tail.
STAGE3_LIMIT="${STAGE3_LIMIT:-4}"
VAL_LIMIT="${VAL_LIMIT:-4}"
TEST_LIMIT="${TEST_LIMIT:-4}"

VAL_OFFSET="${VAL_OFFSET:-1608}"
TEST_OFFSET="${TEST_OFFSET:-1953}"
SPLIT_JSON="${SPLIT_JSON:-${PROJECT_ROOT}/outputs/smoke_cycles/${SOURCE_RUN_ID}/fixed_split.json}"

SOURCE_TRAIN_RUN_ROOT="${PROJECT_ROOT}/outputs/smoke_cycles/${SOURCE_RUN_ID}/train_runs/${SOURCE_RUN_ID}"
CONTROLLER_CKPT="${CONTROLLER_CKPT:-${SOURCE_TRAIN_RUN_ROOT}/stage1_supervised_controller_training/checkpoints/controller_planner_scorer__candidate__${SOURCE_RUN_ID}.pt}"
RETRIEVAL_CKPT="${RETRIEVAL_CKPT:-${SOURCE_TRAIN_RUN_ROOT}/stage2_retrieval_scorer_training/checkpoints/retrieval_reranker__candidate__${SOURCE_RUN_ID}.pt}"

RUN_ROOT="${PROJECT_ROOT}/outputs/smoke_cycles/${RUN_ID}"
TRAIN_RUN_ROOT="${RUN_ROOT}/train_runs"
CHECKPOINT_OUT_DIR="${RUN_ROOT}/checkpoints"
POLICY_PATH="${TRAIN_RUN_ROOT}/${RUN_ID}/stage3_policy_candidate_evaluation/candidate_policy.json"

cd "${PROJECT_ROOT}"

if [[ ! -f "${CONTROLLER_CKPT}" ]]; then
  echo "[error] controller checkpoint not found: ${CONTROLLER_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${RETRIEVAL_CKPT}" ]]; then
  echo "[error] retrieval checkpoint not found: ${RETRIEVAL_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${SPLIT_JSON}" ]]; then
  echo "[error] split json not found: ${SPLIT_JSON}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" python "${PROJECT_ROOT}/scripts/train_learned_components.py" \
  --stages 3,4 \
  --run-id "${RUN_ID}" \
  --records-root "${PROJECT_ROOT}/outputs" \
  --data-root "${PROJECT_ROOT}/data" \
  --output-dir "${TRAIN_RUN_ROOT}" \
  --checkpoint-out-dir "${CHECKPOINT_OUT_DIR}" \
  --training-split train \
  --split-json "${SPLIT_JSON}" \
  --epochs 1 \
  --limit "${STAGE3_LIMIT}" \
  --case-offset "${VAL_OFFSET}" \
  --stage3-data-split val \
  --controller-checkpoint-in "${CONTROLLER_CKPT}" \
  --retrieval-checkpoint-in "${RETRIEVAL_CKPT}" \
  --client-timeout "${CLIENT_TIMEOUT}" \
  --client-max-retries "${CLIENT_MAX_RETRIES}"

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" python "${PROJECT_ROOT}/scripts/compare_agent_vs_qwen.py" \
  --data-root "${PROJECT_ROOT}/data" \
  --limit "${VAL_LIMIT}" \
  --case-offset "${VAL_OFFSET}" \
  --data-split val \
  --split-json "${SPLIT_JSON}" \
  --output-dir "${RUN_ROOT}/comparison" \
  --policy-config "${POLICY_PATH}" \
  --policy-label "${RUN_ID}_candidate_policy" \
  --client-timeout "${CLIENT_TIMEOUT}" \
  --client-max-retries "${CLIENT_MAX_RETRIES}"

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" python "${PROJECT_ROOT}/scripts/run_eval_brief.py" \
  --data-root "${PROJECT_ROOT}/data" \
  --limit "${TEST_LIMIT}" \
  --case-offset "${TEST_OFFSET}" \
  --data-split test \
  --split-json "${SPLIT_JSON}" \
  --output-dir "${RUN_ROOT}/evaluation_protocol" \
  --policy-config "${POLICY_PATH}" \
  --client-timeout "${CLIENT_TIMEOUT}" \
  --client-max-retries "${CLIENT_MAX_RETRIES}"

python "${PROJECT_ROOT}/scripts/audit_experiment_state.py" \
  --records-root "${RUN_ROOT}/evaluation_protocol" \
  --eval-root "${RUN_ROOT}/evaluation_protocol" \
  --split-json "${SPLIT_JSON}" \
  --fail-on-issues

echo "[ok] stage34 probe finished"
echo "run_root=${RUN_ROOT}"
echo "candidate_policy=${POLICY_PATH}"
echo "comparison_dir=${RUN_ROOT}/comparison"
echo "evaluation_dir=${RUN_ROOT}/evaluation_protocol"
