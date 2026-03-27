#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-medium_signal_no_ablation_v1}"
INTERVAL="${2:-60}"

RUN_ROOT="/root/DermAgent/outputs/smoke_cycles/${RUN_ID}/train_runs/${RUN_ID}"
EVAL_ROOT="${RUN_ROOT}/stage3_policy_candidate_evaluation/evaluation"
LOG_PATH="${RUN_ROOT}/progress_watch.log"

mkdir -p "${RUN_ROOT}"

while true; do
  TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  EVAL_PID="$(pgrep -f "scripts/evaluate_policy_candidate.py.*stage3_policy_candidate_evaluation/evaluation" | tr '\n' ',' | sed 's/,$//' || true)"
  TRAIN_PID="$(pgrep -f "scripts/train_learned_components.py --stages 3,4 --run-id ${RUN_ID}" | tr '\n' ',' | sed 's/,$//' || true)"

  STABLE_COUNT="$(python -c "from pathlib import Path; p=Path('${EVAL_ROOT}/stable_run/records/case_execution_records.jsonl'); print(len(p.read_text(encoding='utf-8').splitlines()) if p.exists() else 0)")"
  CANDIDATE_COUNT="$(python -c "from pathlib import Path; p=Path('${EVAL_ROOT}/candidate_run/records/case_execution_records.jsonl'); print(len(p.read_text(encoding='utf-8').splitlines()) if p.exists() else 0)")"
  POLICY_EVAL_COUNT="$(python -c "from pathlib import Path; p=Path('${EVAL_ROOT}'); print(len(list(p.glob('policy_eval_*.json'))) if p.exists() else 0)")"

  echo "${TS} eval_pid=${EVAL_PID:-none} train_pid=${TRAIN_PID:-none} stable=${STABLE_COUNT} candidate=${CANDIDATE_COUNT} policy_eval_files=${POLICY_EVAL_COUNT}" >> "${LOG_PATH}"
  sleep "${INTERVAL}"
done

