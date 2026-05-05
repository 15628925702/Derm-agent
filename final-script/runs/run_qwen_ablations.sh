#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/qwen_final.env
cd "$DERMAGENT_REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_FINAL_ABLATION_OUTPUT_ROOT}"
MODE="${MODE:-full}"
ABLATIONS_ARGS=()
LIMIT_ARGS=()
OPTIONAL_ARGS=()

if [[ -n "${ABLATIONS:-}" ]]; then ABLATIONS_ARGS=(--ablations "$ABLATIONS"); fi
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
if [[ "${INCLUDE_OPTIONAL:-0}" == "1" ]]; then OPTIONAL_ARGS=(--include-optional); fi

DEFAULT_CASE_COUNT="$(python - <<'PY'
import json
from pathlib import Path
data=json.loads(Path('/root/DermAgent/outputs/pad_ufes_20_split.json').read_text())
print(len(data['test']))
PY
)"
if [[ -z "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$DEFAULT_CASE_COUNT")
fi

mkdir -p "$OUTPUT_DIR"

echo "[final] Running Qwen agent ablations"
echo "[final] Output dir: $OUTPUT_DIR"
echo "[final] Mode: $MODE"
echo "[final] Default split: ${DATA_SPLIT:-$DERMAGENT_DEFAULT_DATA_SPLIT}"
echo "[final] Default full case count: $DEFAULT_CASE_COUNT"
if [[ -n "${LIMIT:-}" ]]; then
  echo "[final] LIMIT override active: $LIMIT case(s)"
else
  echo "[final] LIMIT override active: none"
  echo "[final] Effective run case count: $DEFAULT_CASE_COUNT"
fi
if [[ -n "${ABLATIONS:-}" ]]; then
  echo "[final] ABLATIONS override active: $ABLATIONS"
else
  echo "[final] ABLATIONS override active: none"
fi
echo "[progress 0/2] waiting for Qwen service"

for _ in $(seq 1 "${SERVER_WAIT_RETRIES:-24}"); do
  if curl -sS -H "Authorization: Bearer $FINAL_AGENT_API_KEY" "$FINAL_AGENT_BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  sleep "${SERVER_WAIT_INTERVAL_SECONDS:-5}"
done

curl -sS -H "Authorization: Bearer $FINAL_AGENT_API_KEY" "$FINAL_AGENT_BASE_URL/models" >/dev/null

echo "[progress 1/2] running ablation matrix"

env DERMAGENT_POLICY_ROOT="$QWEN_FINAL_POLICY_ROOT" \
    DERMAGENT_SPLIT_STATE_ROOT="$QWEN_FINAL_FROZEN_STATE_ROOT" \
    python scripts/run_ablations.py \
      --data-root "$DERMAGENT_DATA_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --mode "$MODE" \
      --data-split "${DATA_SPLIT:-$DERMAGENT_DEFAULT_DATA_SPLIT}" \
      --split-json "${SPLIT_JSON:-$DERMAGENT_DEFAULT_TEST_SPLIT_JSON}" \
      --policy-config "$QWEN_FINAL_POLICY_JSON" \
      --learned-controller-checkpoint "$QWEN_FINAL_CONTROLLER_CHECKPOINT" \
      --retrieval-scorer-checkpoint "$QWEN_FINAL_RETRIEVAL_CHECKPOINT" \
      --strict-checkpoints \
      "${ABLATIONS_ARGS[@]}" \
      "${LIMIT_ARGS[@]}" \
      "${OPTIONAL_ARGS[@]}"

echo "[progress 2/2] completed"
