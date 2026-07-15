#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/medgemma_final.env
cd "$DERMAGENT_REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-$MEDGEMMA_FINAL_COMPARISON_OUTPUT_ROOT}"
LIMIT_ARGS=()
CASE_OFFSET_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
if [[ -n "${CASE_OFFSET:-}" ]]; then CASE_OFFSET_ARGS=(--case-offset "$CASE_OFFSET"); fi

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

echo "[final] Running direct MedGemma vs agent+MedGemma"
echo "[final] Output dir: $OUTPUT_DIR"
echo "[final] Default split: ${DATA_SPLIT:-$DERMAGENT_DEFAULT_DATA_SPLIT}"
echo "[final] Default full case count: $DEFAULT_CASE_COUNT"
if [[ -n "${LIMIT:-}" ]]; then
  echo "[final] LIMIT override active: $LIMIT case(s)"
else
  echo "[final] LIMIT override active: none"
  echo "[final] Effective run case count: $DEFAULT_CASE_COUNT"
fi
if [[ -n "${CASE_OFFSET:-}" ]]; then
  echo "[final] CASE_OFFSET override active: $CASE_OFFSET"
else
  echo "[final] CASE_OFFSET override active: none"
fi
echo "[progress 1/1] full comparison"

env DERMAGENT_POLICY_ROOT="$MEDGEMMA_FINAL_POLICY_ROOT" \
    DERMAGENT_SPLIT_STATE_ROOT="$MEDGEMMA_FINAL_SPLIT_STATE_ROOT" \
    python scripts/compare_agent_vs_qwen.py \
      --data-root "$DERMAGENT_DATA_ROOT" \
      --split-json "${SPLIT_JSON:-$DERMAGENT_DEFAULT_TEST_SPLIT_JSON}" \
      --data-split "${DATA_SPLIT:-$DERMAGENT_DEFAULT_DATA_SPLIT}" \
      --output-dir "$OUTPUT_DIR" \
      --policy-config "$MEDGEMMA_FINAL_POLICY_JSON" \
      --policy-label "MedGemma final selected agent policy" \
      --agent-base-url "$FINAL_AGENT_BASE_URL" \
      --agent-api-key "$FINAL_AGENT_API_KEY" \
      --agent-model "$FINAL_AGENT_MODEL" \
      --baseline-base-url "$FINAL_BASELINE_BASE_URL" \
      --baseline-api-key "$FINAL_BASELINE_API_KEY" \
      --baseline-model "$FINAL_BASELINE_MODEL" \
      --baseline-label "Direct MedGemma" \
      --baseline-description "Direct MedGemma baseline with no agent evidence package." \
      --phase "${PHASE:-$DERMAGENT_DEFAULT_PHASE}" \
      "${LIMIT_ARGS[@]}" \
      "${CASE_OFFSET_ARGS[@]}"

echo "[progress 1/1] completed"
