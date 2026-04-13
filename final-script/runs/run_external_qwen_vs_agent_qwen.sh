#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/qwen_final.env
cd "$DERMAGENT_REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_FINAL_EXTERNAL_OUTPUT_ROOT}"
EXTERNAL_DATASET="${EXTERNAL_DATASET:-ham10000}"
CONSERVATIVE_GENERALIZATION_LAYER="${CONSERVATIVE_GENERALIZATION_LAYER:-1}"

MANIFEST_PATH="${MANIFEST_PATH:-}"
if [[ -z "$MANIFEST_PATH" ]]; then
  if [[ "$EXTERNAL_DATASET" == "ham10000" ]]; then
    MANIFEST_PATH="/root/DermAgent/outputs/external_eval/ham10000_aligned_subset/manifests/ham10000_aligned_subset_manifest_medium_20260330T173735Z.json"
  elif [[ "$EXTERNAL_DATASET" == "isic2019" ]]; then
    MANIFEST_PATH="/root/DermAgent/outputs/overnight_eval_bundle/stable_overnight_eval_bundle_20260330T195909Z/isic2019_100case/manifests/isic2019_aligned_subset_manifest_100case_20260330T195913Z.json"
  else
    echo "[error] Unsupported EXTERNAL_DATASET=$EXTERNAL_DATASET"
    exit 1
  fi
fi

if [[ "$EXTERNAL_DATASET" == "ham10000" ]]; then
  DATA_ROOT="${DATA_ROOT:-/root/DermAgent/data/ham10000}"
elif [[ "$EXTERNAL_DATASET" == "isic2019" ]]; then
  DATA_ROOT="${DATA_ROOT:-/root/DermAgent/data/isic2019}"
else
  echo "[error] Unsupported EXTERNAL_DATASET=$EXTERNAL_DATASET"
  exit 1
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[error] Missing external manifest: $MANIFEST_PATH"
  exit 1
fi

MANIFEST_CASE_COUNT="$(python - <<PY
import json
from pathlib import Path
data=json.loads(Path('$MANIFEST_PATH').read_text())
print(data.get('total_case_count', len(data.get('selected_case_ids', []))))
PY
)"

mkdir -p "$OUTPUT_DIR"

echo "[final] Running external dataset direct Qwen vs agent+Qwen"
echo "[final] Dataset: $EXTERNAL_DATASET"
echo "[final] Manifest: $MANIFEST_PATH"
echo "[final] Manifest case count: $MANIFEST_CASE_COUNT"
echo "[final] Conservative generalization layer: $CONSERVATIVE_GENERALIZATION_LAYER"
echo "[progress 0/2] waiting for Qwen service"

for _ in $(seq 1 "${SERVER_WAIT_RETRIES:-24}"); do
  if curl -sS -H "Authorization: Bearer $FINAL_AGENT_API_KEY" "$FINAL_AGENT_BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  sleep "${SERVER_WAIT_INTERVAL_SECONDS:-5}"
done

curl -sS -H "Authorization: Bearer $FINAL_AGENT_API_KEY" "$FINAL_AGENT_BASE_URL/models" >/dev/null

export DERMAGENT_POLICY_ROOT="$QWEN_FINAL_POLICY_ROOT"
export DERMAGENT_SPLIT_STATE_ROOT="$QWEN_FINAL_FROZEN_STATE_ROOT"
export DERMAGENT_CONSERVATIVE_GENERALIZATION_LAYER="$CONSERVATIVE_GENERALIZATION_LAYER"

echo "[progress 1/2] running external evaluation"
python final-script/tools/external_final_runner.py \
  --dataset "$EXTERNAL_DATASET" \
  --data-root "$DATA_ROOT" \
  --manifest-path "$MANIFEST_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --client-base-url "$FINAL_AGENT_BASE_URL" \
  --client-api-key "$FINAL_AGENT_API_KEY" \
  --client-model "$FINAL_AGENT_MODEL" \
  --policy-config "$QWEN_FINAL_POLICY_JSON" \
  --policy-label "Qwen final stable-paper agent policy (conservative generalization)" \
  --conservative-generalization-layer

echo "[progress 2/2] completed"
