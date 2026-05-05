#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/qwen_final.env
cd "$DERMAGENT_REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash final-script/runs/run_main_qwen_vs_agent_qwen.sh [options]

Options:
  --smoke-10              Run a 10-case smoke check on the default test split.
  --medium-40             Run a 40-case medium evaluation on the default test split.
  --full-test             Run the full default test split.
  --limit N               Override case count.
  --case-offset N         Override case offset.
  --output-dir PATH       Override output directory.
  --data-split NAME       Override data split. Default: test
  --split-json PATH       Override split json.
  --client-timeout SEC    Override per-request timeout.
  --client-max-retries N  Override client retries.
  --help                  Show this help message.

Environment variables remain supported:
  LIMIT, CASE_OFFSET, OUTPUT_DIR, DATA_SPLIT, SPLIT_JSON,
  CLIENT_TIMEOUT, CLIENT_MAX_RETRIES
EOF
}

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_FINAL_COMPARISON_OUTPUT_ROOT}"
DATA_SPLIT_VALUE="${DATA_SPLIT:-$DERMAGENT_DEFAULT_DATA_SPLIT}"
SPLIT_JSON_VALUE="${SPLIT_JSON:-$DERMAGENT_DEFAULT_TEST_SPLIT_JSON}"
CLIENT_TIMEOUT_VALUE="${CLIENT_TIMEOUT:-}"
CLIENT_MAX_RETRIES_VALUE="${CLIENT_MAX_RETRIES:-}"
LIMIT_VALUE="${LIMIT:-}"
CASE_OFFSET_VALUE="${CASE_OFFSET:-}"

PRESET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-10)
      PRESET="smoke-10"
      shift
      ;;
    --medium-40)
      PRESET="medium-40"
      shift
      ;;
    --full-test)
      PRESET="full-test"
      shift
      ;;
    --limit)
      LIMIT_VALUE="${2:?missing value for --limit}"
      shift 2
      ;;
    --case-offset)
      CASE_OFFSET_VALUE="${2:?missing value for --case-offset}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --data-split)
      DATA_SPLIT_VALUE="${2:?missing value for --data-split}"
      shift 2
      ;;
    --split-json)
      SPLIT_JSON_VALUE="${2:?missing value for --split-json}"
      shift 2
      ;;
    --client-timeout)
      CLIENT_TIMEOUT_VALUE="${2:?missing value for --client-timeout}"
      shift 2
      ;;
    --client-max-retries)
      CLIENT_MAX_RETRIES_VALUE="${2:?missing value for --client-max-retries}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

DEFAULT_CASE_COUNT="$(python - <<'PY'
import json
from pathlib import Path
data=json.loads(Path('/root/DermAgent/outputs/pad_ufes_20_split.json').read_text())
print(len(data['test']))
PY
)"

case "$PRESET" in
  smoke-10)
    LIMIT_VALUE="${LIMIT_VALUE:-10}"
    ;;
  medium-40)
    LIMIT_VALUE="${LIMIT_VALUE:-40}"
    ;;
  full-test)
    LIMIT_VALUE="${LIMIT_VALUE:-$DEFAULT_CASE_COUNT}"
    ;;
esac

LIMIT_VALUE="${LIMIT_VALUE:-$DEFAULT_CASE_COUNT}"

LIMIT_ARGS=(--limit "$LIMIT_VALUE")
CASE_OFFSET_ARGS=()
TIMEOUT_ARGS=()
RETRY_ARGS=()
if [[ -n "$CASE_OFFSET_VALUE" ]]; then CASE_OFFSET_ARGS=(--case-offset "$CASE_OFFSET_VALUE"); fi
if [[ -n "$CLIENT_TIMEOUT_VALUE" ]]; then TIMEOUT_ARGS=(--client-timeout "$CLIENT_TIMEOUT_VALUE"); fi
if [[ -n "$CLIENT_MAX_RETRIES_VALUE" ]]; then RETRY_ARGS=(--client-max-retries "$CLIENT_MAX_RETRIES_VALUE"); fi

mkdir -p "$OUTPUT_DIR"

echo "[final] Running direct Qwen vs agent+Qwen"
echo "[final] Output dir: $OUTPUT_DIR"
echo "[final] Default split: $DATA_SPLIT_VALUE"
echo "[final] Default full case count: $DEFAULT_CASE_COUNT"
if [[ -n "$PRESET" ]]; then
  echo "[final] Preset active: $PRESET"
else
  echo "[final] Preset active: none"
fi
if [[ -n "${LIMIT:-}" || -n "$PRESET" || -n "$LIMIT_VALUE" ]]; then
  echo "[final] LIMIT override active: $LIMIT_VALUE case(s)"
else
  echo "[final] LIMIT override active: none"
  echo "[final] Effective run case count: $DEFAULT_CASE_COUNT"
fi
if [[ -n "$CASE_OFFSET_VALUE" ]]; then
  echo "[final] CASE_OFFSET override active: $CASE_OFFSET_VALUE"
else
  echo "[final] CASE_OFFSET override active: none"
fi
echo "[progress 0/3] waiting for Qwen service"

for _ in $(seq 1 "${SERVER_WAIT_RETRIES:-24}"); do
  if curl -sS -H "Authorization: Bearer $FINAL_AGENT_API_KEY" "$FINAL_AGENT_BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  sleep "${SERVER_WAIT_INTERVAL_SECONDS:-5}"
done

curl -sS -H "Authorization: Bearer $FINAL_AGENT_API_KEY" "$FINAL_AGENT_BASE_URL/models" >/dev/null

echo "[progress 1/3] compare phase"
env \
  OPENAI_BASE_URL="$FINAL_AGENT_BASE_URL" \
  OPENAI_API_KEY="$FINAL_AGENT_API_KEY" \
  OPENAI_MODEL="$FINAL_AGENT_MODEL" \
  DERMAGENT_POLICY_ROOT="$QWEN_FINAL_POLICY_ROOT" \
  DERMAGENT_SPLIT_STATE_ROOT="$QWEN_FINAL_FROZEN_STATE_ROOT" \
  python scripts/compare_agent_vs_qwen.py \
    --data-root "$DERMAGENT_DATA_ROOT" \
    --split-json "$SPLIT_JSON_VALUE" \
    --data-split "$DATA_SPLIT_VALUE" \
    --output-dir "$OUTPUT_DIR" \
    --policy-config "$QWEN_FINAL_POLICY_JSON" \
    --policy-label "Qwen final stable-paper agent policy" \
    "${LIMIT_ARGS[@]}" \
    "${CASE_OFFSET_ARGS[@]}" \
    "${TIMEOUT_ARGS[@]}" \
    "${RETRY_ARGS[@]}"

RUN_ROOT="$(find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
if [[ -n "$RUN_ROOT" ]]; then
  echo "[final] Run root: $RUN_ROOT"
fi

echo "[progress 2/3] artifacts written"

echo "[progress 3/3] completed"
