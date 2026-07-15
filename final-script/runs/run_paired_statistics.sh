#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/common.env
source /root/DermAgent/final-script/configs/final_assets_registry.env
cd "$DERMAGENT_REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-$FINAL_PAIRED_STATS_OUTPUT_ROOT}"
mkdir -p "$OUTPUT_DIR"
echo "[progress 0/1] running paired statistics"

RUN_ROOT_ARGS=()
if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT_ARGS+=(--run-root "$RUN_ROOT")
fi
if [[ -n "${RUN_ROOTS:-}" ]]; then
  for run_root in $RUN_ROOTS; do
    RUN_ROOT_ARGS+=(--run-root "$run_root")
  done
fi

python final-script/tools/paired_statistics.py \
  --output-dir "$OUTPUT_DIR" \
  "${RUN_ROOT_ARGS[@]}"
echo "[progress 1/1] completed"
