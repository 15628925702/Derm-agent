#!/usr/bin/env bash
set -euo pipefail

source /root/DermAgent/final-script/configs/common.env
cd "$DERMAGENT_REPO_ROOT"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-$DERMAGENT_FINAL_OUTPUT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DERMAGENT_FINAL_EXPORT_ROOT/tables}"
mkdir -p "$OUTPUT_DIR"
echo "[progress 0/1] exporting tables"

python scripts/export_paper_tables.py \
  --outputs-root "$OUTPUTS_ROOT" \
  --output-dir "$OUTPUT_DIR"
echo "[progress 1/1] completed"
