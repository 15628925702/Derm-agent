#!/usr/bin/env bash
#
# Rerun ALL failed experiments from both batches
# This ensures complete 6x6 matrix coverage
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[rerun-all] Starting comprehensive rerun of all failed experiments"
echo "[rerun-all] This will queue experiments to run after current tasks complete"

# Batch 1 (193621) failed experiments: ham10000, scin, sd198 for all models
echo ""
echo "=== Batch 1 (193621) - Rerunning ham10000, scin, sd198 ==="
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621" \
  MODELS=(qwen skinvl llama hulumed) \
  DATASETS=(ham10000 scin sd198) \
  bash "${SCRIPT_DIR}/run_6x6_matrix_parallel.sh" &

BATCH1_PID=$!
echo "[rerun-all] Batch 1 rerun started with PID: ${BATCH1_PID}"

# Wait a bit to avoid port conflicts
sleep 5

# Batch 2 (193716) failed experiments: scin, sd198 for remaining models (llama, hulumed)
# Note: qwen and skinvl are already being rerun by the earlier script
echo ""
echo "=== Batch 2 (193716) - Rerunning scin, sd198 for llama and hulumed ==="
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193716" \
  MODELS=(llama hulumed) \
  DATASETS=(scin sd198) \
  bash "${SCRIPT_DIR}/run_6x6_matrix_parallel.sh" &

BATCH2_PID=$!
echo "[rerun-all] Batch 2 rerun started with PID: ${BATCH2_PID}"

echo ""
echo "[rerun-all] All rerun jobs queued:"
echo "  - Batch 1 PID: ${BATCH1_PID}"
echo "  - Batch 2 PID: ${BATCH2_PID}"
echo "[rerun-all] Monitor progress:"
echo "  - Batch 1 log: tail -f ${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621/runner.log"
echo "  - Batch 2 log: tail -f ${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193716/runner.log"
echo "[rerun-all] Summary files:"
echo "  - Batch 1: ${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621/summary.tsv"
echo "  - Batch 2: ${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193716/summary.tsv"

wait
