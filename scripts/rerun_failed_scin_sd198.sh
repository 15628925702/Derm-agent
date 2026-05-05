#!/usr/bin/env bash
#
# Rerun failed scin and sd198 experiments from batch 193716
# Runs in background to not interfere with ongoing experiments
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load the main script functions
source "${SCRIPT_DIR}/run_6x6_matrix_parallel.sh"

# Override to only run failed datasets
MODELS=(qwen skinvl)
DATASETS=(scin sd198)

# Use existing output directory from batch 2
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193716"

echo "[rerun] Starting rerun of failed scin and sd198 experiments"
echo "[rerun] Output directory: ${OUTPUT_ROOT}"
echo "[rerun] Models: ${MODELS[*]}"
echo "[rerun] Datasets: ${DATASETS[*]}"
echo "[rerun] This will run in the background and append to existing logs"

main "$@"
