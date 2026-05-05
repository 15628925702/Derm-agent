#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load the main script functions
source "${SCRIPT_DIR}/run_6x6_matrix_parallel.sh"

# Override to test only qwen and 2 datasets
MODELS=(qwen)
DATASETS=(ham10000 pad20)

main "$@"
