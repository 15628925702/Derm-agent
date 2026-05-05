#!/usr/bin/env bash
#
# Quick smoke test for 6x6 matrix runner
# Tests with 1 model (qwen) and 2 datasets (ham10000, pad20) with 3 cases each
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[test] Running smoke test for 6x6 matrix runner"
echo "[test] This will test qwen model on ham10000 and pad20 datasets with 3 cases each"

# Override configuration for smoke test
export BOOTSTRAP_COUNT=3
export COMPARE_COUNT=3
export OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_smoke_test_$(date -u +%Y%m%d_%H%M%S)"

# Create a minimal test version
cat > "${PROJECT_ROOT}/scripts/run_6x6_matrix_smoke.sh" <<'SMOKE_EOF'
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
SMOKE_EOF

chmod +x "${PROJECT_ROOT}/scripts/run_6x6_matrix_smoke.sh"

echo "[test] Smoke test script created at: ${PROJECT_ROOT}/scripts/run_6x6_matrix_smoke.sh"
echo "[test] To run smoke test:"
echo ""
echo "  cd /root/DermAgent"
echo "  bash scripts/run_6x6_matrix_smoke.sh"
echo ""
echo "[test] Output will be in: ${OUTPUT_ROOT}"
echo "[test] Progress can be monitored via: tail -f ${OUTPUT_ROOT}/runner.log"
echo "[test] Summary will be in: ${OUTPUT_ROOT}/summary.tsv"
