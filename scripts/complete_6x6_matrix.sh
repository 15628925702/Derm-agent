#!/usr/bin/env bash
#
# Complete 6x6 matrix by rerunning all failed experiments
# Batch 1: 10 failed (4 ham10000 + 6 scin/sd198)
# Batch 2: 4 failed (4 scin/sd198) - qwen/skinvl already rerunning
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[complete-6x6] Starting complete 6x6 matrix reruns"
echo "[complete-6x6] Current time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ============================================================================
# Batch 1 (193621): Rerun ham10000 for all 4 models
# ============================================================================
echo ""
echo "=== Batch 1 (193621): Rerunning ham10000 for 4 models ==="
cat > /tmp/rerun_batch1_ham10000.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/root/DermAgent"
source "${PROJECT_ROOT}/scripts/run_6x6_matrix_parallel.sh"
MODELS=(qwen skinvl llama hulumed)
DATASETS=(ham10000)
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621"
main "$@"
EOF
chmod +x /tmp/rerun_batch1_ham10000.sh
nohup bash /tmp/rerun_batch1_ham10000.sh > /tmp/rerun_batch1_ham10000.log 2>&1 &
PID_B1_HAM=$!
echo "[complete-6x6] Batch 1 ham10000 queued (PID: ${PID_B1_HAM})"

sleep 2

# ============================================================================
# Batch 1 (193621): Rerun scin/sd198 for llama and hulumed
# (qwen/skinvl scin/sd198 will be handled separately)
# ============================================================================
echo ""
echo "=== Batch 1 (193621): Rerunning scin/sd198 for llama and hulumed ==="
cat > /tmp/rerun_batch1_scin_sd198.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/root/DermAgent"
source "${PROJECT_ROOT}/scripts/run_6x6_matrix_parallel.sh"
MODELS=(llama hulumed)
DATASETS=(scin sd198)
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621"
main "$@"
EOF
chmod +x /tmp/rerun_batch1_scin_sd198.sh
nohup bash /tmp/rerun_batch1_scin_sd198.sh > /tmp/rerun_batch1_scin_sd198.log 2>&1 &
PID_B1_SCIN=$!
echo "[complete-6x6] Batch 1 scin/sd198 queued (PID: ${PID_B1_SCIN})"

sleep 2

# ============================================================================
# Batch 1 (193621): Rerun scin/sd198 for qwen and skinvl
# ============================================================================
echo ""
echo "=== Batch 1 (193621): Rerunning scin/sd198 for qwen and skinvl ==="
cat > /tmp/rerun_batch1_qwen_skinvl.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/root/DermAgent"
source "${PROJECT_ROOT}/scripts/run_6x6_matrix_parallel.sh"
MODELS=(qwen skinvl)
DATASETS=(scin sd198)
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621"
main "$@"
EOF
chmod +x /tmp/rerun_batch1_qwen_skinvl.sh
nohup bash /tmp/rerun_batch1_qwen_skinvl.sh > /tmp/rerun_batch1_qwen_skinvl.log 2>&1 &
PID_B1_QS=$!
echo "[complete-6x6] Batch 1 qwen/skinvl queued (PID: ${PID_B1_QS})"

sleep 2

# ============================================================================
# Batch 2 (193716): Rerun scin/sd198 for llama and hulumed
# (qwen/skinvl already running from earlier rerun_failed_scin_sd198.sh)
# ============================================================================
echo ""
echo "=== Batch 2 (193716): Rerunning scin/sd198 for llama and hulumed ==="
cat > /tmp/rerun_batch2_scin_sd198.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/root/DermAgent"
source "${PROJECT_ROOT}/scripts/run_6x6_matrix_parallel.sh"
MODELS=(llama hulumed)
DATASETS=(scin sd198)
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193716"
main "$@"
EOF
chmod +x /tmp/rerun_batch2_scin_sd198.sh
nohup bash /tmp/rerun_batch2_scin_sd198.sh > /tmp/rerun_batch2_scin_sd198.log 2>&1 &
PID_B2=$!
echo "[complete-6x6] Batch 2 llama/hulumed queued (PID: ${PID_B2})"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "All rerun jobs queued successfully!"
echo "=========================================="
echo ""
echo "Batch 1 (193621) reruns:"
echo "  - ham10000 (4 models): PID ${PID_B1_HAM}, log: /tmp/rerun_batch1_ham10000.log"
echo "  - scin/sd198 (llama/hulumed): PID ${PID_B1_SCIN}, log: /tmp/rerun_batch1_scin_sd198.log"
echo "  - scin/sd198 (qwen/skinvl): PID ${PID_B1_QS}, log: /tmp/rerun_batch1_qwen_skinvl.log"
echo ""
echo "Batch 2 (193716) reruns:"
echo "  - scin/sd198 (qwen/skinvl): Already running from earlier rerun"
echo "  - scin/sd198 (llama/hulumed): PID ${PID_B2}, log: /tmp/rerun_batch2_scin_sd198.log"
echo ""
echo "Monitor progress:"
echo "  tail -f /tmp/rerun_batch1_ham10000.log"
echo "  tail -f /tmp/rerun_batch1_scin_sd198.log"
echo "  tail -f /tmp/rerun_batch2_scin_sd198.log"
echo ""
echo "Check summaries:"
echo "  cat ${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193621/summary.tsv"
echo "  cat ${PROJECT_ROOT}/outputs/6x6_matrix_20260504_193716/summary.tsv"
