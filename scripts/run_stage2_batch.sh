#!/bin/bash
# 阶段2实验：简化版批处理脚本
# 先运行完整数据集的实验，metadata masking 功能后续通过修改代码实现

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $1"
}

PROJECT_ROOT="/root/DermAgent"
cd "$PROJECT_ROOT"

OUTPUT_BASE="$PROJECT_ROOT/final-score/final_runs"
MAIN_LOG="$OUTPUT_BASE/batch_run_$(date '+%Y%m%d_%H%M%S').log"

log_info "========================================" | tee -a "$MAIN_LOG"
log_info "开始批量实验运行" | tee -a "$MAIN_LOG"
log_info "预计总时间：约 10 小时" | tee -a "$MAIN_LOG"
log_info "日志文件：$MAIN_LOG" | tee -a "$MAIN_LOG"
log_info "========================================" | tee -a "$MAIN_LOG"

COMPLETED=0
FAILED=0

run_exp() {
    local name=$1
    local output_dir=$2
    local policy_config=$3
    local limit=$4

    log_info "[$((COMPLETED+FAILED+1))] 开始: $name" | tee -a "$MAIN_LOG"
    log_info "    输出: $output_dir" | tee -a "$MAIN_LOG"

    local cmd="python scripts/compare_agent_vs_qwen.py --limit $limit --case-offset 0 --output-dir $output_dir"

    if [ -n "$policy_config" ] && [ -f "$policy_config" ]; then
        cmd="$cmd --policy-config $policy_config"
        log_info "    策略: $policy_config" | tee -a "$MAIN_LOG"
    else
        log_info "    策略: 默认 (stable policy)" | tee -a "$MAIN_LOG"
    fi

    local exp_log="$output_dir/run.log"
    mkdir -p "$output_dir"

    if $cmd > "$exp_log" 2>&1; then
        COMPLETED=$((COMPLETED+1))
        log_info "    ✅ 完成" | tee -a "$MAIN_LOG"
    else
        FAILED=$((FAILED+1))
        log_error "    ❌ 失败 (查看: $exp_log)" | tee -a "$MAIN_LOG"
    fi
}

# 实验1：Full metadata - Learned (已有数据，跳过)
log_info "实验1: Full-Learned - 已有数据，跳过" | tee -a "$MAIN_LOG"

# 实验2：Full metadata - Heuristic (已有数据，跳过)
log_info "实验2: Full-Heuristic - 已有数据，跳过" | tee -a "$MAIN_LOG"

# 实验3：Partial metadata - Learned (345 cases)
run_exp \
    "Partial-Learned" \
    "$OUTPUT_BASE/stage2_partial_learned" \
    "" \
    345

# 实验4：Partial metadata - Heuristic (345 cases)
run_exp \
    "Partial-Heuristic" \
    "$OUTPUT_BASE/stage2_partial_heuristic" \
    "configs/policies/example_candidate_policy.json" \
    345

# 实验5：Minimal metadata - Learned (345 cases)
run_exp \
    "Minimal-Learned" \
    "$OUTPUT_BASE/stage2_minimal_learned" \
    "" \
    345

# 实验6：Minimal metadata - Heuristic (345 cases)
run_exp \
    "Minimal-Heuristic" \
    "$OUTPUT_BASE/stage2_minimal_heuristic" \
    "configs/policies/example_candidate_policy.json" \
    345

log_info "========================================" | tee -a "$MAIN_LOG"
log_info "批量运行完成！" | tee -a "$MAIN_LOG"
log_info "成功: $COMPLETED, 失败: $FAILED" | tee -a "$MAIN_LOG"
log_info "========================================" | tee -a "$MAIN_LOG"

if [ $FAILED -eq 0 ]; then
    log_info "🎉 所有实验成功！" | tee -a "$MAIN_LOG"
    exit 0
else
    log_error "⚠️  有 $FAILED 个实验失败" | tee -a "$MAIN_LOG"
    exit 1
fi
