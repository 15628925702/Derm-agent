#!/bin/bash
# 阶段2：Metadata Missingness 实验
# 对比 Direct baseline vs Heuristic agent
# 2 个场景 × 1 个配置对比 = 2 个实验
# 预计总时间：10-12 小时

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO $(date '+%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR $(date '+%H:%M:%S')]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# 项目设置
PROJECT_ROOT="/root/DermAgent"
cd "$PROJECT_ROOT" || exit 1

OUTPUT_BASE="$PROJECT_ROOT/final-score/final_runs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="$OUTPUT_BASE/stage2_baseline_vs_heuristic_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_BASE"

log_section "阶段2：Metadata Missingness 实验" | tee "$MAIN_LOG"
log_info "对比：Direct baseline vs Heuristic agent" | tee -a "$MAIN_LOG"
log_info "开始时间: $(date)" | tee -a "$MAIN_LOG"
log_info "预计运行时间: 10-12 小时" | tee -a "$MAIN_LOG"
log_info "主日志文件: $MAIN_LOG" | tee -a "$MAIN_LOG"
log_info "" | tee -a "$MAIN_LOG"

# 实验配置
TOTAL_CASES=345
HEURISTIC_POLICY="configs/policies/example_candidate_policy.json"

# 实验列表：场景名称:输出目录
EXPERIMENTS=(
    "Partial metadata (region+age):stage2_partial_baseline_vs_heuristic"
    "Minimal metadata (无metadata):stage2_minimal_baseline_vs_heuristic"
)

TOTAL_EXP=${#EXPERIMENTS[@]}
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# 运行单个实验
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local output_dir=$3

    log_section "实验 $exp_num/$TOTAL_EXP: $exp_name" | tee -a "$MAIN_LOG"
    log_info "输出目录: $output_dir" | tee -a "$MAIN_LOG"
    log_info "病例数: $TOTAL_CASES" | tee -a "$MAIN_LOG"
    log_info "对比: Direct baseline vs Heuristic agent" | tee -a "$MAIN_LOG"

    local full_output_dir="$OUTPUT_BASE/$output_dir"
    mkdir -p "$full_output_dir"

    local exp_log="$full_output_dir/experiment_${TIMESTAMP}.log"
    local exp_start=$(date +%s)

    # 构建命令 - 使用 Heuristic policy
    # compare_agent_vs_qwen.py 会同时运行 baseline 和 agent
    local cmd="python scripts/compare_agent_vs_qwen.py \
        --limit $TOTAL_CASES \
        --case-offset 0 \
        --policy-config $HEURISTIC_POLICY \
        --policy-label heuristic_agent \
        --output-dir $full_output_dir"

    log_info "策略配置: $HEURISTIC_POLICY (Heuristic)" | tee -a "$MAIN_LOG"
    log_info "执行命令: $cmd" | tee -a "$MAIN_LOG"
    log_info "实验日志: $exp_log" | tee -a "$MAIN_LOG"
    log_info "" | tee -a "$MAIN_LOG"

    # 运行实验
    if eval "$cmd" > "$exp_log" 2>&1; then
        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        COMPLETED=$((COMPLETED + 1))

        log_info "✅ 实验完成: $exp_name" | tee -a "$MAIN_LOG"
        log_info "   耗时: $((exp_duration / 60)) 分钟 ($((exp_duration / 3600))h $((exp_duration % 3600 / 60))m)" | tee -a "$MAIN_LOG"
        log_info "   进度: $COMPLETED/$TOTAL_EXP 完成" | tee -a "$MAIN_LOG"

        # 提取关键结果
        if [ -f "$exp_log" ]; then
            log_info "   结果摘要:" | tee -a "$MAIN_LOG"
            grep -E '"(baseline|agent)":|"top1":|"malignant_recall":' "$exp_log" | head -20 | tee -a "$MAIN_LOG" || true
        fi

        return 0
    else
        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        FAILED=$((FAILED + 1))

        log_error "❌ 实验失败: $exp_name" | tee -a "$MAIN_LOG"
        log_error "   耗时: $((exp_duration / 60)) 分钟" | tee -a "$MAIN_LOG"
        log_error "   查看日志: $exp_log" | tee -a "$MAIN_LOG"
        log_error "   最后 20 行:" | tee -a "$MAIN_LOG"
        tail -20 "$exp_log" | tee -a "$MAIN_LOG"

        return 1
    fi
}

# 运行所有实验
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name output_dir <<< "${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))

    run_experiment "$exp_num" "$exp_name" "$output_dir"

    # 估算剩余时间
    if [ $exp_num -lt $TOTAL_EXP ]; then
        current_time=$(date +%s)
        elapsed=$((current_time - START_TIME))
        avg_time=$((elapsed / exp_num))
        remaining=$((TOTAL_EXP - exp_num))
        eta=$((avg_time * remaining))

        log_info "" | tee -a "$MAIN_LOG"
        log_info "⏱️  预计剩余时间: $((eta / 3600)) 小时 $(((eta % 3600) / 60)) 分钟" | tee -a "$MAIN_LOG"
        log_info "" | tee -a "$MAIN_LOG"
    fi
done

# 最终总结
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

log_section "实验总结" | tee -a "$MAIN_LOG"
log_info "结束时间: $(date)" | tee -a "$MAIN_LOG"
log_info "总耗时: $((TOTAL_DURATION / 3600)) 小时 $(((TOTAL_DURATION % 3600) / 60)) 分钟" | tee -a "$MAIN_LOG"
log_info "" | tee -a "$MAIN_LOG"
log_info "总实验数: $TOTAL_EXP" | tee -a "$MAIN_LOG"
log_info "成功: $COMPLETED" | tee -a "$MAIN_LOG"
log_info "失败: $FAILED" | tee -a "$MAIN_LOG"
log_info "" | tee -a "$MAIN_LOG"
log_info "主日志: $MAIN_LOG" | tee -a "$MAIN_LOG"
log_info "结果目录: $OUTPUT_BASE" | tee -a "$MAIN_LOG"

# 列出所有结果目录
log_info "" | tee -a "$MAIN_LOG"
log_info "实验结果目录:" | tee -a "$MAIN_LOG"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name output_dir <<< "$exp"
    log_info "  - $exp_name: $OUTPUT_BASE/$output_dir" | tee -a "$MAIN_LOG"
done

log_info "" | tee -a "$MAIN_LOG"
log_info "📊 下一步：分析结果" | tee -a "$MAIN_LOG"
log_info "对比 Full vs Partial vs Minimal 场景下 baseline 和 heuristic 的性能差异" | tee -a "$MAIN_LOG"

if [ $FAILED -eq 0 ]; then
    log_info "" | tee -a "$MAIN_LOG"
    log_info "🎉 所有实验成功完成！" | tee -a "$MAIN_LOG"
    exit 0
else
    log_error "" | tee -a "$MAIN_LOG"
    log_error "⚠️  有 $FAILED 个实验失败，请检查日志" | tee -a "$MAIN_LOG"
    exit 1
fi
