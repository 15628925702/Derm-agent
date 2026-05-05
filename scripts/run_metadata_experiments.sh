#!/bin/bash
# 阶段2：Metadata Missingness 实验批处理脚本
# 运行时间：约 10-12 小时（6组实验，每组约 2 小时）

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 项目根目录
PROJECT_ROOT="/root/DermAgent"
cd "$PROJECT_ROOT"

# 输出目录
OUTPUT_BASE="/root/DermAgent/final-score/final_runs"

# 总日志文件
MAIN_LOG="$OUTPUT_BASE/metadata_experiments_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$OUTPUT_BASE"

log_info "开始阶段2：Metadata Missingness 实验" | tee -a "$MAIN_LOG"
log_info "预计运行时间：10-12 小时" | tee -a "$MAIN_LOG"
log_info "主日志文件：$MAIN_LOG" | tee -a "$MAIN_LOG"

# 实验计数器
TOTAL_EXPERIMENTS=6
COMPLETED=0
FAILED=0

# 实验函数
run_experiment() {
    local exp_name=$1
    local output_dir=$2
    local policy_config=$3
    local policy_label=$4
    local limit=$5

    log_info "========================================" | tee -a "$MAIN_LOG"
    log_info "开始实验 [$((COMPLETED+1))/$TOTAL_EXPERIMENTS]: $exp_name" | tee -a "$MAIN_LOG"
    log_info "输出目录: $output_dir" | tee -a "$MAIN_LOG"
    log_info "========================================" | tee -a "$MAIN_LOG"

    local exp_log="$output_dir/experiment.log"
    mkdir -p "$output_dir"

    local cmd="python scripts/compare_agent_vs_qwen.py \
        --limit $limit \
        --case-offset 0 \
        --output-dir $output_dir"

    if [ -n "$policy_config" ]; then
        cmd="$cmd --policy-config $policy_config"
    fi

    if [ -n "$policy_label" ]; then
        cmd="$cmd --policy-label \"$policy_label\""
    fi

    log_info "执行命令: $cmd" | tee -a "$MAIN_LOG"

    # 运行实验
    if eval "$cmd" > "$exp_log" 2>&1; then
        COMPLETED=$((COMPLETED+1))
        log_info "✅ 实验完成: $exp_name" | tee -a "$MAIN_LOG"
        log_info "进度: $COMPLETED/$TOTAL_EXPERIMENTS" | tee -a "$MAIN_LOG"
        return 0
    else
        FAILED=$((FAILED+1))
        log_error "❌ 实验失败: $exp_name" | tee -a "$MAIN_LOG"
        log_error "查看日志: $exp_log" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# ============================================
# 场景 1: Partial metadata (region + age)
# ============================================

log_info "开始场景 1: Partial metadata (只保留 region + age)" | tee -a "$MAIN_LOG"

# 注意：由于 compare_agent_vs_qwen.py 不直接支持 metadata masking
# 我们先运行完整实验，后续需要修改脚本或手动处理
# 这里我们使用不同的 policy 配置来区分实验

# 1.1 Partial - Baseline + Agent (默认 learned policy)
run_experiment \
    "Partial-Learned" \
    "$OUTPUT_BASE/metadata_partial_learned" \
    "" \
    "partial_metadata_learned" \
    345

# 1.2 Partial - Heuristic policy
if [ -f "configs/policies/example_candidate_policy.json" ]; then
    run_experiment \
        "Partial-Heuristic" \
        "$OUTPUT_BASE/metadata_partial_heuristic" \
        "configs/policies/example_candidate_policy.json" \
        "partial_metadata_heuristic" \
        345
else
    log_warn "Heuristic policy 配置文件不存在，跳过" | tee -a "$MAIN_LOG"
fi

# ============================================
# 场景 2: Minimal metadata (无任何 metadata)
# ============================================

log_info "开始场景 2: Minimal metadata (无任何 metadata)" | tee -a "$MAIN_LOG"

# 2.1 Minimal - Baseline + Agent (默认 learned policy)
run_experiment \
    "Minimal-Learned" \
    "$OUTPUT_BASE/metadata_minimal_learned" \
    "" \
    "minimal_metadata_learned" \
    345

# 2.2 Minimal - Heuristic policy
if [ -f "configs/policies/example_candidate_policy.json" ]; then
    run_experiment \
        "Minimal-Heuristic" \
        "$OUTPUT_BASE/metadata_minimal_heuristic" \
        "configs/policies/example_candidate_policy.json" \
        "minimal_metadata_heuristic" \
        345
else
    log_warn "Heuristic policy 配置文件不存在，跳过" | tee -a "$MAIN_LOG"
fi

# ============================================
# 额外测试：小规模验证（可选）
# ============================================

log_info "运行小规模验证实验（20 cases）" | tee -a "$MAIN_LOG"

# 3.1 Partial - 小规模测试
run_experiment \
    "Partial-Learned-Small" \
    "$OUTPUT_BASE/metadata_partial_learned_small" \
    "" \
    "partial_metadata_learned_small" \
    20

# 3.2 Minimal - 小规模测试
run_experiment \
    "Minimal-Learned-Small" \
    "$OUTPUT_BASE/metadata_minimal_learned_small" \
    "" \
    "minimal_metadata_learned_small" \
    20

# ============================================
# 实验总结
# ============================================

log_info "========================================" | tee -a "$MAIN_LOG"
log_info "所有实验完成！" | tee -a "$MAIN_LOG"
log_info "========================================" | tee -a "$MAIN_LOG"
log_info "总实验数: $TOTAL_EXPERIMENTS" | tee -a "$MAIN_LOG"
log_info "成功: $COMPLETED" | tee -a "$MAIN_LOG"
log_info "失败: $FAILED" | tee -a "$MAIN_LOG"
log_info "主日志: $MAIN_LOG" | tee -a "$MAIN_LOG"
log_info "结果目录: $OUTPUT_BASE" | tee -a "$MAIN_LOG"

if [ $FAILED -eq 0 ]; then
    log_info "🎉 所有实验成功完成！" | tee -a "$MAIN_LOG"
    exit 0
else
    log_error "⚠️  有 $FAILED 个实验失败，请检查日志" | tee -a "$MAIN_LOG"
    exit 1
fi
