#!/bin/bash
# 小规模测试脚本 - 验证批处理流程是否正常

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cd /root/DermAgent

log_info "开始小规模测试（2 cases）..."

# 测试1：默认 policy
log_info "测试1: 默认 policy"
if python scripts/compare_agent_vs_qwen.py \
    --limit 2 \
    --case-offset 0 \
    --output-dir /tmp/test_default \
    > /tmp/test1.log 2>&1; then
    log_info "✅ 测试1 通过"
else
    log_error "❌ 测试1 失败"
    tail -20 /tmp/test1.log
    exit 1
fi

# 测试2：Heuristic policy（如果存在）
if [ -f "configs/policies/example_candidate_policy.json" ]; then
    log_info "测试2: Heuristic policy"
    if python scripts/compare_agent_vs_qwen.py \
        --limit 2 \
        --case-offset 0 \
        --policy-config configs/policies/example_candidate_policy.json \
        --output-dir /tmp/test_heuristic \
        > /tmp/test2.log 2>&1; then
        log_info "✅ 测试2 通过"
    else
        log_error "❌ 测试2 失败"
        tail -20 /tmp/test2.log
        exit 1
    fi
else
    log_info "⚠️  Heuristic policy 不存在，跳过测试2"
fi

log_info "========================================"
log_info "🎉 所有测试通过！可以运行完整实验"
log_info "========================================"
