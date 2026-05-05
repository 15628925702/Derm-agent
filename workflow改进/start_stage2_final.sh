#!/bin/bash
# 一键启动阶段2实验 - Direct baseline vs Heuristic agent

echo "=========================================="
echo "  阶段2：Metadata Missingness 实验"
echo "  Direct baseline vs Heuristic agent"
echo "=========================================="
echo ""
echo "📋 将运行 2 个实验（每个 345 cases）："
echo ""
echo "  1. Partial metadata (只保留 region + age)"
echo "     - Direct baseline"
echo "     - Heuristic agent"
echo ""
echo "  2. Minimal metadata (无任何 metadata)"
echo "     - Direct baseline"
echo "     - Heuristic agent"
echo ""
echo "⏱️  预计总时间：10-12 小时"
echo "    (每个实验约 5-6 小时)"
echo ""
echo "📁 结果保存在："
echo "  /root/DermAgent/final-score/final_runs/"
echo "    ├── stage2_partial_baseline_vs_heuristic/"
echo "    └── stage2_minimal_baseline_vs_heuristic/"
echo ""
echo "📊 已有数据（无需重跑）："
echo "  ✅ Full metadata - Direct baseline"
echo "  ✅ Full metadata - Heuristic agent"
echo ""
echo "=========================================="
echo ""

read -p "确认开始运行？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /root/DermAgent

    echo "🚀 启动后台运行..."
    nohup bash scripts/run_stage2_baseline_vs_heuristic.sh > /tmp/stage2_run.log 2>&1 &

    PID=$!
    echo $PID > /tmp/stage2_pid.txt

    echo ""
    echo "✅ 已启动！"
    echo ""
    echo "进程 ID: $PID"
    echo "PID 文件: /tmp/stage2_pid.txt"
    echo "运行日志: /tmp/stage2_run.log"
    echo ""
    echo "📊 监控命令："
    echo "  查看实时日志: tail -f /tmp/stage2_run.log"
    echo "  查看进程状态: ps -p $PID"
    echo "  停止运行: kill $PID"
    echo ""
    echo "等待 3 秒后显示日志..."
    sleep 3
    echo ""
    echo "=========================================="
    tail -f /tmp/stage2_run.log
else
    echo "❌ 已取消"
    exit 0
fi
