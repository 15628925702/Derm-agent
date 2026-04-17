#!/bin/bash
# 一键启动阶段2实验

echo "=========================================="
echo "  阶段2：Metadata Missingness 实验"
echo "=========================================="
echo ""
echo "📋 将运行 4 个实验（每个 345 cases）："
echo "  1. Partial-Learned"
echo "  2. Partial-Heuristic"
echo "  3. Minimal-Learned"
echo "  4. Minimal-Heuristic"
echo ""
echo "⏱️  预计总时间：10-12 小时"
echo ""
echo "📁 结果保存在："
echo "  /root/DermAgent/final-score/final_runs/stage2_*"
echo ""
echo "=========================================="
echo ""

read -p "确认开始运行？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /root/DermAgent

    echo "🚀 启动后台运行..."
    nohup bash scripts/run_stage2_full.sh > /tmp/stage2_run.log 2>&1 &

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
