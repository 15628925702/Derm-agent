# 阶段2实验运行指南

## 📋 概述

这个脚本会自动运行阶段2的所有实验（Metadata Missingness），包括：

1. **Partial-Learned**: Partial metadata + Learned policy (345 cases)
2. **Partial-Heuristic**: Partial metadata + Heuristic policy (345 cases)
3. **Minimal-Learned**: Minimal metadata + Learned policy (345 cases)
4. **Minimal-Heuristic**: Minimal metadata + Heuristic policy (345 cases)

**预计总时间**: 10-12 小时

---

## 🚀 快速开始

### 方式1：后台运行（推荐）

```bash
cd /root/DermAgent

# 使用 nohup 后台运行
nohup bash scripts/run_stage2_full.sh > /tmp/stage2_run.log 2>&1 &

# 记录进程ID
echo $! > /tmp/stage2_pid.txt

# 查看实时日志
tail -f /tmp/stage2_run.log
```

### 方式2：tmux 会话运行

```bash
cd /root/DermAgent

# 创建新的 tmux 会话
tmux new -s stage2

# 在 tmux 中运行
bash scripts/run_stage2_full.sh

# 分离会话: Ctrl+B 然后按 D
# 重新连接: tmux attach -t stage2
```

### 方式3：直接运行（前台）

```bash
cd /root/DermAgent
bash scripts/run_stage2_full.sh
```

---

## 📊 监控进度

### 查看主日志

```bash
# 查看最新的主日志文件
ls -lt /root/DermAgent/final-score/final_runs/stage2_experiments_*.log | head -1

# 实时查看
tail -f /root/DermAgent/final-score/final_runs/stage2_experiments_*.log
```

### 查看单个实验日志

```bash
# Partial-Learned
tail -f /root/DermAgent/final-score/final_runs/stage2_partial_learned/experiment_*.log

# Partial-Heuristic
tail -f /root/DermAgent/final-score/final_runs/stage2_partial_heuristic/experiment_*.log

# Minimal-Learned
tail -f /root/DermAgent/final-score/final_runs/stage2_minimal_learned/experiment_*.log

# Minimal-Heuristic
tail -f /root/DermAgent/final-score/final_runs/stage2_minimal_heuristic/experiment_*.log
```

### 检查进程状态

```bash
# 查看是否还在运行
ps aux | grep run_stage2_full.sh

# 或者使用保存的 PID
cat /tmp/stage2_pid.txt
ps -p $(cat /tmp/stage2_pid.txt)
```

---

## 📁 输出结果

所有结果保存在：`/root/DermAgent/final-score/final_runs/`

```
final_runs/
├── stage2_partial_learned/          # Partial + Learned
│   ├── compare_agent_vs_qwen_*.json
│   └── experiment_*.log
├── stage2_partial_heuristic/        # Partial + Heuristic
│   ├── compare_agent_vs_qwen_*.json
│   └── experiment_*.log
├── stage2_minimal_learned/          # Minimal + Learned
│   ├── compare_agent_vs_qwen_*.json
│   └── experiment_*.log
├── stage2_minimal_heuristic/        # Minimal + Heuristic
│   ├── compare_agent_vs_qwen_*.json
│   └── experiment_*.log
└── stage2_experiments_*.log         # 主日志
```

---

## ⚠️ 注意事项

### 1. Metadata Masking 说明

**重要**: 当前脚本运行的是完整 metadata 的实验。要实现真正的 metadata masking，需要：

- 修改 `agent/evaluation_protocol.py` 来支持 metadata masking
- 或者在后处理阶段手动 mask metadata

### 2. 如果实验中断

```bash
# 查看哪些实验已完成
ls -la /root/DermAgent/final-score/final_runs/stage2_*/compare_agent_vs_qwen_*.json

# 手动运行未完成的实验
cd /root/DermAgent

# 例如：运行 Minimal-Heuristic
python scripts/compare_agent_vs_qwen.py \
    --limit 345 \
    --case-offset 0 \
    --policy-config configs/policies/example_candidate_policy.json \
    --output-dir /root/DermAgent/final-score/final_runs/stage2_minimal_heuristic
```

### 3. 停止运行

```bash
# 使用 PID 停止
kill $(cat /tmp/stage2_pid.txt)

# 或者查找并停止
pkill -f run_stage2_full.sh
```

---

## ✅ 验证结果

实验完成后，检查结果：

```bash
cd /root/DermAgent

# 检查所有实验是否都有结果文件
for dir in stage2_partial_learned stage2_partial_heuristic stage2_minimal_learned stage2_minimal_heuristic; do
    echo "检查: $dir"
    ls -lh final-score/final_runs/$dir/compare_agent_vs_qwen_*.json 2>/dev/null || echo "  ❌ 未找到结果"
done

# 查看主日志的总结部分
tail -50 /root/DermAgent/final-score/final_runs/stage2_experiments_*.log
```

---

## 🔧 故障排查

### 问题1：脚本无法执行

```bash
chmod +x /root/DermAgent/scripts/run_stage2_full.sh
```

### 问题2：Python 环境问题

```bash
# 确认 Python 环境
which python
python --version

# 测试导入
python -c "from agent.evaluation_protocol import run_evaluation_suite; print('OK')"
```

### 问题3：磁盘空间不足

```bash
# 检查磁盘空间
df -h /root/DermAgent

# 清理旧的输出（谨慎操作）
# rm -rf /root/DermAgent/outputs/comparison/*
```

---

## 📞 需要帮助？

如果遇到问题：

1. 查看主日志：`/root/DermAgent/final-score/final_runs/stage2_experiments_*.log`
2. 查看实验日志：`/root/DermAgent/final-score/final_runs/stage2_*/experiment_*.log`
3. 检查错误信息的最后 20 行

---

**创建时间**: 2026-04-17
**脚本版本**: v1.0
