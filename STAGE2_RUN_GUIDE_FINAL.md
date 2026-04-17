# 阶段2实验运行指南（最终版）

## 📋 实验概述

**目标**：证明"信息缺失时 Heuristic agent 相对 Direct baseline 的稳定性优势"

**对比配置**：
- Direct baseline（直接用 Qwen，无 agent）
- Heuristic agent（规则 agent，无 learned 组件）

**实验场景**：
1. ✅ **Full metadata** - 已完成（阶段1数据）
2. ⏳ **Partial metadata** - 只保留 region + age
3. ⏳ **Minimal metadata** - 无任何 metadata

**需要运行**：2 个新实验（Partial 和 Minimal）
**预计时间**：10-12 小时

---

## 🚀 一键启动（推荐）

```bash
cd /root/DermAgent
bash start_stage2_final.sh
```

这个脚本会：
- 显示实验概览
- 要求你确认
- 后台启动实验
- 自动显示实时日志

---

## 📊 或者手动启动

```bash
cd /root/DermAgent
nohup bash scripts/run_stage2_baseline_vs_heuristic.sh > /tmp/stage2_run.log 2>&1 &
echo $! > /tmp/stage2_pid.txt

# 查看实时日志
tail -f /tmp/stage2_run.log
```

---

## 🔍 监控进度

### 查看实时日志

```bash
# 查看运行日志
tail -f /tmp/stage2_run.log

# 查看主实验日志（更详细）
tail -f /root/DermAgent/final-score/final_runs/stage2_baseline_vs_heuristic_*.log
```

### 查看单个实验日志

```bash
# Partial metadata 实验
tail -f /root/DermAgent/final-score/final_runs/stage2_partial_baseline_vs_heuristic/experiment_*.log

# Minimal metadata 实验
tail -f /root/DermAgent/final-score/final_runs/stage2_minimal_baseline_vs_heuristic/experiment_*.log
```

### 检查进程状态

```bash
# 查看是否还在运行
ps -p $(cat /tmp/stage2_pid.txt)

# 或者
ps aux | grep run_stage2_baseline_vs_heuristic.sh
```

---

## 📁 输出结果

所有结果保存在：`/root/DermAgent/final-score/final_runs/`

```
final_runs/
├── main_qwen_vs_agent_qwen/              # ✅ Full metadata (已有)
│   └── compare_agent_vs_qwen_*.json
├── heuristic_with_penalty/               # ✅ Full metadata (已有)
│   └── compare_agent_vs_qwen_*.json
├── stage2_partial_baseline_vs_heuristic/ # ⏳ Partial metadata (新)
│   ├── compare_agent_vs_qwen_*.json
│   └── experiment_*.log
├── stage2_minimal_baseline_vs_heuristic/ # ⏳ Minimal metadata (新)
│   ├── compare_agent_vs_qwen_*.json
│   └── experiment_*.log
└── stage2_baseline_vs_heuristic_*.log    # 主日志
```

---

## 📊 实验时间线

```
开始 ──────────────────────────────────────────────── 结束
 │                                                    │
 ├─ 实验1: Partial metadata (5-6小时)                │
 │   ├─ Direct baseline (345 cases)                  │
 │   └─ Heuristic agent (345 cases)                  │
 │                                                    │
 └─ 实验2: Minimal metadata (5-6小时)                │
     ├─ Direct baseline (345 cases)                  │
     └─ Heuristic agent (345 cases)                  │
                                                      │
                                    总计: 10-12 小时 ─┘
```

---

## 📈 预期结果

完成后，你将得到 3×2 的对比表格：

| 场景 | 配置 | Top-1 准确率 | 恶性召回率 | 错误率 | 性能下降 |
|------|------|-------------|-----------|--------|---------|
| Full | Direct baseline | 29.6% | 48.2% | 70.4% | - |
| Full | Heuristic | 39.7% | 82.2% | 60.3% | - |
| Partial | Direct baseline | ? | ? | ? | -X% |
| Partial | Heuristic | ? | ? | ? | -Y% |
| Minimal | Direct baseline | ? | ? | ? | -X% |
| Minimal | Heuristic | ? | ? | ? | -Y% |

**关键论点**：Heuristic 的性能下降幅度（Y%）应该小于 Direct baseline（X%）

---

## ⏹️ 停止运行

```bash
kill $(cat /tmp/stage2_pid.txt)
```

---

## ✅ 验证结果

实验完成后：

```bash
cd /root/DermAgent

# 检查结果文件是否存在
ls -lh final-score/final_runs/stage2_partial_baseline_vs_heuristic/compare_agent_vs_qwen_*.json
ls -lh final-score/final_runs/stage2_minimal_baseline_vs_heuristic/compare_agent_vs_qwen_*.json

# 查看主日志的总结
tail -50 final-score/final_runs/stage2_baseline_vs_heuristic_*.log
```

---

## ⚠️ 重要说明

### 1. Metadata Masking 实现

**当前状态**：脚本运行的是完整 metadata 的实验。

**真正的 metadata masking 需要**：
- 修改 `dataio/case_loader.py` 中的 `load_case_with_masking()` 函数
- 在评估流程中集成 metadata masking

**临时方案**：先运行完整实验，后续通过后处理或代码修改实现 masking。

### 2. 已有数据

Full metadata 的数据已经在阶段1完成：
- Direct baseline: `/root/DermAgent/final-score/final_runs/main_qwen_vs_agent_qwen/`
- Heuristic: `/root/DermAgent/final-score/final_runs/heuristic_with_penalty/`

**无需重跑！**

---

## 🔧 故障排查

### 问题1：脚本无法执行

```bash
chmod +x /root/DermAgent/scripts/run_stage2_baseline_vs_heuristic.sh
chmod +x /root/DermAgent/start_stage2_final.sh
```

### 问题2：Heuristic policy 不存在

```bash
# 检查配置文件
ls -la /root/DermAgent/configs/policies/example_candidate_policy.json
```

### 问题3：实验中断

```bash
# 查看哪个实验失败了
tail -100 /tmp/stage2_run.log

# 手动运行失败的实验
cd /root/DermAgent

# 例如：手动运行 Minimal metadata 实验
python scripts/compare_agent_vs_qwen.py \
    --limit 345 \
    --case-offset 0 \
    --policy-config configs/policies/example_candidate_policy.json \
    --output-dir /root/DermAgent/final-score/final_runs/stage2_minimal_baseline_vs_heuristic
```

---

## 📞 快速命令参考

```bash
# 启动
bash /root/DermAgent/start_stage2_final.sh

# 监控
tail -f /tmp/stage2_run.log

# 检查进程
ps -p $(cat /tmp/stage2_pid.txt)

# 停止
kill $(cat /tmp/stage2_pid.txt)

# 查看结果
ls -lh /root/DermAgent/final-score/final_runs/stage2_*/compare_agent_vs_qwen_*.json
```

---

**创建时间**: 2026-04-17  
**版本**: v2.0 (Direct baseline vs Heuristic only)  
**预计完成时间**: 10-12 小时
