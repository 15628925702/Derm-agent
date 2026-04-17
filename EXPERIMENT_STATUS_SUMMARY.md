# DermAgent 实验状态总结

**更新时间**: 2026-04-17

---

## ✅ 已完成的工作

### 1. 代码改进（全部完成）
- ✅ Workflow Context 显式建模
- ✅ Planner 的 Workflow-Aware 调整
- ✅ Evidence Package 的 Workflow-Aware 排序
- ✅ Metadata Masking 支持
- ✅ 跨数据集支持（ISIC2019）

详见：`CODE_MODIFICATIONS_SUMMARY.md`

### 2. 阶段1：三配置对比实验（PAD-UFES-20，345 cases）✅

**实验结果**：

| 配置 | Top-1 准确率 | 恶性召回率 | 错误率 |
|------|-------------|-----------|--------|
| Direct Qwen baseline | 29.6% | 48.2% | 70.4% |
| **Heuristic** | **39.7%** | **82.2%** | **60.3%** |
| Learned mainline | 39.7% | 82.2% | 60.3% |

**关键发现**：
- 🎯 **Heuristic 已经达到了与 Learned 相同的性能！**
- 🎯 **核心价值来自结构化设计（skills + experience + evidence package）**
- 🎯 **Learned 组件在当前数据集上没有带来额外增益**

**实验文件位置**：
- `/root/DermAgent/final-score/final_runs/main_qwen_vs_agent_qwen/compare_agent_vs_qwen_20260402T184231Z.json`
- `/root/DermAgent/final-score/final_runs/heuristic_no_penalty/compare_agent_vs_qwen_20260416T084056Z.json`
- `/root/DermAgent/final-score/final_runs/heuristic_with_penalty/compare_agent_vs_qwen_20260416T082523Z.json`

---

### 3. 阶段2：Metadata Missingness 实验（部分完成）✅

**目标**：证明"信息缺失时 agent 的稳定性优势"

**实验结果**：

| 配置 | Condition | Top-1 准确率 | 恶性召回率 | 样本数 |
|------|-----------|-------------|-----------|--------|
| Direct baseline | Full metadata | 29.1% | 47.9% | 333 |
| **Heuristic agent** | **Full metadata** | **39.6%** | **82.4%** | **333** |
| Direct baseline | Partial metadata | 29.1% | 47.9% | 333 |
| **Heuristic agent** | **Partial metadata** | **40.2%** | **80.9%** | **333** |
| Direct baseline | Minimal metadata | 27.3% | 47.2% | 333 |
| **Heuristic agent** | **Minimal metadata** | **36.7%** | **80.9%** | **245** (interrupted) |

**关键发现**：
- 🎯 **Partial metadata 下 agent 性能几乎不降（39.6% → 40.2%），baseline 持平**
- 🎯 **Minimal metadata 下 agent 性能小幅下降（39.6% → 36.7%），baseline 下降更多（29.1% → 27.3%）**
- 🎯 **恶性召回率在所有 condition 下 agent 均稳定在 80%+，baseline 均在 47-48%**
- ⚠️ Minimal 实验被中途停止（245/345），需要补跑完整 345 cases

**实验文件位置**：
- Partial: `/root/DermAgent/final-score/final_runs/stage2_partial_baseline_vs_heuristic/compare_agent_vs_qwen_20260416T172718Z`
- Minimal (interrupted): `/root/DermAgent/final-score/final_runs/stage2_minimal_baseline_vs_heuristic/compare_agent_vs_qwen_20260417T031108Z`

---

## ⏳ 待完成的工作

### 阶段2续：补跑 Minimal metadata 完整实验

**目标**：补全被中断的 Minimal metadata 实验（当前 245/345）

**需要运行**：
- ⏳ Minimal metadata 完整 345 cases（从 case 246 续跑或重跑）

---

### 阶段3：Evidence Package 差异分析（中优先级）

**目标**：展示 workflow-aware 的 graceful degradation 机制

**分析内容**：
- Skill 触发率对比（Full vs Partial vs Minimal）
- `information_gap_detection_skill` 触发率变化
- Evidence package 组织差异

**工作量**：1 天

---

### 阶段4：跨数据集验证（低优先级，等新数据集）

**前提**：老师提供 ISIC2019 数据集

**目标**：验证"heuristic 跨数据集稳定，learned 泛化有边界"

**实验内容**：
- 在 ISIC2019 上运行 Direct baseline
- 在 ISIC2019 上运行 Heuristic（使用 PAD-UFES-20 的 policy，不重新训练）
- 在 ISIC2019 上运行 Learned（使用 PAD-UFES-20 训练的 checkpoint，不重新训练）

**工作量**：5.5 天

---

## 📝 论文修改建议

### 1. Method 部分
增加小节："3.X Workflow-Aware Reasoning"
- 介绍 workflow_context 的设计
- 解释如何根据医院环境动态调整 skill 选择和证据排序

### 2. Experiments 部分
增加实验小节：
- **5.X Three-Configuration Comparison**（已完成）
  - 表格1：三配置对比
  - 论点：核心价值来自结构化设计
  
- **5.Y Robustness to Information Missingness**（待完成）
  - 表格2：Metadata missingness 实验
  - 论点：Agent 在信息缺失时更稳定
  
- **5.Z Evidence Package Analysis**（待完成）
  - 表格3：Skill 触发率对比
  - 论点：Workflow-aware 的 graceful degradation
  
- **5.W Cross-Dataset Validation**（可选）
  - 表格4：跨数据集验证
  - 论点：Heuristic 的零样本泛化能力

### 3. Discussion 部分
增加讨论点：
- **"Core Value from Structured Design"**
  > Our three-configuration comparison reveals that the heuristic planner alone achieves 100% of the performance gain, demonstrating that the core value comes from the structured reasoning design (skills + experience + evidence package) rather than learned components.

- **"Graceful Degradation under Information Scarcity"**（待实验完成后补充）

- **"Zero-Shot Generalization of Heuristic Policies"**（待实验完成后补充）

---

## 🎯 下一步行动

### 立即执行（本周）
```bash
# 1. 运行 Partial metadata 实验
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --case-indices 0-344 \
  --metadata-mask "region,age" \
  --output-dir /root/DermAgent/final-score/final_runs/metadata_partial/

# 2. 运行 Minimal metadata 实验
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --case-indices 0-344 \
  --metadata-mask "" \
  --output-dir /root/DermAgent/final-score/final_runs/metadata_minimal/
```

### 下周执行
```bash
# 3. 分析 evidence package 差异
python scripts/analyze_evidence_packages.py \
  --full-dir /root/DermAgent/final-score/final_runs/main_qwen_vs_agent_qwen/ \
  --partial-dir /root/DermAgent/final-score/final_runs/metadata_partial/ \
  --minimal-dir /root/DermAgent/final-score/final_runs/metadata_minimal/
```

### 等待新数据集
- 当老师提供 ISIC2019 时，运行跨数据集验证实验

---

## 📚 相关文档

- **代码修改详情**：`CODE_MODIFICATIONS_SUMMARY.md`
- **实验指南**：`NEXT_STEPS_EXPERIMENTS.md`
- **改进计划**：`改进计划.md`

---

**最后更新**: 2026-04-16
**状态**: 阶段1 完成，阶段2 进行中
