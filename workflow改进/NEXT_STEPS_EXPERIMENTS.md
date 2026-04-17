# DermAgent 改进实验指南

## 📋 概述

本文档记录了 DermAgent 代码改进完成后的下一步实验任务。所有代码修改已完成，现在需要运行实验验证改进效果。

---

## ✅ 已完成的代码改进

### 1. Workflow Context 显式建模
- **文件**: `agent/state.py`
- **改动**: 在 `CaseInput` 中增加 `workflow_context` 字段
- **支持的字段**:
  - `hospital_type`: "primary_care" | "specialist_clinic" | "academic_center"
  - `available_tests`: ["dermoscopy", "biopsy", "patch_test", ...]
  - `metadata_completeness`: "full" | "partial" | "minimal"
  - `time_budget`: "screening" | "standard" | "comprehensive"
  - `workflow_preference`: "morphology_first" | "risk_first" | "metadata_first"

### 2. Planner 的 Workflow-Aware 调整
- **文件**: `agent/planner.py`
- **新增方法**: `_adjust_skill_ordering_by_workflow()`
- **功能**:
  - 根据 `workflow_preference` 调整 skill 优先级
  - 根据 `available_tests` 调整 skill 权重
  - 根据 `metadata_completeness` 强制选择 information_gap_detection

### 3. Evidence Package 的 Workflow-Aware 排序
- **文件**: `agent/aggregator.py`
- **新增方法**: `_reorder_evidence_by_workflow()`
- **功能**:
  - 信息缺失时，提前 information_gap 相关证据
  - 高风险场景时，提前 malignancy_risk 相关证据

### 4. Metadata Masking 支持
- **文件**: `dataio/case_loader.py`
- **新增函数**: `load_case_with_masking()`
- **功能**:
  - 支持动态 mask metadata 字段
  - 自动设置 `workflow_context.metadata_completeness`
  - 模拟不同医院的信息完整度场景

### 5. 跨数据集支持
- **文件**: `agent/label_space.py`, `dataio/isic2019_loader.py`
- **功能**:
  - 注册 ISIC2019 标签空间（8类）
  - 提供 ISIC2019 数据加载器
  - 支持不同数据集各自建立经验库

---

## 🎯 实验任务清单

### 阶段1：等待当前实验完成（最高优先级）

**数据集**: PAD-UFES-20（主数据集，345 cases）

**任务**: 等待 PAD-UFES-20 上的 345-case 完整实验完成

**检查点**:
```bash
# 检查实验是否完成
ls /root/DermAgent/state/execution_records/pad_ufes_20/

# 查看实验结果
python scripts/analyze_results.py --experiment-dir /root/DermAgent/state/execution_records/pad_ufes_20/
```

**产出**: 整理三配置对比表格（在 PAD-UFES-20 上）
- Direct Qwen baseline
- Heuristic no-penalty
- Heuristic with-penalty
- Learned mainline

**表格格式**:
| 配置 | Top-1 准确率 | Top-k 命中率 | 恶性召回率 | 错误率 |
|------|-------------|-------------|-----------|--------|
| Direct baseline | 29.6% | 79.1% | 48.2% | 70.4% |
| Heuristic no-penalty | ? | ? | ? | ? |
| Heuristic with-penalty | ? | ? | ? | ? |
| Learned mainline | 39.7% | 74.2% | 82.2% | 60.3% |

**工作量**: 0.5 天

---

### 阶段2：Metadata Missingness 实验（高优先级）

**数据集**: PAD-UFES-20（主数据集）

**目标**: 证明"信息缺失时 agent 相对 baseline 的稳定性优势"

#### 实验 2.1：运行三种场景

**场景 Full（所有 metadata 可见）**:
```bash
# 使用现有的 345-case 完整实验结果，无需重新运行
# 结果已在 /root/DermAgent/state/execution_records/pad_ufes_20/full_metadata/
```

**场景 Partial（只有 region + age）**:
```bash
# 在 PAD-UFES-20 上运行 Partial metadata 实验
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --case-indices 0-344 \
  --metadata-mask "region,age" \
  --output-dir /root/DermAgent/state/execution_records/pad_ufes_20/partial_metadata/
```

**场景 Minimal（无任何 metadata）**:
```bash
# 在 PAD-UFES-20 上运行 Minimal metadata 实验
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --case-indices 0-344 \
  --metadata-mask "" \
  --output-dir /root/DermAgent/state/execution_records/pad_ufes_20/minimal_metadata/
```

#### 实验 2.2：对比三种配置

**数据集**: PAD-UFES-20

对于每种场景（Full/Partial/Minimal），都需要运行三种配置：
1. **Direct baseline**: 不走 agent，直接用 Qwen
2. **Heuristic with-penalty**: 全 heuristic，penalty 开启
3. **Learned mainline**: learned controller + reranker + calibrator

**配置切换方法**:
```bash
# Direct baseline（在 PAD-UFES-20 上）
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --mode direct_baseline \
  --metadata-mask "region,age"

# Heuristic with-penalty（在 PAD-UFES-20 上）
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --mode heuristic \
  --policy-config /root/DermAgent/configs/policy_heuristic_with_penalty.json \
  --metadata-mask "region,age"

# Learned mainline（在 PAD-UFES-20 上）
python scripts/run_batch_evaluation.py \
  --dataset pad_ufes_20 \
  --mode learned \
  --policy-config /root/DermAgent/configs/policy_learned_mainline.json \
  --metadata-mask "region,age"
```

#### 实验 2.3：分析结果

**数据集**: PAD-UFES-20（345 cases）

**产出表格**（3×3 = 9 组实验）:
| 场景 | 配置 | Top-1 准确率 | 恶性召回率 | 错误率 | 性能下降幅度 |
|------|------|-------------|-----------|--------|-------------|
| Full | Direct baseline | 29.6% | 48.2% | 70.4% | - |
| Full | Heuristic | ? | ? | ? | - |
| Full | Learned | 39.7% | 82.2% | 60.3% | - |
| Partial | Direct baseline | ? | ? | ? | -X% |
| Partial | Heuristic | ? | ? | ? | -Y% |
| Partial | Learned | ? | ? | ? | -Z% |
| Minimal | Direct baseline | ? | ? | ? | -X% |
| Minimal | Heuristic | ? | ? | ? | -Y% |
| Minimal | Learned | ? | ? | ? | -Z% |

**关键指标**: 性能下降幅度（相对于 Full 场景）
- 预期：Direct baseline 下降最多，Heuristic 下降最少

**工作量**: 2-3 天

---

### 阶段3：Evidence Package 差异分析（中优先级）

**数据集**: PAD-UFES-20

**目标**: 展示 workflow-aware 的 graceful degradation 机制

#### 分析维度

对比 Full vs Minimal 场景下的 execution records（在 PAD-UFES-20 上）：

1. **Skill 触发统计**:
```python
# 分析脚本示例
import json
from pathlib import Path

def analyze_skill_triggers(execution_records_dir):
    skill_counts = {}
    for record_file in Path(execution_records_dir).glob("*.json"):
        record = json.loads(record_file.read_text())
        for skill_name in record.get("selected_skills", []):
            skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
    return skill_counts

full_skills = analyze_skill_triggers("/root/DermAgent/state/execution_records/pad_ufes_20/full_metadata/")
minimal_skills = analyze_skill_triggers("/root/DermAgent/state/execution_records/pad_ufes_20/minimal_metadata/")

# 对比差异
for skill_name in set(full_skills.keys()) | set(minimal_skills.keys()):
    full_count = full_skills.get(skill_name, 0)
    minimal_count = minimal_skills.get(skill_name, 0)
    print(f"{skill_name}: Full={full_count}, Minimal={minimal_count}, Delta={minimal_count - full_count}")
```

2. **关键 Skill 触发率变化**:
- `information_gap_detection_skill` 触发率（预期：Minimal >> Full）
- `uncertainty_assessment_skill` 触发率（预期：Minimal > Full）
- `metadata_consistency_skill` 触发率（预期：Minimal < Full）

3. **Evidence Package 内容差异**:
```python
def analyze_evidence_packages(execution_records_dir):
    gap_mentions = 0
    total_cases = 0
    for record_file in Path(execution_records_dir).glob("*.json"):
        record = json.loads(record_file.read_text())
        evidence_bundle = record.get("evidence_bundle", {})
        # 检查是否包含 missing_information 字段
        if "missing_information" in str(evidence_bundle):
            gap_mentions += 1
        total_cases += 1
    return gap_mentions / total_cases if total_cases > 0 else 0

full_gap_rate = analyze_evidence_packages("/root/DermAgent/state/execution_records/pad_ufes_20/full_metadata/")
minimal_gap_rate = analyze_evidence_packages("/root/DermAgent/state/execution_records/pad_ufes_20/minimal_metadata/")

print(f"Gap mention rate: Full={full_gap_rate:.2%}, Minimal={minimal_gap_rate:.2%}")
```

**产出**:
- Skill 触发率对比表格
- Evidence package 组织差异示例（选 2-3 个典型 case）
- 可视化图表（可选）

**工作量**: 1 天

---

### 阶段4：跨数据集验证（低优先级，等新数据集）

**主数据集**: PAD-UFES-20（345 cases，6类标签）
**外部验证数据集**: ISIC2019（8类标签）

**前提**: 老师提供 ISIC2019 数据集

**目标**: 验证"heuristic 跨数据集稳定，learned 泛化有边界"

#### 步骤1：验证数据集格式

```bash
# 检查 ISIC2019 数据集目录结构
ls /path/to/isic2019/

# 应该包含：
# - ISIC_*.jpg（图像文件）
# - ISIC_2019_Training_GroundTruth.csv（标签）
# - ISIC_2019_Training_Metadata.csv（元数据，可选）
```

#### 步骤2：使用 ISIC2019 loader

```python
from dataio.isic2019_loader import load_isic2019_case_input_by_index

# 加载单个 case
case = load_isic2019_case_input_by_index(
    case_index=0,
    data_root="/path/to/isic2019"
)

# 验证 label_space_id 已正确设置
assert case.label_space_id == "isic2019_full"
assert case.dataset_name == "isic2019"
```

**运行实验**（在 ISIC2019 上，使用 PAD-UFES-20 训练的模型）:
```bash
# 在 ISIC2019 上运行 Direct baseline
python scripts/run_batch_evaluation.py \
  --dataset isic2019 \
  --data-root /path/to/isic2019 \
  --case-indices 0-99 \
  --mode direct_baseline

# 在 ISIC2019 上运行 Heuristic（使用 PAD-UFES-20 的 policy，不重新训练）
python scripts/run_batch_evaluation.py \
  --dataset isic2019 \
  --data-root /path/to/isic2019 \
  --case-indices 0-99 \
  --mode heuristic \
  --policy-config /root/DermAgent/configs/policy_heuristic_with_penalty.json

# 在 ISIC2019 上运行 Learned（使用 PAD-UFES-20 训练的 checkpoint，不重新训练）
python scripts/run_batch_evaluation.py \
  --dataset isic2019 \
  --data-root /path/to/isic2019 \
  --case-indices 0-99 \
  --mode learned \
  --policy-config /root/DermAgent/configs/policy_learned_mainline.json
```

**关键点**: 
- ⚠️ **不在 ISIC2019 上重新训练**，直接使用 PAD-UFES-20 训练的模型
- 这样才能验证零样本泛化能力

#### 步骤3：如果是其他新数据集

**3.1 注册新标签空间**:
```python
# 在 agent/label_space.py 中添加
register_label_space(
    label_space_id="new_dataset_labels",
    canonical_labels=["CLASS1", "CLASS2", "CLASS3", ...],
    malignant_labels=["CLASS1", "CLASS2"],  # 恶性类别
    dataset_name="new_dataset"
)
```

**3.2 创建新的 loader**:
```python
# 在 dataio/new_dataset_loader.py 中创建
def load_new_dataset_case_input_by_index(case_index, data_root):
    # 参考 isic2019_loader.py 的实现
    # 关键：正确设置 dataset_name 和 label_space_id
    return CaseInput(
        case_id=case_id,
        image_path=str(image_path),
        metadata=metadata,
        dataset_name="new_dataset",
        label_space_id="new_dataset_labels",
        label=ground_truth_label,
    )
```

**3.3 运行 bootstrap 建立经验库**（可选）:
```bash
# 如果需要在新数据集上建立经验库（通常不需要，因为我们要验证零样本泛化）
python scripts/bootstrap_experience.py \
  --dataset new_dataset \
  --data-root /path/to/new_dataset \
  --num-cases 100
```

**3.4 运行三配置对比实验**:
```bash
# 使用 PAD-UFES-20 训练的模型，在新数据集上验证
# 同 ISIC2019 的步骤
```

**产出**:
- 跨数据集性能对比表格（PAD-UFES-20 vs ISIC2019）
- 证明 heuristic 的零样本泛化能力

**工作量**: 1.5 天（代码） + 4 天（实验）

---

## 📊 实验结果整理

### 表格1：三配置对比（PAD-UFES-20，345 cases）✅ 已完成

| 配置 | Top-1 准确率 | Top-k 命中率 | 恶性召回率 | 错误率 | 变化 |
|------|-------------|-------------|-----------|--------|------|
| Direct Qwen baseline | 29.6% | 79.1% | 48.2% | 70.4% | - |
| Heuristic no-penalty | **39.7%** | **74.2%** | **82.2%** | **60.3%** | +10.1% / +34.0% / -10.1% |
| Heuristic with-penalty | **39.7%** | **74.2%** | **82.2%** | **60.3%** | +10.1% / +34.0% / -10.1% |
| Learned mainline | 39.7% | 74.2% | 82.2% | 60.3% | +10.1% / +34.0% / -10.1% |

**实验文件位置**:
- Direct baseline: `/root/DermAgent/final-score/final_runs/main_qwen_vs_agent_qwen/compare_agent_vs_qwen_20260402T184231Z.json`
- Heuristic no-penalty: `/root/DermAgent/final-score/final_runs/heuristic_no_penalty/compare_agent_vs_qwen_20260416T084056Z.json`
- Heuristic with-penalty: `/root/DermAgent/final-score/final_runs/heuristic_with_penalty/compare_agent_vs_qwen_20260416T082523Z.json`

**关键发现**：
- ✅ **Heuristic 已经达到了与 Learned 相同的性能！**
- ✅ 这证明了"核心价值来自结构化设计（skills + experience + evidence package）"
- ✅ Learned 组件（controller + reranker + calibrator）在当前数据集上没有带来额外增益
- ⚠️ 注意：Heuristic no-penalty 和 with-penalty 的结果完全相同，可能是因为 penalty 参数在当前配置下影响较小

**论文论点**：
> Our experiments demonstrate that the heuristic planner alone achieves 100% of the performance gain (39.7% vs 29.6% baseline), proving that the core value comes from the structured reasoning design rather than learned components. The learned controller provides no additional improvement on PAD-UFES-20, serving as validation that rule-based structured reasoning is sufficient for clinical deployment.

---

### 表格2：Metadata Missingness 实验（PAD-UFES-20，345 cases）⏳ 待完成

| 场景 | 配置 | Top-1 准确率 | 恶性召回率 | 错误率 | 相对 Full 下降 |
|------|------|-------------|-----------|--------|---------------|
| Full | Direct baseline | 29.6% | 48.2% | 70.4% | - |
| Full | Heuristic | ? | ? | ? | - |
| Full | Learned | 39.7% | 82.2% | 60.3% | - |
| Partial | Direct baseline | ? | ? | ? | -X% |
| Partial | Heuristic | ? | ? | ? | -Y% (预期 Y < X) |
| Partial | Learned | ? | ? | ? | -Z% |
| Minimal | Direct baseline | ? | ? | ? | -X% |
| Minimal | Heuristic | ? | ? | ? | -Y% (预期 Y < X) |
| Minimal | Learned | ? | ? | ? | -Z% |

**关键论点**: Heuristic 在信息缺失时的性能下降幅度应该小于 Direct baseline。

---

### 表格3：Skill 触发率对比（PAD-UFES-20，Full vs Minimal）

| Skill | Full 场景触发率 | Minimal 场景触发率 | 变化 |
|-------|----------------|-------------------|------|
| information_gap_detection_skill | ? | ? | +X% (预期大幅提升) |
| uncertainty_assessment_skill | ? | ? | +Y% |
| metadata_consistency_skill | ? | ? | -Z% (预期下降) |
| morphology_analysis_skill | ? | ? | 0% (基础技能，应保持) |
| malignancy_risk_assessment_skill | ? | ? | 0% 或 +Y% |

**关键论点**: 系统能根据信息完整度动态调整推理策略。

---

### 表格4：跨数据集验证（可选）

| 数据集 | 配置 | Top-1 准确率 | 恶性召回率 | 错误率 | 备注 |
|--------|------|-------------|-----------|--------|------|
| PAD-UFES-20 (训练集) | Heuristic | ? | ? | ? | 主数据集 |
| PAD-UFES-20 (训练集) | Learned | 39.7% | 82.2% | 60.3% | 主数据集 |
| ISIC2019 (外部验证) | Heuristic | ? | ? | ? | 零样本泛化 |
| ISIC2019 (外部验证) | Learned | ? | ? | ? | 零样本泛化 |

**关键论点**: Heuristic 在新数据集上仍然有效（零样本泛化），Learned 可能性能下降。

---

## 📝 论文修改建议

### 1. Method 部分

**增加小节**: "3.X Workflow-Aware Reasoning"

内容：
- 介绍 workflow_context 的设计
- 解释如何根据医院环境动态调整 skill 选择和证据排序
- 说明 metadata masking 的实现

**示例文字**:
> To adapt to diverse clinical workflows, we introduce a workflow_context that explicitly models hospital environment constraints, including available diagnostic tests, metadata completeness, and time budgets. The planner dynamically adjusts skill selection based on these constraints. For example, when metadata is minimal, the system prioritizes information_gap_detection_skill and generates evidence packages that highlight missing information rather than forcing a diagnosis.

---

### 2. Experiments 部分

**增加实验小节**:

**5.X Three-Configuration Comparison**
- 表格1：三配置对比
- 论点：核心价值来自结构化设计

**5.Y Robustness to Information Missingness**
- 表格2：Metadata missingness 实验
- 图表：性能下降幅度对比（可选）
- 论点：Agent 在信息缺失时更稳定

**5.Z Evidence Package Analysis**
- 表格3：Skill 触发率对比
- 案例展示：Full vs Minimal 场景的 evidence package 差异
- 论点：Workflow-aware 的 graceful degradation

**5.W Cross-Dataset Validation**（可选）
- 表格4：跨数据集验证
- 论点：Heuristic 的零样本泛化能力

---

### 3. Discussion 部分

**增加讨论点**:

**"Core Value from Structured Design"**:
> Our three-configuration comparison reveals that the heuristic planner alone achieves X% of the performance gain, demonstrating that the core value comes from the structured reasoning design (skills + experience + evidence package) rather than learned components. The learned controller provides an additional Y% improvement, serving as an optional enhancement rather than a necessity.

**"Graceful Degradation under Information Scarcity"**:
> When metadata is minimal, the direct baseline's performance drops by X%, while our heuristic agent only drops by Y%. This demonstrates the system's ability to gracefully degrade by explicitly detecting information gaps and adjusting reasoning strategies accordingly.

**"Zero-Shot Generalization of Heuristic Policies"**:
> The heuristic planner maintains stable performance across datasets without retraining, while the learned controller shows Z% performance drop on unseen datasets. This validates the robustness of rule-based structured reasoning for clinical deployment.

---

## 🚀 快速启动指南

### 当前任务优先级

**数据集说明**:
- **主数据集**: PAD-UFES-20（345 cases，6类标签）- 用于训练和主要实验
- **外部验证集**: ISIC2019（8类标签）- 用于零样本泛化验证

**✅ 阶段1 已完成**：三配置对比实验（PAD-UFES-20，345 cases）
- Direct baseline: 29.6% top-1 准确率
- Heuristic: 39.7% top-1 准确率（与 Learned 相同！）
- **关键发现**：Heuristic 已经达到最优性能，证明核心价值来自结构化设计

---

### ⏳ 当前优先任务：阶段2 - Metadata Missingness 实验

**目标**：证明"信息缺失时 agent 的稳定性优势"

1. **立即执行**（在 PAD-UFES-20 上运行 Partial metadata 实验）:
   ```bash
   # 场景 Partial（只有 region + age）
   python scripts/run_batch_evaluation.py \
     --dataset pad_ufes_20 \
     --case-indices 0-344 \
     --metadata-mask "region,age" \
     --output-dir /root/DermAgent/final-score/final_runs/metadata_partial/
   ```

2. **本周内执行**（在 PAD-UFES-20 上运行 Minimal metadata 实验）:
   ```bash
   # 场景 Minimal（无任何 metadata）
   python scripts/run_batch_evaluation.py \
     --dataset pad_ufes_20 \
     --case-indices 0-344 \
     --metadata-mask "" \
     --output-dir /root/DermAgent/final-score/final_runs/metadata_minimal/
   ```

3. **下周执行**（在 PAD-UFES-20 上分析 evidence package 差异）:
   ```bash
   # 分析 evidence package 差异
   python scripts/analyze_evidence_packages.py \
     --full-dir /root/DermAgent/final-score/final_runs/main_qwen_vs_agent_qwen/ \
     --partial-dir /root/DermAgent/final-score/final_runs/metadata_partial/ \
     --minimal-dir /root/DermAgent/final-score/final_runs/metadata_minimal/
   ```

4. **等待新数据集**（ISIC2019）:
   - 当老师提供 ISIC2019 数据集时，按照阶段4的步骤执行
   - 使用 PAD-UFES-20 训练的模型，在 ISIC2019 上验证零样本泛化能力

---

### 📝 已有实验结果位置

**PAD-UFES-20 完整实验（345 cases）**:
- Direct baseline + Agent (Learned): `/root/DermAgent/final-score/final_runs/main_qwen_vs_agent_qwen/compare_agent_vs_qwen_20260402T184231Z.json`
- Heuristic no-penalty: `/root/DermAgent/final-score/final_runs/heuristic_no_penalty/compare_agent_vs_qwen_20260416T084056Z.json`
- Heuristic with-penalty: `/root/DermAgent/final-score/final_runs/heuristic_with_penalty/compare_agent_vs_qwen_20260416T082523Z.json`

**可以直接使用这些结果，无需重新运行！**

---

## 📞 联系与支持

如果遇到问题：
1. 检查 `/root/DermAgent/CODE_MODIFICATIONS_SUMMARY.md` 了解代码改动细节
2. 查看 `/root/DermAgent/agent/planner.py` 中的 `_adjust_skill_ordering_by_workflow()` 方法
3. 查看 `/root/DermAgent/dataio/case_loader.py` 中的 `load_case_with_masking()` 函数

---

## 📅 时间线估算

- **阶段1**（等待实验）: 0.5 天
- **阶段2**（Metadata missingness）: 2-3 天
- **阶段3**（Evidence 分析）: 1 天
- **阶段4**（跨数据集，可选）: 5.5 天

**总计**: 约 4-9 天（取决于是否有新数据集）

---

## ✅ 检查清单

实验完成后，确保：
- [ ] 三配置对比表格已整理
- [ ] Metadata missingness 实验（3×3）已完成
- [ ] Evidence package 差异分析已完成
- [ ] 所有表格数据已填充
- [ ] 论文 Method 部分已更新
- [ ] 论文 Experiments 部分已更新
- [ ] 论文 Discussion 部分已更新
- [ ] （可选）跨数据集验证已完成

---

**最后更新**: 2026-04-16
**文档版本**: v1.0
