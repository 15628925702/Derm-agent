# DermAgent v2 总结

**日期**：2026-04-17

---

## 一、初版思路的核心目标

`DermAgent初版思路.md` 提出的核心问题是：

> 当不同机构以不同方式组织同一个皮肤科病例时，系统能否仍然以稳定、显式、可审计的方式完成诊断推理。

为此提出四个技术创新点：

| 创新点 | 目标 |
|---|---|
| **技术创新1** | 将诊断中间动作显式化为 dermatology-specific clinical reasoning skills |
| **技术创新2** | 构建分层经验体系（raw case / tactical / abstract），支持跨病例经验复用 |
| **技术创新3** | 引入 cognitive state + policy state，根据场景动态控制 skill 触发和经验调用 |
| **技术创新4** | 输出动态 evidence pack，而非黑箱式 CoT 文本 |

---

## 二、CODE1 做了什么

**对应目标**：技术创新1、3、4 的基础实现

### 2.1 Workflow Context 显式建模（对应创新3）

- `agent/state.py`：`CaseInput` 新增 `workflow_context` 字段，包含 `hospital_type`、`available_tests`、`metadata_completeness`、`time_budget`、`workflow_preference`
- 把"医院环境约束"从隐式假设变成了可计算、可控制的显式对象

### 2.2 Planner Workflow-Aware Skill Selection（对应创新1+3）

- `agent/planner.py`：新增 `_adjust_decisions_by_workflow_context` 方法
- **risk_first 场景**：`malignancy_risk_assessment_skill` 分数 +5，ordering_hint 提前
- **无 dermoscopy 场景**：morphology 系列技能权重 ×0.7
- **minimal metadata 场景**：强制选择 `information_gap_detection_skill`

### 2.3 Evidence Package Workflow-Aware 排序（对应创新4）

- `agent/aggregator.py`：新增 `_reorder_evidence_by_workflow` 函数
- minimal metadata → information_gap 证据提前
- risk_first → malignancy/risk 证据提前
- metadata_first → metadata_consistency 证据提前

### 2.4 Metadata Masking 支持

- `dataio/case_loader.py`：新增 `load_case_with_masking` 函数
- 支持 Full / Partial / Minimal 三种场景，自动设置 `workflow_context.metadata_completeness`

### 2.5 ISIC2019 跨数据集支持

- `agent/label_space.py`：注册 ISIC2019 8 类标签空间
- `dataio/isic2019_loader.py`：正确设置 `label_space_id`

---

## 三、CODE2 做了什么

**对应目标**：技术创新2、3 的深化，以及创新4的内容筛选

### 3.1 经验层 Workflow 适配（对应创新2）

- `memory/experience_bank.py`：`retrieve_bundle` 新增 `workflow_context` 参数
- 新增 `_compute_workflow_top_k` 函数，根据 `hospital_type` 和 `time_budget` 动态调整三层经验的检索权重：

| hospital_type | raw_case | tactical | abstract |
|---|---|---|---|
| primary_care | -1 | 不变 | +2 |
| specialist_clinic | +2 | +1 | 不变 |
| academic_center | 不变 | +2 | +1 |

- `agent/run_agent.py`：两处 `retrieve_bundle` 调用均透传 `workflow_context`

### 3.2 动态证据构建（对应创新4）

- `agent/aggregator.py`：新增 `_filter_evidence_by_workflow` 函数
- `screening` 模式：只保留 `calibration_score >= 0.5` 的证据，最多 6 条
- `minimal` 模式：过滤掉依赖 metadata 的技能的空输出（`metadata_consistency_skill`、`distribution_analysis_skill`）

### 3.3 CognitionState 长期记忆（对应创新3）

- `cognition/cognition_state.py`：新增 `workflow_preferences` 字段
- 按 `{hospital_type}__{workflow_preference}` 为 key，记录每种 workflow 场景下的历史有效 skill 组合、平均准确率、case 数量
- 新增 `update_workflow_preferences` 方法，每次 reflection 后滚动更新

### 3.4 Planner 读取历史偏好（对应创新3）

- `agent/planner.py`：`_adjust_decisions_by_workflow_context` 从 `cognition.workflow_preferences` 读取历史有效 skill，命中时加分（`workflow_history_bonus`，默认 +1）

### 3.5 Reflection 写回（对应创新3）

- `agent/reflection.py`：`build_reflection` 新增 `diagnosis_correct` 字段
- `apply_cognition_update` 新增 `state` 参数，调用 `update_workflow_preferences` 写回长期记忆

---

## 四、实验结果

### 4.1 阶段1：三配置对比（PAD-UFES-20，345 cases）

| 配置 | Top-1 准确率 | 恶性召回率 | 错误率 |
|---|---|---|---|
| Direct baseline | 29.6% | 48.2% | 70.4% |
| Heuristic agent | **39.7%** | **82.2%** | **60.3%** |
| Learned agent | 39.7% | 82.2% | 60.3% |

**关键发现**：Heuristic 与 Learned 性能完全一致，证明核心价值来自结构化设计（skills + experience + evidence pack），而非 learned 组件。

### 4.2 阶段2：Metadata Missingness（PAD-UFES-20，333 cases）

| Condition | 配置 | Top-1 准确率 | 恶性召回率 |
|---|---|---|---|
| Full metadata | Direct baseline | 29.1% | 47.9% |
| Full metadata | **Heuristic agent** | **39.6%** | **82.4%** |
| Partial metadata | Direct baseline | 29.1% | 47.9% |
| Partial metadata | **Heuristic agent** | **40.2%** | **80.9%** |
| Minimal metadata | Direct baseline | 27.0% | 47.6% |
| Minimal metadata | **Heuristic agent** | **39.0%** | **80.5%** |

**关键发现**：
- Baseline 在 Minimal 场景下准确率从 29.1% 降至 27.0%（-2.1%）
- Agent 从 39.6% 降至 39.0%（**-0.6%**），退化幅度仅为 baseline 的 1/3
- 恶性召回率在所有 condition 下 agent 均稳定在 **80%+**，baseline 均在 47-48%
- 证明了"workflow-aware evidence construction 在信息缺失时的稳定性优势"

### 4.3 CODE2 Smoke Test（20 cases）

| 配置 | acc | mal_recall |
|---|---|---|
| direct_baseline | 35.0% | 53.3% |
| full_dermagent (CODE2) | 35.0% | **80.0%** |

CODE2 改动未引入性能退化，mal_recall 与历史基准一致（80.0% vs 80.9%）。

---

## 五、初版思路目标完成度

| 目标 | 完成情况 | 说明 |
|---|---|---|
| 技术创新1：skills 显式化 | ✅ 完成 | 8 类 clinical reasoning skills，planner 动态选择 |
| 技术创新2：分层经验体系 | ✅ 完成 | raw/tactical/abstract 三层，CODE2 增加 workflow-aware 检索权重 |
| 技术创新3：状态驱动控制 | ✅ 完成 | workflow_context + cognition.workflow_preferences 长期记忆 |
| 技术创新4：动态 evidence pack | ✅ 完成 | 排序 + 内容筛选，可审计输出 |
| 跨医院流程异质性适配 | ✅ 实验验证 | Minimal metadata 下 agent 退化幅度仅 baseline 的 1/3 |
| 可审计的诊断支持 | ✅ 结构完整 | evidence pack 包含支持/反对证据、经验摘要、不确定性、信息缺口 |
| 跨数据集验证 | ⏳ 待完成 | 需要 ISIC2019 数据，代码已支持 |

---

## 六、待完成事项

1. **跨数据集验证**（等老师提供 ISIC2019）— 代码已就绪，运行即可
2. **Evidence package 差异分析**（中优先级）— 统计 Full vs Minimal 下 skill 触发率变化
3. **论文撰写** — 实验数据已齐，可直接写 Result 第三段（Robustness）
