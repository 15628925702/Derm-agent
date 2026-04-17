# DermAgent CODE2 设计文档

## 修改日期
2026-04-17

## 背景
CODE1.md 完成了 workflow_context 的端到端透传、planner skill 调整、evidence 排序调整、metadata masking。
CODE2 在此基础上补齐四个缺口：

1. **经验层 workflow 适配**：ExperienceBank.retrieve_bundle 不感知 workflow_context
2. **动态证据构建**：aggregator 只做排序，没有内容筛选
3. **长期记忆缺失**：CognitionState 没有记录 workflow 相关的长期经验
4. **实验验证**：需要运行 Full / Partial / Minimal 对比实验

---

## 一、经验层 workflow 适配

### 1.1 问题描述

`ExperienceBank.retrieve_bundle` 当前签名：
```python
def retrieve_bundle(self, *, case_state, top_k_raw=2, top_k_tactical=4, top_k_abstract=4, ...)
```
三层经验（raw_case / tactical / abstract）的 top_k 是固定的，不感知 workflow_context。

**目标**：根据 hospital_type 和 time_budget 动态调整三层经验的检索权重。

设计逻辑：
- `primary_care`（基层医院）：医生经验少，更需要 abstract experience（通用规律）→ top_k_abstract ↑
- `specialist_clinic`（专科诊所）：需要具体病例参考 → top_k_raw ↑
- `academic_center`（学术中心）：需要战术经验和鉴别推理 → top_k_tactical ↑
- `time_budget="screening"`：快速筛查，压缩所有 top_k
- `time_budget="comprehensive"`：全面检查，放大所有 top_k

### 1.2 要改的文件

#### `memory/experience_bank.py`

在 `retrieve_bundle` 方法中增加 `workflow_context: dict | None = None` 参数，调用前先计算有效 top_k：

```python
def retrieve_bundle(
    self,
    *,
    case_state=None,
    workflow_context: dict | None = None,
    top_k_raw: int = 2,
    top_k_tactical: int = 4,
    top_k_abstract: int = 4,
    top_k_merged: int = 6,
    ...
) -> dict:
    effective_top_k = _compute_workflow_top_k(
        workflow_context=workflow_context,
        top_k_raw=top_k_raw,
        top_k_tactical=top_k_tactical,
        top_k_abstract=top_k_abstract,
        top_k_merged=top_k_merged,
    )
    return self.retriever.retrieve_bundle(
        case_state=case_state,
        top_k_raw=effective_top_k["raw"],
        top_k_tactical=effective_top_k["tactical"],
        top_k_abstract=effective_top_k["abstract"],
        top_k_merged=effective_top_k["merged"],
        ...
    )
```

新增私有函数 `_compute_workflow_top_k`（放在文件底部）：

```python
def _compute_workflow_top_k(
    workflow_context: dict | None,
    top_k_raw: int,
    top_k_tactical: int,
    top_k_abstract: int,
    top_k_merged: int,
) -> dict[str, int]:
    if not workflow_context:
        return {"raw": top_k_raw, "tactical": top_k_tactical,
                "abstract": top_k_abstract, "merged": top_k_merged}

    hospital_type = str(workflow_context.get("hospital_type", "")).strip()
    time_budget = str(workflow_context.get("time_budget", "")).strip()

    raw, tactical, abstract = top_k_raw, top_k_tactical, top_k_abstract

    # hospital_type 调整
    if hospital_type == "primary_care":
        abstract = min(abstract + 2, 8)
        raw = max(raw - 1, 1)
    elif hospital_type == "specialist_clinic":
        raw = min(raw + 2, 6)
        tactical = min(tactical + 1, 6)
    elif hospital_type == "academic_center":
        tactical = min(tactical + 2, 8)
        abstract = min(abstract + 1, 6)

    # time_budget 调整（乘以系数后取整）
    if time_budget == "screening":
        raw = max(int(raw * 0.5), 1)
        tactical = max(int(tactical * 0.5), 1)
        abstract = max(int(abstract * 0.5), 1)
    elif time_budget == "comprehensive":
        raw = min(int(raw * 1.5), 8)
        tactical = min(int(tactical * 1.5), 8)
        abstract = min(int(abstract * 1.5), 8)

    merged = min(raw + tactical + abstract, top_k_merged + 4)
    return {"raw": raw, "tactical": tactical, "abstract": abstract, "merged": merged}
```

#### `agent/run_agent.py`

在两处调用 `bank.retrieve_bundle` 的地方，增加 `workflow_context=case_input.workflow_context` 参数（第一次检索在 planner 前，第二次在 skill 执行后）。

---

## 二、动态证据构建（内容筛选）

### 2.1 问题描述

`_reorder_evidence_by_workflow` 只做排序，不做内容筛选。
目标：根据 `time_budget` 和 `metadata_completeness` 决定证据包的**内容深度**，而不只是顺序。

设计逻辑：
- `time_budget="screening"`：只保留高置信度证据（calibration_score > 阈值），截断低优先级技能输出
- `time_budget="comprehensive"`：保留所有证据，包括低置信度的 uncertainty 和 information_gap
- `metadata_completeness="minimal"`：过滤掉依赖 metadata 的技能输出（如 metadata_consistency_skill），避免空证据噪声
- `metadata_completeness="full"`：保留所有证据

### 2.2 要改的文件

#### `agent/aggregator.py`

在 `build_evidence_bundle` 中，在 `_reorder_evidence_by_workflow` 之后，增加调用 `_filter_evidence_by_workflow`：

```python
selected_evidence = _reorder_evidence_by_workflow(selected_evidence, workflow_context)
selected_evidence = _filter_evidence_by_workflow(selected_evidence, workflow_context)  # 新增
```

新增函数 `_filter_evidence_by_workflow`：

```python
def _filter_evidence_by_workflow(
    evidence_items: list[dict],
    workflow_context: dict,
) -> list[dict]:
    if not workflow_context or not evidence_items:
        return evidence_items

    time_budget = str(workflow_context.get("time_budget", "")).strip()
    metadata_completeness = str(workflow_context.get("metadata_completeness", "")).strip()

    result = list(evidence_items)

    # screening 模式：只保留 calibration_score >= 0.5 的证据，最多保留 6 条
    if time_budget == "screening":
        high_conf = [e for e in result if float(e.get("calibration_score", 1.0)) >= 0.5]
        result = high_conf[:6] if high_conf else result[:6]

    # minimal metadata：过滤掉依赖 metadata 的技能输出（空输出噪声）
    if metadata_completeness == "minimal":
        metadata_dependent_skills = {"metadata_consistency_skill", "distribution_analysis_skill"}
        result = [
            e for e in result
            if e.get("skill_name", "") not in metadata_dependent_skills
            or bool(e.get("content", "").strip())
        ]

    return result
```

**注意**：`calibration_score` 字段需要确认 `selected_evidence` 中是否已有该字段。
如果没有，需要在 `_build_selected_evidence` 中从 `calibration.to_dict()` 里提取并附加到每条 evidence item 上。

---

## 三、长期记忆：CognitionState 增加 workflow_preferences

### 3.1 问题描述

`CognitionState` 目前记录的是全局的 skill 统计和 confusion patterns，没有按 workflow 场景分层记录。
目标：记录"在某种 workflow 场景下，哪些 skill 组合更有效"，供下次同类场景复用。

### 3.2 数据结构设计

在 `CognitionState` 中新增字段：

```python
workflow_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
```

结构示例：
```json
{
  "primary_care": {
    "preferred_skills": ["malignancy_risk_assessment_skill", "uncertainty_assessment_skill"],
    "avg_accuracy": 0.72,
    "case_count": 34,
    "top_confusion_pairs": ["MEL/NV", "BCC/AK"]
  },
  "specialist_clinic__risk_first": {
    "preferred_skills": ["malignancy_risk_assessment_skill", "morphology_analysis_skill"],
    "avg_accuracy": 0.81,
    "case_count": 18,
    "top_confusion_pairs": ["MEL/BCC"]
  }
}
```

key 的构造规则：`{hospital_type}` 或 `{hospital_type}__{workflow_preference}`（当 workflow_preference 非空时）。

### 3.3 要改的文件

#### `cognition/cognition_state.py`

1. 新增字段：
```python
workflow_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
```

2. 在 `__post_init__` 中初始化：
```python
self.workflow_preferences = dict(self.workflow_preferences or {})
```

3. 新增方法 `update_workflow_preferences`：
```python
def update_workflow_preferences(
    self,
    workflow_context: dict | None,
    skills_used: list[str],
    correct: bool,
) -> None:
    if not workflow_context:
        return
    hospital_type = str(workflow_context.get("hospital_type", "")).strip()
    preference = str(workflow_context.get("workflow_preference", "")).strip()
    if not hospital_type:
        return
    key = f"{hospital_type}__{preference}" if preference else hospital_type
    entry = self.workflow_preferences.setdefault(key, {
        "preferred_skills": [],
        "avg_accuracy": 0.0,
        "case_count": 0,
        "top_confusion_pairs": [],
    })
    n = entry["case_count"]
    entry["avg_accuracy"] = (entry["avg_accuracy"] * n + int(correct)) / (n + 1)
    entry["case_count"] = n + 1
    # 更新 preferred_skills：把本次用到的 skill 加入，保留出现频率最高的前 5 个
    skill_freq = dict.fromkeys(entry["preferred_skills"], 1)
    for s in skills_used:
        skill_freq[s] = skill_freq.get(s, 0) + 1
    entry["preferred_skills"] = sorted(skill_freq, key=lambda x: -skill_freq[x])[:5]
```

4. 在 `_compute_state_version` 的 payload 中加入 `workflow_preferences`。

#### `agent/reflection.py`

在 `apply_cognition_update` 函数中，增加对 `update_workflow_preferences` 的调用：

```python
cognition.update_workflow_preferences(
    workflow_context=state.case_input.workflow_context,
    skills_used=list(state.skill_outputs.keys()),
    correct=bool(reflection.get("diagnosis_correct")),
)
```

**前提**：`reflection` 字典中需要有 `diagnosis_correct` 字段。检查 `build_reflection` 是否已有该字段，如果没有，需要在 reflection 构建时加入。

#### `agent/planner.py`

在 `_adjust_decisions_by_workflow_context` 中，增加从 `cognition.workflow_preferences` 读取历史偏好的逻辑：

```python
# 从 cognition 的 workflow_preferences 中读取历史有效 skill
workflow_key = f"{hospital_type}__{preference}" if preference else hospital_type
historical_prefs = cognition.workflow_preferences.get(workflow_key, {})
historical_skills = historical_prefs.get("preferred_skills", [])
for decision in decisions:
    if decision.skill_name in historical_skills:
        decision.score += int(policy.get("workflow_history_bonus", 1) or 1)
        decision.reasons = dedupe_reasons(
            decision.reasons + [f"Boosted by workflow history ({workflow_key})."]
        )
```

这需要 `PlannerInput` 中传入 `cognition`（已有），并在 `_adjust_decisions_by_workflow_context` 方法签名中增加 `cognition` 参数。

---

## 四、实验验证脚本

### 4.1 目标

验证 workflow_context 是否真正改善了 metadata 缺失场景下的性能。

对比实验：
- **Condition A**：Full metadata，无 workflow_context
- **Condition B**：Full metadata，有 workflow_context（full）
- **Condition C**：Partial metadata，有 workflow_context（partial）
- **Condition D**：Minimal metadata，有 workflow_context（minimal）

### 4.2 要新增的文件

#### `scripts/run_metadata_masking_experiment.py`

```python
"""
运行 metadata masking 对比实验。
用法：python scripts/run_metadata_masking_experiment.py --n-cases 40 --data-split val
"""
```

核心逻辑：
1. 加载 N 个 val 集病例
2. 对每个病例分别用四种 condition 运行 `run_agent`
3. 收集每次运行的最终诊断结果和 evidence package
4. 计算每种 condition 的 accuracy / macro-F1
5. 输出对比表格到 `outputs/metadata_masking_experiment_{timestamp}/results.json`

关键实现细节：
- 使用 `load_case_with_masking` 构造不同 condition 的 CaseInput
- `enable_writeback=False`，避免污染 experience bank
- `run_mode="eval"` 确保不触发 reflection 写回

#### `scripts/analyze_metadata_masking_results.py`

读取实验结果 JSON，输出：
- 四种 condition 的 accuracy 对比
- 各 condition 下 skill 调用频率分布
- Minimal vs Full 的性能退化幅度（degradation rate）

---

## 五、修改文件清单

| 文件 | 修改类型 | 核心改动 |
|------|---------|---------|
| `memory/experience_bank.py` | 修改 | retrieve_bundle 增加 workflow_context 参数 |
| `agent/run_agent.py` | 修改 | 两处 retrieve_bundle 调用增加 workflow_context |
| `agent/aggregator.py` | 修改 | 新增 _filter_evidence_by_workflow 函数 |
| `cognition/cognition_state.py` | 修改 | 新增 workflow_preferences 字段和 update 方法 |
| `agent/reflection.py` | 修改 | apply_cognition_update 中调用 update_workflow_preferences |
| `agent/planner.py` | 修改 | _adjust_decisions_by_workflow_context 读取历史偏好 |
| `scripts/run_metadata_masking_experiment.py` | 新增 | 实验脚本 |
| `scripts/analyze_metadata_masking_results.py` | 新增 | 结果分析脚本 |

---

## 六、改动依赖关系

```
cognition_state.py（新增字段）
    ↓
reflection.py（调用 update_workflow_preferences）
    ↓
planner.py（读取 workflow_preferences 历史偏好）

experience_bank.py（新增 workflow_context 参数）
    ↓
run_agent.py（透传 workflow_context 到 retrieve_bundle）

aggregator.py（新增 _filter_evidence_by_workflow）
    ↓ 独立，不依赖上面的改动

scripts/（依赖所有上面的改动完成后再运行）
```

**建议实施顺序**：
1. cognition_state.py → reflection.py（长期记忆）
2. experience_bank.py → run_agent.py（经验层适配）
3. aggregator.py（动态证据构建）
4. planner.py（读取历史偏好，依赖步骤1完成）
5. 实验脚本（依赖所有步骤完成）

---

## 七、注意事项

### 7.1 向后兼容
- `workflow_context` 参数全部有默认值 `None`，不传时行为与 CODE1 完全一致
- `workflow_preferences` 字段有默认值 `field(default_factory=dict)`，旧的 cognition_state.json 加载时会自动补全为空 dict

### 7.2 calibration_score 字段确认
在实施 `_filter_evidence_by_workflow` 之前，需要先确认 `selected_evidence` 中每条 item 是否有 `calibration_score` 字段。
如果没有，需要在 `_build_selected_evidence` 中从 calibration 结果里提取并附加。

### 7.3 diagnosis_correct 字段确认
在实施 `reflection.py` 修改之前，需要确认 `build_reflection` 返回的 dict 中是否有 `diagnosis_correct` 字段。
如果没有，需要在 reflection 构建时加入（对比 predicted label 和 ground truth label）。

### 7.4 实验规模
建议先用 40 个 val 集病例做快速验证，确认方向正确后再扩展到完整 val 集。
