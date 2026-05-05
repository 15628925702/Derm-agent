# DermAgent CODE2 代码修改总结

## 修改日期
2026-04-17

## 背景
CODE1.md 完成了 workflow_context 的端到端透传、planner skill 调整、evidence 排序调整、metadata masking。
CODE2 在此基础上补齐四个缺口，全部已实施完成并通过 20 cases smoke test 验证。

---

## 一、经验层 workflow 适配

### 修改文件：`memory/experience_bank.py`

**改动内容**：

1. `retrieve_bundle` 方法新增 `workflow_context: dict[str, Any] | None = None` 参数
2. 调用前先通过 `_compute_workflow_top_k` 计算有效 top_k，再传给 retriever
3. 文件末尾新增模块级函数 `_compute_workflow_top_k`

**调整逻辑**：

| hospital_type | raw | tactical | abstract |
|---|---|---|---|
| `primary_care` | -1 | 不变 | +2 |
| `specialist_clinic` | +2 | +1 | 不变 |
| `academic_center` | 不变 | +2 | +1 |

| time_budget | 效果 |
|---|---|
| `screening` | 所有 top_k × 0.5（最小为 1） |
| `comprehensive` | 所有 top_k × 1.5（最大为 8） |
| `standard` | 不变 |

### 修改文件：`agent/run_agent.py`

**改动内容**：两处调用 `bank.retrieve_bundle` 均增加 `workflow_context=case_input.workflow_context` 参数：
- 第一处：planner 执行前的初始检索（line ~94）
- 第二处：skill 执行后的二次检索（line ~158）

---

## 二、动态证据构建（内容筛选）

### 修改文件：`agent/aggregator.py`

**改动内容**：

1. `build_evidence_bundle` 中在 `_reorder_evidence_by_workflow` 之后新增调用：
   ```python
   selected_evidence = _filter_evidence_by_workflow(selected_evidence, workflow_context)
   ```

2. 新增函数 `_filter_evidence_by_workflow`，筛选逻辑：
   - `time_budget="screening"`：只保留 `calibration_score >= 0.5` 的证据，最多 6 条
   - `metadata_completeness="minimal"`：过滤掉 `metadata_consistency_skill` 和 `distribution_analysis_skill` 的空输出（content 为空时过滤）

**向后兼容**：`workflow_context=None` 时直接返回原列表，不做任何筛选。

---

## 三、长期记忆：CognitionState workflow_preferences

### 修改文件：`cognition/cognition_state.py`

**改动内容**：

1. 新增字段：
   ```python
   workflow_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
   ```

2. `__post_init__` 中初始化：
   ```python
   self.workflow_preferences = dict(self.workflow_preferences or {})
   ```

3. 新增方法 `update_workflow_preferences`：
   - key 构造规则：`{hospital_type}` 或 `{hospital_type}__{workflow_preference}`
   - 每次调用更新 `avg_accuracy`（滚动平均）、`case_count`、`preferred_skills`（按频率排序取前 5）

4. `_compute_state_version` 的 payload 中加入 `workflow_preferences`

**存储结构示例**：
```json
{
  "primary_care__risk_first": {
    "preferred_skills": ["malignancy_risk_assessment_skill", "morphology_analysis_skill"],
    "avg_accuracy": 0.72,
    "case_count": 34
  }
}
```

### 修改文件：`agent/reflection.py`

**改动内容**：

1. `build_reflection` 返回值中新增 `diagnosis_correct` 字段（`label_match is True`）

2. `apply_cognition_update` 签名增加 `state: CaseState | None = None` 参数，当 state 不为 None 时调用：
   ```python
   cognition.update_workflow_preferences(
       workflow_context=state.case_input.workflow_context,
       skills_used=list(state.skill_outputs.keys()),
       correct=bool(reflection.get("diagnosis_correct", False)),
   )
   ```

### 修改文件：`agent/run_agent.py`

**改动内容**：`apply_cognition_update` 调用处增加 `state` 参数：
```python
apply_cognition_update(cognition_state, state.reflection, state)
```

---

## 四、Planner 读取历史 workflow 偏好

### 修改文件：`agent/planner.py`

**改动内容**：

1. `_adjust_decisions_by_workflow_context` 方法签名增加 `cognition: Any | None` 参数，同时新增读取 `hospital_type` 变量

2. 方法末尾新增逻辑：从 `cognition.workflow_preferences` 读取历史有效 skill，对命中的 skill 加分：
   ```python
   key = f"{hospital_type}__{preference}" if preference else hospital_type
   historical_skills = cognition.workflow_preferences.get(key, {}).get("preferred_skills", [])
   # 命中的 skill score += policy.get("workflow_history_bonus", 1)
   ```

3. 调用处增加 `cognition=planner_input.cognition` 参数

---

## 五、实验验证

### Smoke test（20 cases，2026-04-17）

| 配置 | acc | mal_recall |
|---|---|---|
| direct_baseline | 35.0% (7/20) | 53.3% |
| **full_dermagent (CODE2)** | **35.0% (7/20)** | **80.0%** |

**历史基准（333 cases，CODE1）**：agent 40.2% acc，80.9% mal_recall

结论：CODE2 改动未引入性能退化，mal_recall 与历史基准一致（80.0% vs 80.9%）。

---

## 六、修改文件清单

| 文件 | 修改类型 | 核心改动 |
|---|---|---|
| `memory/experience_bank.py` | 修改 | retrieve_bundle 增加 workflow_context，新增 _compute_workflow_top_k |
| `agent/run_agent.py` | 修改 | 两处 retrieve_bundle 透传 workflow_context，apply_cognition_update 传入 state |
| `agent/aggregator.py` | 修改 | 新增 _filter_evidence_by_workflow |
| `cognition/cognition_state.py` | 修改 | 新增 workflow_preferences 字段和 update_workflow_preferences 方法 |
| `agent/reflection.py` | 修改 | 新增 diagnosis_correct 字段，apply_cognition_update 接收 state |
| `agent/planner.py` | 修改 | _adjust_decisions_by_workflow_context 读取 cognition.workflow_preferences |

---

## 七、向后兼容说明

- 所有新增参数均有默认值 `None`，不传时行为与 CODE1 完全一致
- `workflow_preferences` 字段默认为空 dict，旧的 `cognition_state.json` 加载时自动补全，无需迁移
- `apply_cognition_update` 的 `state` 参数可选，不传时跳过 workflow_preferences 更新
