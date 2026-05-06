# DermAgent 代码修改总结

## 修改日期
- CODE1：2026-04-16
- CODE2：2026-04-17

## 背景与目标

根据 `改进计划.md` 的要求，分两批实现以下核心改进：

**CODE1（2026-04-16）**：
1. 显式建模 workflow_context（医院环境约束）
2. 支持 metadata masking（模拟信息缺失场景）
3. 确保跨数据集标签空间对齐（ISIC2019 支持）

**CODE2（2026-04-17）**：
4. 经验层 workflow 适配（experience_bank）
5. 动态证据构建（内容筛选）
6. 长期记忆：CognitionState workflow_preferences
7. Planner 读取历史 workflow 偏好

---

## 一、修改文件总览

| 文件 | 修改批次 | 核心改动 |
|---|---|---|
| `agent/state.py` | CODE1 | CaseInput 增加 workflow_context 字段 |
| `agent/planner.py` | CODE1 + CODE2 | 增加 workflow_context 调整逻辑；读取 cognition.workflow_preferences |
| `agent/aggregator.py` | CODE1 + CODE2 | 增加证据排序调整；新增 _filter_evidence_by_workflow |
| `agent/run_agent.py` | CODE1 + CODE2 | 透传 workflow_context；两处 retrieve_bundle 透传；apply_cognition_update 传入 state |
| `dataio/case_loader.py` | CODE1 | 增加 load_case_with_masking 函数 |
| `agent/label_space.py` | CODE1 | 验证（无需修改） |
| `dataio/isic2019_loader.py` | CODE1 | 验证（无需修改） |
| `memory/experience_bank.py` | CODE2 | retrieve_bundle 增加 workflow_context；新增 _compute_workflow_top_k |
| `cognition/cognition_state.py` | CODE2 | 新增 workflow_preferences 字段和 update_workflow_preferences 方法 |
| `agent/reflection.py` | CODE2 | 新增 diagnosis_correct 字段；apply_cognition_update 接收 state |

---

## 二、CODE1 详细改动

### 1. `agent/state.py` — CaseInput 增加 workflow_context 字段

在 `CaseInput` dataclass 中新增：
```python
workflow_context: dict[str, Any] | None = None
```

字段包含：
- `hospital_type`：`"primary_care"` | `"specialist_clinic"` | `"academic_center"`
- `available_tests`：可用检查手段列表（`["dermoscopy", "biopsy", "patch_test", ...]`）
- `metadata_completeness`：`"full"` | `"partial"` | `"minimal"`
- `time_budget`：`"screening"` | `"standard"` | `"comprehensive"`
- `workflow_preference`：`"morphology_first"` | `"risk_first"` | `"metadata_first"`

---

### 2. `agent/planner.py` — workflow_context 调整 skill ordering（CODE1 部分）

#### 2.1 PlannerInput 增加字段
```python
workflow_context: dict[str, Any] | None = None
```
`to_dict()` 方法中同步包含该字段。

#### 2.2 新增 `_adjust_decisions_by_workflow_context` 方法

| 场景 | 调整逻辑 |
|---|---|
| `workflow_preference == "risk_first"` | `malignancy_risk_assessment_skill` 分数 +bonus，ordering_hint 设为 5 |
| `available_tests` 不含 `"dermoscopy"` | morphology 系列技能（morphology_analysis、border_surface、color_pattern）× penalty_factor（默认 0.7） |
| `metadata_completeness == "minimal"` | 强制选择 `information_gap_detection_skill`，分数 +3 |
| `metadata_completeness == "partial"` | `information_gap_detection_skill` 分数 +2 |

#### 2.3 调用位置
在 `_apply_learned_controller_sparsification` 之后、`_apply_case_budget_gate` 之前调用。

---

### 3. `agent/aggregator.py` — 证据排序调整（CODE1 部分）

#### 3.1 修改 `build_evidence_bundle`
从 `state.case_input.workflow_context` 读取 workflow_context，构建 selected_evidence 后调用 `_reorder_evidence_by_workflow`。

#### 3.2 新增 `_reorder_evidence_by_workflow`

| 场景 | 排序逻辑 |
|---|---|
| `metadata_completeness == "minimal"` | 含 `"information_gap"` 的证据项提到最前 |
| `workflow_preference == "risk_first"` | 含 `"malignancy"` / `"risk"` 或 `section == "risk"` 的证据项提到最前 |
| `workflow_preference == "metadata_first"` | 含 `"metadata"` 的证据项提到最前 |

---

### 4. `agent/run_agent.py` — 透传 workflow_context（CODE1 部分）

构建 `PlannerInput` 时增加：
```python
workflow_context=case_input.workflow_context
```

---

### 5. `dataio/case_loader.py` — metadata masking

新增函数：
```python
def load_case_with_masking(
    case_index: int,
    data_root: str | Path,
    metadata_mask: list[str] | None = None,
) -> CaseInput
```

| metadata_mask 值 | 行为 | 自动设置 metadata_completeness |
|---|---|---|
| `None` | 保留所有字段 | `"full"` |
| `["region", "age"]` | 只保留指定字段 | `"partial"`（≤2 字段）|
| `[]` | mask 所有字段 | `"minimal"` |

---

### 6. `agent/label_space.py` + `dataio/isic2019_loader.py` — ISIC2019 验证

验证结果（无需修改）：
- `ISIC2019_FULL_LABEL_SPACE` 已定义，包含 8 类（MEL, NV, BCC, AK, BKL, DF, VASC, SCC）
- 已注册到 `LABEL_SPACES`，`label_space_id = "isic2019_full"`
- 恶性标签：`(MEL, BCC, AK, SCC)`；良性标签：`(NV, BKL, DF, VASC, UNK)`

---

## 三、CODE2 详细改动

### 7. `memory/experience_bank.py` — 经验层 workflow 适配

#### 7.1 `retrieve_bundle` 新增参数
```python
workflow_context: dict[str, Any] | None = None
```
调用前先通过 `_compute_workflow_top_k` 计算有效 top_k。

#### 7.2 新增模块级函数 `_compute_workflow_top_k`

按 hospital_type 调整 top_k：

| hospital_type | raw | tactical | abstract |
|---|---|---|---|
| `primary_care` | -1 | 不变 | +2 |
| `specialist_clinic` | +2 | +1 | 不变 |
| `academic_center` | 不变 | +2 | +1 |

按 time_budget 缩放：

| time_budget | 效果 |
|---|---|
| `screening` | 所有 top_k × 0.5（最小为 1） |
| `comprehensive` | 所有 top_k × 1.5（最大为 8） |
| `standard` | 不变 |

---

### 8. `agent/aggregator.py` — 动态证据内容筛选（CODE2 部分）

在 `_reorder_evidence_by_workflow` 之后新增调用：
```python
selected_evidence = _filter_evidence_by_workflow(selected_evidence, workflow_context)
```

新增函数 `_filter_evidence_by_workflow`：

| 场景 | 筛选逻辑 |
|---|---|
| `time_budget == "screening"` | 只保留 `calibration_score >= 0.5` 的证据，最多 6 条 |
| `metadata_completeness == "minimal"` | 过滤 `metadata_consistency_skill` 和 `distribution_analysis_skill` 的空输出 |
| `workflow_context is None` | 直接返回原列表，不做任何筛选 |

---

### 9. `cognition/cognition_state.py` — 长期记忆 workflow_preferences

#### 9.1 新增字段
```python
workflow_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
```

#### 9.2 新增方法 `update_workflow_preferences`
- key 构造：`{hospital_type}` 或 `{hospital_type}__{workflow_preference}`
- 每次调用更新 `avg_accuracy`（滚动平均）、`case_count`、`preferred_skills`（按频率排序取前 5）

存储结构示例：
```json
{
  "primary_care__risk_first": {
    "preferred_skills": ["malignancy_risk_assessment_skill", "morphology_analysis_skill"],
    "avg_accuracy": 0.72,
    "case_count": 34
  }
}
```

#### 9.3 其他
- `__post_init__` 中初始化 `workflow_preferences`
- `_compute_state_version` 的 payload 中加入 `workflow_preferences`

---

### 10. `agent/reflection.py` — 反思层更新

1. `build_reflection` 返回值新增 `diagnosis_correct` 字段（`label_match is True`）
2. `apply_cognition_update` 签名增加 `state: CaseState | None = None` 参数，当 state 不为 None 时调用：
```python
cognition.update_workflow_preferences(
    workflow_context=state.case_input.workflow_context,
    skills_used=list(state.skill_outputs.keys()),
    correct=bool(reflection.get("diagnosis_correct", False)),
)
```

---

### 11. `agent/run_agent.py` — 多处透传（CODE2 部分）

1. 两处 `bank.retrieve_bundle` 调用均增加 `workflow_context=case_input.workflow_context`：
   - planner 执行前的初始检索（line ~94）
   - skill 执行后的二次检索（line ~158）
2. `apply_cognition_update` 调用处增加 `state` 参数：
```python
apply_cognition_update(cognition_state, state.reflection, state)
```

---

### 12. `agent/planner.py` — 读取历史 workflow 偏好（CODE2 部分）

`_adjust_decisions_by_workflow_context` 方法末尾新增逻辑：
```python
key = f"{hospital_type}__{preference}" if preference else hospital_type
historical_skills = cognition.workflow_preferences.get(key, {}).get("preferred_skills", [])
# 命中的 skill score += policy.get("workflow_history_bonus", 1)
```

方法签名增加 `cognition: Any | None` 参数，调用处传入 `cognition=planner_input.cognition`。

---

## 四、使用示例

### 4.1 workflow_context

```python
from agent.state import CaseInput

# 基层医院，无 dermoscopy，信息缺失
case = CaseInput(
    case_id="case_001",
    image_path="/path/to/image.jpg",
    metadata={"region": "arm", "age": 45},
    workflow_context={
        "hospital_type": "primary_care",
        "available_tests": ["biopsy"],
        "metadata_completeness": "partial",
        "time_budget": "screening",
        "workflow_preference": "risk_first",
    }
)
```

### 4.2 metadata masking

```python
from dataio.case_loader import load_case_with_masking

case_full    = load_case_with_masking(0, data_root, metadata_mask=None)   # 全量
case_partial = load_case_with_masking(0, data_root, metadata_mask=["region", "age"])
case_minimal = load_case_with_masking(0, data_root, metadata_mask=[])     # 纯图像
```

### 4.3 实验脚本

```python
for metadata_scenario in ["full", "partial", "minimal"]:
    metadata_mask = None if metadata_scenario == "full" \
                    else ["region", "age"] if metadata_scenario == "partial" \
                    else []
    for case_index in range(50):
        case = load_case_with_masking(case_index, data_root, metadata_mask)
        state, evidence_package = run_agent(case_input=case, client=qwen_client)
```

---

## 五、实验验证

### Smoke test（20 cases，2026-04-17，CODE2）

| 配置 | acc | mal_recall |
|---|---|---|
| direct_baseline | 35.0% (7/20) | 53.3% |
| full_dermagent (CODE2) | 35.0% (7/20) | **80.0%** |

历史基准（333 cases，CODE1）：agent 40.2% acc，80.9% mal_recall

结论：CODE2 改动未引入性能退化，mal_recall 与历史基准一致（80.0% vs 80.9%）。

---

## 六、向后兼容说明

- 所有新增字段均有默认值 `None`，不传时行为与改动前完全一致
- `workflow_preferences` 字段默认为空 dict，旧的 `cognition_state.json` 加载时自动补全，无需迁移
- `apply_cognition_update` 的 `state` 参数可选，不传时跳过 workflow_preferences 更新
- `load_case_with_masking` 为新增函数，原有 `load_case_by_index` 不受影响

---

## 七、Policy 参数

新增的 workflow 调整逻辑使用以下 policy 参数（均有默认值，可在 policy JSON 中覆盖）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `workflow_risk_first_bonus` | 5 | risk_first 场景加分 |
| `workflow_no_dermoscopy_penalty` | 0.7 | 无 dermoscopy 时 morphology 技能惩罚系数 |
| `workflow_minimal_metadata_bonus` | 3 | minimal metadata 场景加分 |
| `workflow_partial_metadata_bonus` | 2 | partial metadata 场景加分 |
| `workflow_history_bonus` | 1 | 历史有效 skill 加分 |
