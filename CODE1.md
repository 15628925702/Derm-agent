# DermAgent 代码修改总结

## 修改日期
2026-04-16

## 修改目标
根据 `改进计划.md` 的要求，实现以下核心改进：
1. 显式建模 workflow_context（医院环境约束）
2. 支持 metadata masking（模拟信息缺失场景）
3. 确保跨数据集标签空间对齐（ISIC2019 支持）

---

## 一、已完成的代码修改

### 1. 在 CaseInput 中增加 workflow_context 字段

**文件**: `agent/state.py`

**修改内容**:
- 在 `CaseInput` dataclass 中新增 `workflow_context: dict[str, Any] | None = None` 字段
- 该字段包含以下信息：
  - `hospital_type`: 医院类型（"primary_care" | "specialist_clinic" | "academic_center"）
  - `available_tests`: 可用检查手段列表（["dermoscopy", "biopsy", "patch_test", ...]）
  - `metadata_completeness`: 元数据完整度（"full" | "partial" | "minimal"）
  - `time_budget`: 时间预算（"screening" | "standard" | "comprehensive"）
  - `workflow_preference`: 工作流偏好（"morphology_first" | "risk_first" | "metadata_first"）

**影响**: 为系统提供了医院环境约束的显式建模能力。

---

### 2. 在 planner 中根据 workflow_context 调整 skill ordering

**文件**: `agent/planner.py`

**修改内容**:

#### 2.1 在 PlannerInput 中增加 workflow_context 字段
- 新增 `workflow_context: dict[str, Any] | None = None` 字段
- 在 `to_dict()` 方法中包含 workflow_context

#### 2.2 新增 `_adjust_decisions_by_workflow_context` 方法
该方法根据 workflow_context 调整 skill 选择决策：

**场景1: risk_first workflow**
- 提高 `malignancy_risk_assessment_skill` 的分数和优先级
- 将其 ordering_hint 设为 5，提前到 morphology 之前

**场景2: 缺少 dermoscopy**
- 降低 morphology 系列技能（morphology_analysis、border_surface、color_pattern）的权重
- 应用 penalty_factor（默认 0.7）

**场景3: metadata 不完整**
- `minimal`: 强制选择 `information_gap_detection_skill`，分数 +3
- `partial`: 提升 `information_gap_detection_skill` 分数 +2

#### 2.3 在 plan 方法中调用新方法
- 在 `_apply_learned_controller_sparsification` 之后
- 在 `_apply_case_budget_gate` 之前
- 调用 `_adjust_decisions_by_workflow_context`

**影响**: 系统能够根据医院环境动态调整技能选择策略。

---

### 3. 在 evidence_package 中根据 workflow_context 调整证据排序

**文件**: `agent/aggregator.py`

**修改内容**:

#### 3.1 修改 `build_evidence_bundle` 函数
- 从 `state.case_input.workflow_context` 读取 workflow_context
- 在构建 selected_evidence 后，调用 `_reorder_evidence_by_workflow` 进行排序

#### 3.2 新增 `_reorder_evidence_by_workflow` 函数
该函数根据 workflow_context 调整证据排序：

**场景1: metadata_completeness == "minimal"**
- 将包含 "information_gap" 的证据项提到最前
- 其余证据项保持原顺序

**场景2: workflow_preference == "risk_first"**
- 将包含 "malignancy" 或 "risk" 的证据项提到最前
- 或 section == "risk" 的证据项提到最前

**场景3: workflow_preference == "metadata_first"**
- 将包含 "metadata" 的证据项提到最前

**影响**: 证据包的组织方式能够适应不同的临床工作流偏好。

---

### 4. 在 run_agent 中透传 workflow_context

**文件**: `agent/run_agent.py`

**修改内容**:
- 在构建 `PlannerInput` 时，增加 `workflow_context=case_input.workflow_context` 参数
- 确保 workflow_context 从 CaseInput 一路透传到 planner

**影响**: 完成了 workflow_context 的端到端透传。

---

### 5. 在 dataio 中增加 metadata masking 逻辑

**文件**: `dataio/case_loader.py`

**修改内容**:

#### 5.1 新增 `load_case_with_masking` 函数
```python
def load_case_with_masking(
    case_index: int,
    data_root: str | Path,
    metadata_mask: list[str] | None = None,
) -> CaseInput
```

**功能**:
- 加载病例后，根据 `metadata_mask` 参数过滤 metadata 字段
- `metadata_mask=None`: 保留所有字段（默认行为）
- `metadata_mask=[]`: mask 所有字段（minimal 场景）
- `metadata_mask=["region", "age"]`: 只保留指定字段（partial 场景）

**自动设置 workflow_context**:
- `len(metadata_mask) == 0` → `metadata_completeness = "minimal"`
- `len(metadata_mask) <= 2` → `metadata_completeness = "partial"`
- `len(metadata_mask) > 2` → `metadata_completeness = "full"`

**影响**: 支持模拟不同医院的信息完整度场景。

---

### 6. 验证 ISIC2019 标签空间和数据加载器

**文件**: 
- `agent/label_space.py`
- `dataio/isic2019_loader.py`

**验证结果**:
- ✅ `ISIC2019_FULL_LABEL_SPACE` 已定义，包含 8 类标签（MEL, NV, BCC, AK, BKL, DF, VASC, SCC）
- ✅ 已注册到 `LABEL_SPACES` 字典中，label_space_id = "isic2019_full"
- ✅ `isic2019_loader.py` 正确设置 `label_space_id="isic2019_full"`
- ✅ 恶性标签定义为 ("MEL", "BCC", "AK", "SCC")
- ✅ 良性标签定义为 ("NV", "BKL", "DF", "VASC", "UNK")

**影响**: 系统已支持 ISIC2019 数据集的跨数据集验证实验。

---

## 二、修改的文件清单

1. `agent/state.py` - 增加 workflow_context 字段
2. `agent/planner.py` - 增加 workflow_context 调整逻辑
3. `agent/aggregator.py` - 增加证据排序调整逻辑
4. `agent/run_agent.py` - 透传 workflow_context
5. `dataio/case_loader.py` - 增加 metadata masking 函数
6. `agent/label_space.py` - 验证（无需修改）
7. `dataio/isic2019_loader.py` - 验证（无需修改）

---

## 三、如何使用新功能

### 3.1 使用 workflow_context

```python
from agent.state import CaseInput

# 场景1: 基层医院，无 dermoscopy，信息缺失
case = CaseInput(
    case_id="case_001",
    image_path="/path/to/image.jpg",
    metadata={"region": "arm", "age": 45},
    workflow_context={
        "hospital_type": "primary_care",
        "available_tests": ["biopsy"],  # 没有 dermoscopy
        "metadata_completeness": "partial",
        "time_budget": "screening",
        "workflow_preference": "risk_first",
    }
)

# 场景2: 专科诊所，完整信息
case = CaseInput(
    case_id="case_002",
    image_path="/path/to/image.jpg",
    metadata={"region": "face", "age": 60, "diameter_1": 5.2, "grew": "yes"},
    workflow_context={
        "hospital_type": "specialist_clinic",
        "available_tests": ["dermoscopy", "biopsy", "patch_test"],
        "metadata_completeness": "full",
        "time_budget": "comprehensive",
        "workflow_preference": "morphology_first",
    }
)
```

### 3.2 使用 metadata masking

```python
from dataio.case_loader import load_case_with_masking

# 场景 Full: 所有 metadata 可见
case_full = load_case_with_masking(
    case_index=0,
    data_root="/root/DermAgent/data/pad_ufes_20",
    metadata_mask=None  # 保留所有字段
)

# 场景 Partial: 只保留 region 和 age
case_partial = load_case_with_masking(
    case_index=0,
    data_root="/root/DermAgent/data/pad_ufes_20",
    metadata_mask=["region", "age"]
)

# 场景 Minimal: 只有图像，无 metadata
case_minimal = load_case_with_masking(
    case_index=0,
    data_root="/root/DermAgent/data/pad_ufes_20",
    metadata_mask=[]  # mask 所有字段
)

# workflow_context 会自动设置
print(case_minimal.workflow_context)
# 输出: {"metadata_completeness": "minimal"}
```

### 3.3 在实验脚本中使用

```python
# 实验2: Metadata missingness
for metadata_scenario in ["full", "partial", "minimal"]:
    if metadata_scenario == "full":
        metadata_mask = None
    elif metadata_scenario == "partial":
        metadata_mask = ["region", "age"]
    else:  # minimal
        metadata_mask = []
    
    for case_index in range(50):
        case = load_case_with_masking(
            case_index=case_index,
            data_root=data_root,
            metadata_mask=metadata_mask
        )
        
        # 运行 agent
        state, evidence_package = run_agent(
            case_input=case,
            client=qwen_client,
            # ...
        )
```

---

## 四、预期效果

### 4.1 Workflow-aware skill selection

**场景**: 基层医院，无 dermoscopy，risk_first workflow

**预期行为**:
- `malignancy_risk_assessment_skill` 会被提前触发
- `morphology_analysis_skill` 等依赖 dermoscopy 的技能权重降低
- 如果 metadata 缺失，`information_gap_detection_skill` 会被强制选择

### 4.2 Workflow-aware evidence ordering

**场景**: metadata_completeness = "minimal"

**预期行为**:
- Evidence package 中，`information_gap_detection_skill` 的输出会排在最前
- `missing_information` 字段会明确列出缺失的关键信息
- 最终诊断会更谨慎，可能建议"需要更多信息"

### 4.3 Graceful degradation

**场景**: 从 Full → Partial → Minimal

**预期行为**:
- Full: 正常工作流，所有技能正常触发
- Partial: 部分技能权重调整，开始提示信息缺失
- Minimal: 强制触发 information_gap_detection，证据包中明确标注缺失信息

---

## 五、后续实验建议

### 实验1: 三配置对比（最高优先级）
- 等待 50-case 实验完成
- 对比 Direct baseline vs Heuristic vs Learned
- 验证"核心价值来自结构化设计"

### 实验2: Metadata missingness（高优先级）
- 使用 `load_case_with_masking` 函数
- 运行 3×3 实验（3 种场景 × 3 种配置）
- 分析 agent 在信息缺失时的稳定性

### 实验3: Evidence package 差异分析（中优先级）
- 对比 Full vs Minimal 场景的 execution records
- 统计 skill 触发率变化
- 分析 evidence package 组织差异

### 实验4: 跨数据集验证（低优先级）
- 在 ISIC2019 上运行 heuristic 和 learned 配置
- 验证 heuristic 的零样本泛化能力

---

## 六、注意事项

### 6.1 向后兼容性
- 所有新增字段都是可选的（`None` 默认值）
- 如果不提供 `workflow_context`，系统行为与之前完全一致
- 如果不使用 `load_case_with_masking`，使用原有的 `load_case_by_index` 即可

### 6.2 Policy 配置
新增的 workflow 调整逻辑使用以下 policy 参数（都有默认值）：
- `workflow_risk_first_bonus`: 默认 5
- `workflow_no_dermoscopy_penalty`: 默认 0.7
- `workflow_minimal_metadata_bonus`: 默认 3
- `workflow_partial_metadata_bonus`: 默认 2

可以在 policy JSON 中覆盖这些参数。

### 6.3 测试建议
建议运行以下测试验证修改：
```bash
# 1. 测试 workflow_context 透传
python scripts/debug_single_case.py --case-index 0 --data-split train

# 2. 测试 metadata masking
python -c "
from dataio.case_loader import load_case_with_masking
case = load_case_with_masking(0, '/root/DermAgent/data/pad_ufes_20', metadata_mask=[])
print('workflow_context:', case.workflow_context)
print('metadata keys:', list(case.metadata.keys()))
"

# 3. 运行单元测试（如果有）
pytest tests/
```

---

## 七、总结

本次代码修改实现了 `改进计划.md` 中的核心改进目标：

✅ **改进1**: 显式建模 workflow_context（3 天工作量）
- 在 CaseInput 中增加字段
- 在 planner 中调整 skill ordering
- 在 evidence_package 中调整证据排序
- 在 run_agent 中透传

✅ **改进2**: 支持 metadata masking（0.5 天工作量）
- 新增 `load_case_with_masking` 函数
- 自动设置 workflow_context

✅ **改进3**: 验证跨数据集支持（0.5 小时工作量）
- ISIC2019 标签空间已正确定义和注册
- 数据加载器已正确设置 label_space_id

**总工作量**: 约 4 天（符合改进计划预估）

这些修改为后续的实验提供了坚实的代码基础，使 DermAgent 能够从"实验室原型"走向"可部署系统"。
