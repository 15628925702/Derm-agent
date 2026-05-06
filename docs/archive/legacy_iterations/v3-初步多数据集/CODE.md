# v3 代码改动记录

本文档记录 v3 分支中为支持多数据集应用所做的所有代码改动。

---

## 1. `agent/confusion_clusters.py`

### 改动类型
重构：从硬编码 PAD-UFES-20 混淆对 → 数据集可配置注册表

### 新增内容

**`TERM_ALIASES` 扩展**
新增 HAM10000 / ISIC 标签别名：`nv`, `bkl`, `df`, `vasc`, `akiec`

**注册表机制**
```python
_DATASET_CLUSTER_REGISTRY: dict[str, dict[str, dict[str, Any]]]

def register_confusion_clusters(dataset_name: str, clusters: dict) -> None
def get_confusion_cluster_definitions(dataset_name: str | None = None) -> dict
```
- `register_confusion_clusters()` 注册新数据集的混淆对定义
- `get_confusion_cluster_definitions()` 按 `dataset_name` 查找，找不到则回退到默认 PAD-UFES-20 定义

**函数签名变更**（全部新增 `dataset_name: str | None = None` 参数）
- `detect_confusion_clusters()`
- `cluster_guidance_snapshot()`
- `cluster_priority_bonus()`
- `cluster_ordering_hints()`
- `cluster_related_keywords()`
- `cluster_pairs()`
- `cluster_match_bonus()`
- `preferred_abstract_section()`

### 向后兼容性
完全向后兼容。所有新参数均有默认值 `None`，不传时行为与 v2 完全一致。

---

## 2. `agent/policy_evaluation.py`

### 改动类型
重构：`DEFAULT_KEY_CONFUSION_SUBSETS` 从模块级常量 → 数据集可配置注册表

### 新增内容

**注册表**
```python
_DATASET_KEY_CONFUSION_SUBSETS: dict[str, tuple[str, ...]] = {
    "pad_ufes_20": ("melanoma->nev", "ack->scc"),
    "isic2019":    ("melanoma->nv", "ak->bcc"),
    "ham10000":    ("melanoma->nv", "bkl->nv"),
}

def get_key_confusion_subsets(dataset_name: str | None = None) -> tuple[str, ...]
def register_key_confusion_subsets(dataset_name: str, subsets: tuple[str, ...]) -> None
```

**函数签名变更**
- `build_policy_summary(case_results, dataset_name=None)` — 新增 `dataset_name` 参数，自动从 results 推断（单数据集时）
- `compare_policy_summaries(stable, candidate, dataset_name=None)` — 新增 `dataset_name` 参数，用于选择正确的混淆对集合

### 向后兼容性
完全向后兼容。不传 `dataset_name` 时行为与 v2 完全一致。

---

## 3. `agent/evaluation_protocol.py`

### 改动类型
重构：`SPECIALIST_SKILLS` 从模块级常量 → 数据集可配置注册表

### 新增内容

**注册表**
```python
_DATASET_SPECIALIST_SKILLS: dict[str, set[str]] = {
    "pad_ufes_20": {"mel_nev_specialist_skill", "ack_scc_specialist_skill"},
    "isic2019":    {"mel_nev_specialist_skill", "ack_scc_specialist_skill"},
    "ham10000":    {"mel_nev_specialist_skill"},
}

def register_specialist_skills(dataset_name: str, skills: set[str]) -> None
def get_specialist_skills(dataset_name: str | None = None) -> set[str]
```

**函数签名变更**
- `default_ablation_target_specs(dataset_name=None)` — 新增 `dataset_name` 参数，`no_specialists` ablation 会自动使用对应数据集的 specialist skills

### 向后兼容性
完全向后兼容。`SPECIALIST_SKILLS` 常量保留，不传 `dataset_name` 时行为与 v2 完全一致。

---

## 4. `memory/experience_schema.py`

### 改动类型
扩展：`ExperienceRecord` 新增 `dataset_name` 字段

### 改动内容
```python
@dataclass
class ExperienceRecord:
    ...
    dataset_name: str | None = None  # 新增，默认 None
```

### 用途
- 记录经验来源的数据集，便于未来按数据集过滤检索
- 当前经验库通过独立资产仓（`DERMAGENT_SPLIT_STATE_ROOT`）实现物理隔离，此字段作为逻辑标记
- 新字段有默认值 `None`，不影响现有序列化/反序列化

### 向后兼容性
完全向后兼容。旧的 JSON 记录反序列化时 `dataset_name` 为 `None`。

---

## 改动汇总

| 文件 | 改动类型 | 核心变化 |
|------|----------|----------|
| `agent/confusion_clusters.py` | 重构 | 混淆对定义改为注册表，所有函数新增 `dataset_name` 参数 |
| `agent/policy_evaluation.py` | 重构 | 关键混淆对改为注册表，`build_policy_summary` / `compare_policy_summaries` 新增 `dataset_name` |
| `agent/evaluation_protocol.py` | 重构 | Specialist skills 改为注册表，`default_ablation_target_specs` 新增 `dataset_name` |
| `memory/experience_schema.py` | 扩展 | `ExperienceRecord` 新增 `dataset_name: str | None = None` |

所有改动均向后兼容，v2 的调用方式无需修改。

---

## 5. `agent/confusion_clusters.py` — HAM10000 混淆对注册

### 改动类型
新增：文件末尾调用 `register_confusion_clusters("ham10000", {...})` 注册3个集群

### 新增集群

| 集群 ID | 标签对 | 依据 |
|---------|--------|------|
| `mel_nv` | Melanoma / Nevus | HAM10000 最高频混淆，NV占67%，MEL占11% |
| `bkl_nv` | Benign Keratosis / Nevus | BKL+NV合计占78%，形态相似 |
| `mel_bkl` | Melanoma / Benign Keratosis | 色素性BKL可模拟黑色素瘤 |

### Priority skills 配置
- `mel_nv`：`mel_nev_specialist_skill` 2.0，`malignancy_risk_assessment_skill` 1.5
- `bkl_nv`：`differential_compare_skill` 2.0，`border_surface_analysis_skill` 1.2
- `mel_bkl`：`malignancy_risk_assessment_skill` 2.0

### 向后兼容性
完全向后兼容。PAD-UFES-20 默认集群不受影响。

---

## 6. `dataio/ham10000_loader.py` — metadata 字段映射

### 改动类型
扩展：`_sanitize_ham10000_metadata()` 新增字段映射和诊断置信度标记

### 改动内容
```python
# localization -> region（skills 期望 "region" 字段）
metadata["region"] = metadata["localization"]

# dx_type -> diagnosis_confidence + has_histopathology
metadata["diagnosis_confidence"] = "histopathology_confirmed" | "follow_up_confirmed" | ...
metadata["has_histopathology"] = True | False
```

### 用途
- `metadata_consistency_skill` 现在可以读取 `region` 字段（之前只有 `localization`）
- `diagnosis_confidence` 和 `has_histopathology` 可被 evidence aggregator 利用，提升高置信度 case 的证据权重

### 向后兼容性
完全向后兼容。原有字段保留，只新增字段。

---

## 7. `agent/planner.py` + `agent/run_agent.py` — dataset_name 传递

### 改动类型
修复：`PlannerInput` 新增 `dataset_name` 字段，`detect_confusion_clusters` 调用时传入，`run_agent.py` 构建时从 `case_input.dataset_name` 读取

### 问题
v3a 评测发现所有 case 的 `active_confusion_clusters=[]`，原因是 `detect_confusion_clusters` 没有收到 `dataset_name`，回退到 PAD-UFES-20 默认集群，HAM10000 的 `mel_nv`/`bkl_nv` 无法被激活

### 改动内容
```python
# agent/planner.py
@dataclass
class PlannerInput:
    ...
    dataset_name: str | None = None  # 新增

# _build_signal_profile() 中
active_confusion_clusters = detect_confusion_clusters(
    ...,
    dataset_name=planner_input.dataset_name,  # 新增
)

# agent/run_agent.py
PlannerInput(
    ...,
    dataset_name=case_input.dataset_name,  # 新增
)
```

### 向后兼容性
完全向后兼容。`dataset_name` 默认 `None`，PAD-UFES-20 行为不变。

---

## 8. `agent/policy_evaluation.py` — per-class metrics

### 改动类型
新增：`_per_class_metrics()` 函数，`build_policy_summary()` 输出增加 `per_class` 字段

### 新增内容
```python
summary["per_class"] = {
    "MEL": {"support": 10, "tp": 8, "recall": 0.8, "precision": 0.9, "f1": 0.85},
    "NV":  {"support": 67, "tp": 60, "recall": 0.9, ...},
    ...
}
```

### 用途
- 长尾数据集（HAM10000/SD-198）分析每类的识别能力
- 发现 VASC/DF 等稀有类别的性能瓶颈
- 计算 macro-averaged F1

### 向后兼容性
完全向后兼容。只新增 `per_class` 字段，不影响现有字段。
