# DermAgent v3 — 多数据集应用指南

v3 在 v2 的基础上，将混淆对、specialist skills、policy evaluation 等原本硬编码 PAD-UFES-20 的部分改为数据集可配置的注册表机制，使新数据集可以无侵入地接入。

---

## 快速开始：接入新数据集

### Step 1：注册标签空间

在 `agent/label_space.py` 中添加（或在你的初始化脚本中调用）：

```python
from agent.label_space import LabelSpace, register_label_space

MY_DATASET_LABEL_SPACE = LabelSpace(
    label_space_id="my_dataset_full",
    canonical_labels=("MEL", "NV", "BCC", ...),
    aliases=(
        LabelAlias("MEL", ("mel", "melanoma")),
        LabelAlias("NV",  ("nv", "nevus", "mole")),
        ...
    ),
    malignant_labels=("MEL", "BCC", ...),
    benign_labels=("NV", ...),
)
register_label_space(MY_DATASET_LABEL_SPACE, dataset_names=["my_dataset"])
```

---

### Step 2：注册混淆对（confusion clusters）

```python
from agent.confusion_clusters import register_confusion_clusters

register_confusion_clusters("my_dataset", {
    "mel_nv": {
        "label": "Melanoma / Nevus",
        "pairs": ("melanoma->nv", "malignant melanoma->nv"),
        "term_groups": (("mel", "nv"),),
        "keywords": ("mel", "melanoma", "nv", "nevus", "pigmented", "asymmetry"),
        "supporting_clues": ("Contrast irregular border and color variation against uniform nevus pattern.",),
        "opposing_clues": ("Symmetric, uniform pigmentation weakens melanoma confidence.",),
        "missing_evidence": ("Missing dermoscopic detail for atypical network.",),
        "watch_outs": ("Do not call melanoma based on size alone.",),
        "priority_skills": {
            "mel_nev_specialist_skill": 2.0,
            "malignancy_risk_assessment_skill": 1.5,
            "differential_compare_skill": 1.0,
        },
    },
    # 添加更多混淆对...
})
```

---

### Step 3：注册 policy evaluation 混淆对

```python
from agent.policy_evaluation import register_key_confusion_subsets

register_key_confusion_subsets("my_dataset", ("melanoma->nv", "bkl->nv"))
```

---

### Step 4：注册 specialist skills（可选）

```python
from agent.evaluation_protocol import register_specialist_skills

register_specialist_skills("my_dataset", {"mel_nev_specialist_skill"})
```

---

### Step 5：注册 split

在 `configs/dataset_splits.py` 的 `FIXED_SPLITS` 中添加：

```python
FIXED_SPLITS["my_dataset_balanced_v1"] = FixedSplitDefinition(
    split_id="my_dataset_balanced_v1",
    dataset_name="my_dataset",
    metadata_relpath="my_dataset/metadata.csv",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    strategy="stratified_by_dx_group",   # 长尾数据集用分层 split
    notes="Balanced split for my_dataset.",
)
```

---

### Step 6：实现 loader

新建 `dataio/my_dataset_loader.py`，参考 `dataio/ham10000_loader.py`：

```python
from agent.state import CaseInput

def load_my_dataset_case_inputs(data_root, split_json=None, split="test") -> list[CaseInput]:
    # 读取 CSV，解析 metadata，返回 CaseInput 列表
    ...
```

在 `dataio/case_loader.py` 的路由逻辑中添加：

```python
from dataio.my_dataset_loader import load_my_dataset_case_inputs

DEFAULT_MY_DATASET_ROOT = PROJECT_ROOT / "data" / "my_dataset"

# 在 load_case_by_index() 中添加：
if root.resolve() == DEFAULT_MY_DATASET_ROOT.resolve():
    return load_my_dataset_case_input_by_index(...)
```

---

### Step 7：初始化资产仓

```bash
python scripts/manage_dataset_experiment_assets.py init \
  --experiment-id my_dataset_v3 \
  --base-policy-config state/policy/versions/heuristic_with_penalty.json

source state/dataset_adaptation/my_dataset_v3/experiment.env
```

---

### Step 8：评测

```bash
# Smoke test（10 cases）
python scripts/compare_agent_vs_qwen.py \
  --data-root data/my_dataset \
  --split-json outputs/dataset_adaptation/my_dataset_v3/split.json \
  --data-split test \
  --limit 10

# Full test
python scripts/compare_agent_vs_qwen.py \
  --data-root data/my_dataset \
  --split-json outputs/dataset_adaptation/my_dataset_v3/split.json \
  --data-split test
```

---

## 已支持数据集的配置

### PAD-UFES-20（默认）

| 配置项 | 值 |
|--------|-----|
| 标签空间 | `derm_six`（BCC/ACK/NEV/SEK/SCC/MEL） |
| 混淆对 | `ack_bcc_scc`, `ack_sek`, `mel_nev`, `bcc_sek` |
| Key confusion subsets | `melanoma->nev`, `ack->scc` |
| Specialist skills | `mel_nev_specialist_skill`, `ack_scc_specialist_skill` |
| Split | `pad_ufes_20_contiguous_v1` |

### ISIC2019

| 配置项 | 值 |
|--------|-----|
| 标签空间 | `isic2019_full`（MEL/NV/BCC/AK/BKL/DF/VASC/SCC） |
| 混淆对 | 使用 PAD-UFES-20 默认（待扩展） |
| Key confusion subsets | `melanoma->nv`, `ak->bcc` |
| Specialist skills | `mel_nev_specialist_skill`, `ack_scc_specialist_skill` |
| Split | `isic2019_contiguous_v1` |

### HAM10000

| 配置项 | 值 |
|--------|-----|
| 标签空间 | `ham10000_full`（MEL/BCC/NV/BKL/DF/VASC/AKIEC） |
| 混淆对 | 使用 PAD-UFES-20 默认（待扩展，见下方 TODO） |
| Key confusion subsets | `melanoma->nv`, `bkl->nv` |
| Specialist skills | `mel_nev_specialist_skill` |
| Split | `ham10000_balanced_v1`（推荐，分层 split） |

**HAM10000 注意事项**：
- 必须使用 `ham10000_balanced_v1` 而非 `ham10000_contiguous_v1`，否则 smoke test 可能全是 NV
- 图像为皮肤镜图像，与 PAD-UFES-20 的临床照片不同
- 需要独立资产仓，不能复用 PAD-UFES-20 的经验库

---

## v3 新增 API 速查

### `agent/confusion_clusters.py`

```python
# 注册新数据集的混淆对
register_confusion_clusters(dataset_name: str, clusters: dict) -> None

# 获取数据集的混淆对定义（找不到则回退到 PAD-UFES-20 默认）
get_confusion_cluster_definitions(dataset_name: str | None = None) -> dict

# 所有函数新增 dataset_name 参数（可选，默认 None）
detect_confusion_clusters(..., dataset_name=None)
cluster_priority_bonus(cluster_names, skill_name, dataset_name=None)
cluster_ordering_hints(cluster_names, dataset_name=None)
cluster_related_keywords(cluster_names, dataset_name=None)
cluster_pairs(cluster_names, dataset_name=None)
cluster_match_bonus(..., dataset_name=None)
cluster_guidance_snapshot(cluster_names, dataset_name=None)
preferred_abstract_section(..., dataset_name=None)
```

### `agent/policy_evaluation.py`

```python
# 注册数据集的关键混淆对
register_key_confusion_subsets(dataset_name: str, subsets: tuple[str, ...]) -> None

# 获取数据集的关键混淆对
get_key_confusion_subsets(dataset_name: str | None = None) -> tuple[str, ...]

# 函数新增 dataset_name 参数
build_policy_summary(case_results, dataset_name=None)
compare_policy_summaries(stable, candidate, dataset_name=None)
```

### `agent/evaluation_protocol.py`

```python
# 注册数据集的 specialist skills
register_specialist_skills(dataset_name: str, skills: set[str]) -> None

# 获取数据集的 specialist skills
get_specialist_skills(dataset_name: str | None = None) -> set[str]

# 函数新增 dataset_name 参数
default_ablation_target_specs(dataset_name=None)
```

### `memory/experience_schema.py`

```python
@dataclass
class ExperienceRecord:
    ...
    dataset_name: str | None = None  # v3 新增
```

---

## 待完成（v3 后续）

- [ ] HAM10000 专用混淆对定义（`mel_nv`, `bkl_nv`）
- [ ] `dataio/case_loader.py` 路由改为注册表模式
- [ ] 增加 per-class metrics（`agent/evaluation.py`）
- [ ] DermNet loader
- [ ] SCIN loader
- [ ] SD-198 loader（198类，需要标签合并策略）
