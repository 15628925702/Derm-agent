# v3 代码改动记录

本文档记录 v3 分支中消除 PAD-UFES-20 硬编码、实现完整多数据集适配所做的改动。

---

## 1. `agent/confusion_clusters.py` — 元数据字段注册表

### 改动类型
新增：数据集感知的元数据字段注册表

### 新增内容

```python
_METADATA_FIELDS_REGISTRY: dict[str, dict[str, tuple[str, ...]]] = {
    "pad_ufes_20": {
        "temporal":      ("grew", "changed", "bleed", "itch", "hurt", "elevation"),
        "location_size": ("region", "age", "diameter_1", "diameter_2"),
        "risk":          ("changed", "bleed", "hurt"),
    },
    "isic2019": {
        "temporal":      (),
        "location_size": ("anatom_site_general", "age_approx"),
        "risk":          (),
    },
    "ham10000": {
        "temporal":      (),
        "location_size": ("localization", "age"),
        "risk":          (),
    },
}

def get_metadata_fields(dataset_name: str | None, field_group: str) -> tuple[str, ...]
```

- `field_group` 取值：`"temporal"` / `"location_size"` / `"risk"`
- 找不到 `dataset_name` 时回退到 PAD-UFES-20 字段（向后兼容）
- ISIC2019 / HAM10000 无症状性时序字段（grew/changed/bleed 等），`temporal` 和 `risk` 均为空元组

### 向后兼容性
完全向后兼容。不传 `dataset_name` 时行为与 v3 完全一致。

---

## 2. `agent/planner.py` — `_build_signal_profile` 去硬编码

### 改动类型
修复：三处 PAD-UFES-20 硬编码替换为数据集感知逻辑

### 问题
`_build_signal_profile()` 中：
- `temporal_metadata` 信号硬编码 PAD 字段 `("grew", "changed", "bleed", "itch", "hurt", "elevation")`
- `location_or_size_metadata` 信号硬编码 PAD 字段 `("region", "age", "diameter_1", "diameter_2")`
- `mel_nev_confusion` / `ack_scc_confusion` / `ack_sek_confusion` 信号用硬编码 `has_confusion_pair()` 检测，不走集群注册表

### 改动内容

```python
# 修复前
"temporal_metadata": any(str(metadata.get(field, "")).strip()
    for field in ("grew", "changed", "bleed", "itch", "hurt", "elevation")),
"mel_nev_confusion": has_confusion_pair(ddx_candidates, ("mel", "melanoma"), ("nev", "nevus", ...)),

# 修复后
"temporal_metadata": any(str(metadata.get(field, "")).strip()
    for field in get_metadata_fields(planner_input.dataset_name, "temporal")),
"mel_nev_confusion": any(c in active_confusion_clusters for c in ("mel_nev", "mel_nv")),
"ack_scc_confusion": any(c in active_confusion_clusters for c in ("ack_bcc_scc", "ack_scc")),
"ack_sek_confusion": "ack_sek" in active_confusion_clusters,
```

### 效果
- HAM10000 的 `mel_nv` 集群激活后，`mel_nev_confusion=True` → `mel_nev_specialist_skill` 正常触发
- ISIC2019 / HAM10000 无时序元数据字段，`temporal_metadata=False`，不再误触发 `temporal_evolution_skill`

---

## 3. `memory/experience_retriever.py` — 元数据提取去硬编码

### 改动类型
修复：三个静态方法替换硬编码 PAD 字段

### 问题
- `_extract_metadata_patterns()` 硬编码 10 个 PAD 字段，ISIC/HAM 的 `localization`/`age_approx` 等字段完全被忽略
- `_extract_risk_patterns()` 硬编码 PAD 风险字段 `("changed", "bleed", "hurt")`
- `_detect_confusion_pair()` 硬编码 PAD 混淆对（`melanoma->nev`、`ack->scc` 等），HAM10000 的 `melanoma->nv` 永远无法匹配

### 改动内容

**方法签名新增 `dataset_name` 参数：**
```python
def _extract_metadata_patterns(metadata, dataset_name=None) -> list[str]
def _extract_risk_patterns(perception, metadata, risk_flags, dataset_name=None) -> list[str]
def _detect_confusion_pair(ddx_candidates, dataset_name=None) -> str | None
```

**`_extract_metadata_patterns` / `_extract_risk_patterns`：**
```python
fields = get_metadata_fields(dataset_name, "temporal") + get_metadata_fields(dataset_name, "location_size")
```

**`_detect_confusion_pair`：** 完全重写，改用集群注册表：
```python
clusters = detect_confusion_clusters(ddx_candidates=ddx_candidates, dataset_name=dataset_name)
# 取第一个激活集群的第一个 pair 返回
```

**删除：** `_has_confusion_pair()` 静态方法（已无调用方）

**`build_query()` 调用处同步更新：**
```python
metadata_patterns=self._extract_metadata_patterns(query_metadata, dataset_name=dataset_name),
risk_patterns=self._extract_risk_patterns(..., dataset_name=dataset_name),
resolved_confusion_pair = self._detect_confusion_pair([...], dataset_name=dataset_name),
```

### 效果
- HAM10000 检索时 `confusion_pair="melanoma->nv"`，可正确命中经验库中的 HAM10000 tactical experiences
- ISIC2019 的 `anatom_site_general` / `age_approx` 字段现在参与元数据模式匹配

---

## 4. `skills/mel_nev_specialist.py` — `select_related_abstract_experiences` 去硬编码

### 改动类型
修复：移除硬编码 PAD 混淆对回退

### 问题
```python
# 修复前：硬编码 PAD 混淆对作为兜底
combined_pairs = tuple(dict.fromkeys(
    list(cluster_pairs(cluster_names))
    + ["melanoma->nev", "malignant melanoma->nev", "malignant melanoma_vs_nev"]
))
```
HAM10000 上，`cluster_pairs` 返回 `("melanoma->nv", ...)` 但硬编码追加了 PAD 的 `"melanoma->nev"`，导致检索偏向 PAD 经验。

### 改动内容
```python
# 修复后：完全依赖集群注册表
dataset_name = getattr(state, "dataset_name", None)
combined_pairs = cluster_pairs(cluster_names, dataset_name=dataset_name)
combined_keywords = cluster_related_keywords(cluster_names, dataset_name=dataset_name)
```

---

## 5. `skills/ack_scc_specialist.py` — `select_related_abstract_experiences` 去硬编码

### 改动类型
修复：移除硬编码 PAD 混淆对回退（同上）

### 问题
硬编码追加了 9 个 PAD 专属混淆对（`ack->scc`、`scc->bcc`、`seborrheic keratosis->bcc` 等），在 HAM10000 / ISIC2019 上会检索到无关经验。

### 改动内容
与 `mel_nev_specialist.py` 相同，改为完全依赖集群注册表。

---

## 改动汇总

| 文件 | 改动类型 | 核心变化 |
|------|----------|----------|
| `agent/confusion_clusters.py` | 新增 | `_METADATA_FIELDS_REGISTRY` + `get_metadata_fields()` |
| `agent/planner.py` | 修复 | 元数据字段 + 混淆信号全部改为数据集感知 |
| `memory/experience_retriever.py` | 修复 | 三个静态方法新增 `dataset_name`，`_detect_confusion_pair` 改用集群注册表 |
| `skills/mel_nev_specialist.py` | 修复 | 移除硬编码 PAD 混淆对，改用集群注册表 |
| `skills/ack_scc_specialist.py` | 修复 | 移除硬编码 PAD 混淆对，改用集群注册表 |

所有改动均向后兼容，PAD-UFES-20 行为不变。

---

## 6. `agent/planner.py` — `detect_confusion_pair` 去硬编码

### 改动类型
修复：本地函数 `detect_confusion_pair()` 改用集群注册表

### 问题
`_build_signal_profile()` 内部调用的 `detect_confusion_pair()` 仍是硬编码 PAD 混淆对，HAM10000 的 `"nv"` 标签无法匹配，导致 `known_confusion_match` 信号在 HAM10000 上永远为 `False`，`known_confusion_bonus` 加分失效。

### 改动内容
```python
# 修复前：硬编码 if/elif 链
def detect_confusion_pair(ddx_candidates):
    if has_confusion_pair(ddx_candidates, ("mel", "melanoma"), ("nev", "nevus", ...)):
        return "melanoma->nev"
    ...

# 修复后：走集群注册表
def detect_confusion_pair(ddx_candidates, dataset_name=None):
    clusters = detect_confusion_clusters(ddx_candidates=ddx_candidates, dataset_name=dataset_name)
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in clusters:
        pairs = list(cluster_defs.get(cluster_name, {}).get("pairs", ()))
        if pairs:
            return str(pairs[0]).strip().lower()
    return None
```

调用处同步传入 `dataset_name`：
```python
current_confusion_pair = detect_confusion_pair(ddx_candidates, dataset_name=planner_input.dataset_name)
```

---

## 7. `agent/confusion_clusters.py` — `register_metadata_fields()` 注册函数

### 改动类型
新增：元数据字段注册函数，使新数据集无需修改源码即可接入

### 新增内容
```python
def register_metadata_fields(dataset_name: str, fields: dict[str, tuple[str, ...]]) -> None:
    """Register dataset-specific metadata field groups. Keys: temporal, location_size, risk."""
    _METADATA_FIELDS_REGISTRY[str(dataset_name).strip().lower()] = fields
```

### 用法（新数据集接入示例）
```python
register_confusion_clusters("sd198", { ... })
register_metadata_fields("sd198", {
    "temporal": (),
    "location_size": ("body_part", "age"),
    "risk": (),
})
```

接入新数据集只需在 loader 或初始化文件中调用这两个函数，不需要改任何核心代码。

---

---

## 8. `memory/experience_retriever.py` — 混淆对规范化与家族标签修复

### 改动类型
修复：`_normalize_confusion_pair` 别名规范化 + `_confusion_family_tags` 新增 HAM10000/ISIC 标签

### 问题
- `_normalize_confusion_pair` 只做小写和箭头规范化，不做别名替换。经验库中存储的 `"Malignant Melanoma->NV"` 规范化后为 `"malignant melanoma->nv"`，与查询生成的 `"melanoma->nv"` 不匹配，精确匹配（+6分）失效，只能靠子串匹配（+3分）
- `_confusion_family_tags` 只有 PAD 标签（mel/nev/ack/scc/bcc/sek/inflammatory），HAM10000 的 `nv`/`bkl`/`df`/`vasc`/`akiec` 完全无法识别，导致 `_is_same_confusion_family` 对 HAM10000 混淆对永远返回 False，家族匹配（+3分）失效

### 改动内容

**`_PAIR_TERM_NORMALIZATIONS`（新增模块级常量）：**
```python
_PAIR_TERM_NORMALIZATIONS: tuple[tuple[str, str], ...] = (
    ("malignant melanoma", "melanoma"),
    ("melanocytic nevus", "nv"),
    ("squamous cell carcinoma", "scc"),
    ("basal cell carcinoma", "bcc"),
    ("actinic keratosis", "ack"),
    ("benign keratosis", "bkl"),
    ("seborrheic keratosis", "sek"),
    ("dermatofibroma", "df"),
    ("vascular lesion", "vasc"),
    ...
)
```

**`_normalize_confusion_pair` 新增别名替换：**
```python
left, right = [part.strip() for part in text.split("->", 1)]
for alias, canonical in _PAIR_TERM_NORMALIZATIONS:
    if left == alias: left = canonical
    if right == alias: right = canonical
return f"{left}->{right}"
```

**`_confusion_family_tags` 重写：**
- 使用 `re.split(r"[^a-z0-9]+", text)` 分词，避免 `"nv" in "invasive"` 误匹配
- 新增 HAM10000/ISIC 标签，并与 PAD 标签合并为统一语义组：
  - `nev` 组：`nev`（PAD）+ `nv`（HAM/ISIC）统一为 `"nev"` 标签
  - `ack` 组：`ack`（PAD）+ `akiec`（HAM/ISIC）统一为 `"ack"` 标签
  - `sek` 组：`sek`（PAD）+ `bkl`（HAM/ISIC）统一为 `"sek"` 标签
  - 新增：`df`、`vasc` 标签

### 效果
- `"Malignant Melanoma->NV"` 规范化后 = `"melanoma->nv"` → 精确匹配（+6分）
- `"melanoma->nv"` 与 `"melanoma->nev"` 家族标签均为 `{mel, nev}` → 家族匹配（+3分）
- `"melanoma->bkl"` 与 `"melanoma->sek"` 家族标签均为 `{mel, sek}` → 家族匹配（+3分）

---

## 9. `skills/malignancy_risk.py` — 数据集类别分布先验

### 改动类型
修复：`build_prompt` 新增数据集类别分布校准提示，防止在良性类别主导的数据集上过度预测恶性

### 问题
HAM10000 上 agent 预测 17/30 = MEL（57%），但 MEL 实际只占 11%；ISIC2019 上 14/30 = MEL（47%），实际只占 17%。`malignancy_risk_assessment_skill` 无数据集先验，对所有数据集一视同仁，导致 `risk_level=high` 过于频繁，进而偏置 Qwen 最终诊断向 MEL。

### 改动内容

**新增 `_DATASET_PRIOR_NOTES` 字典：**
```python
_DATASET_PRIOR_NOTES: dict[str, str] = {
    "ham10000": "Dataset prior: HAM10000 class distribution is NV 67%, BKL 11%, MEL 11%, ...",
    "isic2019": "Dataset prior: ISIC2019 class distribution is NV ~49%, ...",
    "pad_ufes_20": "Dataset prior: PAD-UFES-20 class distribution is NEV 24%, BCC 22%, ...",
}
```

**`build_prompt` 注入先验：**
```python
dataset_name = getattr(state.case_input, "dataset_name", None) or ""
prior_note = _DATASET_PRIOR_NOTES.get(dataset_name.lower(), "")
prior_section = f"Calibration note: {prior_note}\n" if prior_note else ""
```

### 效果
- HAM10000：提示模型 NV 占 67%，只有强形态学证据才应给出 high risk
- ISIC2019：提示模型良性类别（NV+BKL+DF）总体多于恶性类别
- PAD-UFES-20：提示模型恶性/癌前类别在该临床数据集中相对常见（行为不变）
- 新数据集：不传 `dataset_name` 时无先验注入，行为与修复前一致

---

## 10. `integrations/openai_client.py` — 最终诊断 prompt 注入数据集先验

### 改动类型
修复：`final_diagnosis()` 的 prompt 新增数据集类别分布先验，防止 `risk_only` 模式下 Qwen 因 caution flag 偏向 MEL

### 问题
评测发现 HAM10000 上 3/4 NV 案例被 agent 错误预测为 MEL，而 Qwen baseline 正确预测为 NV。根因：
- `evidence_decision_policy.override_mode = "risk_only"`（无足够证据支持 override）
- `risk_layer.caution_flags = ["malignancy_risk_high"]`（malignancy_risk_assessment_skill 输出 high）
- final_diagnosis prompt 说"preserve risk warnings"，Qwen 将此解读为"倾向恶性"
- 结果：agent 的 risk caution 反而把 Qwen 从正确的 NV 推向错误的 MEL

### 改动内容

**新增 `_FINAL_DIAGNOSIS_DATASET_PRIORS` 字典：**
```python
_FINAL_DIAGNOSIS_DATASET_PRIORS: dict[str, str] = {
    "ham10000": (
        "Dataset calibration: HAM10000 class distribution is NV 67%, BKL 11%, MEL 11%, ... "
        "When the override layer says risk_only, do NOT shift toward MEL based on risk caution flags alone — "
        "stay close to the image-based baseline and only note the risk concern in follow_up_considerations."
    ),
    "isic2019": (...),
    "pad_ufes_20": (...),
}
```

**`final_diagnosis()` 注入先验（非 SkinVL 分支）：**
```python
dataset_prior = _FINAL_DIAGNOSIS_DATASET_PRIORS.get(
    str(getattr(case_input, "dataset_name", "") or "").strip().lower(), ""
)
dataset_prior_line = f"{dataset_prior}\n" if dataset_prior else ""
# 注入到 prompt 末尾，Evidence package 之前
```

### 效果
- HAM10000：明确告知 Qwen NV 占 67%，`risk_only` 模式下不得因 caution flag 改变诊断方向
- ISIC2019：明确告知 Qwen 良性类别总体更多，risk caution 不应单独驱动诊断
- PAD-UFES-20：行为不变（恶性类别在该数据集中本就常见）
- 新数据集：不传 `dataset_name` 时无先验注入，行为与修复前一致

---

## 改动汇总（完整版）

| 文件 | 改动类型 | 核心变化 |
|------|----------|----------|
| `agent/confusion_clusters.py` | 新增 | `_METADATA_FIELDS_REGISTRY` + `get_metadata_fields()` + `register_metadata_fields()` |
| `agent/planner.py` | 修复 | 元数据字段 + 混淆信号 + `detect_confusion_pair()` 全部改为数据集感知 |
| `memory/experience_retriever.py` | 修复 | 三个静态方法新增 `dataset_name`；`_detect_confusion_pair` 改用集群注册表；`_normalize_confusion_pair` 新增别名规范化；`_confusion_family_tags` 新增 HAM/ISIC 标签并统一语义组 |
| `skills/mel_nev_specialist.py` | 修复 | 移除硬编码 PAD 混淆对，改用集群注册表 |
| `skills/ack_scc_specialist.py` | 修复 | 移除硬编码 PAD 混淆对，改用集群注册表 |
| `skills/malignancy_risk.py` | 修复 | `build_prompt` 注入数据集类别分布先验，防止 HAM10000/ISIC2019 上过度输出 high risk |
| `integrations/openai_client.py` | 修复 | `final_diagnosis()` prompt 注入数据集先验，`risk_only` 模式下明确禁止因 caution flag 偏向 MEL |
| `skills/ack_scc_specialist.py` | 修复 | 移除硬编码 PAD 混淆对，改用集群注册表 |
| `skills/malignancy_risk.py` | 修复 | `build_prompt` 注入数据集类别分布先验，防止 HAM10000/ISIC2019 上过度预测 MEL |

所有改动均向后兼容，PAD-UFES-20 行为不变。新数据集接入只需调用 `register_confusion_clusters()` + `register_metadata_fields()`。
