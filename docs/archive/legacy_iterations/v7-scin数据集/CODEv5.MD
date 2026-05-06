# CODEv5 代码改动记录

本文档记录 `v5` 阶段围绕“多数据集分仓、heuristic-only 运行、跨数据集稳定性提升”所做的关键代码改动。

---

## 一、v5 总目标

v5 的改动目标不是单纯继续优化 PAD-UFES-20 主线，而是把 DermAgent 从“单主线可用”推进到“多数据集可运行、可分仓、可控退回 baseline、尽量跨数据集稳定”的状态。

核心目标分成四类：

1. 多数据集独立分仓
2. 默认走 heuristic-only，而不是 learned 组件
3. bootstrap / compare 入口不再误用旧根或错误 split
4. agent 在证据不足时不再把 baseline 的正确答案推坏

---

## 二、改动总览

| 文件 | 改动类型 | 目的 |
|---|---|---|
| `agent/state.py` | 扩展字段 / 防泄漏 | 增强 metadata 标签泄漏防护，支持 baseline 锚点 |
| `agent/run_agent.py` | 主链路修改 | 在 agent 主链中引入 baseline diagnosis 和保守融合 |
| `agent/aggregator.py` | evidence policy 强化 | 把真实 baseline 注入 evidence policy，并在空证据时禁止 override |
| `agent/conservative_fusion.py` | 新增 | 对 agent final diagnosis 做确定性保守融合 |
| `integrations/openai_client.py` | prompt 与 evidence canonicalization 修改 | 让 final diagnosis 阶段明确参考 baseline anchor |
| `scripts/bootstrap_ham10000_train_cases.sh` | bootstrap 入口修复 | 默认切到 `ham10000_v2`，并按 split json 的 train case indices 建库 |
| `scripts/bootstrap_isic2019_train_cases.sh` | bootstrap 入口修复 | 按 split json 的 train case indices 建库 |
| `scripts/manage_dataset_experiment_assets.py` | 资产初始化修复 | 新建数据集资产时自动清洗为 heuristic-only |
| `final-script/tools/external_final_runner.py` | external 评测修复 | 改用安全 loader，去掉手搓 `CaseInput` 带来的标签泄漏 |
| `final-script/runs/run_external_qwen_vs_agent_qwen.sh` | external 分仓修复 | external compare 改成用数据集专用根，而不是主线 Qwen frozen 资产 |
| `state/dataset_adaptation/ham10000_v1/*` | 元数据整理 | 把旧 HAM10000 v1 资产标识也同步成 heuristic-only |
| `state/dataset_adaptation/ham10000_v2/*` | 元数据整理 | 把 HAM10000 v2 资产 manifest / policy 元数据统一成 heuristic-only |
| `使用规范.md` | 使用文档更新 | 写清三条数据集线路的切换和评测方法 |

---

## 三、核心代码改动

### 1. `agent/state.py`

#### 1.1 扩大标签泄漏屏蔽范围

`LEAKY_METADATA_KEYS` 增加了以下字段：

- `dx`
- `original_label`
- `aligned_label`
- `binary_label`
- `canonical_label`
- `ground_truth`
- `mel`
- `nv`
- `bcc`
- `ak`
- `bkl`
- `df`
- `vasc`
- `scc`
- `akiec`
- 等

目的：

- 阻止 HAM10000 / ISIC2019 等数据集把原始标签字段经 `clinical_metadata()` 暴露给 prompt
- 降低手工构造 `CaseInput` 时的标签泄漏风险

#### 1.2 `CaseState` 新增 `baseline_diagnosis`

新增字段：

```python
baseline_diagnosis: dict[str, Any] = field(default_factory=dict)
```

目的：

- 在 agent 主链中显式保存 image+metadata-only 的 baseline 结果
- 后续用于 evidence policy 的 baseline anchor

---

### 2. `agent/run_agent.py`

#### 2.1 在正式 agent 推理前显式计算 baseline diagnosis

在 initial perception 后加入：

```python
state.baseline_diagnosis = qwen_client.baseline_diagnosis(case_input)
```

目的：

- baseline 不再只是评测脚本外部单独算一次
- agent 主链内部也能拿到真实 baseline，用于后续 final diagnosis 约束

#### 2.2 在 final diagnosis 后增加 conservative fusion

原来：

```python
state.final_diagnosis = qwen_client.final_diagnosis(case_input, evidence_package)
```

现在：

```python
raw_final_diagnosis = qwen_client.final_diagnosis(case_input, evidence_package)
state.final_diagnosis = apply_conservative_agent_fusion(
    baseline_output=state.baseline_diagnosis,
    agent_output=raw_final_diagnosis,
    evidence_bundle=evidence_package.to_dict(),
)
```

这意味着：

- Qwen 仍然先给出 evidence-aware diagnosis
- 但 agent 侧会再根据 evidence policy 决定是否保留 baseline label

效果：

- 避免“空证据 / 不允许 override / 高不确定性”时 agent 把 baseline 正确答案推坏

---

### 3. `agent/aggregator.py`

#### 3.1 `baseline_preview` 不再只是 early ddx

原来 `_build_baseline_preview()` 只记录：

- `early_ddx_candidates`
- `image_summary`

现在新增：

- `baseline_final_diagnosis`
- `baseline_differential_diagnoses`
- `baseline_confidence`

并在 `_build_evidence_decision_policy()` 中调用：

```python
"baseline_preview": _build_baseline_preview(
    initial_perception_summary=initial_perception_summary,
    baseline_diagnosis=state.baseline_diagnosis,
),
```

目的：

- 给 final diagnosis 阶段一个真正的 baseline anchor，而不是只有早期 perception ddx

#### 3.2 空 `selected_evidence` 时，禁止 override

新增：

```python
selected_evidence_present = bool(selected_evidence)
```

然后把它纳入：

```python
malignancy_override_allowed = (
    selected_evidence_present
    and risk_level == "high"
    and support_margin >= 6.0
    ...
)
```

同时把 `selected_evidence_present` 写入 `diagnosis_override_layer`。

效果：

- 如果 evidence calibrator 没选出任何 curated evidence，就不能再走 override
- 避免空证据条件下由风险/不确定性语气把标签推走

#### 3.3 `why_not_confident_enough_to_override` 增加空证据原因

新增文案：

```python
"no curated selected evidence is available to justify overriding the baseline diagnosis"
```

目的：

- 让后续 prompt / 调试记录能看出为什么不能 override

---

### 4. `agent/conservative_fusion.py`（新增）

新增文件，提供两层接口：

```python
apply_conservative_agent_fusion(...)
decide_conservative_agent_fusion(...)
```

#### 4.1 作用

对 `baseline_output` 与 `agent_output` 做一层 agent 侧保守裁决。

#### 4.2 主要规则

以下情况回退到 baseline：

- `selected_evidence_present == False`
- `malignancy_override_allowed == False`
- 只是 `risk_only`，且没有 subtype override
- `uncertainty_level` 太高，且 `support_margin` 不够
- `subtype_support_margin` 不足
- baseline 置信度不低，但 agent 证据不够强

#### 4.3 目的

这层是为了把“prompt 约束”升级成“确定性规则”，避免 Qwen 在明明 `override_allowed = false` 时仍然改 label。

这是 v5 非常关键的一步。

---

### 5. `integrations/openai_client.py`

#### 5.1 final diagnosis prompt 加强 baseline anchoring

在 final diagnosis prompt 中新增明确要求：

- 把 `risk_layer.baseline_preview` 当作 image+metadata-only anchor
- 如果 `selected_evidence` 为空，不要偏离 baseline
- generic risk / uncertainty / follow-up 不能单独当 subtype 证据
- 当 evidence 主要在说“信息不足”时，应保留 baseline label，只在 rationale / follow-up 中表达不确定性

#### 5.2 evidence decision policy canonicalization 增加 `selected_evidence_present`

在 `_canonicalize_evidence_decision_policy()` 和 `_compact_evidence_decision_policy()` 中，把：

```python
"selected_evidence_present": bool(...)
```

同步进 evidence payload。

目的：

- 让 final diagnosis prompt 能明确知道当前是否存在 curated evidence

---

## 四、bootstrap / 数据集分仓改动

### 6. `scripts/bootstrap_ham10000_train_cases.sh`

改动：

- 默认根从 `ham10000_v1` 切换到 `ham10000_v2`
- 新增：
  - `DATA_SPLIT`
  - `SPLIT_JSON`
  - `SPLIT_ID`
- 不再默认按连续全局 index 扫病例
- 改为从 split json 中解析 `train_case_indices`

效果：

- 真正按 train split 建库
- 默认不再写进旧的参数化资产仓

### 7. `scripts/bootstrap_isic2019_train_cases.sh`

同样改为：

- 优先读取 split json
- 从 `train_case_indices` 或 `train_range` 中取 train 病例

效果：

- ISIC2019 的 bootstrap 也变成 split-aware

### 8. `scripts/manage_dataset_experiment_assets.py`

新增：

```python
_sanitize_policy_for_heuristic_only(...)
```

作用：

- 新建数据集实验资产时，自动把 copied / seeded policy 清洗成 heuristic-only
- 关闭：
  - learned controller
  - learned retrieval reranker
  - learned calibrator

效果：

- 新数据集接入时，默认不会把旧主线 learned 配置带进去

---

## 五、external / final 入口修复

### 9. `final-script/tools/external_final_runner.py`

原来问题：

- 手工读取原始 CSV
- 手工构造 `CaseInput`
- 容易把 `dx` / one-hot ground truth 等标签字段带进 metadata

v5 改成：

- HAM10000 走 `load_ham10000_case_inputs()` / `load_ham10000_records()`
- ISIC2019 走 `load_isic2019_case_inputs()` / `load_isic2019_records()`

效果：

- 使用已有安全 loader
- 避免标签泄漏

### 10. `final-script/runs/run_external_qwen_vs_agent_qwen.sh`

原来：

- 复用主线 `QWEN_FINAL_POLICY_ROOT`
- 复用主线 `QWEN_FINAL_FROZEN_STATE_ROOT`

v5 改成：

- HAM10000 external 默认走 `ham10000_v2`
- ISIC2019 external 默认走 `isic2019_v1`

效果：

- external compare 不再误用主线 Qwen frozen 资产

---

## 六、状态元数据整理

### 11. `state/dataset_adaptation/ham10000_v1/*`

把旧 `ham10000_v1` 的：

- `current_stable_policy.json`
- `policy/manifest.json`
- `asset_manifest.json`

统一标记成 heuristic-only 版本。

虽然这条线不是默认新入口，但做这步能避免有人手工 source 老环境时又被误导回 parameterized 线路。

### 12. `state/dataset_adaptation/ham10000_v2/*`

同样把：

- `policy_id`
- `version`
- `manifest`
- `asset manifest`

统一成 heuristic-only 的描述。

效果：

- 运行状态和元数据描述一致
- 后续审计和结果记录不再显示为 parameterized 主线

---

## 七、文档改动

### 13. `使用规范.md`

重写成可直接运行的使用规范，覆盖：

- PAD-UFES-20 / HAM10000 / ISIC2019 三条线路
- 如何切换分仓
- 如何跑 12 / 24 / 30 case compare
- 一行版命令
- 常见错误（尤其是 bash 续行反斜杠后有空格）

---

## 八、v5 的实际效果变化

基于当前已跑结果，v5 相关改动带来的变化大致是：

### PAD-UFES-20

- round1 agent 已优于 baseline
- round2 进一步提升
- 说明保守融合没有伤到原本正收益的数据集

### ISIC2019

- 从 round1 基本持平
- 到 round2 top-1、恶性召回、error rate 都优于 baseline
- 说明“baseline anchoring + conservative fusion” 有效减少了错误漂移

### HAM10000

- 从 round1 明显负收益
- 到 round2 接近 baseline
- 说明 v5 把最坏的错误翻转压住了，但 HAM10000 还没有完全稳定超过 baseline

---

## 九、总结

v5 的本质不是简单多加几条 prompt，而是做了三层收口：

1. 把多数据集分仓和 bootstrap 入口理顺
2. 把 baseline 结果引入 agent 主链，作为真正的保守锚点
3. 在 prompt 约束之外，再加一层确定性的 conservative fusion

所以 v5 的核心价值是：

- 更适合往更多数据集扩展
- 更不容易在证据不足时“越帮越忙”
- 在不伤 PAD 的前提下，开始把 ISIC2019 拉成正收益，并把 HAM10000 从明显负收益修到接近持平
