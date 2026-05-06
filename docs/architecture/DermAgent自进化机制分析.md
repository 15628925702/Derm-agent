# DermAgent 自进化机制分析

更新时间：2026-04-29

---

## 概述

DermAgent 的"自进化"不是训练 backbone，而是在 backbone 外部构建了一套完整的**经验积累 → 认知更新 → 决策偏置**闭环。每次病例执行后，系统会把推理过程中的有效信号沉淀为多层经验资产，并让这些资产持续影响后续的技能检索、规划和证据组织方式。

自进化体现在以下六个层面：

1. 单病例反思与写回
2. 三层经验体系的分层沉淀
3. 经验库的整体状态重建
4. 认知状态的长期积累与反向影响
5. 批量再学习机制
6. 向更高阶知识演化的接口

---

## 1. 单病例反思：成长闭环的起点

每次病例执行完成后，`agent/reflection.py` 的 `build_reflection()` 会生成一份结构化反思，这是整个成长机制的触发点。

### 1.1 病例分类

反思首先把当前病例归入三类：

| 类型 | 触发条件 | 用途 |
|---|---|---|
| `raw_case_experience` | 正常执行完成 | 通用经验积累 |
| `confusion_experience` | 预测标签 ≠ 参考标签 | 混淆对追踪 |
| `hard_case_experience` | 高不确定性或低置信度 | 错题本 |

### 1.2 技能评估（skill_assessments）

反思最核心的产出是对每个被调用技能的逐一评估，判断维度包括：

```
helpfulness:          success / partially_helpful / harmful
evidence_strength:    float（证据强度）
uncertainty_reduction: bool（是否降低了不确定性）
contradiction_detection: bool（是否发现了矛盾）
malignant_flag_support: bool（是否支持了恶性风险判断）
evidence_usage_score: float（证据被最终诊断实际使用的程度）
```

这不是简单记录"调用了哪些技能"，而是判断每个技能在这次推理中的实际价值。这份评估会直接进入后续的技能统计和规划偏置。

### 1.3 认知更新信号

反思同时生成一份 `cognition_update`，包含：

- `total_cases_increment`：+1
- `failed_cases_increment`：预测错误时 +1
- `confusion_cases_increment`：标签不匹配时 +1
- `hard_cases_increment`：高不确定性时 +1
- `preferred_skills`：本次有帮助的技能列表
- `confusion_pair`：本次出现的混淆对（如 `"melanoma->nevus"`）

### 1.4 写回触发

`agent/run_agent.py` 主链在病例执行完成后：

```python
# run_agent.py:219-228
state.reflection = build_reflection(state, cognition_state)
if effective_writeback:
    if state.reflection.get("write_experience"):
        written_bundle = bank.writeback(state.reflection.get("writeback_bundle", {}))
    apply_cognition_update(cognition_state, state.reflection, state)
    cognition_state.save(cognition_path)
```

frozen evaluation 阶段会关闭 `effective_writeback`，保证正式评测不受在线成长污染。

---

## 2. 三层经验体系：分层沉淀而非平铺日志

`memory/experience_transform.py` 把一次病例转成三个粒度的经验，这是 DermAgent 区别于普通 prompt chaining 的核心设计。

### 2.1 第一层：原始病例记忆（raw_case_memory）

保存完整病例快照：

```
case_id / dataset_name / true_label / predicted_label
correctness: {is_correct, status}
skill_outputs: {skill_name: output_dict}
perception: 初始感知结果
metadata: 临床元数据
case_outcome: 结果分类
skill_assessments: 技能评估列表
```

用途：具体病例检索、相似案例参考、后续批量分析的原始数据源。

### 2.2 第二层：战术经验（tactical_experience）

把病例中每个技能的局部决策提炼成可复用的战术片段：

```
skill_name / evidence_type / evidence_summary
recommendation_type / impact / evidence_strength
applicable_scenarios: [适用场景列表]
failure_modes: [常见失败模式]
```

用途：在新病例中检索"在什么条件下，某个技能做了什么，结果怎样"，支持中层战术复用。

### 2.3 第三层：抽象经验（abstract_experience）

进一步跨病例抽象，产出四类资产：

**a) 原型（prototype）**
来自多次正确病例的稳定模式，记录某个诊断标签的典型视觉特征、常见解剖部位、元数据分布。

**b) 混淆记忆（confusion_memory）**
来自反复出现的混淆对，记录两个诊断之间的鉴别要点、常见误判方向、有效区分技能。

**c) 规则候选（rule_candidate）**
来自重复出现的战术决策模式，提炼成"在 X 条件下，优先做 Y"的可审阅规则。

**d) 复合技能种子（composite_skill_seed）**
来自重复成功的多技能序列，记录哪些技能组合在特定场景下稳定有效，为后续技能提案提供依据。

这三层的关系是：具体例子 → 中层战术 → 跨病例抽象原则，而不是简单的日志堆积。

---

## 3. 经验库的整体状态重建：不是追加日志

`memory/experience_store.py` 的设计决定了经验库不是"往末尾加一行"，而是每次写入后触发全库重建。

### 3.1 写入触发的全局操作

任何一条新经验（raw / tactical / abstract）写入后，都会调用 `refresh_metadata()`，执行：

**manifest 和计数更新**
- `raw_case_count / tactical_count / abstract_count`
- 当前文件路径记录
- `state_partition`（split 标识）

**版本哈希重算**
- 基于全体 `raw_case_ids / tactical_ids / abstract_ids` 重算 `record_hash`
- 生成新的 `split_aware_version`
- 经验库被当成整体状态版本管理，而不是松散文件集合

**索引文件重建**
- `case_id_to_raw.json`
- `tactical_by_case.json`
- `abstract_by_type.json`
- `confusion_memory_index.json`

### 3.2 抽象经验的 upsert 合并

对 `abstract_experience`，不是简单重复插入，而是按 `abs_id` 做 upsert：

```
已有同一抽象经验时，合并：
  supporting_cases（支持案例列表）
  counter_cases（反例列表）
  provenance（来源追踪）
  composite_skill_seed（复合技能种子）
  promotion_interface（晋升候选状态）
```

这意味着同一个混淆模式或原型，会随着更多病例的加入而不断被强化，而不是产生重复条目。

### 3.3 检索空间的整体变化

`ExperienceRetriever` 每次检索都重新遍历当前全库并重新打分。一条新经验加入后，变化的不只是"多了一条记录"，而是整个候选池和 top-k 结果都可能被重排。

---

## 4. 认知状态：长期积累反向影响决策

`cognition/cognition_state.py` 维护系统级长期状态，持久化到 `state/cognition_state.json`。这是"成长真正影响后续行为"的关键环节。

### 4.1 维护的状态字段

```
known_confusion_patterns: dict[str, int]
    # {"melanoma->nevus": 5, "bcc->scc": 3}
    # 记录哪些混淆对反复出现，以及出现次数

preferred_skills: list[str]
    # 按历史帮助率排序的技能列表

skill_statistics: dict[str, dict]
    # 每个技能的：
    #   call_count / selected_count / success_count
    #   partially_helpful_count / harmful_count
    #   avg_evidence_strength / helpful_rate / harmful_rate

workflow_preferences: dict[str, dict]
    # 在某类 workflow 场景下：
    #   preferred_skills（历史有效技能）
    #   avg_accuracy / case_count

failure_statistics:
    # total_cases / failed_cases / confusion_cases / hard_cases

evolution_generation: int
    # 每次反思后递增，用于追踪成长代数
```

### 4.2 认知状态如何影响技能检索

`agent/skill_retriever.py:182` 中，技能召回分数会受认知状态影响：

- `helpful_rate` 高的技能得分提升
- `failure_rate` 高的技能得分降低
- `known_confusion_patterns` 中出现的混淆对会触发对应专科技能的优先召回

### 4.3 认知状态如何影响规划

`agent/planner.py` 中，认知状态在两个位置介入：

**planner.py:381** — 基础分数调整：
- `preferred_skills` 中的技能加分
- `helpful_rate` 高的技能加分
- `failure_rate / harmful_rate` 高的技能减分

**planner.py:735** — workflow 条件化加分：
- `workflow_preferences` 中，特定 workflow 场景下历史有效的技能组合再次加分

这形成了明确的行为闭环：前面的病例改变 `cognition_state`，`cognition_state` 改变后面的技能检索和规划，后面的病例再继续更新 `cognition_state`。

---

## 5. 批量再学习：从大量病例压缩更高阶知识

除了单病例在线写回，仓库还提供了三类批处理层面的再学习机制。

### 5.1 Hard Case Mining（agent/hard_case_miner.py:69）

从大量 execution records 中挖出：

- agent 相对 baseline 出现回退的病例
- 高不确定性失败病例
- 反复出现的混淆簇
- 重要但容易错的病例群

产出：结构化的"错题本"，供后续经验整合和人工审查使用。

### 5.2 Skill Helpfulness Analysis（agent/skill_helpfulness_analyzer.py:45）

跨病例汇总每个技能的：

- 帮助率和伤害率（跨所有病例的统计）
- 适用场景分布
- 常见失败模式
- evidence strength 分布

产出：技能级别的"复盘报告"，可用于调整 planner 的先验偏置。

### 5.3 Experience Consolidation（memory/experience_consolidator.py:31）

把大量 raw/tactical/abstract 经验再压缩成更稳定的抽象资产：

| 产出类型 | 来源 | 含义 |
|---|---|---|
| `prototype` | 多次正确病例的稳定模式 | 某诊断的典型表现 |
| `confusion_memory` | 反复出现的混淆对 | 两个诊断的鉴别要点 |
| `rule_candidate` | 重复出现的战术决策模式 | 可审阅的诊断规则 |
| `composite_skill_seed` | 重复成功的多技能序列 | 稳定有效的技能组合 |

这一步的意义在于：系统不只会"记"，还会"压缩和提炼"，把偶然成功的模式和跨病例稳定规律区分开来。

---

## 6. 向更高阶知识演化的接口

### 6.1 Composite Skill Proposal（agent/composite_skill_proposal_generator.py:43）

成长路径：

```
单病例 composite_skill_seed 生成
    ↓ experience_transform.py:512
批量 seed 整合（experience_consolidator.py:330）
    ↓ 支持度达到阈值
正式 proposal 生成（composite_skill_proposal_generator.py:43）
    ↓ 人工 review checklist
（可选）注册为新技能
```

系统已经能自动提出"未来可以新增哪种复合 workflow 技能"，但不会自动上线执行。proposal 仍需人工审查，这是当前保持的审慎边界。

### 6.2 Controller Training Data（agent/controller_training.py:36）

执行记录被转换为监督学习样本：

```
case_state_features:  当前病例的状态特征
candidate_skills:     候选技能列表
selected_skills:      实际选择的技能
helpful_skills:       事后判断有帮助的技能
harmful_skills:       事后判断有害的技能
delta_vs_baseline:    相对 baseline 的增益
reward:               综合奖励信号
```

这为后续离线训练 learned controller 提供了完整数据面，使得 heuristic 运行历史可以转化为监督信号。

---

## 7. 自进化的边界：已落地 vs 尚未自动闭环

### 已明确落地

| 机制 | 状态 |
|---|---|
| 单病例 reflection 生成 | 完整运行 |
| 三层经验分层写回 | 完整运行 |
| CognitionState 增量更新 | 完整运行 |
| 认知状态反向影响 skill retrieval | 完整运行 |
| 认知状态反向影响 planner | 完整运行 |
| Hard case 挖掘 | 完整运行 |
| Skill helpfulness 分析 | 完整运行 |
| Experience consolidation | 完整运行 |
| Composite skill seed 生成 | 完整运行 |
| Controller training data 导出 | 完整运行 |

### 尚未自动闭环

| 机制 | 当前状态 |
|---|---|
| Qwen backbone 权重更新 | 不做，设计边界 |
| 新技能自动注册上线 | 需人工 review |
| Learned controller 自动替换 heuristic | 候选线，非默认 |
| Frozen evaluation 阶段在线成长 | 故意关闭 |

---

## 8. 整体闭环结构

```
单病例执行
    ↓
reflection（skill_assessments + cognition_update + writeback_bundle）
    ↓
experience_writer（raw / tactical / abstract 分层写回）
    ↓
experience_store（全库 manifest / 版本哈希 / 索引重建 / abstract upsert）
    ↓
cognition_state（confusion_patterns / skill_statistics / workflow_preferences 更新）
    ↓
cognition_state.save()
    ↓
下一病例执行
    ├─ skill_retriever（helpful_rate / confusion_patterns 影响召回分数）
    ├─ planner（preferred_skills / workflow_preferences 影响规划）
    └─ experience_retriever（全库重排后的检索结果）
```

批量层面额外叠加：

```
大量 execution records
    ↓
hard_case_miner → 错题本
skill_helpfulness_analyzer → 技能复盘
experience_consolidator → prototype / confusion_memory / rule_candidate / composite_seed
composite_skill_proposal_generator → 新技能候选（人工审查）
controller_training → learned controller 训练数据
```

---

## 总结

DermAgent 自进化的核心不是"参数自训练"，而是：

> 把每次病例执行后的推理信号沉淀为可复用的多层经验资产，并让这些资产持续塑造后续的检索、技能选择和证据组织方式。

它的成长体现在六个可审计的层面：单病例反思、三层经验沉淀、经验库整体重建、认知状态长期积累、批量再学习、以及向新技能和 learned controller 演化的接口。当前最稳定落地的是前四层；批量再学习和高阶演化接口已经工程化，但仍保持人工审查边界。
