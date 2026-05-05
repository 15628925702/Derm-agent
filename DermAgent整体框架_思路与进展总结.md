# DermAgent 整体框架、核心思路与当前进展总结

更新时间：2026-04-29  
阅读范围：`README.md`、`当前最佳版本说明.md`、`当前实验结果与大规模运行建议_20260424.md`、`使用规范.md`、`design/`、`workflow改进/`、`v3/v4/v7` 阶段文档，以及 `agent/`、`skills/`、`memory/`、`cognition/`、`dataio/`、`scripts/`、`tests/` 等核心代码。

---

## 1. 一句话概括 DermAgent 是什么

DermAgent 不是一个替代底层视觉语言模型的“新分类器”，而是一个加在皮肤科视觉语言模型前后的**结构化推理支架**。  
它的目标不是自己直接给最终疾病标签，而是把“最终诊断之前的临床推理过程”拆成可调用、可审计、可积累经验的模块，再把这些模块产出的证据组织成一个 `Evidence Package`，交回给同一个 backbone 做最终诊断。

仓库想回答的核心问题也非常清楚：

> 在固定同一个皮肤科 backbone 的前提下，结构化、经验驱动、workflow-aware 的 agent reasoning，能否稳定优于 direct prompting？

当前默认 backbone 是本地 OpenAI-compatible Qwen 服务，因此主对照一直是：

- `direct Qwen`
- `DermAgent + Qwen`

---

## 2. 核心设计边界

这个仓库最重要的原则不是“功能多”，而是边界清楚：

1. DermAgent 不负责最终疾病标签。
2. 最终诊断责任始终保留给 backbone。
3. DermAgent 只负责组织证据、暴露不确定性、调用经验、控制 skill 执行顺序。
4. 正式评测必须保证同模型、同 case list、同 split、同 frozen state，对 direct baseline 和 full agent 做公平对照。

因此它不是：

- 多模型投票系统
- end-to-end 训练的皮肤病分类器
- 用技能直接替代 final diagnosis 的 rule engine

它更像是一个“临床推理中间层”。

---

## 3. 实际代码中的主执行链路

真正的主入口是 `agent/run_agent.py` 里的 `run_agent()`。  
按代码执行顺序，单个病例的大致流程如下：

### 3.1 输入层

- `CaseInput` 定义在 `agent/state.py`
- 包含 `case_id`、`image_path`、`metadata`、`label/reference_label`、`dataset_name`、`label_space_id`、`workflow_context`
- `clinical_metadata()` 会主动过滤泄漏字段，避免把 `dx / label / ground_truth / one-hot label` 之类信息直接塞进 prompt

### 3.2 初始感知

`integrations/openai_client.py` 里的 `initial_perception()` 先让 Qwen 做一次早期观察，输出：

- `image_summary`
- `ddx_candidates`
- `uncertainty`
- `notes`

这一步明确要求：

- 只能做早期观察和粗鉴别
- 不能给最终诊断

### 3.3 baseline anchor

随后 `baseline_diagnosis()` 会先算一个**不用 agent 证据包的 direct baseline 结果**。  
这个 baseline 不只是评测对照，也会被后续 evidence policy 当作 anchor 使用。

### 3.4 经验检索

`memory/experience_bank.py` + `memory/experience_retriever.py` 负责经验库检索。  
当前经验被拆成三层：

- `raw_case_memory`：原始病例记忆
- `tactical_experience`：中层“遇到什么情况优先做什么”
- `abstract_experience`：跨病例抽象规则、混淆模式、原型经验

检索 query 不是简单“相似图像”，而是综合：

- 初始 ddx
- morphology clues
- metadata patterns
- risk patterns
- confusion pair / confusion clusters
- uncertainty level

### 3.5 skill retrieval + planner

当前 skill 不是全部盲跑，而是两层控制：

1. `agent/skill_retriever.py`
   - 先从技能库里召回候选 skill
2. `agent/planner.py`
   - 再从候选里选出本病例真正要执行的技能顺序

planner 的默认稳定路线仍是 **heuristic rule-based**，但代码里已经留好了 learned controller 的接口。

### 3.6 skill 执行

技能系统定义在 `skills/`，统一由 `skills/base.py` 和 `skills/registry.py` 管理。  
每个 skill 都必须满足同一约束：

- 只能产出结构化证据
- 不能直接输出 final diagnosis
- 输出写入 `state.skill_outputs[skill_name]`

当前技能大致分成几类：

- 观察类：`morphology / color_pattern / border_surface / distribution / lesion_description_structuring`
- 推理类：`metadata_consistency / differential_compare / exclusion_reasoning / temporal_evolution`
- 风险与不确定性：`malignancy_risk / uncertainty / contradiction_check / information_gap_detection / escalation_recommendation`
- specialist：`mel_nev_specialist / ack_scc_specialist / benign_mimic_specialist`

### 3.7 第二次检索与证据聚合

skill 执行后，系统会再次检索经验，然后由 `agent/aggregator.py` 生成 `Evidence Package`。  
这一层会把以下内容组织起来：

- 初始感知摘要
- skill outputs
- raw / tactical / abstract 经验摘要
- risk / contradiction / uncertainty / information gap / escalation 摘要
- planner rationale
- confusion cluster summary
- `selected_evidence`
- `serialized_evidence_text`
- `evidence_decision_policy`

这一步不是简单拼接文本，而是做了 evidence calibration、排序、筛选、去重、workflow-aware 重排。

### 3.8 final diagnosis

`openai_client.py` 的 `final_diagnosis()` 让 backbone 读取结构化证据包后输出最终诊断。  
这里仍然强调：

- backbone 才是唯一 final diagnosis maker
- agent evidence 是 supporting context，不是硬指令

### 3.9 conservative fusion、reflection、writeback

final diagnosis 之后还有三件事：

1. `agent/conservative_fusion.py`
   - 提供保守融合机制，避免 agent 在证据不足时把 baseline 推坏
   - 代码中模块完整存在，但当前 stable policy 的正式默认并不强依赖它

2. `agent/reflection.py`
   - 判断这次病例是普通经验、hard case 还是 confusion case
   - 产出 skill helpfulness、case outcome、reflection extract

3. writeback 到经验库和认知状态
   - 训练 / bootstrap 阶段可写回
   - frozen compare 阶段禁止在线写回

---

## 4. 整体架构可以怎么理解

如果把整个系统抽象成一张图，大致是：

`CaseInput -> Initial Perception -> Experience Retrieval -> Skill Retrieval -> Planner -> Skills -> Evidence Package -> Final Diagnosis -> Reflection -> Writeback`

其中真正的“结构化智能体增益”主要来自 5 个层面：

### 4.1 Skill System

把皮肤科推理从“一次性长文本”拆成原子化临床动作。  
这样 workflow 的差异就变成：

- 哪些 skill 被触发
- 顺序如何
- 哪些 skill 被抑制

而不是整段 prompt 风格差异。

### 4.2 Experience System

把经验从“参数里的隐式记忆”外显成三层显式资产：

- 具体病例
- 中层战术
- 抽象原则

这使得系统不是简单找相似病例，而是能在不同粒度上复用经验。

### 4.3 Cognition System

`cognition/cognition_state.py` 维护系统级长期状态，包括：

- 已知混淆模式
- 偏好技能
- 检索偏好
- 失败统计
- skill statistics
- workflow preferences

它相当于“这个 agent 在长期运行后，对自己擅长什么、容易错什么、在哪类 workflow 里哪些技能更有效”的自我画像。

### 4.4 Workflow-Aware Routing

当前仓库一个很重要的工程判断是：

> 运行时行为应该优先由 `workflow_profile / workflow_capabilities` 决定，而不是直接按 dataset name 写特殊开关。

这体现在 `agent/workflow_profiles.py`。  
系统会根据 metadata、label granularity、presentation mode 等，推断病例属于哪种 workflow。

### 4.5 Evidence Package

Evidence Package 是整个仓库的“临床接口”。  
它把多步推理、中间证据、经验摘要、不确定性和反对证据收敛成一个可审阅的对象，替代黑箱式 CoT。

---

## 5. 当前已经形成的 6 条 workflow

目前正式主线不是一个 workflow，而是统一 core 下的 6 条可审计 workflow：

1. `PAD-UFES-20`
   - `clinical_full_taxonomy_lesion_workflow`

2. `ISIC2019`
   - `image_archive_full_taxonomy_lesion_workflow`

3. `HAM10000`
   - `sparse_lesion_workflow`

4. `SCIN grouped`
   - `family_routing_workflow`

5. `SD-198 grouped`
   - `coarse_taxonomy_workflow`

6. `Xiangya SFT`
   - `eczematous_family_routing_workflow`

这 6 条线共享同一套 agent core，但在以下方面分流：

- label granularity
- presentation mode
- workflow capabilities
- prompt hint
- confusion logic
- specialist skill 策略
- grouped vs full-taxonomy 评测口径

这也是仓库从“PAD 主线原型”升级成“多数据集平台”的关键一步。

---

## 6. 仓库目录的真实职责

### 6.1 `agent/`

最核心的业务逻辑都在这里：

- `run_agent.py`：主链
- `planner.py`：skill 选择
- `skill_retriever.py`：候选 skill 检索
- `aggregator.py`：证据聚合
- `reflection.py`：反思与写回
- `evaluation_protocol.py`：正式 frozen eval 协议
- `policy_config.py`：stable / candidate policy 管理
- `conservative_fusion.py`：保守融合
- `controller_training.py`、`retrieval_scorer.py`、`supervised_controller.py`：可训练组件

### 6.2 `skills/`

这里不是“prompt 模板堆”，而是严格 schema 化的 atomic reasoning skills。

### 6.3 `memory/`

经验库的 schema、检索、变换、写回、持久化都在这里，属于 DermAgent 区别于普通 prompt chaining 的关键模块。

### 6.4 `cognition/`

维护跨病例长期状态，是“agent 持续学习但不训练 backbone”的主要载体。

### 6.5 `dataio/`

已经从手工路由逐步收敛成注册式 loader 机制。  
当前重点是：

- 每个数据集必须明确 `dataset_name`
- 明确 `label_space_id`
- metadata 必须做防泄漏清洗
- 必须能进入 split / bootstrap / compare 主流程

### 6.6 `scripts/`

是真正的实验操作层：

- 启服务
- bootstrap 经验库
- compare direct vs agent
- final-round 总控
- 训练 learned components
- 导表、导图、审计、hard-case mining

### 6.7 `tests/`

当前有 `48` 个测试文件，覆盖的不是单纯 unit test，而是：

- loader / split / label space
- evaluation protocol
- training pipeline
- final-round 脚本
- contamination guard
- policy config
- evidence calibration
- skill helpfulness

这说明仓库已经不是单纯实验脚本集合，而是带协议约束的工程化研究仓库。

---

## 7. 这套系统背后的核心思路

从设计文档和版本演进来看，DermAgent 的真正 thesis 不是“多加几个模块”，而是下面这几个判断：

### 7.1 皮肤科诊断不是纯图像分类，而是证据组织问题

很多病例需要同时处理：

- 视觉形态
- 解剖部位
- 病程演化
- 风险提示
- 临床 metadata
- 不确定性
- 常见混淆对

因此仓库试图优化的对象不是“让模型更会答题”，而是“让系统更会在诊断前组织证据”。

### 7.2 真实世界的问题不是只有 dataset shift，还有 workflow heterogeneity

仓库早期文档一直反复强调：

- 不同医院
- 不同信息完整度
- 不同专科层级
- 不同检查可得性
- 不同记录习惯

这些不是简单的数据分布变化，而是**流程组织方式的差异**。  
所以 DermAgent 不是让模型只学一条专家路径，而是让 agent 根据当前 workflow 动态编排技能、经验和证据。

### 7.3 优化边界放在 backbone 外部

仓库设计非常克制：

- 不训练 Qwen
- 不做 end-to-end
- 允许训练 controller / retrieval scorer / calibrator
- 但所有 learned component 都必须可审计、可回滚、可 frozen replay

这使得系统更适合做公平比较，也更适合论文中的方法学论证。

### 7.4 正式主线以 heuristic 为主，不靠 learned line 讲故事

从 `当前最佳版本说明.md`、`使用规范.md` 和 stable policy 来看，当前默认稳定口径是：

- planner：`heuristic`
- retrieval reranker：默认关闭
- evidence calibrator：`heuristic`

也就是说，仓库目前最有说服力的结论不是“learned controller 很强”，而是：

> 即便不用 learned 组件，仅靠结构化 workflow、skills、experience、cognition 和 evidence package，本身就能带来可观增益。

---

## 8. Agent 的自进化与成长机制

这一部分非常值得单独说明，因为 DermAgent 的“成长”不是传统意义上的模型在线训练，也不是让 Qwen 自己边推理边改权重。  
它的成长发生在 **backbone 外部的 agent 组织层**，核心闭环是：

`单病例执行 -> reflection -> 经验写回 -> cognition 更新 -> 下一病例的检索/规划偏置变化`

也就是说，DermAgent 进化的不是最终分类器，而是：

- 更会记住哪些病例值得参考
- 更会判断哪些 skill 在什么场景下有帮助
- 更会识别哪些混淆对在反复出现
- 更会在特定 workflow 下复用历史上有效的技能组合

### 8.1 成长不是“训练 backbone”，而是更新 agent 外部状态

`run_agent.py` 里主链已经把这件事接通了。  
在一次病例执行完成后，系统会：

1. 生成 `reflection`
2. 决定是否写回经验
3. 更新 `CognitionState`
4. 将更新后的状态保存到 split 对应状态目录

这部分逻辑直接写在 [run_agent.py](/root/DermAgent/agent/run_agent.py:219) 到 [run_agent.py](/root/DermAgent/agent/run_agent.py:228)。

因此，DermAgent 的自进化首先体现为：

- 它不是每次都从零开始
- 它会把前一批病例的结果变成下一批病例的决策偏置

### 8.2 reflection 是成长闭环的起点

`agent/reflection.py` 是整个成长机制的起点。  
在 [reflection.py](/root/DermAgent/agent/reflection.py:18) 开始，系统会基于：

- 最终预测是否正确
- 当前不确定性高不高
- 有没有明确 confusion pair
- 哪些 skill 真正输出了有意义的证据

把当前病例归入三类经验之一：

- `raw_case_experience`
- `hard_case_experience`
- `confusion_experience`

同时还会生成：

- `case_outcome`
- `skill_assessments`
- `reflection_extract`
- `writeback_bundle`
- `cognition_update`

其中最关键的是 `skill_assessments`。  
它不是简单记录“调用了哪些技能”，而是进一步判断每个 skill：

- 是 `success / partially_helpful / harmful / failure`
- 提供了什么证据强度
- 是不是减少了不确定性
- 是不是发现了矛盾
- 是不是提供了恶性风险支持
- 有没有形成可复用经验

这使得系统的成长不是“记住结果”，而是“记住这次推理过程中哪些动作有价值”。

### 8.3 经验库的成长：从病例到多层经验

`memory/experience_transform.py` 把一次病例转成三层经验：

1. `raw_case_memory`
   - 见 [experience_transform.py](/root/DermAgent/memory/experience_transform.py:302)
   - 保存完整病例、最终结果、skill outputs、引用过的经验、正确性状态

2. `tactical_experience`
   - 见 [experience_transform.py](/root/DermAgent/memory/experience_transform.py:350)
   - 把一次病例中的局部决策提炼成“在什么条件下，某个 skill 做了什么，结果怎样”

3. `abstract_experience`
   - 见 [experience_transform.py](/root/DermAgent/memory/experience_transform.py:404)
   - 进一步抽象成：
     - `confusion_memory`
     - `rule`
     - `prototype`
     - `composite_skill_seed`

这意味着 DermAgent 的成长不是平铺日志，而是分层沉淀：

- 先保留具体例子
- 再保留中层战术
- 再保留跨病例抽象原则

这是整个仓库里最“agentic”的成长机制之一。

### 8.4 认知状态的成长：经验如何反过来影响下一次决策

如果只有经验写回，但下一次决策根本不读这些状态，那就不能算真正成长。  
DermAgent 在这点上是打通的。

`cognition/cognition_state.py` 中，长期状态至少包括：

- `known_confusion_patterns`
- `preferred_skills`
- `retrieval_preferences`
- `failure_statistics`
- `skill_statistics`
- `workflow_preferences`

对应更新逻辑见 [cognition_state.py](/root/DermAgent/cognition/cognition_state.py:99) 之后。

这里最关键的几类增长信号是：

1. `known_confusion_patterns`
   - 系统会累计哪些 confusion pair 反复出现

2. `skill_statistics`
   - 会累计每个 skill 的
     - `call_count`
     - `helpful_rate`
     - `failure_rate`
     - `harmful_count`
     - `average_evidence_strength`

3. `workflow_preferences`
   - 会记录在某类 workflow 场景下，哪些 skill 组合历史上更有效

这些状态不是摆设。  
它们会直接进入后续检索和 planner 决策：

- 在 [skill_retriever.py](/root/DermAgent/agent/skill_retriever.py:182)，`helpful_rate / failure_rate` 会影响 skill 召回分数
- 在 [planner.py](/root/DermAgent/agent/planner.py:381)，`preferred_skills`、`helpful_rate`、`failure_rate`、`harmful_rate` 会直接加减分
- 在 [planner.py](/root/DermAgent/agent/planner.py:735)，`workflow_preferences` 会给特定 workflow 下历史有效的 skill 再加分

换句话说，DermAgent 的成长已经形成了明确的行为闭环：

- 前面的病例改变 `cognition`
- `cognition` 改变后面的 skill retrieval 和 planner
- 后面的病例再继续更新 `cognition`

这才是它真正意义上的“越跑越像一个积累过经验的 agent”。

### 8.5 经验写回已经工程化，而不是概念停留

经验写回逻辑在 `memory/experience_writer.py`。  
[experience_writer.py](/root/DermAgent/memory/experience_writer.py:13) 会把：

- `raw_case_memory`
- `tactical_experiences`
- `abstract_experiences`

分别 upsert 到持久化 store 中。  
它还会进一步刷新 `reflection_extract`，特别是标记：

- 产生了多少 `abstract_experience`
- 产生了多少 `composite_skill_seed`
- 哪些 seed 已经达到了“可以晋升候选”的支持数

这说明仓库里的成长不是纸上设计，而是已经进入状态资产管理层。

这里还需要特别澄清一件事：

> DermAgent 的经验库更新，不是“单纯往 jsonl 末尾再加一行”这么简单；每加入一个新经验后，系统会对经验库的若干全局结构做重算和刷新。

这部分真正由 `memory/experience_store.py` 驱动。  
在 [experience_store.py](/root/DermAgent/memory/experience_store.py:73)、[experience_store.py](/root/DermAgent/memory/experience_store.py:82)、[experience_store.py](/root/DermAgent/memory/experience_store.py:92) 可以看到，不管写入的是 `raw`、`tactical` 还是 `abstract` 经验，最终都会触发 `refresh_metadata()`；而 [experience_store.py](/root/DermAgent/memory/experience_store.py:103) 之后这一步会做全库刷新，而不是局部追加。

一次新经验写入后，至少会有下面几类“全面更新”：

1. **manifest 和全库计数更新**
   - `raw_case_count`
   - `tactical_count`
   - `abstract_count`
   - 当前文件路径记录
   - `state_partition`

2. **split-aware 版本和整体哈希更新**
   - `ExperienceStore` 会基于全体 `raw_case_ids / tactical_ids / abstract_ids` 重算 `record_hash`
   - 然后生成新的 `split_aware_version`
   - 这意味着经验库被当成一个整体状态版本，而不是松散日志集合

3. **索引文件重建**
   - `case_id_to_raw.json`
   - `tactical_by_case.json`
   - `abstract_by_type.json`
   - `confusion_memory_index.json`

4. **抽象经验的全局 merge**
   - 对 `abstract_experience`，不是简单重复插入，而是按 `abs_id` 做 upsert
   - 如果已有同一个抽象经验，会合并：
     - `supporting_cases`
     - `counter_cases`
     - `provenance`
     - `composite_skill_seed`
     - `promotion_interface`
   - 见 [experience_store.py](/root/DermAgent/memory/experience_store.py:190)

5. **composite seed 的支持度和候选状态更新**
   - 一个新 seed 进来后，不只是“多一条 seed”
   - 如果命中已有 seed，会把 `supporting_cases` 合并、`success_count` 重算
   - 因而“这个组合是不是已经从偶然成功，变成跨病例稳定模式”会被整体刷新

6. **后续检索分布整体变化**
   - `ExperienceRetriever` 每次检索都会重新遍历当前全库的 `raw / tactical / abstract` 记录并重新打分，见 [experience_retriever.py](/root/DermAgent/memory/experience_retriever.py:89) 之后
   - 所以一条新经验加入后，变化的不是“某条记录多了”，而是整个候选池和 top-k 结果都可能被重排

换句话说，DermAgent 的经验库更像一个会持续重建索引和整体状态版本的知识库，而不是简单追加日志：

- 局部新增的是单条经验
- 全局变化的是 manifest、版本、索引、抽象合并结果，以及后续所有检索排序空间

这也是为什么前面说它的成长不是“记多几条例子”，而是“让整个检索-规划-证据组织空间随着经验库变化而重构”。

### 8.6 批量成长：从 many cases 再压缩成更高阶知识

除了单病例在线写回，仓库还提供了批处理层面的“再学习”机制。

#### 8.6.1 hard-case mining

`agent/hard_case_miner.py` 会从大量 execution records 中挖出：

- agent regression vs baseline
- 高不确定性失败
- 反复出现的 confusion cluster
- 重要但容易错的病例簇

入口见 [hard_case_miner.py](/root/DermAgent/agent/hard_case_miner.py:69)。

这相当于 agent 的“错题本”。

#### 8.6.2 skill helpfulness analysis

`agent/skill_helpfulness_analyzer.py` 会跨病例总结：

- 每个 skill 的帮助率和伤害率
- 适用场景
- 常见失败模式
- evidence strength 分布

入口见 [skill_helpfulness_analyzer.py](/root/DermAgent/agent/skill_helpfulness_analyzer.py:45)。

这相当于 agent 的“技能复盘器”。

#### 8.6.3 experience consolidation

`memory/experience_consolidator.py` 会把大量 raw/tactical/abstract 经验再压成更稳定的抽象经验，见 [experience_consolidator.py](/root/DermAgent/memory/experience_consolidator.py:31)。

它会批量产出：

- `prototype`
- `confusion_memory`
- `rule_candidate`
- 更稳定的 `composite_skill_seed`

特别是：

- `prototype` 来自多次正确病例的稳定模式
- `confusion_memory` 来自反复出现的混淆对
- `rule_candidate` 来自重复出现的 tactical 决策模式
- `composite_skill_seed` 来自重复成功的多 skill 序列

这说明系统不仅会“记”，还会“压缩和提炼”。

### 8.7 更高层的成长：向新技能或新控制器演化

仓库里还有两类非常重要但尚未完全自动上线的成长机制。

#### 8.7.1 composite skill seed -> composite skill proposal

在单病例阶段，`experience_transform.py` 已经会为稳定成功的多 skill 序列生成 `composite_skill_seed`，见 [experience_transform.py](/root/DermAgent/memory/experience_transform.py:512)。

在批量阶段，`experience_consolidator.py` 会把跨病例重复出现的 seed 再整合，见 [experience_consolidator.py](/root/DermAgent/memory/experience_consolidator.py:330)。

随后，`agent/composite_skill_proposal_generator.py` 会把这些 seed 变成正式 proposal，见 [composite_skill_proposal_generator.py](/root/DermAgent/agent/composite_skill_proposal_generator.py:43)。

这里最关键的一点是：  
它已经能自动提出“未来可以新增哪种 composite workflow skill”，但**不会自动上线执行**。  
proposal 仍然需要人工 review。

这说明仓库已经具备“长出新技能候选”的能力，但当前仍保持审慎边界。

#### 8.7.2 execution record -> learned controller training data

`agent/controller_training.py` 会把 execution records 转成 controller training examples，见 [controller_training.py](/root/DermAgent/agent/controller_training.py:36)。

这些样本里已经包含：

- case state features
- candidate skills
- selected skills
- helpful / harmful skills
- delta vs baseline
- reward

也就是说，仓库已经具备：

- 从运行记录中自动生成监督信号
- 把 heuristic 运行历史转换成 learned controller 的训练数据

因此它的成长并不只停留在 heuristic state update，还为后续离线学习留好了完整数据面。

### 8.8 当前“自进化”已经做到哪一步

这部分一定要讲清楚，否则容易高估当前自动化程度。

#### 已经明确落地的

1. 单病例 reflection
2. 经验分层写回
3. `CognitionState` 增量更新
4. 认知状态反向影响后续 skill retrieval / planner
5. hard-case 挖掘
6. skill helpfulness 分析
7. experience consolidation
8. composite skill seed 生成
9. controller training data 导出

#### 还没有完全自动闭环的

1. Qwen 不会在线更新权重
2. 新 skill 不会自动注册上线
3. candidate controller / reranker / calibrator 不会自动替换 stable 主线
4. 正式 frozen evaluation 阶段会故意关闭在线成长

因此最准确的表述应该是：

> DermAgent 当前已经是一个会积累病例经验、会复盘技能价值、会更新长期认知状态，并让这些状态反过来改变后续决策的结构化 agent；但它还不是一个会自动改写自身技能语义、自动更新 backbone、自动上线新策略的完全自治系统。

### 8.9 为什么这部分是仓库里最重要的“agent 性”

如果没有这套成长闭环，DermAgent 只会是“结构化 prompt pipeline”。  
真正让它更接近 agent 的，是它具备了以下性质：

- 有长期记忆
- 有错题本
- 有技能复盘
- 有 workflow 条件化偏好
- 有经验压缩与抽象
- 有未来技能/控制器的生长接口

所以从研究视角看，DermAgent 的自进化核心不是“参数自训练”，而是：

> 它把每次病例执行后的结果沉淀为可复用经验和长期认知状态，并让这些状态持续塑造之后的检索、技能选择和证据组织方式。

---

## 9. 版本演进脉络

### 9.1 初版阶段

初版文档提出的四个核心创新点已经很明确：

- skills 显式化
- hierarchical experience
- cognition / policy state
- dynamic evidence pack

这套 conceptual design 后来基本都落到了代码里。

### 9.2 v2 阶段

重点是把 workflow-aware 逻辑真正接到主链上：

- `workflow_context` 显式进入 `CaseInput`
- planner 开始感知 workflow
- evidence package 开始做 workflow-aware 排序
- metadata missingness 开始进入实验验证

这是从“概念原型”走向“有实验闭环”的关键阶段。

### 9.3 v3 阶段

核心工作是多数据集化：

- label space 注册化
- confusion clusters 数据集可配置
- specialist skills 数据集可配置
- split 和 loader 逐步标准化

这一步把很多 PAD 专属 hardcode 拆掉了。

### 9.4 v4/v5 阶段

从文档看，重点转向跨数据集稳定性，尤其是 HAM10000 上的过度恶性化问题。  
这阶段引入和强化了几类东西：

- baseline anchoring
- evidence decision policy
- conservative fusion 模块
- 独立分仓
- heuristic-only stable runtime

可以看出作者已经开始从“能不能提高指标”转向“怎么避免 agent 把 baseline 推坏”。

### 9.5 当前阶段

当前仓库已经进入“统一 core + 6 条 workflow + final-round 结果 + 论文资产”的状态。  
这说明项目已经不只是探索框架，而是在整理正式对外叙事。

---

## 10. 当前进展与结果

根据 `当前实验结果与大规模运行建议_20260424.md`，当前 final-round 的主结果如下：

| Workflow | Cases | Direct top1 | Agent top1 | Direct topk | Agent topk | Direct err | Agent err | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PAD-UFES-20 | 80 | 28.75% | 35.00% | 55.00% | 61.25% | 71.25% | 65.00% | 明确正向 |
| ISIC2019 | 80 | 32.50% | 33.75% | 62.50% | 58.75% | 67.50% | 66.25% | 轻微正向 |
| SCIN grouped | 48/80 可评分 | 47.92% | 50.00% | 47.92% | 54.17% | 52.08% | 50.00% | 正向但协议仍需稳固 |
| SD-198 grouped | 80 | 47.50% | 50.00% | 61.25% | 63.75% | 52.50% | 50.00% | 正向 |
| Xiangya SFT | 13 | 38.46% | 69.23% | 38.46% | 76.92% | 61.54% | 30.77% | 强正向但样本极小 |
| HAM10000 | 100 | 17.00% | 17.00% | 25.00% | 37.00% | 83.00% | 83.00% | top-k 正向，top1 持平 |

如果压缩成一句话：

- 6 条线里，`5/6` 是正向或轻微正向
- `HAM10000` 仍然没有把 top1 拉起来
- 当前最稳的主结果线是 `PAD-UFES-20`、`SD-198 grouped`
- `SCIN` 有正向信号，但评测协议口径还需要更稳
- `Xiangya SFT` 信号很强，但样本太小

---

## 11. 当前最稳的结论与最需要谨慎的地方

### 11.1 可以比较稳地说的

1. DermAgent 已经从单数据集原型发展成统一 core + 多 workflow 平台。
2. 结构化 reasoning scaffold 本身能带来可见增益，不依赖 learned line 才成立。
3. 系统已经具备可审计的中间证据、经验写回、长期认知状态、frozen evaluation 协议。
4. 代码、脚本、资产、测试和论文目录都已形成完整研究工程链条。

### 11.2 需要谨慎说的

1. 不能说“所有数据集都强正向”。
2. 不能说“当前默认依赖 learned components”。
3. 不能把 `SCIN grouped` 和 `PAD / SD-198` 放在同等稳定性上。
4. 不能把 `HAM10000` 现在这版表述成明确成功。

---

## 12. 当前仓库真正完成到什么程度

如果按工程成熟度看，我会把当前仓库判断为：

### 已完成

- 统一 agent core
- skills / memory / cognition / evidence package 主闭环
- 6 条 workflow
- 多数据集 loader + split + compare + final-round
- frozen evaluation 协议
- 经验写回与反思
- 训练组件边界定义
- 较完整测试集

### 已有但不是当前默认主线

- learned controller
- retrieval reranker
- evidence calibrator 训练线
- staged training pipeline
- checkpoint / candidate policy 管理

### 仍在探索或尚未完全收敛

- HAM10000 top1 提升机制
- SCIN grouped 的稳定计分协议
- Xiangya 更大规模评测
- 跨 workflow 的统计强度进一步放大
- learned line 是否能稳定超过 heuristic 主线

---

## 13. 我对这个仓库当前形态的总体判断

DermAgent 现在最像的是一个**已经完成核心方法落地、正在收敛正式实验叙事的研究工程仓库**。

它的强项不在于某个单点模块特别花哨，而在于下面三点已经同时具备：

1. 方法边界清楚：agent 负责证据组织，backbone 负责最终诊断。
2. 工程闭环完整：运行、写回、冻结评测、分仓、测试、论文资产都在。
3. 结果形态有说服力：不是单线偶然成功，而是 6 条 workflow 中多数有正向信号。

它当前最值得强调的主叙事不是“我们训练了一个更强模型”，而是：

> 我们把皮肤科 AI 从一次性答案生成器，重构成了一个会围绕病例主动组织证据、能积累经验、能适配不同 workflow 的结构化智能体层。

---

## 14. 建议的后续工作重点

如果继续推进，这个仓库最合理的下一步不是再无差别加模块，而是：

1. 继续扩大 `PAD-UFES-20`、`ISIC2019`、`SD-198 grouped` 的正式 compare 分母。
2. 先固定 `SCIN grouped` 的可评分协议，再放大样本。
3. 把 `HAM10000` 的目标聚焦到“如何把 top-k 增益转成 top1 增益”。
4. 将 learned components 继续作为候选增强线，但不要替代 heuristic 主叙事。
5. 保持 frozen evaluation、独立分仓和无标签泄漏约束，不要为追指标破坏方法论可信度。

---

## 15. 最终结论

从整个 `DermAgent` 文件夹来看，这个项目已经不再是“一个想法 + 一堆脚本”，而是：

- 有明确 clinical reasoning framing
- 有稳定主执行链
- 有多数据集 workflow 分流
- 有经验与认知写回机制
- 有正式 frozen evaluation 协议
- 有当前可用于论文叙事的阶段性结果

当前最准确的概括是：

> DermAgent 已经完成从 PAD 主线原型到统一多 workflow 皮肤科结构化智能体平台的过渡；其当前最稳定的价值主张是，**在不训练 backbone 的前提下，通过 skills、experience、cognition 和 evidence package 重构诊断前推理流程，可以在多条皮肤科 workflow 上带来总体正向的性能与可审计性收益。**
