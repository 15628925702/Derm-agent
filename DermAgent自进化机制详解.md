DermAgent 自进化机制详解

更新时间：2026-04-29

概述

DermAgent 的"自进化"不是训练 backbone，也不是在线更新 Qwen 的权重。它的进化发生在 backbone 外部的 agent 组织层，核心是一条从单病例执行到长期状态更新的闭环：

单病例执行
  → reflection（反思）
  → 经验分层写回
  → CognitionState 增量更新
  → 下一病例的检索 / 规划偏置变化

这条闭环让系统不是每次从零开始，而是把前面病例的结果变成后面病例的决策偏置。

一、单病例闭环：reflection → 写回 → 认知更新

1.1 reflection 是起点

每个病例执行完成后，agent/reflection.py 的 build_reflection() 会做以下判断：

经验类型分类

条件

经验类型

预测与参考标签不匹配

confusion_experience

不确定性高 / 置信度低

hard_case_experience

其余有效病例

raw_case_experience

产出内容

case_outcome：最终预测是否正确、混淆对、错误类型

skill_assessments：每个 skill 的帮助性评估（success / partially_helpful / harmful / failure）、证据强度、是否减少不确定性、是否发现矛盾

tactical_experiences：从本次病例提炼的中层战术（"在什么条件下，某 skill 做了什么，结果如何"）

abstract_experiences：跨病例抽象（confusion_memory / rule / prototype / composite_skill_seed）

cognition_update：传递给 CognitionState 的增量更新信号

writeback_bundle：打包好的写回数据

写回触发条件（_should_write_experience()）

if experience_type in {"hard_case_experience", "confusion_experience"}:
    return True
return bool(useful_skills)

即：只要是 hard case 或 confusion case，或者有任何 skill 产出了有效证据，就写回。

1.2 经验分三层写入

memory/experience_writer.py 把一次病例的产出分别写入三层持久化存储：

raw_case_memory.jsonl      ← 完整病例记录、skill outputs、引用经验、正确性
tactical_experience.jsonl  ← 局部决策提炼："条件 → skill → 结果"
abstract_experience.jsonl  ← 跨病例抽象：confusion_memory / rule / prototype / seed

这不是简单追加日志。每次写入后，ExperienceStore.refresh_metadata() 会触发全库重建：

manifest 和全库计数更新：raw_case_count / tactical_count / abstract_count

split-aware 版本和整体哈希重算：基于全体 raw_case_ids / tactical_ids / abstract_ids 重算 record_hash，生成新的 split_aware_version

索引文件重建：case_id_to_raw.json / tactical_by_case.json / abstract_by_type.json / confusion_memory_index.json

abstract 经验全局 merge：同一 abs_id 的经验会合并 supporting_cases / counter_cases / provenance / composite_skill_seed / promotion_interface，而不是重复插入

composite seed 支持度更新：命中已有 seed 时，合并 supporting_cases，重算 success_count，判断是否从偶然成功变成跨病例稳定模式

因此，一条新经验加入后，变化的不只是"某条记录多了"，而是整个候选池、索引结构和版本状态都被重构。

1.3 CognitionState 增量更新

apply_cognition_update()（agent/reflection.py）在每次病例结束后更新长期认知状态：

cognition.update_failure_statistics(...)       # 总病例数、失败数、混淆数、hard case 数
cognition.update_known_confusion_patterns(...) # 累计混淆对
cognition.update_preferred_skills(...)         # 更新偏好技能列表
cognition.update_skill_statistics(...)         # 每个 skill 的统计数据
cognition.update_workflow_preferences(...)     # workflow 条件化技能偏好

cognition/cognition_state.py 中维护的长期状态至少包括：

字段

含义

known_confusion_patterns

已知混淆对及其出现频次

preferred_skills

历史上有帮助的技能列表

failure_statistics

总病例数、失败率、混淆率、hard case 率

skill_statistics

每个 skill 的 call_count / helpful_rate / failure_rate / harmful_count / average_evidence_strength

workflow_preferences

特定 workflow 下历史有效的技能组合

retrieval_preferences

检索偏好（top_k 等）

二、认知状态如何反向影响决策

经验写回和认知更新如果不影响后续决策，就只是日志。DermAgent 在这点上是打通的。

2.1 影响 skill 检索

agent/skill_retriever.py 的 RuleMetadataHybridSkillRetriever 在评分时会读取 cognition.skill_statistics：

helpful_rate 高的 skill → 检索分数加成

failure_rate 高的 skill → 检索分数惩罚

harmful_count 非零的 skill → 额外降权

2.2 影响 planner 决策

agent/planner.py 的 heuristic planner 在选择技能时会：

读取 cognition.preferred_skills，对历史有效技能加分

读取 cognition.skill_statistics，对 helpful_rate / failure_rate / harmful_rate 做加减分

读取 cognition.workflow_preferences，对特定 workflow 下历史有效的技能组合再加分

读取 cognition.known_confusion_patterns，在检测到当前病例可能涉及已知混淆对时，优先触发对应的 specialist skill

2.3 影响经验检索

memory/experience_retriever.py 的 ExperienceRetriever 每次检索都会遍历当前全库的 raw / tactical / abstract 记录并重新打分。由于经验库在每次写入后都完成了全局重建，新加入的经验会立即参与后续所有检索的候选排序。

完整的行为闭环：

前面的病例 → 更新 cognition + 写回经验库
                ↓
cognition 改变 skill_retriever 的召回分数
cognition 改变 planner 的选择偏置
经验库变化改变 experience_retriever 的 top-k 结果
                ↓
后面的病例使用新的检索结果和规划偏置
                ↓
后面的病例再继续更新 cognition + 经验库

三、批量层面的再学习

除了单病例在线写回，仓库还提供了三类批处理层面的"再学习"机制。

3.1 hard-case mining

agent/hard_case_miner.py 从大量 execution records 中挖出：

agent 相对 baseline 的回归病例（agent 错但 baseline 对）

高不确定性失败病例

反复出现的 confusion cluster

重要但容易错的病例簇

产出 HardCaseCandidate 列表，包含 failure_type / hardness_reasons / involved_skills / confusion_tags。这相当于系统的"错题本"，可以用于后续针对性的经验补充或 bootstrap。

3.2 skill helpfulness analysis

agent/skill_helpfulness_analyzer.py 跨病例总结每个 skill 的：

帮助率和伤害率

适用场景（哪类 workflow、哪类 ddx 候选下更有效）

常见失败模式

evidence strength 分布

这相当于系统的"技能复盘器"，产出可以直接用于更新 cognition.skill_statistics，也可以用于调整 policy 中的 skill 权重配置。

3.3 experience consolidation

memory/experience_consolidator.py 把大量 raw/tactical/abstract 经验再压缩成更稳定的高阶知识：

产出类型

来源

含义

prototype

多次正确病例的稳定模式

某类疾病的典型表现原型

confusion_memory

反复出现的混淆对

已知的系统性混淆模式

rule_candidate

重复出现的 tactical 决策模式

可提升为规则的经验

composite_skill_seed

重复成功的多 skill 序列

候选复合技能的种子

这说明系统不只会"记"，还会"压缩和提炼"——把偶然成功的模式和反复出现的失败模式都沉淀为更稳定的知识资产。

四、更高层的成长：向新技能和新控制器演化

4.1 composite skill seed → composite skill proposal

成长路径：

单病例 experience_transform.py
  → 生成 composite_skill_seed（稳定成功的多 skill 序列）
      ↓
批量 experience_consolidator.py
  → 跨病例合并 seed，重算 supporting_cases 和 success_count
      ↓
agent/composite_skill_proposal_generator.py
  → 把达到支持度阈值的 seed 变成正式 proposal
      ↓
人工 review → 决定是否注册为新 skill

关键约束：proposal 不会自动上线执行，仍需人工审核。系统已经具备"长出新技能候选"的能力，但保持了审慎边界。

4.2 execution record → learned controller training data

agent/controller_training.py 把 execution records 转成 controller training examples，每条样本包含：

case_state_features：病例特征向量

candidate_skills：候选技能列表

selected_skills：实际选择的技能

helpful_skills / harmful_skills：事后评估

delta_vs_baseline：相对 baseline 的增益

reward：监督信号

这意味着系统已经具备：

从运行记录中自动生成监督信号

把 heuristic 运行历史转换成 learned controller 的训练数据

因此它的成长并不只停留在 heuristic state update，还为后续离线学习留好了完整数据面。

五、自进化的边界与约束

5.1 已经明确落地的

机制

代码位置

单病例 reflection

agent/reflection.py

经验分层写回（raw/tactical/abstract）

memory/experience_writer.py

经验库全局重建（manifest/index/version）

memory/experience_store.py

CognitionState 增量更新

cognition/cognition_state.py

认知状态反向影响 skill retrieval

agent/skill_retriever.py

认知状态反向影响 planner

agent/planner.py

hard-case 挖掘

agent/hard_case_miner.py

skill helpfulness 分析

agent/skill_helpfulness_analyzer.py

experience consolidation

memory/experience_consolidator.py

composite skill seed 生成

memory/experience_transform.py

controller training data 导出

agent/controller_training.py

5.2 还没有完全自动闭环的

Qwen 不会在线更新权重

新 skill 不会自动注册上线（需人工 review proposal）

candidate controller / reranker / calibrator 不会自动替换 stable 主线

正式 frozen evaluation 阶段会故意关闭在线成长（enforce_writeback_policy() 强制禁止写回）

5.3 frozen evaluation 的隔离保证

agent/contamination_guard.py 的 enforce_writeback_policy() 在 run_mode="compare" 或 strict_frozen_writeback_guard=True 时会强制关闭写回。这保证了正式评测的公平性：frozen compare 阶段的经验库和认知状态是固定的，不会因为评测本身而改变。

六、自进化的层次结构

把整个自进化机制按时间尺度和抽象层次整理如下：

【即时层】单病例执行后
  reflection → skill_assessments → cognition_update
  经验写回 → 经验库全局重建 → 检索空间重排

【短期层】数十到数百病例后
  CognitionState 积累 → skill_statistics 趋于稳定
  known_confusion_patterns 形成 → planner 偏置显现
  workflow_preferences 形成 → workflow 条件化路由改善

【中期层】批量运行后
  hard_case_miner → 错题本形成
  skill_helpfulness_analyzer → 技能价值图谱形成
  experience_consolidator → prototype / confusion_memory / rule_candidate 沉淀

【长期层】跨批次积累后
  composite_skill_seed 达到支持度阈值 → proposal 生成
  execution records 积累 → learned controller 训练数据就绪
  candidate policy 可与 stable policy 做正式对比评测

七、最准确的表述

DermAgent 当前已经是一个会积累病例经验、会复盘技能价值、会更新长期认知状态，并让这些状态反过来改变后续检索、技能选择和证据组织方式的结构化 agent。

但它还不是一个会自动改写自身技能语义、自动更新 backbone、自动上线新策略的完全自治系统。

它的自进化核心是：把每次病例执行后的结果沉淀为可复用经验和长期认知状态，并让这些状态持续塑造之后的检索、技能选择和证据组织方式——在不触碰 backbone 参数的前提下，让 agent 组织层越跑越像一个积累过经验的临床推理者

---

八、新增自进化机制（2026-04-29）

在原有闭环基础上，新增了以下机制。所有新机制均不影响 Qwen backbone 和现有实验评测路径。

8.1 进化代数计数器

代码位置：cognition/cognition_state.py、agent/reflection.py

CognitionState 新增 evolution_generation 字段，每次 apply_cognition_update() 执行后自动 +1。该字段参与 state_version 哈希计算，因此每代认知状态都有唯一版本标识。

对外可表述为：系统已完成第 N 代自进化，每处理一个病例即完成一次认知迭代。

8.2 认知状态 Diff 日志

代码位置：agent/evolution_logger.py、agent/reflection.py

每次认知更新前后，compute_cognition_diff() 对比以下维度的变化：

新增或频次增加的混淆对（known_confusion_patterns）

preferred_skills 列表变化

各 skill 的 helpful_rate / harmful_rate / call_count 变化

failure_statistics 变化

变化记录以 JSONL 格式追加写入 state/cognition_evolution_log.jsonl，每条记录包含 generation 编号和 timestamp。这条日志是系统自我进化轨迹的完整历史记录。

8.3 Skill 适应度快照

代码位置：agent/skill_fitness_snapshot.py、agent/batch_reflection.py

每次 batch reflection 结束后，save_skill_fitness_snapshot() 从 cognition.skill_statistics 提取所有 skill 的适应度指标（helpful_rate / harmful_rate / call_count / average_evidence_strength），按 helpful_rate 排序后写入：

state/skill_fitness_snapshots/snapshot_gen{N}_{timestamp}.json

N 取自当前 evolution_generation。多个快照文件构成 skill 适应度随代际变化的时间序列，可直接用于绘制"技能进化曲线"。

8.4 自我描述报告

代码位置：scripts/generate_self_report.py

手动运行后，脚本读取 cognition_state.json 和 batch_critique.json，构造包含当前代数、准确率、top 混淆对、top 技能的 prompt，调用 Qwen 生成一段自然语言自我评估报告，写入：

state/self_reports/self_report_gen{N}_{timestamp}.md

报告内容不回流到任何决策路径，纯输出。可对外展示为"系统能用自然语言描述自己学到了什么"。

运行方式：

python scripts/generate_self_report.py

8.5 Composite Skill 自动生成 → 人工审核 → 一键部署

代码位置：agent/composite_skill_codegen.py、scripts/generate_skill_from_proposal.py、scripts/approve_skill.py

完整管道：

run_batch_reflection.py
  → generate_composite_skill_proposals.py（已有，输出到 proposals/composite_skills/）
  → generate_skill_from_proposal.py（新增，生成 Python 文件到 skills/pending/）
  → 人工审核 skills/pending/*.py
  → approve_skill.py --skill-id <id>（新增，一键部署）

generate_skill_code() 从 proposal 的 proposed_workflow_text、trigger_pattern、skill_sequence 生成完整的 Python skill 类，结构与现有 specialist skill 一致（name / description / output_fields / skill_object / build_prompt）。

approve_skill.py 执行以下操作：
1. 将 skills/pending/{id}.py 复制到 skills/{id}.py
2. 在 skills/registry.py 的 build_default_registry() 中追加 import 和实例化行
3. 将 proposal 的 review_status 更新为 "approved"
4. 删除 pending 目录中的临时文件

这使得系统具备"从运行经验中长出新技能候选，经人工确认后正式上线"的完整闭环。

8.6 Confusion 阈值触发 → 新 Specialist Skill 提案

代码位置：agent/confusion_skill_proposal_generator.py、scripts/check_confusion_thresholds.py

generate_confusion_triggered_proposals() 遍历 cognition.known_confusion_patterns，对频次超过阈值（默认 3）且尚未被现有 specialist skill 覆盖的混淆对，自动生成 specialist skill 提案，写入：

proposals/confusion_triggered_skills/{pair_slug}_{timestamp}.json

提案结构与 composite skill proposal 一致，可直接通过 generate_skill_from_proposal.py → approve_skill.py 管道部署。

运行方式：

python scripts/check_confusion_thresholds.py --threshold 3

生成的提案同样需要人工审核后才能部署，保持了审慎边界。

8.7 元学习适应层（默认关闭）

代码位置：meta_learning/few_shot_adapter.py、scripts/run_meta_learning_adaptation.py

当遇到新数据集时，可手动运行 few-shot 适应：

python scripts/run_meta_learning_adaptation.py \
  --dataset-name <name> \
  --cases-path <few_shot_cases.jsonl>

FewShotAdapter.adapt() 从少量标注案例中统计各 skill 的有效率，生成：

retrieval_bias：helpful_rate > 0.6 的 skill 加权字典
workflow_preferences_patch：按疾病类别的技能偏好补丁

适应结果写入 state/meta_learning/adaptations/{dataset_name}.json。

激活方式：设置环境变量 DERMAGENT_META_LEARNING_ENABLED=1。激活后：

agent/planner.py 的 _evaluate_skill() 末尾读取适应文件，对高 bias skill 加 1-2 分
agent/skill_retriever.py 的 _evaluate_skill() 末尾读取适应文件，对高 bias skill 加 0.5-1.0 分

两处均通过 try/except 包裹，env var 未设置时完全不执行，不影响原有实验路径。

---

九、更新后的自进化层次结构

【即时层】单病例执行后
  reflection → skill_assessments → cognition_update
  evolution_generation +1 → cognition diff 写入 evolution_log
  经验写回 → 经验库全局重建 → 检索空间重排

【短期层】数十到数百病例后
  CognitionState 积累 → skill_statistics 趋于稳定
  known_confusion_patterns 形成 → planner 偏置显现
  evolution_log 记录完整进化轨迹

【中期层】批量运行后
  hard_case_miner → 错题本形成
  skill_helpfulness_analyzer → 技能价值图谱形成
  experience_consolidator → prototype / confusion_memory / rule_candidate 沉淀
  skill_fitness_snapshot → 技能适应度时间序列
  check_confusion_thresholds → 新 specialist skill 提案自动生成
  generate_composite_skill_proposals → composite skill 提案生成
  generate_self_report → 自然语言自我评估报告

【长期层】跨批次积累后
  composite/confusion skill 提案经人工审核 → approve_skill 一键部署 → 技能库扩展
  execution records 积累 → learned controller 训练数据就绪
  candidate policy 可与 stable policy 做正式对比评测
  新数据集 → run_meta_learning_adaptation → 快速偏置适应（手动开启）