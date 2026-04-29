DermAgent 自进化机制详解

更新时间：2026-04-29

概述

DermAgent 的"自进化"不是训练 backbone，也不是在线更新 Qwen 的权重。它的进化发生在 backbone 外部的 agent 组织层，分为四个层级：

【感知层】每个病例执行后，系统感知自身表现并写回经验
【认知层】长期认知状态积累，反向影响后续检索和规划决策
【提炼层】批量运行后，从经验中压缩出更稳定的高阶知识
【演化层】从经验中生长出新技能候选，经人工审核后扩展技能库

Qwen 始终是唯一的最终诊断决策者，自进化只发生在 agent 组织层。

---

一、感知层：单病例执行后的反思与写回

每个病例执行完成后，agent/reflection.py 的 build_reflection() 将病例分类：

  预测与参考标签不匹配  →  confusion_experience
  不确定性高 / 置信度低  →  hard_case_experience
  其余有效病例           →  raw_case_experience

产出内容包括：skill_assessments（每个 skill 的帮助性评估）、tactical_experiences（局部决策提炼）、abstract_experiences（跨病例抽象）、cognition_update（认知更新信号）。

写回触发条件：hard case 或 confusion case 必写；其余只要有 skill 产出有效证据就写。

经验分三层持久化（memory/experience_writer.py）：

  raw_case_memory.jsonl      ← 完整病例记录
  tactical_experience.jsonl  ← 条件 → skill → 结果
  abstract_experience.jsonl  ← confusion_memory / rule / prototype / seed

每次写入后，ExperienceStore.refresh_metadata() 触发全库重建：manifest 更新、split-aware 版本哈希重算、索引文件重建、同一 abs_id 的 abstract 经验全局 merge。一条新经验加入后，整个候选池、索引结构和版本状态都被重构。

进化记录（新增）：每次病例结束后，evolution_generation 自动 +1，cognition diff 写入 state/cognition_evolution_log.jsonl，记录本代新增的混淆对、skill 统计变化、preferred_skills 变化。

---

二、认知层：长期状态积累与决策反馈

感知层的写回如果不影响后续决策，就只是日志。认知层负责把积累的状态转化为决策偏置。

CognitionState（cognition/cognition_state.py）维护的长期状态：

  known_confusion_patterns   已知混淆对及出现频次
  preferred_skills           历史有效技能列表
  skill_statistics           每个 skill 的 helpful_rate / harmful_rate / call_count / evidence_strength
  failure_statistics         总病例数、失败率、混淆率、hard case 率
  workflow_preferences       特定 workflow 下历史有效的技能组合
  evolution_generation       已完成的进化代数（新增）

这些状态反向影响三个决策点：

skill 检索（agent/skill_retriever.py）
  helpful_rate 高 → 检索分数加成
  failure_rate 高 → 检索分数惩罚
  harmful_count 非零 → 额外降权
  known_confusion_patterns → 混淆 cluster 触发 specialist skill 优先召回

planner 决策（agent/planner.py）
  preferred_skills → 历史有效技能加分
  skill_statistics → helpful/failure/harmful rate 加减分
  workflow_preferences → workflow 条件化技能偏好加分
  known_confusion_patterns → 检测到已知混淆对时优先触发 specialist skill

经验检索（memory/experience_retriever.py）
  经验库每次写入后完成全局重建，新经验立即参与后续所有检索的候选排序

完整闭环：

  前面的病例 → 更新 cognition + 写回经验库
                ↓
  cognition 改变 skill_retriever 的召回分数
  cognition 改变 planner 的选择偏置
  经验库变化改变 experience_retriever 的 top-k 结果
                ↓
  后面的病例使用新的检索结果和规划偏置
                ↓
  后面的病例再继续更新 cognition + 经验库

---

三、提炼层：批量运行后的知识压缩

单病例写回积累的是原始经验，提炼层负责把大量原始经验压缩成更稳定的高阶知识。

3.1 错题本：hard-case mining

agent/hard_case_miner.py 从 execution records 中挖出：agent 相对 baseline 的回归病例、高不确定性失败病例、反复出现的 confusion cluster、重要但容易错的病例簇。产出 HardCaseCandidate 列表，包含 failure_type / hardness_reasons / involved_skills / confusion_tags。

3.2 技能复盘：skill helpfulness analysis

agent/skill_helpfulness_analyzer.py 跨病例总结每个 skill 的帮助率、伤害率、适用场景、常见失败模式、evidence strength 分布。产出可直接用于更新 cognition.skill_statistics 或调整 policy 权重。

技能适应度快照（新增）：每次 batch reflection 结束后，自动将当前所有 skill 的适应度指标按 helpful_rate 排序，写入 state/skill_fitness_snapshots/snapshot_gen{N}_{timestamp}.json，构成技能适应度随代际变化的时间序列。

3.3 知识压缩：experience consolidation

memory/experience_consolidator.py 把大量 raw/tactical/abstract 经验压缩成：

  prototype          多次正确病例的稳定模式 → 某类疾病的典型表现原型
  confusion_memory   反复出现的混淆对 → 已知的系统性混淆模式
  rule_candidate     重复出现的 tactical 决策模式 → 可提升为规则的经验
  composite_skill_seed  重复成功的多 skill 序列 → 候选复合技能的种子

3.4 批量反思：batch reflection

agent/batch_reflection.py 跨病例聚合成功/失败 cluster、skill 序列分析、emergent confusion memories、rule candidates、composite skill seed candidates，为后续演化层提供输入。

自我描述报告（新增）：手动运行 scripts/generate_self_report.py，系统读取当前认知状态和 batch critique，调用 Qwen 生成自然语言自我评估报告，写入 state/self_reports/self_report_gen{N}.md。报告不回流任何决策路径，纯输出。

---

四、演化层：从经验中生长新技能

提炼层产出的 seed 和 confusion pattern 是演化层的原料。演化层负责把这些原料转化为可部署的新技能。

4.1 Composite Skill 自动生成管道（新增）

从 batch reflection 产出的 composite_skill_seed_candidates 出发，完整管道：

  run_batch_reflection.py
    → generate_composite_skill_proposals.py（已有）
        输出到 proposals/composite_skills/
    → generate_skill_from_proposal.py（新增）
        从 proposal 生成 Python skill 类，写入 skills/pending/
    → 人工审核 skills/pending/*.py
    → approve_skill.py --skill-id <id>（新增）
        复制到 skills/，patch registry.py，标记 proposal 为 approved

生成的 skill 类结构与现有 specialist skill 一致，workflow_text 来自 proposal 的 proposed_workflow_text，trigger 来自 trigger_pattern。

4.2 Confusion 阈值触发 → 新 Specialist Skill（新增）

当某个混淆对在 known_confusion_patterns 中的频次超过阈值（默认 3），且尚未被现有 specialist skill 覆盖时，自动生成 specialist skill 提案：

  python scripts/check_confusion_thresholds.py --threshold 3

提案写入 proposals/confusion_triggered_skills/，结构与 composite skill proposal 一致，同样通过 generate_skill_from_proposal.py → approve_skill.py 管道部署。

两类新技能均需人工审核后才能上线，保持了审慎边界。

4.3 Learned Controller 训练数据积累

agent/controller_training.py 把 execution records 转成 controller training examples，每条样本包含 case_state_features、candidate_skills、selected_skills、helpful/harmful_skills、delta_vs_baseline、reward。系统持续为后续离线训练 learned controller 积累完整数据面。

4.4 元学习适应层（默认关闭，新增）

当遇到新数据集时，可手动运行 few-shot 适应：

  python scripts/run_meta_learning_adaptation.py \
    --dataset-name <name> \
    --cases-path <few_shot_cases.jsonl>

从少量标注案例中统计各 skill 的有效率，生成 retrieval_bias 和 workflow_preferences_patch，写入 state/meta_learning/adaptations/{dataset_name}.json。

激活：设置 DERMAGENT_META_LEARNING_ENABLED=1。激活后 planner 和 skill_retriever 各自在评分末尾叠加 bias（加法，不替换原有逻辑）。env var 未设置时完全不执行，不影响原有实验路径。

---

五、边界与约束

已落地的机制：

  单病例 reflection                    agent/reflection.py
  经验分层写回（raw/tactical/abstract） memory/experience_writer.py
  经验库全局重建                        memory/experience_store.py
  CognitionState 增量更新               cognition/cognition_state.py
  进化代数计数 + diff 日志              agent/evolution_logger.py（新增）
  认知状态反向影响 skill retrieval      agent/skill_retriever.py
  认知状态反向影响 planner              agent/planner.py
  hard-case 挖掘                        agent/hard_case_miner.py
  skill helpfulness 分析                agent/skill_helpfulness_analyzer.py
  skill 适应度快照                      agent/skill_fitness_snapshot.py（新增）
  experience consolidation              memory/experience_consolidator.py
  batch reflection                      agent/batch_reflection.py
  自我描述报告                          scripts/generate_self_report.py（新增）
  composite skill seed 生成             memory/experience_transform.py
  composite skill 代码生成              agent/composite_skill_codegen.py（新增）
  composite/confusion skill 审核部署    scripts/approve_skill.py（新增）
  confusion 阈值触发提案                agent/confusion_skill_proposal_generator.py（新增）
  controller training data 导出         agent/controller_training.py
  元学习适应层                          meta_learning/few_shot_adapter.py（新增）

尚未自动闭环的：

  Qwen 不会在线更新权重
  新 skill 不会自动注册上线（需人工 review proposal）
  candidate controller / reranker / calibrator 不会自动替换 stable 主线
  正式 frozen evaluation 阶段强制关闭写回（contamination_guard.py）

frozen evaluation 隔离：agent/contamination_guard.py 的 enforce_writeback_policy() 在 run_mode="compare" 或 strict_frozen_writeback_guard=True 时强制禁止写回，保证评测公平性。

---

六、自进化层级总览

【感知层】单病例执行后
  reflection → skill_assessments → cognition_update
  evolution_generation +1 → cognition diff 写入 evolution_log
  经验写回 → 经验库全局重建 → 检索空间重排

【认知层】数十到数百病例后
  CognitionState 积累 → skill_statistics 趋于稳定
  known_confusion_patterns 形成 → planner 偏置显现
  workflow_preferences 形成 → workflow 条件化路由改善
  evolution_log 记录完整进化轨迹

【提炼层】批量运行后
  hard_case_miner → 错题本形成
  skill_helpfulness_analyzer → 技能价值图谱形成
  skill_fitness_snapshot → 技能适应度时间序列
  experience_consolidator → prototype / confusion_memory / rule_candidate 沉淀
  batch_reflection → composite_skill_seed_candidates / rule_candidates
  generate_self_report → 自然语言自我评估报告（手动）

【演化层】跨批次积累后
  composite/confusion skill 提案经人工审核 → approve_skill 一键部署 → 技能库扩展
  confusion 阈值触发 → 新 specialist skill 提案自动生成
  execution records 积累 → learned controller 训练数据就绪
  candidate policy 可与 stable policy 做正式对比评测
  新数据集 → run_meta_learning_adaptation → 快速偏置适应（手动开启）
