摘要

皮肤科多模态诊断中的一个核心挑战在于，基础视觉语言模型虽然能够给出初步判断，但其诊断过程通常以内隐生成形式完成，难以显式执行临床上关键的比较、排除、不确定性评估、信息缺口识别与经验积累。因此，系统往往难以审计、难以分析错误来源，也难以在冻结评测条件下对外围增强策略进行独立优化。本文提出 DermAgent，一个面向皮肤科诊断的结构化智能体推理框架，其目标不是替代骨干模型，而是在保持 Qwen 作为唯一最终诊断者的前提下，为最终诊断前的上下文构建提供结构化证据增强外骨骼。

DermAgent 将诊断过程组织为初始感知、分层经验检索、技能检索与选择、技能执行、证据组织、最终诊断以及反思写回等阶段。与将外围模块设计为病种分类器或投票器的方法不同，本文中的 skill 模块被定义为原子化临床推理动作，用于产生结构化证据而非最终疾病标签；experience bank 不是普通的相似病例检索，而是由 raw case memory、tactical experience 与 abstract experience 构成的分层经验系统，并共同影响 planner、skill use 与 evidence routing；cognition、experience 与 policy 在系统中被显式分离，以支持跨病例的策略记忆、经验沉淀与可审计版本控制。基于这些状态，系统进一步通过 evidence package 在最终诊断前组织技能输出、经验线索、风险与不确定性信息，其中 Qwen remains the sole final diagnostician，the agent does not replace the backbone。

在实现层面，当前代码库已完成主链路、状态对象、技能库、经验写回、认知更新、可训练外围策略模块以及冻结评测协议的基本实现。系统支持对 planner/controller、retrieval reranker、evidence calibration 与 policy layer 进行独立训练或版本化，但不训练 backbone 权重。为避免评测污染，正式评测被设计为 frozen and fair：固定 split、固定 case list、关闭在线 writeback，并要求 direct baseline 与 agent variant 在一致协议下比较。

当前项目的实验验证仍处于 preliminary 阶段。Early controlled comparisons 与若干分析脚本已经具备，但完整的冻结结果、最终对比数字与外部验证仍在 ongoing validation 中。因而，本文摘要的重点在于系统问题设定、方法设计、实现边界与评测约束，而非宣称已经充分建立的结论性性能优势。
