## 2 Related Work

### 2.1 Dermatology vision-language models and reasoning models

近年来，面向皮肤科场景的视觉语言模型与领域专用多模态模型逐渐成为医学 AI 的重要方向。SkinGPT-4、PanDerm、SkinGPT-R1、Skin-R1 与 SkinFlow 所代表的主线工作，分别从通用皮肤科多模态交互、foundation model、reasoning-oriented tuning 与流程化推理等角度推动了皮肤科大模型的发展 [Citation: skingpt4_2024; panderm_2025; skingptr1_2025; skinr1_2025; skinflow_2026]。这类工作共同表明，皮肤科 AI 的核心问题已经不再只是“能否分类”，而 increasingly 是“能否在开放临床情境中进行更可信的视觉-语言推理”。

然而，这类方法整体上仍主要把改进重点放在 backbone 本体，包括模型权重、领域知识注入、推理风格塑形或多模态对齐能力。即便模型开始输出更丰富的 reasoning text，关键临床动作通常仍以内隐方式嵌入在 backbone 的生成过程之中。DermAgent 与这一路线的边界在于：本文的核心不是重新训练一个新的 dermatology backbone，而是在固定 backbone 的条件下，把 structured evidence 的生成、经验使用与证据组织显式外置为 agent scaffold。因而，DermAgent 更接近围绕 dermatology VLM 构建 structured reasoning exoskeleton，而不是再做一个新的皮肤科 foundation model。

### 2.2 Medical agents and tool-augmented reasoning

随着大模型 agent 框架的发展，医学场景中也出现了越来越多围绕工具调用、流程分解与多步推理展开的方法。这类工作通常通过检索器、知识库、规则工具、外部计算模块或多轮规划机制，使模型在单次回答之外具备更强的任务分解与证据整合能力 [Citation: healthcare_agents_review_2026]。相较于直接 prompting，medical agents 的一个重要贡献是把原本封闭在一次生成中的中间步骤显式化，使系统能够在推理过程中调用外部资源，并在一定程度上提升可解释性。

不过，现有 medical agents 很多仍然集中在文本问答、文献检索、报告生成或通用医疗咨询等任务上。即使涉及医学图像，其 agent 结构也常常偏向“调用若干工具完成一个多步任务”，而非围绕某一高风险视觉诊断场景构建长期、可演化、可公平评测的推理外骨骼。对皮肤科而言，这一区别尤其重要：工具的价值并不只是补充一个外部 API，而在于把具有临床语义边界的中间动作组织成诊断前的 structured evidence 流程，并使这些动作和证据在跨病例尺度上可积累、可分析、可训练。DermAgent 与一般 medical agents 的边界因此十分明确：它不是一个纯文本 QA 医疗助手，而是一个专门服务于皮肤图像诊断的 structured reasoning layer；并且在系统责任分工上，它始终保持 backbone 作为 sole final diagnostician，而不把 agent 变成新的最终决策者。

### 2.3 Skill- and experience-based agent learning

在 agent learning 领域，XSKILL 与 TARSE 提供了与 DermAgent 最直接相关的方法学启发。它们共同强调，skill 与 experience 应被显式区分：skill 更像可复用的动作能力或过程模板，experience 则更像与历史实例相关的轨迹、条件和反馈信号。这样的双结构设计使 agent 不必把所有历史信息混入单一 memory pool，也让 controller 能在“调用什么能力”和“参考什么经验”之间做更精细的选择 [Citation: xskill_2026; tarse_2026]。

同时，MACRO 与 MemSkill 进一步推动了这一方向的发展。MACRO 强调从历史成功执行中发现并封装可复用的多步程序，使经验不再只是被动参考，而能够上升为新的高层能力；MemSkill 则把 memory operation 本身重写为可选择、可演化的 skills，并通过 controller 与 designer 机制推动 skill selection 与 skill set evolution 的闭环 [Citation: macro_2026; memskill_2026]。这些工作共同说明，一个更强的 agent 往往不是依赖单轮 prompt，而是依赖 skill、experience 与控制策略之间的结构化协同。

不过，DermAgent 与这些工作并不等同。首先，它们多发生在通用 agent、文本任务或更广义的工具使用环境中，而非皮肤科图像诊断这一高风险多模态任务。其次，在这些环境中，skill 往往可以直接等价于子任务工具，而在医学诊断中，如果不限制语义边界，skill 很容易退化为 disease classifier。DermAgent 因而有意把 skill 限定为 structured clinical reasoning actions，把 experience 组织为 layered experience bank，并把 controller 的作用限定在 skill selection、experience selection 与 evidence routing 上，而不允许这些外围结构直接替代 final diagnosis。

### 2.4 Evolving cognition / memory / self-improving agents

关于 agent 如何持续改进，AutoAgent、MemSkill 等工作提供了另一组关键启发。AutoAgent 将 evolving cognition 与 elastic memory orchestration 作为核心组件，强调 agent 若要在长程任务中持续提高适应性，就不能只依赖静态 prompt，而需要维护跨回合的结构化 cognition、压缩与抽象 memory，以及基于结果反馈的闭环更新 [Citation: autoagent_2026]。MemSkill 也表明，memory 并不只是存储容器，它可以被重新理解为由 skill-like procedures 驱动、并可被 controller 选择和持续进化的对象 [Citation: memskill_2026]。

这些工作共同说明，自进化 agent 的核心不只是“存更多记忆”，而是把经验、状态与控制信号转化为未来可用的决策偏置。不过，在医学诊断场景中，这类思想不能被简单理解为开放式在线学习。医学 agent 不仅要考虑如何持续积累经验，还要考虑在线 writeback 是否会污染正式评测、策略偏好是否会与病例知识混淆，以及系统是否还能在 frozen protocol 下保持可归因性。DermAgent 在这一点上的定位因此更克制：它并不声称已经实现了一个充分验证的 self-improving medical agent，而是实现了一个具备认知自进化结构条件的框架，即通过 layered experience bank、cognition state 与 parameterized policy layer 的分离，使经验更新、策略偏置与参数化控制可以在代码与评测协议中被分别管理、分别冻结、分别分析。

因此，DermAgent 与一般意义上的 self-improving agent 的差异不在于是否承认长期演化的重要性，而在于它把这种演化置于更严格的 clinical reasoning 和 frozen evaluation 约束之下。系统可以在非正式运行中通过 reflection/writeback 累积经验与 cognition，但在正式 same-backbone comparison 中必须关闭 online writeback，并使用 split-isolated frozen state。正是在这种约束下，认知进化才不会与实验污染混为一谈。

### 2.5 Positioning of DermAgent

综上，现有相关工作分别从不同方向推动了 dermatology VLM、medical agents、skill/experience learning 与 evolving cognition 的发展，但它们的重点并不相同。SkinGPT-4、PanDerm、SkinGPT-R1、Skin-R1 与 SkinFlow 的主要贡献在于提升 dermatology vision-language backbone 的建模能力、开放诊断能力或推理能力；XSKILL、TARSE、MemSkill 与 MACRO 则更关注 agent 如何组织技能、经验、memory 与可复用程序；AutoAgent 则强调 cognition 与 elastic memory 在长期自适应中的作用 [Citation: skingpt4_2024; panderm_2025; skingptr1_2025; skinr1_2025; skinflow_2026; xskill_2026; tarse_2026; memskill_2026; macro_2026; autoagent_2026]。

DermAgent 位于这些方向的交叉点，但其问题定义与方法边界更具体。它的核心不是重新训练一个新的 dermatology backbone，也不是构造一个通用医疗问答 agent，而是在固定 backbone 的条件下，为皮肤图像诊断构建一个由 skill bank、layered experience bank、cognition state 与 parameterized policy layer 组成的外围结构化推理外骨骼。与现有工作相比，DermAgent 最强调的不是“agent 能做更多事”，而是“agent 如何在不抢走最终诊断权的前提下，改进 structured evidence 的生成、组织与跨病例演化”，以及“这种改进如何在 frozen and fair evaluation 中被公平比较”。
