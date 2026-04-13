# DermAgent：固定皮肤科视觉语言模型骨干条件下的结构化智能体推理框架

摘要

皮肤科多模态诊断中的一个核心挑战在于，基础视觉语言模型虽然能够给出初步判断，但其诊断过程通常以内隐生成形式完成，难以显式执行临床上关键的比较、排除、不确定性评估、信息缺口识别与经验积累。因此，系统往往难以审计、难以分析错误来源，也难以在冻结评测条件下对外围增强策略进行独立优化。本文提出 DermAgent，一个面向皮肤科诊断的结构化智能体推理框架，其目标不是替代骨干模型，而是在保持 Qwen 作为 sole final diagnostician 的前提下，为最终诊断前的上下文构建提供 structured evidence 外骨骼。

DermAgent 将诊断过程组织为初始感知、分层经验检索、skill 检索与选择、skill 执行、证据组织、最终诊断以及反思写回等阶段。与将外围模块设计为病种分类器或投票器的方法不同，本文中的 skill 模块被定义为 atomic clinical reasoning actions，用于产生 structured evidence 而非最终疾病标签；experience bank 不是普通的相似病例检索，而是由 raw case memory、tactical experience 与 abstract experience 构成的分层经验系统，并共同影响 planner、skill use 与 evidence routing；cognition state、experience bank 与 policy layer 在系统中被显式分离，以支持跨病例的策略记忆、经验沉淀与可审计版本控制。基于这些状态，系统进一步通过 evidence package 在最终诊断前组织 skill 输出、经验线索、风险与不确定性信息，其中 Qwen remains the sole final diagnostician，the agent does not replace the backbone。

在实现层面，当前代码库已完成主链路、状态对象、skill bank、经验写回、认知更新、可训练外围策略模块以及 frozen evaluation 协议的基本实现。系统支持对 planner/controller、retrieval reranker、evidence calibration 与 policy layer 进行独立训练或版本化，但不训练 backbone 权重。为避免评测污染，正式评测被设计为 frozen and fair：固定 split、固定 case list、关闭在线 writeback，并要求 direct baseline 与 agent variant 在 same-backbone comparison 条件下进行比较。

当前项目的实验验证仍处于持续补充阶段。基于当前已完成的主对比，在固定 `345` 个测试病例、same-backbone comparison 与 frozen evaluation 条件下，DermAgent 相对于 direct Qwen baseline 已观察到明确的主指标改善：`top-1 accuracy` 从 `102/345 (29.6%)` 提升至 `137/345 (39.7%)`，`malignant recall` 从 `133/276 (48.2%)` 提升至 `227/276 (82.2%)`，`error rate` 从 `70.4%` 下降至 `60.3%`；与此同时，`top-k hit rate` 从 `79.1%` 降至 `74.2%`。这些结果表明，当前 structured agent reasoning scaffold 在主要诊断准确性与恶性召回上具有积极信号，但完整的消融结果、最终误差分析与外部验证仍在 ongoing validation 中。因而，本文的重点仍然是系统问题设定、方法设计、实现边界与评测约束，而不是提前宣称已经建立全面结论性优势。

关键词：皮肤科诊断；医学多模态模型；智能体推理；structured evidence；skill bank；experience bank；cognition state；frozen evaluation

## 1 Introduction

皮肤科诊断具有高度依赖视觉表型、临床上下文与鉴别推理的特点。即使在单张病灶图像可提供较强外观线索的情形下，真实诊断过程通常也并非一次性的图像分类，而是围绕“看到了什么”“哪些鉴别诊断需要保留”“哪些候选应被排除”“目前还缺少哪些关键信息”“风险是否足以支持进一步检查”等问题逐步展开。对于黑色素细胞病变、角化性病变及若干高混淆病种而言，局部形态相似、metadata 不充分、时间演变信息缺失以及风险信号与表面视觉线索不一致等因素，都会使诊断推理呈现出明显的多阶段、约束收缩式特征。因而，皮肤科场景不仅要求模型具备视觉识别能力，也要求其具备较强的结构化临床推理能力与可审计性。

近年来，通用视觉语言模型及医学场景中的多模态大模型为皮肤科智能诊断带来了新的机会。这类模型能够在统一接口下处理图像、文本与部分临床 metadata，并生成较为丰富的观察描述、鉴别候选和解释性文本。因此，相较于传统的专病分类器，它们更接近开放式临床推理的交互范式，也为构建更通用的医学 AI 系统提供了基础 [@skingpt4_2024; @panderm_2025; @skingptr1_2025; @skinr1_2025; @skinflow_2026]。然而，这类模型在实际使用中仍存在若干关键不足。首先，其推理过程大多以内隐生成的方式完成，难以判断模型是否显式执行了比较、排除、不确定性评估与信息缺口检测等临床关键动作。其次，当模型面对高混淆病例、高风险病例或信息不足病例时，单次生成往往难以稳定区分“证据支持不足”与“结论已足够成立”之间的边界。再次，直接调用基础模型进行一次性回答，通常无法形成可持续积累的跨病例经验结构，也难以将历史错误、混淆模式和策略偏好转化为可训练、可审计的外围能力 [@skingptr1_2025; @skinr1_2025; @skinflow_2026]。

基于此，一个自然的方向是通过 prompting 或 prompt chaining 对基础模型施加额外引导，例如要求其先描述病灶、再给出鉴别诊断、最后输出结论。尽管这类方法在实现上简洁，并且在若干任务中能够带来一定收益，但其局限同样明显。直接 prompting 的控制粒度通常停留在输出格式或浅层步骤提示上，难以把“哪些推理动作应被触发”“哪些历史经验应被检索”“哪些证据应进入最终诊断上下文”显式建模为独立对象。更重要的是，这类方法往往难以积累长期状态：一次推理中出现的有益 skill、无效 skill、典型混淆对或高价值经验片段，难以在后续病例中以结构化方式被复用。于是，系统仍然更像一次性问答，而不是一个能够跨病例调整策略、组织证据并持续演化的临床推理框架 [@xskill_2026; @tarse_2026; @autoagent_2026]。

这促使我们进一步考虑结构化 agent reasoning 的必要性。对于皮肤科诊断而言，更合理的方向不是让外围模块直接输出病种结论，而是将一些临床上具有明确语义边界的局部推理动作外显化，并让这些动作围绕最终诊断前的 structured evidence 构建发挥作用。沿着这一思路，skill 可以被定义为原子化的临床推理动作，例如形态分析、颜色模式分析、边界与表面审查、鉴别比较、排除推理、恶性风险评估、不确定性评估和信息缺口检测；experience 不应仅是相似病例缓存，而应进一步区分具体病例记忆、战术层经验和跨病例抽象经验；在此基础上，系统还需要一种跨病例的 cognition state，用于记录混淆模式、skill 的帮助性统计、检索偏好和策略倾向，使 skill selection、experience selection 与 evidence organization 不再是静态模板，而成为可演化的外围 policy layer [@xskill_2026; @tarse_2026; @macro_2026; @memskill_2026; @autoagent_2026]。这样的结构一方面能够将原本隐含于单次生成中的临床动作显式化，另一方面也为后续训练、审计与错误分析提供接口。

然而，在引入 agent reasoning 时，另一个同样重要的问题是系统边界。如果将外围模块设计成多个子诊断器，再通过投票或聚合决定最终病种，系统很容易退化为多模块投票式诊断框架；如果将 skill 写成若干刚性的 heuristic classifier，则又会把结构化推理简化为外挂式规则分类器。这两条路线虽然表面上增强了系统，但都会削弱最终决策责任的清晰性，并增加系统行为解释与公平比较的难度。特别是在医学场景中，如果 agent 与 baseline 之间同时改变了骨干模型、病例列表、在线记忆状态或比较协议，那么所谓的 agent 增益就很难被归因。基于这一考虑，本文刻意避免三类路线：多模块投票式诊断、刚性 heuristic classifier 外挂，以及非冻结、非同条件的 agent-vs-baseline 对比。相反，我们将 agent 限定为结构化证据增强层，而将最终诊断责任明确保留给底层视觉语言模型本身。

基于上述动机，本文提出 DermAgent，一个面向皮肤科诊断的结构化智能体推理框架。DermAgent 的核心设计是：在保持底层视觉语言模型不变的前提下，由 agent 承担经验驱动的 skill 组织、经验检索、证据路由、反思写回与 cognition 更新，而由基础模型完成初始感知与最终诊断。换言之，DermAgent 并不替代 backbone，也不试图通过外围模块直接改判，而是通过构造更可审计、更具临床语义的 evidence package 来影响最终诊断前的推理上下文。在这一框架中，skill 模块只产出 structured evidence 而非最终疾病标签；experience bank、cognition state 与 policy layer 被显式分层；planner/controller、retrieval reranker 与 evidence calibration 等能力被实现为可训练或可版本化的外围策略组件；正式评测则被设计为 frozen evaluation，以便在 same-backbone comparison 条件下，将 structured agent reasoning 与 direct prompting 进行可复核比较。

围绕这一系统设计，本文关注的核心研究问题是：在底层视觉语言模型保持不变时，结构化、经验驱动的 agent reasoning 是否能够优于 direct prompting？这一问题并不只是比较“是否增加了几个模块”，而是检验以下命题是否成立：当诊断动作被组织为 skill bank、当跨病例经验被实现为分层 experience bank、当 cognition state 能够对后续 skill use 与 evidence routing 产生影响时，外围推理结构本身是否能够为最终诊断提供稳定且公平可评估的增益。由于当前项目仍处于持续验证阶段，本文的重点首先放在系统问题设定、方法设计、实现边界与评测纪律上，而不是预先宣称结论性性能优势。

本文的主要贡献如下。

1. 本文提出 DermAgent，一种面向皮肤科诊断的结构化智能体推理框架，在不改变底层视觉语言模型最终诊断职责的前提下，将 skill bank、experience bank、cognition state 与 policy layer 组织为可审计的外围临床推理层。
2. 本文设计并实现了 skill bank 与 layered experience bank 的双结构体系，其中 skill 被定义为原子化临床推理动作，experience 被组织为 raw case memory、tactical experience 与 abstract experience，并共同作用于 skill selection、experience selection 与 evidence organization。
3. 本文引入显式的 cognition state 与反思写回机制，使系统能够在跨病例层面记录混淆模式、skill 帮助性与策略偏好，从而为 agent 的自进化提供结构基础，同时保持 experience bank、cognition state 与 policy layer 的分层分离。
4. 本文将 planner/controller、retrieval reranker、evidence calibration 与 policy versioning 实现为可训练或可版本化的外围组件，并明确限定这些组件只影响 structured evidence 的生产与组织，不替代 backbone 的最终诊断。
5. 本文给出一套面向 agent-vs-baseline 比较的 frozen evaluation 设计，强调 same-backbone comparison、相同 case list、相同协议和关闭在线 writeback 的对照约束，以减少不公平比较和状态污染带来的结论偏差。

## 2 Related Work

### 2.1 Dermatology vision-language models and reasoning models

近年来，面向皮肤科场景的视觉语言模型与领域专用多模态模型逐渐成为医学 AI 的重要方向。SkinGPT-4、PanDerm、SkinGPT-R1、Skin-R1 与 SkinFlow 所代表的主线工作，分别从通用皮肤科多模态交互、foundation model、reasoning-oriented tuning 与流程化推理等角度推动了皮肤科大模型的发展 [@skingpt4_2024; @panderm_2025; @skingptr1_2025; @skinr1_2025; @skinflow_2026]。这类工作共同表明，皮肤科 AI 的核心问题已经不再只是“能否分类”，而 increasingly 是“能否在开放临床情境中进行更可信的视觉-语言推理”。

然而，这类方法整体上仍主要把改进重点放在 backbone 本体，包括模型权重、领域知识注入、推理风格塑形或多模态对齐能力。即便模型开始输出更丰富的 reasoning text，关键临床动作通常仍以内隐方式嵌入在 backbone 的生成过程之中。DermAgent 与这一路线的边界在于：本文的核心不是重新训练一个新的 dermatology backbone，而是在固定 backbone 的条件下，把 structured evidence 的生成、经验使用与证据组织显式外置为 agent scaffold。因而，DermAgent 更接近围绕 dermatology VLM 构建 structured reasoning exoskeleton，而不是再做一个新的皮肤科 foundation model。

### 2.2 Medical agents and tool-augmented reasoning

随着大模型 agent 框架的发展，医学场景中也出现了越来越多围绕工具调用、流程分解与多步推理展开的方法。这类工作通常通过检索器、知识库、规则工具、外部计算模块或多轮规划机制，使模型在单次回答之外具备更强的任务分解与证据整合能力 [@healthcare_agents_review_2026]。相较于直接 prompting，medical agents 的一个重要贡献是把原本封闭在一次生成中的中间步骤显式化，使系统能够在推理过程中调用外部资源，并在一定程度上提升可解释性。

不过，现有 medical agents 很多仍然集中在文本问答、文献检索、报告生成或通用医疗咨询等任务上。即使涉及医学图像，其 agent 结构也常常偏向“调用若干工具完成一个多步任务”，而非围绕某一高风险视觉诊断场景构建长期、可演化、可公平评测的推理外骨骼。对皮肤科而言，这一区别尤其重要：工具的价值并不只是补充一个外部 API，而在于把具有临床语义边界的中间动作组织成诊断前的 structured evidence 流程，并使这些动作和证据在跨病例尺度上可积累、可分析、可训练。DermAgent 与一般 medical agents 的边界因此十分明确：它不是一个纯文本 QA 医疗助手，而是一个专门服务于皮肤图像诊断的 structured reasoning layer；并且在系统责任分工上，它始终保持 backbone 作为 sole final diagnostician，而不把 agent 变成新的最终决策者。

### 2.3 Skill- and experience-based agent learning

在 agent learning 领域，XSKILL 与 TARSE 提供了与 DermAgent 最直接相关的方法学启发。它们共同强调，skill 与 experience 应被显式区分：skill 更像可复用的动作能力或过程模板，experience 则更像与历史实例相关的轨迹、条件和反馈信号。这样的双结构设计使 agent 不必把所有历史信息混入单一 memory pool，也让 controller 能在“调用什么能力”和“参考什么经验”之间做更精细的选择 [@xskill_2026; @tarse_2026]。

同时，MACRO 与 MemSkill 进一步推动了这一方向的发展。MACRO 强调从历史成功执行中发现并封装可复用的多步程序，使经验不再只是被动参考，而能够上升为新的高层能力；MemSkill 则把 memory operation 本身重写为可选择、可演化的 skills，并通过 controller 与 designer 机制推动 skill selection 与 skill set evolution 的闭环 [@macro_2026; @memskill_2026]。这些工作共同说明，一个更强的 agent 往往不是依赖单轮 prompt，而是依赖 skill、experience 与控制策略之间的结构化协同。

不过，DermAgent 与这些工作并不等同。首先，它们多发生在通用 agent、文本任务或更广义的工具使用环境中，而非皮肤科图像诊断这一高风险多模态任务。其次，在这些环境中，skill 往往可以直接等价于子任务工具，而在医学诊断中，如果不限制语义边界，skill 很容易退化为 disease classifier。DermAgent 因而有意把 skill 限定为 structured clinical reasoning actions，把 experience 组织为 layered experience bank，并把 controller 的作用限定在 skill selection、experience selection 与 evidence routing 上，而不允许这些外围结构直接替代 final diagnosis。

### 2.4 Evolving cognition / memory / self-improving agents

关于 agent 如何持续改进，AutoAgent、MemSkill 等工作提供了另一组关键启发。AutoAgent 将 evolving cognition 与 elastic memory orchestration 作为核心组件，强调 agent 若要在长程任务中持续提高适应性，就不能只依赖静态 prompt，而需要维护跨回合的结构化 cognition、压缩与抽象 memory，以及基于结果反馈的闭环更新 [@autoagent_2026]。MemSkill 也表明，memory 并不只是存储容器，它可以被重新理解为由 skill-like procedures 驱动、并可被 controller 选择和持续进化的对象 [@memskill_2026]。

这些工作共同说明，自进化 agent 的核心不只是“存更多记忆”，而是把经验、状态与控制信号转化为未来可用的决策偏置。不过，在医学诊断场景中，这类思想不能被简单理解为开放式在线学习。医学 agent 不仅要考虑如何持续积累经验，还要考虑在线 writeback 是否会污染正式评测、策略偏好是否会与病例知识混淆，以及系统是否还能在 frozen protocol 下保持可归因性。DermAgent 在这一点上的定位因此更克制：它并不声称已经实现了一个充分验证的 self-improving medical agent，而是实现了一个具备认知自进化结构条件的框架，即通过 layered experience bank、cognition state 与 parameterized policy layer 的分离，使经验更新、策略偏置与参数化控制可以在代码与评测协议中被分别管理、分别冻结、分别分析。

因此，DermAgent 与一般意义上的 self-improving agent 的差异不在于是否承认长期演化的重要性，而在于它把这种演化置于更严格的 clinical reasoning 和 frozen evaluation 约束之下。系统可以在非正式运行中通过 reflection/writeback 累积经验与 cognition，但在正式 same-backbone comparison 中必须关闭 online writeback，并使用 split-isolated frozen state。正是在这种约束下，认知进化才不会与实验污染混为一谈。

### 2.5 Positioning of DermAgent

综上，现有相关工作分别从不同方向推动了 dermatology VLM、medical agents、skill/experience learning 与 evolving cognition 的发展，但它们的重点并不相同。SkinGPT-4、PanDerm、SkinGPT-R1、Skin-R1 与 SkinFlow 的主要贡献在于提升 dermatology vision-language backbone 的建模能力、开放诊断能力或推理能力；XSKILL、TARSE、MemSkill 与 MACRO 则更关注 agent 如何组织技能、经验、memory 与可复用程序；AutoAgent 则强调 cognition 与 elastic memory 在长期自适应中的作用 [@skingpt4_2024; @panderm_2025; @skingptr1_2025; @skinr1_2025; @skinflow_2026; @xskill_2026; @tarse_2026; @memskill_2026; @macro_2026; @autoagent_2026]。

DermAgent 位于这些方向的交叉点，但其问题定义与方法边界更具体。它的核心不是重新训练一个新的 dermatology backbone，也不是构造一个通用医疗问答 agent，而是在固定 backbone 的条件下，为皮肤图像诊断构建一个由 skill bank、layered experience bank、cognition state 与 parameterized policy layer 组成的外围结构化推理外骨骼。与现有工作相比，DermAgent 最强调的不是“agent 能做更多事”，而是“agent 如何在不抢走最终诊断权的前提下，改进 structured evidence 的生成、组织与跨病例演化”，以及“这种改进如何在 frozen and fair evaluation 中被公平比较”。

## 3 Method

### 3.1 Problem Formulation

本文关注的任务是皮肤科多模态诊断中的结构化推理增强问题。给定一张皮肤病灶图像以及可选的临床 metadata，例如年龄、部位、病灶直径、症状变化与部分病史信息，系统需要输出最终诊断及其相关解释信息。与直接将输入一次性送入视觉语言模型并要求其给出结论的设定不同，本文研究的问题是：在底层视觉语言模型保持不变的条件下，是否可以通过一个结构化、经验驱动的 agent 中间层，为最终诊断前的推理过程提供更可审计、更可训练、也更公平可比较的 structured evidence 增强。

在这一设定中，底层 backbone 不是被替换或重训练的对象，而是被视为固定的 sole final diagnostician。DermAgent 的目标并不是让外围模块竞争最终决策权，而是将原本隐含在一次性生成中的局部临床动作、经验检索与证据整合过程显式化，从而在不改变最终诊断责任归属的前提下，增强最终诊断所依赖的上下文。形式上，可将系统记为

\[
y = F_{\text{backbone}}(x, m, \mathcal{E}(x,m;\theta_{\text{agent}})),
\]

其中 \(x\) 表示皮肤图像，\(m\) 表示可选 metadata，\(F_{\text{backbone}}\) 表示固定的视觉语言模型，\(\mathcal{E}\) 表示由 agent 产生的 structured evidence package，\(\theta_{\text{agent}}\) 则对应 planner、retrieval scoring、evidence calibration 等外围策略参数。本文的核心问题并非构造一个新的诊断 backbone，而是研究在固定 \(F_{\text{backbone}}\) 时，structured agent reasoning 是否能够优于 direct prompting。

### 3.2 System Overview

DermAgent 是一个围绕皮肤图像诊断构建的结构化推理外骨骼。其基本思想是将诊断过程拆分为两个由 backbone 承担的关键阶段与一个由 agent 承担的中间增强阶段：首先，Qwen 对输入图像及 metadata 执行初始视觉理解，生成病灶概述、初始鉴别诊断候选以及初步不确定性信息；随后，agent 中间层基于该初始感知结果开展经验检索、skill 选择、skill 执行与证据组织；最后，Qwen 在接收到整理后的 evidence package 后输出最终诊断。因而，系统中存在两次与 backbone 的核心交互：一次用于初始视觉理解和候选诊断生成，另一次用于最终诊断生成，而 final answer 始终由 Qwen 输出。

如 Figure 1 所示，DermAgent 的整体推理流程并不是在 backbone 之外再增加一个并行诊断器，而是在固定 Qwen 的前提下，对最终诊断前的 structured evidence 生产与组织过程进行显式建模。输入病例首先进入 `qwen_direct_skill`，生成 initial diagnosis、top-k differential、confidence 与初步 reasoning；随后 `perception_skill` 对视觉观察进行结构化整理，retrieval 模块基于 top-k 与 perception 结果从 layered experience bank 中提取相关经验，planner/controller 则据此决定需要触发哪些 skills。被选中的 skills 经由 router 调度执行，输出比较、风险、metadata 一致性与 specialist confusion 等 structured evidence，并最终由 aggregator 融合为 evidence package。Qwen remains the sole final diagnostician throughout this process; the agent serves only as an evidence augmentation layer.

如 Figure 2 所示，DermAgent 的内部结构由 skill bank、experience bank、cognition state 与 parameterized policy layer 共同构成。skill bank 提供可执行的 structured clinical reasoning actions，experience bank 提供分层经验内容，cognition state 提供跨病例的策略记忆与偏好，而 policy layer 负责把这些状态转化为当前病例中的 skill selection、experience routing 与 evidence organization。由此，系统的核心并不是“重新分类”，而是通过 skill 与 experience 的双系统生成、筛选并融合 structured evidence，再由 Qwen 在此基础上完成最终诊断。在训练或非冻结模式下，reflection/writeback 会把病例 outcome、skill helpfulness 与 evidence usage 写回经验与状态层，从而形成受控的认知进化闭环；而在 frozen evaluation 中，这些在线更新路径被显式关闭。

### 3.3 Design Principles

DermAgent 的方法设计基于以下原则。首先，单一最终诊断者原则。系统明确保持 Qwen 作为 sole final diagnostician，agent 不直接输出最终疾病标签，也不通过多模块投票或外围 override 来取代 backbone 的决策。其次，structured evidence 原则。agent 的作用不是产生另一个并行诊断答案，而是显式生成、筛选和组织进入最终诊断上下文的 structured evidence。第三，经验与认知分层原则。系统将 experience bank、cognition state 与 policy layer 视为不同层级的状态对象：experience 侧重病例级与跨病例知识内容，cognition 侧重策略记忆与行为偏好，policy 承载可训练和可版本化的外围控制参数。第四，外围可训练、骨干冻结原则。系统允许 controller、retrieval scoring、evidence calibration 等外围层独立训练或版本化，但不训练 backbone 权重，也不改变最终诊断责任。第五，公平评测原则。agent 带来的收益必须在 frozen evaluation、统一病例列表和 same-backbone comparison 条件下进行评估，以尽量减少不可控状态漂移和不公平对比。

### 3.4 Inference Pipeline

在推理阶段，DermAgent 采用多阶段流水线。给定输入图像与可选 metadata，Qwen 首先执行初始视觉理解，输出病灶描述、初始鉴别诊断候选以及不确定性线索。基于这一结果，系统从分层 experience bank 中检索与当前病例相关的历史经验，并从 skill bank 中检索当前最可能有帮助的临床推理动作。随后，planner/controller 结合初始 perception、clinical metadata、retrieved experience、skill retrieval signal 以及 cognition state 中积累的偏好与统计信息，选择本次病例需要执行的 skills。

被选中的 skill 模块围绕当前病例产生 structured evidence，例如观察性线索、比较性线索、风险提示、不确定性描述和信息缺口说明。之后，agent 对这些输出与检索到的经验进行重新整理，构建一个供最终诊断使用的 evidence package。该 package 不是最终答案，而是 final diagnosis 之前的结构化上下文接口，其中包含经组织后的初始感知摘要、经验摘要、skill 输出、风险与不确定性信息以及必要的说明性注释。最后，Qwen 读取原始输入与 evidence package，生成最终诊断结果。若运行模式允许 writeback，则系统在诊断后执行 reflection，提取病例级 outcome、skill helpfulness、经验候选与 cognition update；在正式评测模式下，这一步不会写回在线状态。

![Figure 1. DermAgent 的整体推理流程。](/root/DermAgent/paper/figures/figure1_overall_framework.png)

**Figure 1. DermAgent 的整体推理流程。** 系统接收输入病例，包括皮肤图像与可选 metadata。Qwen 首先执行 `qwen_direct_skill`，生成初始诊断候选、top-k differential、置信度与初步推理线索；随后 `perception_skill` 将视觉观察进一步组织为 structured evidence。基于初始 top-k 与感知结果，系统从 experience bank 中执行 retrieval，并由 planner/controller 结合当前病例状态、经验线索与策略偏好选择需要触发的 skills。被选中的 skills 经由 router 调度执行，产生比较、风险、metadata 一致性与 specialist confusion 等 structured evidence。aggregator 将 perception、retrieval 与 selected skills 的输出融合为 evidence package，并将其提供给 Qwen 进行最终诊断 refinement。Qwen 始终是 sole final diagnostician，agent 仅承担 evidence augmentation 的角色。图中的虚线表示仅在训练或非冻结模式下启用的 reflection/writeback 路径；在 frozen evaluation setting 中，该路径被显式关闭。

### 3.5 Skill Bank

在 DermAgent 中，一个 skill 被定义为一个 atomic clinical reasoning action。这里的“action”并不意味着技能模块独立完成诊断闭环，而是指其承担一个具有明确临床语义边界的局部推理职责，例如显式提取病灶表型线索、比较候选诊断之间的差异、审查 metadata 与视觉线索的一致性、指出潜在风险信号，或识别当前推理中的不确定性与信息缺口。由此，skill 在系统中的基本单位不是“病种预测器”，而是“证据生产器”：它围绕当前病例生成 structured evidence，供后续证据组织与最终诊断使用。

这一设计与将外围模块构造成 disease classifier 的路线有本质差异。首先，DermAgent 中的 skill 不直接输出最终病种，也不承担最终诊断责任。系统始终要求 Qwen 作为 sole final diagnostician，因此 skill 的职责是补充和组织 backbone 在单次生成中未必会显式完成的临床动作，而不是与 backbone 争夺最终决策权。其次，将 skill 设计为 atomic clinical reasoning actions，可以避免外围模块退化为一组刚性的子分类器。如果每个 skill 都被训练成判断某一类病种或某一组标签，那么整个系统很容易演化成“若干外挂分类器 + 一个聚合器”的结构，不仅削弱了单一最终诊断者的边界，也会使模块语义与误差来源变得难以审计。相反，当 skill 的输出被限制为 structured evidence 而非最终标签时，系统便能够在不引入额外诊断权竞争的前提下，将比较、排除、风险审查与不确定性处理等中间动作显式化。

从方法设计角度看，这种 skill 定义同时服务于可解释性、可审计性与后续可训练性。由于每个 skill 对应一个较为稳定的临床动作语义，planner/controller 可以围绕“当前病例需要哪些动作”进行显式选择，而不是围绕“哪个子分类器可能更准”进行隐式组合；reflection 也可以围绕每个 skill 的帮助性、冗余性或潜在误导性进行病例后分析；进一步地，skill 的触发条件、执行结果与对最终 evidence package 的贡献，都可以在 execution record 中被单独追踪。这使得 skill bank 不再只是若干静态函数的集合，而成为一个可被选择、可被分析、可被版本化、也可被外围策略持续利用的推理动作库。

在当前实现中，skill bank 覆盖了若干面向皮肤科诊断的证据生产类型。其一是 perception-oriented evidence extraction，即围绕病灶形态、颜色模式、边界与表面、分布特征以及病灶描述结构化等内容，对初始视觉理解进行更细粒度的显式整理。其二是 differential comparison 相关技能，用于围绕候选诊断之间的相似性与差异性生成比较性证据，并支持进一步的 exclusion reasoning。其三是 metadata consistency checking，用于审查图像表型、病史元数据与输入描述之间是否存在冲突、缺失或可疑之处。其四是 malignancy risk assessment，通过风险导向而非结论导向的方式前置潜在警示信号。其五是 specialist confusion auditing，即围绕若干高混淆诊断对提供更聚焦的专门审查。除此之外，当前系统还实现了 uncertainty assessment，并包含 contradiction checking、information-gap detection 与 escalation recommendation 等与不确定性和安全边界相关的技能；这些模块已经进入现有执行链路，但其最终作用范围与收益仍应在后续冻结评测中进一步验证。

这些 skill 类型虽然在功能上有所差异，但共享同一方法学约束：skill 输出的是 structured evidence，而不是最终 disease label。作为系统组成部分，skill bank 同时承担三个角色：作为一组版本化的源码资产被共享和维护；由 planner/controller 根据当前病例状态、经验检索结果与 cognition state 动态选择；其输出进入 evidence package，而非直接以投票方式汇总为答案。

### 3.6 Experience Bank

DermAgent 中的 experience bank 并不是一个普通的 nearest-neighbor case retrieval 模块，也不是对历史病例进行无差别缓存的记忆池。其核心思想在于：对皮肤科诊断而言，系统需要积累的不仅是“见过哪些相似图像”，还包括“哪些推理动作在何种情境下有帮助”“哪些混淆模式值得优先关注”“哪些跨病例规律可以被抽象为可复用的证据偏置”。因此，DermAgent 将 experience 组织为分层结构，而不是将所有历史信息压缩为单一相似度空间中的若干近邻样本。

在当前实现中，experience bank 主要包含三个层级。第一层是 raw case memory，其作用是保留与具体病例直接对应的经验内容，包括病例级输入、历史输出、正确性线索与反思结果等。第二层是 tactical experience，其关注点不再是病例整体相似，而是条件到动作、动作到结果的局部关联，例如某类不确定性条件下哪些 skill 组合更有帮助、哪些风险线索需要被前置、哪些比较动作在特定情形下具有更高价值。第三层是 abstract experience，它进一步跨越单个病例和单次动作，将重复出现的模式提升为更抽象的原型、规则候选或混淆记忆，用于表达跨病例可转移的知识结构。

这一分层设计带来的关键差异在于，experience 不再只在 retrieval 阶段起作用。experience 不仅为当前病例提供具体病例、战术模式和抽象经验的补充上下文，还会影响 planner 的 skill selection、skill use 本身以及最终 evidence organization。因而，experience bank 在 DermAgent 中不是一个附加的“相似病例检索器”，而是贯穿检索、规划、执行与证据组织的知识内容层。

### 3.7 Cognition State

如果说 experience bank 主要负责存储“知道什么”，那么 cognition state 更接近于系统跨病例积累的“如何做”的状态。DermAgent 将 cognition 定义为跨 case 的策略记忆和行为偏好状态，而不是普通缓存或简单统计表。它的作用不在于替代经验内容层，而在于记录系统在持续运行过程中形成的偏好、失败模式和控制先验，并将这些状态反馈到后续病例的 skill use、retrieval preference 与 evidence routing 中。

在当前实现中，cognition state 包含若干跨病例层面的策略性变量，例如已知混淆模式、偏好的 skills、检索偏好、失败统计以及 skill-level helpful/harmful 统计等。experience 与 cognition 的区别是明确的：experience 更偏向知识内容层，其基本单位是病例、动作经验或跨病例抽象模式；cognition 更偏向策略状态层，其基本单位是偏好、统计、习惯与控制先验。前者告诉系统“历史中有哪些内容值得参考”，后者告诉系统“面对类似情境时，哪些策略倾向更值得优先采用”。

### 3.8 Policy Layer

在 DermAgent 中，experience bank 与 cognition state 提供了知识内容和策略状态，但它们并不直接决定系统行为。真正把这些信息转化为当前病例中的具体控制决策的是 policy layer。该层的目标不是修改 Qwen 权重，也不是改变 backbone 的最终诊断责任，而是在 frozen-backbone 的前提下，对外围的中间策略进行参数化、可学习化、可版本化与可回滚化管理。

当前系统中的 policy layer 主要包括四个部分。第一是 planner policy，用于规定 skill 选择阶段的控制行为。第二是 retrieval policy，用于规定经验检索与 skill 检索的候选范围、重排序行为以及相关控制参数。第三是 evidence policy，用于控制 evidence calibration 与 evidence organization，例如不同类型证据的排序、配额、裁剪与保留优先级。第四是 evaluation gate，用于在候选 policy 与稳定 policy 之间建立保守的比较与回滚机制，限制外围策略更新在正式使用中的风险。

从整体上看，experience bank、cognition state 与 policy layer 共同构成了 DermAgent 的认知进化闭环。experience bank 负责沉淀病例知识、战术模式与抽象规律，cognition state 负责积累跨病例的策略统计与行为偏好，而 policy layer 则把这些内容转化为当前病例中的可执行控制决策。病例运行结束后，reflection/writeback 将新的 outcome、skill helpfulness 与经验候选回写到 experience 与 cognition 中；在训练或版本化阶段，policy 层可以利用这些执行记录进行保守更新。

### 3.9 Evidence Package

在 DermAgent 中，agent 中间层的终点并不是另一个诊断结论，而是一个结构化的 evidence package。该对象构成了最终诊断前 backbone 可见的主要增强上下文，也是系统边界设计中的关键接口。DermAgent 的核心并不是让多个外围模块分别给出病种判断再进行聚合，而是让 skill 输出、经验检索结果、风险提示、不确定性线索与规划依据在最终诊断之前被组织为一个可审计的 structured evidence 包，并由 Qwen 在此基础上生成 final answer。

将 evidence package 作为统一接口有三个直接目的。首先，它把 agent 的职责严格限制在“证据生产与证据组织”这一层面，从结构上避免了外围模块与 backbone 争夺最终诊断权。其次，它使中间推理过程具备更好的可解释性和可审计性。第三，它为后续训练和策略优化提供了自然接口。由于进入最终诊断阶段的并不是一串无结构的中间文本，而是带有分组与来源语义的 structured evidence，planner、retrieval 与 evidence calibration 的贡献便能够被更细粒度地归因和分析。

在当前实现中，evidence package 由若干互补的信息块组成，包括初始视觉理解摘要、raw/tactical/abstract 三层经验检索摘要、skill 输出汇总、风险标记、不确定性信息、矛盾检查、信息缺口说明、升级建议以及 planner rationale 等。这里重要的不是字段数量本身，而是其组织方式：DermAgent 并不简单拼接所有中间输出，而是通过 evidence organization 与 evidence calibration 对不同来源的证据进行筛选、分组、排序、裁剪与配额控制。因而，哪些证据最终进入 Qwen 的上下文、以什么顺序进入、哪些信息被突出、哪些信息被降权，本身就是一个被显式建模的外围策略问题。

![Figure 2. DermAgent 的内部状态结构与证据交互机制。](/root/DermAgent/paper/figures/figure2_internal_state_evidence.png)

**Figure 2. DermAgent 的内部状态结构与证据交互机制。** 该图从架构层面展示 skill bank、experience bank、cognition state、planner/controller、router、aggregator、evidence package 与 reflection/writer 之间的关系。在 inference 阶段，experience bank 提供分层经验并通过 retrieval 进入证据链路，skill bank 提供可被 planner/controller 动态选择的 structured clinical reasoning actions，aggregator 将 retrieval 结果与 skill outputs 融合为 evidence package，并将其送入 Qwen 完成最终诊断。evidence package 是 agent 与 Qwen 之间的核心中枢接口，承载 structured evidence 而非最终标签。在 training 或非冻结模式下，reflection/writer 根据病例 outcome、skill helpfulness 与 evidence usage 将新经验写回 experience bank，并更新 cognition state 与相关 policy statistics，从而形成受控的认知进化闭环。在 frozen evaluation setting 下，experience bank 仅作为只读状态使用，online writeback 与 online update 被禁止，以确保 same-backbone comparison 的可归因性与可复核性。

### 3.10 Reflection and Writeback

如果 evidence package 对应的是病例前向推理中的“证据汇聚”，那么 reflection and writeback 对应的则是病例后向更新中的“经验沉淀”。DermAgent 的设计并不把一次病例运行视为孤立事件，而是将其视为后续推理可以利用的经验来源。因此，在每次病例完成最终诊断后，系统会生成结构化 reflection，对当前推理过程进行病例级总结，并在允许的运行模式下将其中可复用的部分写回到 experience bank 与 cognition state 中。

在当前实现中，reflection 的内容不仅包括简要的 reasoning summary，还包括对当前病例 outcome 的结构化判断、对各个已执行 skill 的 helpfulness assessment、对潜在错误或混淆模式的识别，以及后续可写回的 raw/tactical/abstract experience 候选。writeback 则是在 reflection 基础上的受控状态更新机制。对 experience bank 而言，writeback 会将病例级结果转换为 raw case memory、tactical experience 与 abstract experience 的增量；对 cognition state 而言，writeback 会更新混淆模式、失败统计、skill helpful/harmful 统计与偏好信息。

不过，本文同样强调 writeback 的边界。writeback 发生在 backbone 之外，它不会修改 Qwen 权重，也不会通过在线更新改变最终诊断模型本身；在正式 frozen evaluation 中，在线 writeback 被关闭，split state 被固定，从而避免当前评测样本通过状态污染影响后续样本。

### 3.11 Training and Optimization

DermAgent 的训练目标并不是对 Qwen 进行 finetuning，也不是进行端到端的诊断模型重训练。相反，本文将优化重点放在 frozen-backbone 条件下的外围 policy layer，即那些决定“哪些 skills 被调用”“哪些经验被优先使用”“哪些 structured evidence 被保留并进入最终诊断上下文”的中间组件。这样的设计与本文的问题设定保持一致：我们关注的是，在底层视觉语言模型保持不变时，结构化、经验驱动的 agent reasoning 是否能够优于 direct prompting。因此，训练对象必须位于 backbone 之外，并且能够在相同 backbone 条件下被独立评估。

在当前实现中，DermAgent 的主要可训练对象包括 controller、retrieval reranker/scorer 与 evidence calibrator。controller training 关注的是 skill selection 问题，即在给定初始 perception、clinical metadata、experience context 与 cognition state 的条件下，学习哪些 clinical reasoning actions 更值得在当前病例中被激活。retrieval reranker 或 scorer 的训练则服务于 experience selection 与 skill candidate ordering。evidence calibrator training 对应 evidence organization 层，关注如何对已经生成的 skill outputs、experience summaries、risk flags、uncertainty signals 与 contradiction cues 进行更稳健的分组、排序、裁剪与权重分配，使最终 evidence package 更适合 backbone 的最终诊断使用。

这些训练信号主要来自四类来源：execution records、frozen split state、helpful/harmful signals，以及 downstream evaluation-oriented supervision。execution records 保存输入摘要、初始 perception、retrieval bundle、planner 决策、skill outputs、evidence package、final diagnosis 与 reflection summary；frozen split state 为训练数据的采样与状态隔离提供可控边界；helpful/harmful signals 来自反思阶段对 skills、经验引用与证据使用情况的后验判断；downstream evaluation-oriented supervision 则利用最终病例级评估结果、agent 与 direct baseline 的差异表现，以及关键混淆子集上的保守指标，对外围策略更新进行筛选和约束。

在训练流程上，DermAgent 采用 staged training pipeline，而不是将所有外围模块放入一个难以归因的统一黑箱中联合优化。当前代码中，stage 0 负责从已有 execution records 中收集并导出训练数据；stage 1 侧重 controller training；stage 2 面向 retrieval scorer/reranker；stage 3 将训练得到的外围 checkpoint 组装为 candidate policy，并在 frozen evaluation 协议下进行候选评估；stage 4 则负责稳定 checkpoint 的导出与版本化管理。就当前完成度而言，controller training、retrieval scorer/reranker training 与 staged training pipeline 已经进入主线代码路径；evidence calibrator 已经作为主链路中的独立外围组件被实际调用，并在配置层被纳入可训练对象，但其训练与最终冻结评测仍处于持续迭代中。

将训练限制在外围策略层而保持 backbone 固定，有几个直接优势：更好的稳定性、更强的可审计性、更易于因果归因，以及更适合 same-backbone comparison。由于 baseline 与 DermAgent 使用相同的 backbone、相同的病例列表与相同的冻结状态，系统间差异主要集中在外围推理结构是否被启用，以及这些外围策略是否经过训练优化。

### 3.12 Frozen Evaluation Setting

为了评估结构化 agent reasoning 相对于 direct prompting 的真实增益，本文采用 frozen evaluation。正式评测时，在线 writeback 被关闭，系统不允许在评测过程中把当前病例的反思结果继续写回 experience bank 或 cognition state；同时，split state 被固定，意味着 experience、cognition 与相关 policy 状态均来自预先确定的分割与快照，而非在评测过程中持续变化。除此之外，baseline 与 agent 共享相同的 backbone 和相同的 case list，并在一致的输入条件、病例顺序与协议下进行比较。

上述设计的目的，是尽量把性能差异归因于“是否使用 structured agent reasoning”这一因素，而不是归因于 backbone 不一致、病例列表不同、在线状态污染或隐式额外训练。因而，本文的方法框架不仅关心如何构建 skill-experience-cognition 外骨骼，也同样强调在 same-backbone comparison、同 case list、同协议约束下进行可复核比较。

## 4 Experimental Setup

### 4.1 Datasets

本文当前实验框架围绕皮肤科图像诊断任务构建，输入由病灶图像及可选临床 metadata 组成。根据现有代码与固定 split 配置，可确认的主数据源为 `pad_ufes_20`，其默认固定划分定义在 `pad_ufes_20_contiguous_v1` 中，并由代码自动生成确定性的 train/val/test case lists。对于该内置 split，代码中明确设定的比例为 70%/15%/15%，划分策略为按 metadata 索引连续切分，以保证 case-offset/limit 驱动的 frozen evaluation 与重复运行的一致性。该设置目前是仓库中最明确、可复核的主实验划分来源。

除主数据源外，代码结构也预留并支持在其它皮肤科数据目录上运行同一评测协议，但具体外部数据集名称、目录组织、病例数量和最终使用范围需要根据实际运行日志、split JSON 或实验 manifest 补全。为避免引入未冻结的信息，本文在当前版本中将这些外部数据条件记为 “to be filled from logs/configs/manifests”，而不凭印象预填规模或样本数。

### 4.2 Label Space

当前代码中的 canonical dermatology label space 已有明确实现，主要覆盖六类标准化标签：`BCC`、`ACK`、`NEV`、`SEK`、`SCC` 和 `MEL`。系统通过统一的 label normalization 逻辑将同义词、全称与常见变体映射到上述 canonical labels，因此正式实验中应基于 canonicalized labels 报告结果。

对于恶性相关评测，当前代码中的 malignant definition 同样是显式的：`BCC`、`ACK`、`SCC` 和 `MEL` 被视为 malignant or clinically high-risk labels，而 `NEV` 与 `SEK` 被视为 benign labels。这里需要说明的是，该定义服务于当前项目中的 malignant recall 与安全性相关评测，并不等同于对所有外部数据集和所有临床场景的普适肿瘤学结论。特别是在外部数据集标签空间不一致、标注粒度不同或存在中间风险类别时，恶性/良性映射应被视为协议级近似，而非天然可直接推广的统一医学事实。

对于 OOD external evaluation，标签空间的一致性是主要限制之一。如果外部数据集与内部 canonical labels 不能一一对应，则不应直接报告完整六类多分类结果，而应根据可对齐程度选择 malignant-vs-benign、部分标签映射或受限子集评测。

### 4.3 Data Split and Reproducibility

为保证结果可复核性，DermAgent 使用确定性的固定 split 机制而不是运行时随机切分。主 split 配置在代码中以显式 ID、元数据路径、比例和切分策略保存，并能够导出为固定 JSON 文件，供训练、验证和测试阶段共享。正式评测时，具体病例选择由 split name、case offset 与 limit 决定，相关 case indices、case IDs、split ID 与 split JSON 路径都会记录在 evaluation manifest 中。因此，同一配置在重复运行时应返回一致的病例列表与顺序。

与此同时，state 也采用 split-aware 管理。训练、验证和测试可分别绑定不同的 experience root、cognition state 与 policy snapshot，以减少跨 split 的状态泄漏。最终稿中的样本数量、所用 split ID、病例范围和 offset/limit 组合，建议全部从正式 evaluation manifest 或固定 split JSON 中自动填充。

### 4.4 Backbone and Serving Setup

本文的核心比较建立在 same-backbone setting 下。当前主实验路径默认以 Qwen 作为 backbone，并通过统一服务接口提供初始视觉理解与最终诊断能力。更具体地说，在 DermAgent 路径中，Qwen 首先根据原始输入生成初始 perception 和候选诊断，再在接收 evidence package 后生成最终诊断；在 direct baseline 中，则由相同 backbone/service 直接基于原始输入生成最终输出。因而，主比较中的 backbone 本体、服务地址、运行时模型配置与 prompt stack 需要在 evaluation manifest 中共同记录。

代码中也存在用于其它服务后端的脚本与运行目录，例如 MedGemma 相关启动脚本和状态目录；但根据本文的主问题设定，这类跨 backbone 对照不应替代 same-backbone direct baseline comparison。若论文最终版本包含不同服务后端的实验，应当在结果章节中明确区分为扩展性分析，而不是与主基线混写。

### 4.5 Baselines

本文的主 baseline 为 direct Qwen baseline，即在与 DermAgent 完全相同的 backbone/service、输入条件和病例列表下，不启用经验检索、skill selection、evidence package、reflection/writeback 等中间层，而直接由 Qwen 输出最终诊断。这一 baseline 与本文的研究问题严格对应，因为它回答的是：在底层模型保持不变时，引入结构化 agent reasoning 是否带来增益。

在此基础上，实验协议还支持若干 matched ablations 作为内部对照，包括移除 experience retrieval、移除 skill retrieval、去除 cognition bias、限制 skill 层级或移除 uncertainty/escalation 相关模块等。这些设置不构成独立 baseline，而是用于拆解 full-agent 中各个中间层的贡献。

### 4.6 Evaluation Protocol

DermAgent 的评测协议建立在一个基本前提之上：本文的研究问题不是“任意 agent 系统是否可以优于任意 baseline”，而是“在相同 backbone 条件下，结构化、经验驱动的 agent reasoning 是否能够优于 direct prompting”。因此，评测设计的核心并非仅仅给出一组结果数字，而是确保所有比较都在可复核、可归因且尽可能公平的条件下进行。

本文采用 same-backbone comparison 原则。对于主结果比较，baseline 被定义为 direct Qwen with the same backbone/service，即在不启用 DermAgent 中间层的情况下，使用与 agent 完全相同的 backbone 服务、相同的模型运行环境与一致的输入条件，直接从图像和可用 metadata 生成最终输出。与之对应，DermAgent 版本在相同 backbone/service 上增加结构化 agent 中间层，包括经验检索、skill 选择、evidence organization 和最终 evidence package 构造。由此，agent 与 baseline 之间的差异被尽量限定在“是否引入 structured reasoning scaffold”这一点上，而不是混入 backbone 更换、服务差异或运行配置变化。

正式比较采用 frozen evaluation。其含义包括两个层面：第一，split state 固定；第二，正式比较时不允许 online writeback。这样的冻结设定有助于将实验结果解释为“在既定状态快照下，结构化推理框架对当前病例集的增益”，而不是“系统边评边学后逐步适应这组样本”的结果。

所有正式 agent-vs-baseline 比较都基于 matched-case comparison，即 agent 与 baseline 使用完全相同的 case list、相同的病例顺序、相同的数据 split 以及一致的输入可见条件。这样，任一病例上的差异都可以在病例级 execution records 中被直接对齐，支持逐例分析 agent 是否改善、保持或削弱了最终预测。

在指标层面，本文优先报告与临床诊断和公平比较直接相关的结果，包括 top-1 accuracy、top-k hit rate、malignant recall、subgroup metrics、calibration-related metrics（若相关信号已稳定）以及 error analysis。[Metrics Placeholder: exact reported metric set to be finalized from manifests]

由于 DermAgent 显式使用 memory、cognition 与 policy 状态，污染控制是评测协议中的关键组成部分。本文采用三类保护机制：policy versioning、state isolation 与 contamination guard。除此之外，病例 metadata 本身也需要进行 leakage prevention。对于可能直接泄露标签的信息字段，正式输入应进行过滤或屏蔽；若某些经验记录与当前评测样本存在直接身份级重合或可逆映射，也应通过 split-aware state 管理与快照隔离加以防止。

为了理解 DermAgent 各组成部分的实际贡献，本文采用组件级 ablation protocol，而不是仅报告 full-agent 与 direct baseline 的单点差异。ablation 的基本原则仍然遵循 same-backbone、same-case-list 与 frozen-state 约束。外部验证用于考察 DermAgent 在不同数据来源、不同病例分布或不同标签体系下的可迁移性，但其解释必须比内部 matched evaluation 更为谨慎。若标签空间不一致，则只能在 malignant-vs-benign、部分标签映射或受限子集层面进行解释。

### 4.7 Implementation Details

当前代码库已经提供了训练、评测与分析的主要运行入口，但并非所有实验超参数都适合在当前草稿中手动填写。可确认的实现细节包括：内置固定 split 定义、多个 end-to-end run profiles、frozen evaluation 入口、staged training pipeline、policy snapshot/manifest 机制以及 paper export 工具链。对于训练 epoch、不同 profile 的病例预算、是否运行 ablations、以及 controller/retrieval scorer 的 checkpoint 使用方式，代码中均已有明确配置项或脚本参数。

为避免手工转录错误，本文建议将以下具体实现细节在最终稿中由脚本或日志自动补全：各实验 profile 的名称与预算、所使用的 stable policy ID、controller 与 retrieval checkpoint 路径、服务端模型名、推理 timeout/retry 设置，以及每次正式评测对应的 output root 与 frozen state manifest 路径。对于目前尚未冻结确认的字段，本文在本节中统一采用 “to be filled from logs/configs” 的保守写法。

### 4.8 Evaluation Outputs and Logging

DermAgent 的评测输出不仅包括聚合指标，还包括一组面向可审计性和后续训练的结构化日志。正式评测运行会生成 evaluation manifest、result manifest、各 target 的 summary 文件以及病例级 `case_execution_records.jsonl`。其中，evaluation manifest 记录数据源、病例列表、split、fairness constraints、model context、frozen state、run targets 与 execution config；result manifest 记录 protocol version、contamination check、各 target summary 与 matched comparisons；而病例级 execution records 则保存 input summary、初始 perception、retrieval bundle、planner 决策、skill outputs、evidence package、final diagnosis、reflection summary 以及 baseline delta 等细粒度信息。

[Table Placeholder: Dataset and split summary]
[Table Placeholder: Backbone/service and implementation settings]

## 5 Results

本节报告当前已完成的主对比结果，并对仍在运行或尚未冻结完成的实验保持占位。所有尚未确认的显著性结论、完整消融结果与外部验证结果均继续保留为待补充内容，不在当前草稿中擅自补数。除已确认的主对比指标外，其余表格、图与指标仍建议在最终版本中由 evaluation manifests、result manifests、summary files 与 paper export 脚本自动回填。

### 5.1 Main Controlled Comparison: Agent vs Direct Qwen

本节报告 `full DermAgent` 与 `direct Qwen baseline` 在 same-backbone comparison、相同 `345` 个测试病例以及 frozen evaluation 条件下的主结果。基础服务在两组之间保持一致，均使用同一 Qwen 服务与相同 case list；agent 组额外启用 structured reasoning scaffold、layered experience bank、dynamic skill selection 与 evidence organization。

当前已完成的主对比结果如下。direct Qwen baseline 在 `345` 个带有 ground truth 的测试病例上达到 `top-1 = 102/345 (29.6%)`、`top-k = 273/345 (79.1%)`、`malignant recall = 133/276 (48.2%)`、`error rate = 243/345 (70.4%)`。在完全匹配的 same-backbone comparison 条件下，DermAgent 达到 `top-1 = 137/345 (39.7%)`、`top-k = 256/345 (74.2%)`、`malignant recall = 227/276 (82.2%)`、`error rate = 208/345 (60.3%)`。对应的 agent-versus-baseline delta 分别为：`top-1 +10.1` 个百分点、`malignant recall +34.1` 个百分点、`error rate -10.1` 个百分点，而 `top-k` 为 `-4.9` 个百分点。

这些结果表明，在固定 backbone 的条件下，DermAgent 当前版本在 `top-1 accuracy`、`malignant recall` 与 `error rate` 上都表现出明确改善，尤其是在恶性相关召回上获得了较大幅度提升。这一结果与本文的方法目标一致，即通过 structured evidence、layered experience 与外围策略层改善最终诊断前的推理上下文，而不是通过替换 backbone 获得提升。与此同时，`top-k` 的下降也说明当前 agent scaffold 可能在提高最终决断集中度的同时牺牲了部分鉴别候选覆盖范围，因此不能将本轮结果简单概括为“所有指标均改善”。更合适的解释是：preliminary controlled evidence suggests，在当前 frozen and fair evaluation 条件下，DermAgent 对主要诊断准确性与恶性召回具有积极信号，但其对 differential coverage 的影响仍需结合后续消融与错误分析进一步解释。

**表 1.** 在相同 backbone、相同 `345` 个测试病例与 frozen evaluation 条件下，DermAgent 与 direct Qwen baseline 的主对比结果。

| Metric | Direct Qwen baseline | DermAgent | Delta |
|---|---:|---:|---:|
| Top-1 accuracy | `102/345 = 29.6%` | `137/345 = 39.7%` | `+10.1` pct pts |
| Top-k hit rate | `273/345 = 79.1%` | `256/345 = 74.2%` | `-4.9` pct pts |
| Malignant recall | `133/276 = 48.2%` | `227/276 = 82.2%` | `+34.1` pct pts |
| Error rate | `243/345 = 70.4%` | `208/345 = 60.3%` | `-10.1` pct pts |

[Figure Placeholder: Matched-case delta distribution between full DermAgent and direct Qwen]

### 5.2 Ablation Study

本节将呈现组件级 ablations，包括去除 experience retrieval、去除 skill retrieval、去除 cognition bias、仅保留 foundational skills、移除 uncertainty/escalation 层，以及不同 experience layer 组合等。消融的目标是分解 full-agent 的收益来源，而不是仅报告 full-agent 与 baseline 的单点差异。

截至当前写作时点，正式消融仍在运行中。按 target-case 总工作量计算，本轮正式消融共包含 `10` 个 targets、每个 target `345` 个病例，总计 `3450` 个 target-case；当前已完成约 `694/3450 (20.1%)`。按完整 target 数计算，`direct_qwen_baseline` 与 `first_stage_basic_skills_only` 已完成，`skill_bank_without_experience` 刚开始运行。因此，本节当前仅保留结构与占位，不提前填入未完成 target 的结果。

[Table Placeholder: Component-level ablation study]
[Figure Placeholder: Relative change from full-agent across ablations]
[Metrics Placeholder: ablation metrics to be filled from frozen runs]

建议解读方式：重点讨论各层是否承担不同功能角色，以及其影响是否主要体现在总体指标、恶性召回或关键混淆子集上。在结果完全冻结前，不应提前认定某一模块“贡献最大”或“确定无效”。

### 5.3 Effect of Retrieval

本节应分析 experience bank 的作用，包括不同 experience layers 的贡献差异，以及 retrieval 对 planner、skill use 与 evidence organization 的影响。若有 heuristic retrieval 与 learned retrieval reranker 的对照，也应在本节中呈现。

[Table Placeholder: Retrieval variants and experience-layer comparisons]
[Figure Placeholder: Experience usage distribution and referenced-experience analysis]

建议解读方式：重点不是“检索到了多少”，而是“检索到的 experience 是否改善了中间推理与最终 structured evidence 质量”。在最终验证完成前，不能提前写成 layered experience 已稳定优于普通 retrieval。

### 5.4 Effect of Controller / Learned Policy

本节应比较 heuristic planner 与 learned controller/policy variants 的差异，包括 skill 选择稀疏性、helpful/harmful skill 比例、病例级 delta，以及 candidate policy 相对 stable policy 的表现。

[Table Placeholder: Heuristic vs learned controller/policy comparison]
[Figure Placeholder: Skill selection sparsity and helpful/harmful ratio]

建议解读方式：应从“是否更好地选择了中间推理动作”来解释，而不是从“是否学到了新的诊断器”来解释。若 learned controller 的收益仍不稳定，应明确写为 under ongoing validation。

### 5.5 Effect of Evidence Calibration

本节应分析 evidence calibration / evidence policy 对最终结果和上下文组织的影响，例如是否更好地突出高价值 structured evidence、抑制冗余或低支持证据，以及这种变化是否影响 final diagnosis。

[Table Placeholder: With/without evidence calibration]
[Figure Placeholder: Evidence composition and calibration effects]

建议解读方式：强调 evidence calibration 作用于 final reasoning context，而不是直接输出疾病标签。若总体指标变化有限，但证据组织的 qualitative traces 显示更合理的排序，也应据实呈现。最终冻结前，不应提前写 evidence calibration 已显著提升诊断性能。

### 5.6 Error Analysis

本节应对错误病例进行结构化分析，包括关键混淆对、agent 相对 baseline 的新增错误与修正错误、恶性漏检类型、metadata inconsistency、high-uncertainty cases 与 contradiction-rich cases 等。

[Table Placeholder: Error taxonomy and key confusion subsets]
[Figure Placeholder: Corrected vs regressed cases / confusion-pair analysis]

建议解读方式：重点不只是列出失败，而是解释 agent 在哪些地方帮助了 backbone，又在哪些地方引入了额外偏置、冗余证据或误导性风险信号。任何关于“主要修复了哪类错误”的结论都应等待最终错误分析冻结后再表述。

### 5.7 External Validation

本节应呈现 external validation 结果，并明确说明标签映射规则、不可比较类别、恶性/良性二分类边界，以及哪些结论仅适用于部分可对齐标签空间。若外部数据集标签空间不完全一致，应限制为 malignant-vs-benign、部分标签映射或受限子集评测。

[Table Placeholder: External validation summary with mapping notes]
[Figure Placeholder: Internal vs external relative comparison]

建议解读方式：外部验证应更强调可迁移性边界，而不是被解释为完整多类别泛化能力。在最终外部结果冻结前，不能提前写成“外部泛化已经建立”。

### 5.8 Qualitative Case Study

本节应展示若干具有代表性的病例，最好覆盖：agent 明显帮助的病例、agent 与 baseline 一致的病例、agent 引入退化的病例。每个病例应包含初始 perception、被选中的 skills、关键 experience 引用、evidence package 摘要、最终输出与病例级分析。

[Figure Placeholder: Qualitative case timelines]
[Table Placeholder: Case-level structured comparison]

建议解读方式：定性病例的作用是说明机制，而不是替代主结果。应同时纳入成功和失败病例，避免 cherry-picking。

### 5.9 Limitations of Current Evidence

本节应主动说明当前结果证据的边界，包括样本规模是否仍有限、训练与评测是否仍处于迭代阶段、learned controller 或 evidence calibrator 是否尚未完全冻结、以及 external validation 是否仍受标签映射约束等。

[Metrics Placeholder: explicitly note all still-pending fields]

建议解读方式：本节的目标是防止过度结论。可以说 preliminary controlled evidence suggests 该框架具有可行性，但 full validation、broader external validation 和 long-horizon writeback effects 仍 under ongoing validation。

## 6 Discussion

DermAgent 的核心贡献并不只在于增加了一组中间模块，而在于对医学 agent 在诊断任务中的角色边界作出了明确限定。本文选择保持单一最终诊断者，即始终由 backbone 负责最终诊断，而让 agent 只承担 structured evidence 增强与中间推理组织。这一设计首先体现为一种安全边界。对于皮肤科诊断这类高风险、多混淆的任务而言，如果外围模块同时拥有独立诊断权，再通过投票、加权或 override 决定最终输出，系统将很难回答“最终诊断究竟由谁负责”这一问题，也难以在错误分析中清晰定位失败来自 backbone 还是来自中间模块。相反，单一最终诊断者架构将责任边界维持在 backbone 一侧，而把 agent 的价值限定在证据生产、证据排序与证据组织上，从而使系统在解释性、审计性与公平比较上都更可控。

这一边界也解释了本文为什么坚持将 skill 设计为 reasoning action，而不是 disease classifier。若 skill 被设计为外围病种分类器，那么即便其局部性能较强，整个系统也会迅速滑向“多个模块共同猜病”的结构，导致中间层与 backbone 之间出现职责重叠。更重要的是，disease-specific 子分类器往往容易退化为刚性的 heuristic attachment：它们也许在部分标签上有效，却难以保留临床动作本身的通用语义。与之相比，将 skill 定义为 atomic clinical reasoning actions 可以更自然地对应真实诊断过程中的观察、比较、排除、风险评估与不确定性审计，使外围能力的优化对象从“病种输出”转变为“中间推理动作质量”。这不仅提升了模块级可解释性，也使 planner、reflection 和 downstream training 更容易围绕 help/harm、evidence utility 与 routing quality 等中间信号展开。

本文的另一项核心设计，是将 experience bank、cognition state 与 policy layer 分层。这样做的意义并不只是软件架构上的整洁，而在于它为持续演化提供了更清晰的状态语义。experience 更偏向知识内容，回答的是系统从历史病例中保留了哪些具体案例、局部战术和抽象模式；cognition 更偏向策略状态，记录系统在跨病例尺度上形成了哪些偏好、混淆先验与行为统计；policy 则进一步把可训练、可版本化、可回滚的外围控制参数从经验内容和策略记忆中分离出来。若缺少这种分层，系统很容易把所有长期变化都混入一个统一记忆池中，既不利于诊断错误归因，也不利于 formal evaluation。相反，这种分层使“知识增长”“策略偏移”和“参数更新”能够被分别记录、分别冻结、分别比较，从而为 agent 的认知自进化提供了受控接口。

在本文看来，same-backbone frozen comparison 对 agent 论文尤其重要。对于传统模型论文而言，更换 backbone 或增加训练数据往往本身就是主要创新来源；但对于 DermAgent 这类以中间推理结构为研究对象的工作，如果比较时同时改变 backbone、病例列表、在线状态或评测协议，则任何性能变化都很难被可靠归因。因此，本文将 fairness、state isolation、writeback disabling 与 matched-case comparison 视为方法定义的一部分，而不是附属实验技巧。只有在 direct baseline 与 full-agent 共享同一 backbone/service、同一 case list、同一 split state 且正式比较中不允许 online writeback 的前提下，研究者才有可能较为严谨地回答“structured agent reasoning 本身是否有价值”这一问题。

皮肤科之所以是适合研究 structured agent reasoning 的场景，也与其任务特征密切相关。首先，皮肤科诊断高度依赖可视表型，但这些表型往往不能直接一一映射到最终诊断标签，而是需要通过鉴别诊断、局部排除、时间演变解释和风险权衡逐步收缩诊断空间。其次，皮肤科中存在若干具有代表性的高混淆对与高风险子群，使得 skill selection、specialist auditing、uncertainty handling 和 experience reuse 具有明确的临床动机。再次，该场景同时具备图像、metadata 和部分病史文本，使 agent 可以在多模态上下文中组织 structured evidence，而不是仅依赖自由文本问答。正因如此，皮肤科既足够复杂，能够暴露直接 prompting 的局限；又足够结构化，适合作为分层 experience、reasoning actions 和 evidence package 的实验平台。

尽管如此，本文也应当明确当前证据的局限。首先，系统的完整结果仍在补充验证之中。当前代码库已经具备 frozen evaluation、matched ablation、外围训练与结果导出链路；主对比已经在 `345` 个测试病例上完成，并显示 `top-1`、`malignant recall` 与 `error rate` 的积极变化，但完整消融、子组汇总和外部验证仍未全部冻结完成，因此当前证据仍不应被表述为最终定论。其次，learned controller 与 evidence calibrator 虽然都已进入主线实现和配置体系，但它们的训练稳定性、泛化边界以及在不同病例分布下的收益仍需进一步验证。第三，外部数据集的标签空间并不总与内部 canonical dermatology label space 一致，因此 external validation 往往只能在 malignant-vs-benign、部分映射或受限子集层面进行解释，而不能简单地外推为完整多类别泛化能力。第四，writeback 的在线收益与正式 frozen evaluation 之间存在天然张力：如果允许系统在评测期间持续写回经验和认知，可能更接近“会成长的 agent”这一设想，但同时也会削弱结果的公平性与可归因性。如何在长期自进化收益与严格冻结比较之间取得更合理平衡，仍是本文尚未完全解决的问题。

另一个必须正视的问题是系统复杂度与实际收益之间的平衡。DermAgent 明确引入了 skill bank、layered experience、cognition state、policy versioning、evidence package 与 reflection/writeback 等多个层次。这样的设计带来了更强的结构化表达与审计能力，但也显著增加了系统复杂性、实现成本和实验空间。对于医学 AI 而言，更复杂的系统并不天然更优；只有当这种复杂度能够转化为更稳健的错误控制、更清晰的行为归因，或在 same-backbone comparison 条件下带来可复核的性能收益时，其设计才真正成立。

基于当前代码与实验框架，未来工作可以沿几个自然方向推进。第一，可以进一步稳定并系统评估 learned controller、retrieval reranker 与 evidence calibrator 的训练流程，特别是在固定 split state 和候选 policy gate 约束下，分析各外围组件的单独收益与交互作用。第二，可以将当前的 abstract experience 与 composite skill proposal 机制进一步发展为更稳定的 reusable multi-step reasoning templates，但仍需保持其不越过最终诊断边界。第三，可以在标签可对齐的外部皮肤科数据集上建立更明确的 malignant-vs-benign 与 partial-label evaluation protocol，以补足当前 external validation 的证据范围。第四，可以系统研究 writeback 的两种模式：一种面向 frozen formal evaluation，强调公平与可复核；另一种面向长期 online adaptation，强调认知自进化与经验积累，并通过独立协议评估其长期收益。最后，随着更多 execution records、paper exports 与训练快照被积累，DermAgent 还可以进一步支持更严格的 case-level attribution analysis，从而把 agent 的“为什么有用”从事后叙述推进到更系统化的经验驱动解释框架。

总体而言，DermAgent 所提出的并非一个已经完全收敛的最终系统，而是一种针对医学多模态诊断任务的结构化 agent 设计立场：在保持 backbone 最终诊断权不变的前提下，把 skill bank、experience bank、cognition state 与 policy layer 组织为可演化、可审计、可训练且可公平评测的外围推理外骨骼。本文认为，这一立场的价值不只在于是否获得更高的当前分数，更在于它为今后研究医学 agent 如何在安全边界内持续学习、如何被严格评测，以及如何将中间推理显式化提供了一个较为清晰的出发点。

## 7 Conclusion

本文提出并实现了 DermAgent，一个面向皮肤科多模态诊断的结构化智能体推理框架。与通过重新训练 backbone 或引入并行诊断器来提升性能的路线不同，DermAgent 保持 dermatology VLM backbone 的最终诊断职责不变，而将 skill bank、experience bank、cognition state 与可训练外围 policy layer 组织为最终诊断前的 structured evidence 外骨骼。本文的核心主张是：在固定 backbone 的前提下，将中间推理动作、分层经验、跨病例策略状态与证据组织过程显式化，能够为医学 agent 提供一种更可审计、可演化且可公平评测的系统方向。

本文当前更强调系统设计、实现边界与评测纪律，而非提前宣称已经建立全面结论性性能优势。不过，在当前已完成的主对比中，DermAgent 在 `345` 个测试病例上相对于 direct Qwen baseline 实现了 `top-1` 从 `29.6%` 提升到 `39.7%`、`malignant recall` 从 `48.2%` 提升到 `82.2%`，同时将 `error rate` 从 `70.4%` 降低到 `60.3%`；这一 same-backbone comparison 结果为 structured reasoning scaffold 的有效性提供了实质性支持。与此同时，`top-k` 从 `79.1%` 降至 `74.2%`，也提示当前系统的 gain 并非无代价提升，而是可能伴随 differential coverage 的变化。未来工作仍需在冻结协议下补齐完整消融、稳定 learned controller 与 evidence calibration 的训练，并扩展外部验证范围；但现阶段的受控结果已经表明，围绕固定 dermatology backbone 构建 structured reasoning scaffold 是一条具有真实经验信号和方法学价值的医学 agent 研究方向。

## References

当前草稿正文采用 `[@bibkey]` 引用语法，建议与同目录下的 [`references_suggested.bib`](/root/DermAgent/paper/references_suggested.bib) 配合使用。该 `.bib` 文件已根据本地 PDF 文件名与可核实的公开来源补入主要参考文献信息；在正式投稿前，仍建议逐条核对作者顺序、页码、出版状态与 arXiv 编号，并根据最终采用的排版工具（如 Pandoc、Quarto、LaTeX/BibTeX）统一格式化。
