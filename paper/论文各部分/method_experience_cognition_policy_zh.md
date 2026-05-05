### 3.X Experience Bank

DermAgent 中的 experience bank 并不是一个普通的 nearest-neighbor case retrieval 模块，也不是对历史病例进行无差别缓存的记忆池。其核心思想在于：对皮肤科诊断而言，系统需要积累的不仅是“见过哪些相似图像”，还包括“哪些推理动作在何种情境下有帮助”“哪些混淆模式值得优先关注”“哪些跨病例规律可以被抽象为可复用的证据偏置”。因此，DermAgent 将 experience 组织为分层结构，而不是将所有历史信息压缩为单一相似度空间中的若干近邻样本。

在当前实现中，experience bank 主要包含三个层级。第一层是 raw case memory，其作用是保留与具体病例直接对应的经验内容，包括病例级输入、历史输出、正确性线索与反思结果等。这一层解决的是实例保真问题：当系统面对与既往病例具有局部相似性或处于相同混淆簇中的样本时，raw case memory 能够提供最接近原始案例层面的参考。第二层是 tactical experience，其关注点不再是病例整体相似，而是条件到动作、动作到结果的局部关联，例如某类不确定性条件下哪些 skill 组合更有帮助、哪些风险线索需要被前置、哪些比较动作在特定情形下具有更高价值。这一层解决的是“如何做”的问题，即将病例后反思中可复用的局部推理策略沉淀为战术层经验。第三层是 abstract experience，它进一步跨越单个病例和单次动作，将重复出现的模式提升为更抽象的原型、规则候选或混淆记忆，用于表达跨病例可转移的知识结构。这一层解决的是“应优先关注什么模式”的问题，为系统提供高层先验与概念性 guidance。

这一分层设计带来的关键差异在于，experience 不再只在 retrieval 阶段起作用。首先，experience 确实参与检索，用于为当前病例提供具体病例、战术模式和抽象经验的补充上下文。其次，experience 会影响 planner 的 skill selection：当某些 tactical 或 abstract 经验表明当前病例更接近特定混淆模式、高风险模式或信息不足模式时，planner 可以据此调整本轮优先执行的 reasoning actions。再次，experience 也会影响 skill use 本身，因为 skill 在执行时可以显式引用历史经验中的学习点、混淆提示或风险线索。最后，experience 进入 evidence organization，决定哪些历史证据值得被纳入最终 evidence package，哪些经验应作为背景线索保留，哪些则应在校准后被压低权重。因而，experience bank 在 DermAgent 中不是一个附加的“相似病例检索器”，而是贯穿检索、规划、执行与证据组织的知识内容层。

### 3.X Cognition State

如果说 experience bank 主要负责存储“知道什么”，那么 cognition state 更接近于系统跨病例积累的“如何做”的状态。DermAgent 将 cognition 定义为跨 case 的策略记忆和行为偏好状态，而不是普通缓存或简单统计表。它的作用不在于替代经验内容层，而在于记录系统在持续运行过程中形成的偏好、失败模式和控制先验，并将这些状态反馈到后续病例的 skill use、retrieval preference 与 evidence routing 中。

在当前实现中，cognition state 包含若干跨病例层面的策略性变量，例如已知混淆模式、偏好的 skills、检索偏好、失败统计以及 skill-level helpful/harmful 统计等。与 raw case 或 abstract experience 不同，这些内容本身并不试图描述某一个具体病例发生了什么，而是试图总结系统在过去执行中表现出的行为规律。例如，一个 skill 在某类病例中是否经常带来有益补充，一个混淆对是否反复出现，一个 retrieval preference 是否值得继续维持，都属于 cognition 所记录的内容。

因此，experience 与 cognition 的区别是明确的。experience 更偏向知识内容层，其基本单位是病例、动作经验或跨病例抽象模式；cognition 更偏向策略状态层，其基本单位是偏好、统计、习惯与控制先验。前者告诉系统“历史中有哪些内容值得参考”，后者告诉系统“面对类似情境时，哪些策略倾向更值得优先采用”。在 DermAgent 中，planner 会读取 cognition 中的偏好与统计信息，以影响 skill 选择；retrieval 模块也可以利用 cognition 所提供的混淆模式与偏好先验，对经验候选进行更适应当前状态的排序。换言之，cognition 并不提供新的病例知识，而是改变系统使用已有知识和技能的方式。

这一分层设计还有一个重要意义，即它使系统能够在不混淆“内容更新”和“策略更新”的情况下实现跨病例演化。如果没有 cognition 层，所有长期变化都只能被写入经验库，导致“经验内容增长”和“行为偏好变化”难以区分；而显式 cognition state 使得 system-level adaptation 可以在一个受控、可审计的状态对象中发生。这对于医学场景尤为重要，因为我们既希望 agent 能从历史执行中逐步形成更合理的策略倾向，又希望这种变化可以与病例知识本身严格区分，并在正式评测中被冻结或隔离。

### 3.X Parameterized Policy Layer

在 DermAgent 中，experience 与 cognition 提供了知识内容和策略状态，但它们并不直接决定系统行为。真正把这些信息转化为当前病例中的具体控制决策的是 parameterized policy layer。该层的目标不是修改 Qwen 权重，也不是改变 backbone 的最终诊断责任，而是在 frozen-backbone 的前提下，对外围的中间策略进行参数化、可学习化、可版本化与可回滚化管理。通过这一层，DermAgent 中若干原本可能散落在规则逻辑中的控制行为，被显式提升为可以独立记录和比较的策略对象。

当前系统中，parameterized policy layer 主要包括四个部分。第一是 planner policy，用于规定 skill 选择阶段的控制行为，例如选择阈值、稀疏化策略、可启用或禁用的 skill 集合，以及规则骨架与 learned controller 之间的组合方式。planner policy 决定的是在给定 perception、experience 与 cognition 条件下，系统应如何组织本次病例的 reasoning actions。第二是 retrieval policy，用于规定经验检索与 skill 检索的候选范围、重排序行为以及相关控制参数。它不仅影响检索返回什么，还影响检索结果如何在后续规划中被使用。第三是 evidence policy，用于控制 evidence calibration 与 evidence organization，例如不同类型证据的排序、配额、裁剪与保留优先级。它直接影响最终 evidence package 的结构，而不会直接生成最终病种标签。第四是 evaluation gate，用于在候选 policy 与稳定 policy 之间建立保守的比较与回滚机制，限制外围策略更新在正式使用中的风险。

这些 policy 层之所以重要，是因为它们将外围优化从“隐式改动系统行为”转化为“显式可管理的策略版本”。在当前实现中，planner/controller、retrieval scoring 或 reranking、evidence calibration 等外围模块都可以通过 policy 配置、checkpoint 与评测记录进行独立版本化；候选更新可以在冻结协议下进行比较，必要时也可以回滚到稳定版本。这意味着 DermAgent 的优化方向是外围策略优化，而非 backbone finetuning。系统可以学习更好的 skill selection、更好的 experience routing 与更稳健的 evidence organization，但 Qwen 的权重与最终诊断职责保持不变。这一点构成了本文方法边界的核心。

从整体上看，experience bank、cognition state 与 parameterized policy layer 共同构成了 DermAgent 的认知进化闭环。experience bank 负责沉淀病例知识、战术模式与抽象规律，cognition state 负责积累跨病例的策略统计与行为偏好，而 parameterized policy layer 则把这些内容转化为当前病例中的可执行控制决策。病例运行结束后，reflection/writeback 将新的 outcome、skill helpfulness 与经验候选回写到 experience 与 cognition 中；在训练或版本化阶段，policy 层可以利用这些执行记录进行保守更新。由此，DermAgent 的“进化”不是对 backbone 的在线改写，而是一个围绕知识内容、策略状态与外围控制层展开的受控闭环，其目标是在保持最终诊断责任不变的前提下，逐步改进结构化 reasoning scaffold 的质量与稳定性。
