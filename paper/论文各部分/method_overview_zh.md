## 3 Method Overview

### 3.1 Problem formulation

本文关注的任务是皮肤科多模态诊断中的结构化推理增强问题。给定一张皮肤病灶图像以及可选的临床 metadata，例如年龄、部位、病灶直径、症状变化与部分病史信息，系统需要输出最终诊断及其相关解释信息。与直接将输入一次性送入视觉语言模型并要求其给出结论的设定不同，本文研究的问题是：在底层视觉语言模型保持不变的条件下，是否可以通过一个结构化、经验驱动的 agent 中间层，为最终诊断前的推理过程提供更可审计、更可训练、也更公平可比较的证据增强。

在这一设定中，底层 backbone 不是被替换或重训练的对象，而是被视为固定的最终诊断者。DermAgent 的目标并不是让外围模块竞争最终决策权，而是将原本隐含在一次性生成中的局部临床动作、经验检索与证据整合过程显式化，从而在不改变最终诊断责任归属的前提下，增强最终诊断所依赖的上下文。形式上，可将系统记为

\[
y = F_{\text{backbone}}(x, m, \mathcal{E}(x,m;\theta_{\text{agent}})),
\]

其中 \(x\) 表示皮肤图像，\(m\) 表示可选 metadata，\(F_{\text{backbone}}\) 表示固定的视觉语言模型，\(\mathcal{E}\) 表示由 agent 产生的结构化 evidence package，\(\theta_{\text{agent}}\) 则对应 planner、retrieval scoring、evidence calibration 等外围策略参数。本文的核心问题并非构造一个新的诊断 backbone，而是研究在固定 \(F_{\text{backbone}}\) 时，结构化 agent reasoning 是否能够优于 direct prompting。

### 3.2 System overview

DermAgent 是一个围绕皮肤图像诊断构建的结构化推理外骨骼。其基本思想是将诊断过程拆分为两个由 backbone 承担的关键阶段与一个由 agent 承担的中间增强阶段：首先，Qwen 对输入图像及 metadata 执行初始视觉理解，生成病灶概述、初始鉴别诊断候选以及初步不确定性信息；随后，agent 中间层基于该初始感知结果开展经验检索、skill 选择、skill 执行与证据组织；最后，Qwen 在接收到整理后的 evidence package 后输出最终诊断。因而，系统中存在两次与 backbone 的核心交互：一次用于初始视觉理解和候选诊断生成，另一次用于最终诊断生成，而 final answer 始终由 Qwen 输出。

中间层 agent 的职责可以概括为四类：第一，检索与当前病例相关的分层经验，包括具体病例记忆、战术经验与抽象经验；第二，根据初始感知、metadata、经验上下文与 cognition 状态选择适当的 clinical reasoning skills；第三，将 skill 输出、经验线索、风险提示与不确定性信息组织为结构化 evidence package；第四，在允许的运行模式下，对病例执行结果进行 reflection/writeback，并更新跨病例 cognition 状态。由此，DermAgent 并不改变 backbone 的最终诊断职责，而是通过一层显式、可审计的中间推理结构对最终上下文进行增强。

### 3.3 Design principles

DermAgent 的方法设计基于以下原则。首先，单一最终诊断者原则。系统明确保持 Qwen 作为唯一最终诊断者，agent 不直接输出最终疾病标签，也不通过多模块投票或外围 override 来取代 backbone 的决策。其次，结构化证据原则。agent 的作用不是产生另一个并行诊断答案，而是显式生成、筛选和组织进入最终诊断上下文的结构化证据。第三，经验与认知分层原则。系统将 experience、cognition 与 policy 视为不同层级的状态对象：experience 侧重病例级与跨病例知识内容，cognition 侧重策略记忆与行为偏好，policy 则承载可训练和可版本化的外围控制参数。第四，外围可训练、骨干冻结原则。系统允许 controller、retrieval scoring、evidence calibration 等外围层独立训练或版本化，但不训练 backbone 权重，也不改变最终诊断责任。第五，公平评测原则。agent 带来的收益必须在冻结状态、统一病例列表和一致 backbone 条件下进行评估，以尽量减少不可控状态漂移和不公平对比。

这些原则共同限定了 DermAgent 的边界：它不是多分类器投票系统，不是外挂式 heuristic classifier 集合，也不是通过更换 backbone 获得性能变化的另一种大模型方案。它更接近一个结构化、经验驱动的诊断推理增强层。

### 3.4 Inference pipeline

在推理阶段，DermAgent 采用多阶段流水线。给定输入图像与可选 metadata，Qwen 首先执行初始视觉理解，输出病灶描述、初始鉴别诊断候选以及不确定性线索。基于这一结果，系统从分层 experience bank 中检索与当前病例相关的历史经验，并从 skill bank 中检索当前最可能有帮助的临床推理动作。随后，planner/controller 结合初始 perception、clinical metadata、retrieved experience、skill retrieval signal 以及 cognition 中积累的偏好与统计信息，选择本次病例需要执行的 skills。

被选中的 skill 模块围绕当前病例产生结构化证据，例如观察性线索、比较性线索、风险提示、不确定性描述和信息缺口说明。之后，agent 对这些输出与检索到的经验进行重新整理，构建一个供最终诊断使用的 evidence package。该 package 不是最终答案，而是 final diagnosis 之前的结构化上下文接口，其中包含经组织后的初始感知摘要、经验摘要、skill 输出、风险与不确定性信息以及必要的说明性注释。最后，Qwen 读取原始输入与 evidence package，生成最终诊断结果。若运行模式允许 writeback，则系统在诊断后执行 reflection，提取病例级 outcome、skill helpfulness、经验候选与 cognition update；在正式评测模式下，这一步不会写回在线状态。

### 3.5 Trainable peripheral components

尽管 DermAgent 保持 backbone 固定，系统外围仍包含若干可以独立训练或版本化的策略组件。第一类是 planner/controller，用于对 skill candidates 进行排序、筛选和稀疏选择，从而控制每个病例执行哪些临床推理动作。第二类是 retrieval scoring 或 retrieval reranking，用于对经验检索结果与 skill 候选进行重排序，使检索结果更贴合当前病例与既有认知状态。第三类是 evidence calibration，用于对 skill 输出、经验线索、风险/不确定性信息进行打分、分组、裁剪与排序，决定哪些证据被优先纳入最终 evidence package。第四类是 policy layer，它对 planner、retrieval 与 evidence 相关参数进行统一版本化与候选评估。

这些组件的共同特征在于：它们可以改变证据生产、证据路由与证据组织方式，但不能替代 backbone 给出最终疾病标签。换言之，可训练部分始终位于外围策略层，而不进入最终诊断权本身。这一设计使得系统既可以从执行记录中学习更好的中间策略，又保持了最终诊断责任边界的清晰性。

### 3.6 Frozen evaluation setting

为了评估结构化 agent reasoning 相对于 direct prompting 的真实增益，本文采用冻结且公平的评测设定。正式评测时，在线 writeback 被关闭，系统不允许在评测过程中把当前病例的反思结果继续写回 experience bank 或 cognition state；同时，split state 被固定，意味着 experience、cognition 与相关 policy 状态均来自预先确定的分割与快照，而非在评测过程中持续变化。除此之外，baseline 与 agent 共享相同的 backbone 和相同的 case list，并在一致的输入条件、病例顺序与协议下进行比较。

上述设计的目的，是尽量把性能差异归因于“是否使用 structured agent reasoning”这一因素，而不是归因于 backbone 不一致、病例列表不同、在线状态污染或隐式额外训练。因而，本文的方法框架不仅关心如何构建 skill-experience-cognition 外骨骼，也同样强调在同 backbone、同 case list、同协议约束下进行可复核比较。这一 frozen evaluation setting 构成了 DermAgent 方法设计中的重要组成部分，而不仅仅是实验阶段的附加细节。
