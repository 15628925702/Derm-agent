### 3.X Training and Optimization

DermAgent 的训练目标并不是对 Qwen 进行 finetuning，也不是进行端到端的诊断模型重训练。相反，本文将优化重点放在 frozen-backbone 条件下的外围策略层，即那些决定“哪些 skills 被调用”“哪些经验被优先使用”“哪些证据被保留并进入最终诊断上下文”的中间组件。这样的设计与本文的问题设定保持一致：我们关注的是，在底层视觉语言模型保持不变时，结构化、经验驱动的 agent reasoning 是否能够优于 direct prompting。因此，训练对象必须位于 backbone 之外，并且能够在相同 backbone 条件下被独立评估。

在当前实现中，DermAgent 的主要可训练对象包括 controller、retrieval reranker/scorer 与 evidence calibrator。controller training 关注的是 skill selection 问题，即在给定初始 perception、clinical metadata、experience context 与 cognition state 的条件下，学习哪些 clinical reasoning actions 更值得在当前病例中被激活。为此，系统从 execution records 中导出 controller training examples，其中包含可用 skill 候选、planner 决策轨迹、病例状态特征以及病例后反思所给出的 helpful、partially helpful 或 harmful 信号。换言之，controller 的监督并不来自一个额外的 disease label classifier，而是来自中间推理动作在下游病例结果中的有益性与风险性表现。这种训练方式使 controller 优化的对象始终是“中间决策质量”，而不是“直接替代最终诊断器”。

retrieval reranker 或 scorer 的训练则服务于 experience selection 与 skill candidate ordering。其目标不是学习一个新的病例诊断器，而是在已有 heuristic retrieval 的基础上，对经验候选和 skill candidates 进行更细粒度的重排序，使真正对当前病例有帮助的经验或技能更容易进入后续规划阶段。当前实现中，这一训练信号同样来源于 execution records 和 reflection 结果，例如哪些被引用的经验最终对应 helpful 或 harmful skill outcomes，哪些 skill candidate 在类似状态下更可能带来正向贡献，以及 agent 相对于 direct baseline 的下游表现变化。由此，retrieval 模块学习的是“检索对象对后续 reasoning 的实用性”，而不是对病种本身进行重新判断。

evidence calibrator training 则对应 evidence organization 层。其关注点是：在 skill outputs、experience summaries、risk flags、uncertainty signals 与 contradiction cues 已经生成之后，系统应如何对这些证据进行分组、排序、裁剪与权重分配，使最终 evidence package 更适合 backbone 的最终诊断使用。在现有代码结构中，evidence calibrator 已作为独立外围组件接入主链路，并支持与 evidence policy 联动。与 controller 和 retrieval 模块类似，它的优化方向也不是替代 Qwen 进行分类，而是根据病例 outcome、skill helpfulness 与下游诊断表现，学习更稳健的 evidence routing 与 packaging 规则。相较于直接改动 backbone，这种训练方式更容易把性能变化归因到“证据组织是否更合理”这一中间层问题。

这些训练信号主要来自四类来源。第一类是 execution records，即系统在病例级运行后保存的结构化执行记录，其中包含输入摘要、初始 perception、retrieval bundle、planner 决策、skill outputs、evidence package、final diagnosis 与 reflection summary。第二类是 frozen split state，它为训练数据的采样与状态隔离提供了可控边界，使不同 split 下的经验、认知与策略状态能够被独立管理，并为后续 frozen evaluation 保留一致的对照条件。第三类是 helpful/harmful signals，这些信号来自反思阶段对各个 skills、经验引用与证据使用情况的后验判断，用于构造比单纯“最终答对/答错”更细粒度的外围监督。第四类是 downstream evaluation-oriented supervision，即利用最终病例级评估结果、agent 与 direct baseline 的差异表现，以及关键混淆子集上的保守指标，对外围策略的候选更新进行筛选和约束。

在训练流程上，DermAgent 采用 staged training pipeline，而不是将所有外围模块放入一个难以归因的统一黑箱中联合优化。当前代码中，stage 0 负责从已有 execution records 中收集并导出训练数据；stage 1 侧重 controller training；stage 2 面向 retrieval scorer/reranker；stage 3 将训练得到的外围 checkpoint 组装为 candidate policy，并在冻结协议下进行候选评估；stage 4 则负责稳定 checkpoint 的导出与版本化管理。这样的分阶段设计有两个直接好处：一方面，不同外围组件的训练信号与行为变化可以被分别观察和审计；另一方面，候选更新可以在进入 stable policy 之前经过保守 gate，从而降低策略层更新对正式评测和后续部署的扰动。

就当前完成度而言，controller training、retrieval scorer/reranker training 与 staged training pipeline 已经进入主线代码路径，具备明确的训练数据导出、checkpoint 加载与候选策略评估接口。evidence calibrator 目前已经作为主链路中的独立外围组件被实际调用，并在配置层被纳入可训练对象，但其训练与最终冻结评测仍处于持续迭代中。类似地，policy-level optimization、hard-case prioritization 与若干辅助分析组件也已形成较完整的工具链，但其最终作用边界和稳定收益仍应在后续实验中进一步确认。基于此，本文将这些模块表述为“已实现并进入 ongoing optimization”，而不将其写成已经充分收敛的最终训练结论。

从方法论上看，将训练限制在外围策略层而保持 backbone 固定，有几个直接优势。首先，这种设计具有更好的稳定性，因为 backbone 的诊断行为边界保持不变，系统更新主要体现在 skill selection、experience routing 与 evidence packaging 等中间层。其次，它具有更强的可审计性，因为每一次策略更新都可以映射到具体的 controller、retrieval 或 evidence 模块，而不必将变化归因到不可分解的模型权重漂移。再次，它更易于进行因果归因：当性能发生变化时，研究者可以追踪是 skill 选择变了、经验排序变了，还是 evidence calibration 变了。最后，这种设计天然更适合 same-backbone fair comparison。由于 baseline 与 DermAgent 使用相同的 backbone、相同的病例列表与相同的冻结状态，系统间差异主要集中在外围推理结构是否被启用，以及这些外围策略是否经过训练优化。这样可以更清楚地回答本文的核心研究问题，而不会把比较结果混杂为“换了 backbone”“改变了数据条件”或“加入了在线训练”的综合效应。

#### Implementation Details Placeholder

本文目前将具体实现细节保留为可由脚本与运行清单自动补全的占位部分。后续版本可根据训练脚本与导出结果，系统性补充以下内容：各阶段使用的数据路径与 split 设置、controller 与 retrieval scorer 的关键超参数、训练轮数与早停策略、candidate policy 的生成与评测入口、checkpoint 目录组织方式、stable pointer 与 rollback 规则、以及不同运行 profile 下的默认配置。由于这些信息已在当前代码库的脚本、配置文件与 checkpoint/manifest 组织中具备明确入口，后续论文版本可在不改变方法定义的前提下自动填充对应实现细节。
