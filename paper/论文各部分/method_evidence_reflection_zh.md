### 3.X Evidence Package

在 DermAgent 中，agent 中间层的终点并不是另一个诊断结论，而是一个结构化的 evidence package。该对象构成了最终诊断前 backbone 可见的主要增强上下文，也是系统边界设计中的关键接口。换言之，DermAgent 的核心并不是让多个外围模块分别给出病种判断再进行聚合，而是让 skill 输出、经验检索结果、风险提示、不确定性线索与规划依据在最终诊断之前被组织为一个可审计的证据包，并由 Qwen 在此基础上生成 final answer。

将 evidence package 作为统一接口有三个直接目的。首先，它把 agent 的职责严格限制在“证据生产与证据组织”这一层面，从结构上避免了外围模块与 backbone 争夺最终诊断权。其次，它使中间推理过程具备更好的可解释性和可审计性，因为系统可以明确记录哪些 skill 被触发、哪些经验被引用、哪些风险或不确定性被前置，以及这些信息最终如何进入诊断上下文。第三，它为后续训练和策略优化提供了自然接口。由于进入最终诊断阶段的并不是一串无结构的中间文本，而是带有分组与来源语义的 evidence package，planner、retrieval 与 evidence calibration 的贡献便能够被更细粒度地归因和分析。

在当前实现中，evidence package 由若干互补的信息块组成，包括初始视觉理解的摘要、raw/tactical/abstract 三层经验的检索摘要、skill 输出汇总、风险标记、不确定性信息、矛盾检查、信息缺口说明、升级建议以及 planner rationale 等。这里重要的不是字段数量本身，而是其组织方式：DermAgent 并不简单拼接所有中间输出，而是通过 evidence organization 与 evidence calibration 对不同来源的证据进行筛选、分组、排序、裁剪与配额控制。因而，哪些证据最终进入 Qwen 的上下文、以什么顺序进入、哪些信息被突出、哪些信息被降权，本身就是一个被显式建模的外围策略问题。

这一设计对于皮肤科诊断尤其重要。许多病例的难点并不在于基础模型是否“看到了”某个局部形态，而在于这些线索是否在最终诊断前被以合理方式组织起来。例如，某些病例需要先前置 metadata inconsistency 或 contradiction signal，避免后续 reasoning 直接建立在不可靠前提上；另一些病例则需要先突出 malignancy risk 或 unresolved differential，防止系统过早收缩诊断空间。因此，DermAgent 将 evidence package 视为 final reasoning context 的结构化承载体，而不是若干模块输出的简单容器。

更重要的是，evidence package 明确体现了单一最终诊断者架构。即便其中包含丰富的比较性、风险性与不确定性证据，它仍然不直接给出最终病种裁决，而是把这些材料组织成供 backbone 二次推理使用的上下文。在这一意义上，evidence package 既是 DermAgent 与 Qwen 之间的接口，也是系统安全边界与责任分工的具体实现。

### 3.X Reflection and Writeback

如果 evidence package 对应的是病例前向推理中的“证据汇聚”，那么 reflection and writeback 对应的则是病例后向更新中的“经验沉淀”。DermAgent 的设计并不把一次病例运行视为孤立事件，而是将其视为后续推理可以利用的经验来源。因此，在每次病例完成最终诊断后，系统会生成结构化 reflection，对当前推理过程进行病例级总结，并在允许的运行模式下将其中可复用的部分写回到 experience bank 与 cognition state 中。

在当前实现中，reflection 的内容不仅包括简要的 reasoning summary，还包括对当前病例 outcome 的结构化判断、对各个已执行 skill 的 helpfulness assessment、对潜在错误或混淆模式的识别、以及后续可写回的 raw/tactical/abstract experience 候选。与仅生成一段自由文本总结不同，这种 reflection 设计试图回答三个更具体的问题：该病例在当前条件下属于何种 outcome 类型；哪些 skills 对当前证据组织和最终诊断具有帮助、部分帮助或潜在误导；哪些信息值得被沉淀为未来病例可利用的经验或策略信号。由此，reflection 不只是诊断后的解释性附注，而是连接单病例执行与跨病例演化的桥梁。

writeback 则是在 reflection 基础上的受控状态更新机制。对 experience bank 而言，writeback 会将病例级结果转换为 raw case memory、tactical experience 与 abstract experience 的增量；对 cognition state 而言，writeback 会更新混淆模式、失败统计、skill helpful/harmful 统计与偏好信息。这样，当前病例中的局部诊断动作、失败模式和有用证据便不再停留在一次性执行记录中，而能够以结构化形式影响未来病例中的 skill selection、experience routing 与 evidence organization。这也是 DermAgent 能够形成认知进化闭环的必要条件。

不过，本文同样强调 writeback 的边界。首先，writeback 发生在 backbone 之外，它不会修改 Qwen 权重，也不会通过在线更新改变最终诊断模型本身。其次，writeback 并不在所有运行模式下都开启。在训练、调试或经验积累阶段，允许系统将 reflection 结果写回分层经验与 cognition，以支持外围策略优化；但在正式 frozen evaluation 中，在线 writeback 被关闭，split state 被固定，从而避免当前评测样本通过状态污染影响后续样本。这一点尤为关键，因为若不加区分地在评测过程中持续写回，就很难判断性能变化究竟来自结构化 reasoning，还是来自在线状态泄漏与分布内适应。

因此，Reflection and Writeback 在 DermAgent 中承担的是一种受控的后验学习接口：它既允许系统在非正式评测阶段积累经验、更新认知并形成后续策略偏置，又通过冻结评测协议严格限制这种更新在正式比较中的作用范围。通过这一设计，DermAgent 将“可自进化”与“可公平评测”同时纳入系统框架之内，而不是将前者建立在后者失效的基础上。
