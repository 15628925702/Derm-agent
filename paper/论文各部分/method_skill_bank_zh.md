### 3.X Skill Bank

在 DermAgent 中，一个 skill 被定义为一个 atomic clinical reasoning action。这里的“action”并不意味着技能模块独立完成诊断闭环，而是指其承担一个具有明确临床语义边界的局部推理职责，例如显式提取病灶表型线索、比较候选诊断之间的差异、审查 metadata 与视觉线索的一致性、指出潜在风险信号，或识别当前推理中的不确定性与信息缺口。由此，skill 在系统中的基本单位不是“病种预测器”，而是“证据生产器”：它围绕当前病例生成结构化 evidence，供后续证据组织与最终诊断使用。

这一设计与将外围模块构造成 disease classifier 的路线有本质差异。首先，DermAgent 中的 skill 不直接输出最终病种，也不承担最终诊断责任。系统始终要求 Qwen 作为唯一最终诊断者，因此 skill 的职责是补充和组织 backbone 在单次生成中未必会显式完成的临床动作，而不是与 backbone 争夺最终决策权。其次，将 skill 设计为 atomic clinical reasoning actions，可以避免外围模块退化为一组刚性的子分类器。如果每个 skill 都被训练成判断某一类病种或某一组标签，那么整个系统很容易演化成“若干外挂分类器 + 一个聚合器”的结构，不仅削弱了单一最终诊断者的边界，也会使模块语义与误差来源变得难以审计。相反，当 skill 的输出被限制为结构化证据而非最终标签时，系统便能够在不引入额外诊断权竞争的前提下，将比较、排除、风险审查与不确定性处理等中间动作显式化。

从方法设计角度看，这种 skill 定义同时服务于可解释性、可审计性与后续可训练性。由于每个 skill 对应一个较为稳定的临床动作语义，planner/controller 可以围绕“当前病例需要哪些动作”进行显式选择，而不是围绕“哪个子分类器可能更准”进行隐式组合；reflection 也可以围绕每个 skill 的帮助性、冗余性或潜在误导性进行病例后分析；进一步地，skill 的触发条件、执行结果与对最终 evidence package 的贡献，都可以在 execution record 中被单独追踪。这使得 skill bank 不再只是若干静态函数的集合，而成为一个可被选择、可被分析、可被版本化、也可被外围策略持续利用的推理动作库。

在当前实现中，skill bank 覆盖了若干面向皮肤科诊断的证据生产类型。其一是 perception-oriented evidence extraction，即围绕病灶形态、颜色模式、边界与表面、分布特征以及病灶描述结构化等内容，对初始视觉理解进行更细粒度的显式整理。其二是 differential comparison 相关技能，用于围绕候选诊断之间的相似性与差异性生成比较性证据，并支持进一步的 exclusion reasoning。其三是 metadata consistency checking，用于审查图像表型、病史元数据与输入描述之间是否存在冲突、缺失或可疑之处。其四是 malignancy risk assessment，通过风险导向而非结论导向的方式前置潜在警示信号。其五是 specialist confusion auditing，即围绕若干高混淆诊断对提供更聚焦的专门审查。除此之外，当前系统还实现了 uncertainty assessment，并包含 contradiction checking、information-gap detection 与 escalation recommendation 等与不确定性和安全边界相关的技能；这些模块已经进入现有执行链路，但其最终作用范围与收益仍应在后续冻结评测中进一步验证。

值得注意的是，这些 skill 类型虽然在功能上有所差异，但它们共享同一方法学约束：skill 输出的是结构化 evidence，而不是最终 disease label。即便某一 specialist-oriented skill 聚焦于特定混淆对，其职责也只是生成更细致的比较与审查线索，而不是直接裁决病种。对 DermAgent 而言，这一点尤为关键，因为它保证了所有外围能力都服务于 final reasoning context 的构造，而不会改变系统中的最终责任分工。

作为系统组成部分，skill bank 还承担三个更高层面的角色。首先，它作为一组版本化的源码资产被共享和维护，使技能定义、输出结构与语义边界保持显式可追踪，而不是隐含在临时 prompt 片段之中。其次，它不是被固定顺序机械执行，而是由 planner/controller 根据当前病例状态、经验检索结果与 cognition 偏好进行动态选择，因此同一病例并不要求激活全部技能。最后，skill 的输出不会以投票方式直接汇总为答案，而是进入 evidence package，与经验检索结果、风险标记、不确定性信息及其他上下文一起被组织后送入最终诊断阶段。由此，skill bank 在 DermAgent 中既是能力库，也是中间推理的可组合接口，但始终不是一个并行诊断器集合。
