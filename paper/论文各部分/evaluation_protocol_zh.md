### 4.X Evaluation Protocol

DermAgent 的评测协议建立在一个基本前提之上：本文的研究问题不是“任意 agent 系统是否可以优于任意 baseline”，而是“在相同 backbone 条件下，结构化、经验驱动的 agent reasoning 是否能够优于 direct prompting”。因此，评测设计的核心并非仅仅给出一组结果数字，而是确保所有比较都在可复核、可归因且尽可能公平的条件下进行。基于这一目标，本文从 fairness principle、frozen evaluation、matched-case comparison、state isolation 与 contamination prevention 等方面约束正式实验协议。

#### 4.X.1 Fairness principle

本文采用 same-backbone fair comparison 原则。对于主结果比较，baseline 被定义为 direct Qwen with the same backbone/service，即在不启用 DermAgent 中间层的情况下，使用与 agent 完全相同的 backbone 服务、相同的模型运行环境与一致的输入条件，直接从图像和可用 metadata 生成最终输出。与之对应，DermAgent 版本在相同 backbone/service 上增加结构化 agent 中间层，包括经验检索、skill 选择、证据组织和最终 evidence package 构造。由此，agent 与 baseline 之间的差异被尽量限定在“是否引入 structured reasoning scaffold”这一点上，而不是混入 backbone 更换、服务差异或运行配置变化。

这一原则之所以重要，是因为在医学 AI 场景中，很多表面上的 agent 增益都可能来自隐式不公平因素，例如 baseline 使用了更弱的 prompting、agent 使用了不同模型服务、两者在病例列表或状态初始化上不一致，或者 agent 在评测过程中获得了额外的在线学习机会。本文明确排除这类比较方式，将 fair comparison 视为方法定义的一部分，而不仅仅是实验实现细节。

#### 4.X.2 Frozen evaluation

正式比较采用 frozen evaluation setting。其含义包括两个层面。第一，split state 固定。用于正式评测的 experience bank、cognition state 与相关 policy 状态来自预先确定的 split-specific snapshot，而不是评测过程中动态累积形成的状态。第二，正式比较时不允许 online writeback。即便 DermAgent 在训练或调试模式下支持 reflection/writeback 以积累分层经验和认知更新，进入正式评测后，这些在线更新必须关闭，防止前序样本通过状态写回影响后续样本。

这样的冻结设定有助于将实验结果解释为“在既定状态快照下，结构化推理框架对当前病例集的增益”，而不是“系统边评边学后逐步适应这组样本”的结果。对于包含 memory、cognition 与 policy 的 agent 系统而言，如果不采用 frozen evaluation，则每个病例都可能改变后续病例可访问的状态，从而使性能增益难以归因，也会削弱与 direct baseline 的可比性。

#### 4.X.3 Baseline definition

本文主 baseline 明确定义为 direct Qwen baseline。具体而言，baseline 使用与 DermAgent 完全相同的 backbone/service，输入为同一病例图像及相同可用 metadata，但不启用 experience retrieval、skill selection、evidence package 与 writeback 等中间机制。baseline 的输出直接来自 backbone 对原始输入的诊断生成。该定义的目的，是构造一个最直接回答研究问题的对照：在保持 Qwen 作为同一最终诊断者的前提下，引入结构化 agent reasoning 是否带来增益。

这一 baseline 定义也意味着，本文不将“不同 backbone 之间的比较”与“agent versus direct prompting 的比较”混为一体。若后续实验包含跨 backbone 对照，则其性质属于扩展性分析或外部比较，而不能替代主问题下的公平基线。主结果必须围绕 same-backbone direct baseline 报告。

#### 4.X.4 Matched-case comparison

所有正式 agent-vs-baseline 比较都基于 matched-case comparison，即 agent 与 baseline 使用完全相同的 case list、相同的病例顺序、相同的数据 split 以及一致的输入可见条件。这样，任一病例上的差异都可以在病例级 execution records 中被直接对齐，支持逐例分析 agent 是否改善、保持或削弱了最终预测。本文认为，对 agent 系统而言，这种逐病例对齐比仅报告聚合平均值更重要，因为外围结构化推理的价值往往集中体现在高混淆病例、高风险病例或信息不足病例上。

Matched-case comparison 还服务于下游训练与错误分析。由于 controller、retrieval 与 evidence organization 的监督信号部分来自病例级 outcome 与 baseline delta，只有在病例严格对齐的前提下，执行记录中的正负信号才具有明确含义。否则，所谓“agent 优于 baseline”将无法被可靠映射回 skill helpfulness、经验路由质量或 evidence calibration 质量。

#### 4.X.5 Metrics

在指标层面，本文优先报告与临床诊断和公平比较直接相关的结果。第一类是整体诊断指标，包括 top-1 accuracy 与 top-k hit rate，用于反映最终诊断与鉴别候选层面的整体表现。第二类是 malignant recall，用于评估系统对恶性或高风险病变的召回能力；鉴于皮肤科诊断中漏掉恶性病变的代价通常高于一般误分类，该指标应作为主结果中的关键安全性指标。第三类是 subgroup metrics，用于在关键混淆子集、恶性相关子集或其它预定义病例簇中报告分层表现，以避免总体平均结果掩盖局部失败模式。第四类是 calibration-related metrics：当实验记录中包含稳定可比较的置信度、uncertainty 或 evidence calibration 信号时，可进一步报告与置信度校准、一致性或不确定性分布相关的指标；若相关测量在某一版本中尚未充分稳定，则应将其作为补充分析或 ongoing validation，而不应夸大为主结论。第五类是 error analysis，包括病例级错误类别、关键混淆对、agent 相对 baseline 的改善/退化分布，以及与 skill helpfulness、经验引用和 contradiction/uncertainty 信号相关的后验分析。

本文强调，指标报告不仅应包含“平均性能”，还应包含“在哪些病例上发生改善或退化”以及“改善或退化与哪些中间推理信号相关”。对于 DermAgent 这类中间层可审计的系统，仅给出单一 aggregate number 不足以支撑方法分析。

#### 4.X.6 Contamination and leakage prevention

由于 DermAgent 显式使用 memory、cognition 与 policy 状态，污染控制是评测协议中的关键组成部分。本文采用三类保护机制。其一是 policy versioning，即每次正式比较都绑定明确的 policy snapshot，使 planner policy、retrieval policy、evidence policy 与 evaluation gate 的版本具有可追踪性，从而避免在同一实验中发生隐式策略漂移。其二是 state isolation，即按 split 固定并隔离 experience 与 cognition 状态，防止训练 split、验证 split 与测试 split 之间通过共享状态发生信息泄漏。其三是 contamination guard，用于检测和约束正式比较中的状态写回、快照一致性与 case-selection 合法性，确保当前评测不会被在线更新或错误状态引用所污染。

除此之外，病例 metadata 本身也需要进行 leakage prevention。对于可能直接泄露标签的信息字段，正式输入应进行过滤或屏蔽，避免系统通过非临床合理路径获得答案。类似地，若某些经验记录与当前评测样本存在直接身份级重合或可逆映射，也应通过 split-aware state 管理与快照隔离加以防止。总体而言，本文将 contamination/leakage prevention 视为 agent 评测不可缺少的基础设施，而不是事后补充的工程措施。

#### 4.X.7 Ablation protocol

为了理解 DermAgent 各组成部分的实际贡献，本文采用组件级 ablation protocol，而不是仅报告 full-agent 与 direct baseline 的单点差异。ablation 的基本原则仍然遵循 same-backbone、same-case-list 与 frozen-state 约束，即所有消融实验必须与 full-agent 和 direct baseline 使用相同 backbone/service、相同病例列表与一致的正式评测协议。

在这一前提下，消融主要围绕以下维度展开：移除 experience retrieval；移除 skill retrieval；去除 specialist confusion-focused skills；保留 raw+tactical 或 raw+abstract 等不同经验层组合；移除 cognition bias；仅保留 foundational observation skills；移除 uncertainty / contradiction / escalation 相关层等。通过这些 matched ablations，可以更清楚地识别分层经验、动态 skill 选择、认知偏置与证据组织层在整体性能和关键子集表现中的作用。重要的是，消融实验的目标并不是追求任意模块的“存在即可有益”，而是检验各层在严格控制条件下是否对最终 same-backbone diagnosis 带来可解释的贡献。

#### 4.X.8 External validation protocol

外部验证用于考察 DermAgent 在不同数据来源、不同病例分布或不同标签体系下的可迁移性，但其解释必须比内部 matched evaluation 更为谨慎。特别是在皮肤科外部数据集中，标签空间往往与训练或开发数据不完全一致：某些数据集只提供粗粒度标签，某些数据集的诊断类别与内部 canonical labels 并不一一对应，另一些数据集则只适合支持 malignant-versus-benign 或部分映射评测。在这些情况下，本文不主张将外部验证结果直接解释为完整多类别诊断性能，而应根据标签空间的一致性程度，采用 malignant-vs-benign、部分标签映射或子集评测等受限协议。

外部验证的另一个关键原则，是保持主问题与扩展问题的区分。若外部数据集仍支持 same-backbone、same-protocol 的 direct-baseline 对照，则可以检验 structured agent reasoning 在分布外条件下是否保持相对收益；若标签空间或输入条件不允许这种严格对照，则外部验证应被定位为补充性分析，而不是替代主结果。无论何种情形，论文都应明确说明标签映射规则、不可比较类别、恶性/良性二分类边界以及哪些结论仅适用于部分可对齐标签空间。

综上，DermAgent 的评测协议并不把 fairness、frozen state 和 contamination prevention 视为附属工程，而是将其作为回答核心研究问题的必要条件。只有在 baseline 被严格定义为 direct Qwen with the same backbone/service、agent 与 baseline 使用相同 case list、正式比较中不允许 online writeback、split state 固定且 contamination guard 生效的条件下，结构化 agent reasoning 的增益才具有可信的解释基础。
