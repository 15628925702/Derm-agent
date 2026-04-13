### 4.X Experimental Setup

#### 4.X.1 Datasets

本文当前实验框架围绕皮肤科图像诊断任务构建，输入由病灶图像及可选临床 metadata 组成。根据现有代码与固定 split 配置，可确认的主数据源为 `pad_ufes_20`，其默认固定划分定义在 `pad_ufes_20_contiguous_v1` 中，并由代码自动生成确定性的 train/val/test case lists。对于该内置 split，代码中明确设定的比例为 70%/15%/15%，划分策略为按 metadata 索引连续切分，以保证 case-offset/limit 驱动的冻结评测与重复运行的一致性。该设置目前是仓库中最明确、可复核的主实验划分来源。

除主数据源外，代码结构也预留并支持在其它皮肤科数据目录上运行同一评测协议，但具体外部数据集名称、目录组织、病例数量和最终使用范围需要根据实际运行日志、split JSON 或实验 manifest 补全。为避免引入未冻结的信息，本文在当前版本中将这些外部数据条件记为 “to be filled from logs/configs/manifests”，而不凭印象预填规模或样本数。对于论文最终稿，建议所有数据集名称、样本量、图像数量和有效标签覆盖范围均从固定 split 文件、evaluation manifest 或 dataset inspection 脚本自动回填。

#### 4.X.2 Label space

当前代码中的 canonical dermatology label space 已有明确实现，主要覆盖六类标准化标签：`BCC`、`ACK`、`NEV`、`SEK`、`SCC` 和 `MEL`。系统通过统一的 label normalization 逻辑将同义词、全称与常见变体映射到上述 canonical labels，例如 basal cell carcinoma 映射到 `BCC`，melanoma 映射到 `MEL`，seborrheic keratosis 映射到 `SEK`，nevus/naevus/mole 等映射到 `NEV`。这一映射被用于病例级结果对齐、top-1/top-k 统计和错误分析，因此正式实验中应基于 canonicalized labels 报告结果。

对于恶性相关评测，当前代码中的 malignant definition 同样是显式的：`BCC`、`ACK`、`SCC` 和 `MEL` 被视为 malignant or clinically high-risk labels，而 `NEV` 与 `SEK` 被视为 benign labels。这里需要说明的是，该定义服务于当前项目中的 malignant recall 与安全性相关评测，并不等同于对所有外部数据集和所有临床场景的普适肿瘤学结论。特别是在外部数据集标签空间不一致、标注粒度不同或存在中间风险类别时，恶性/良性映射应被视为协议级近似，而非天然可直接推广的统一医学事实。

对于 OOD external evaluation，标签空间的一致性是主要限制之一。如果外部数据集与内部 canonical labels 不能一一对应，则不应直接报告完整六类多分类结果，而应根据可对齐程度选择 malignant-vs-benign、部分标签映射、或受限子集评测。换言之，外部验证中可报告的结果范围取决于标签映射是否可信，而不是取决于系统是否能够输出任意字符串诊断。

#### 4.X.3 Data split and reproducibility

为保证结果可复核性，DermAgent 使用确定性的固定 split 机制而不是运行时随机切分。主 split 配置在代码中以显式 ID、元数据路径、比例和切分策略保存，并能够导出为固定 JSON 文件，供训练、验证和测试阶段共享。正式评测时，具体病例选择由 split name、case offset 与 limit 决定，相关 case indices、case IDs、split ID 与 split JSON 路径都会记录在 evaluation manifest 中。因此，同一配置在重复运行时应返回一致的病例列表与顺序。

与此同时，state 也采用 split-aware 管理。训练、验证和测试可分别绑定不同的 experience root、cognition state 与 policy snapshot，以减少跨 split 的状态泄漏。对于论文写作而言，这意味着 reproducibility 不仅来自于病例列表固定，还来自于状态对象的分割与快照固定。最终稿中的样本数量、所用 split ID、病例范围和 offset/limit 组合，建议全部从正式 evaluation manifest 或固定 split JSON 中自动填充。

#### 4.X.4 Backbone and serving setup

本文的核心比较建立在 same-backbone setting 下。当前主实验路径默认以 Qwen 作为 backbone，并通过统一服务接口提供初始视觉理解与最终诊断能力。更具体地说，在 DermAgent 路径中，Qwen 首先根据原始输入生成初始 perception 和候选诊断，再在接收 evidence package 后生成最终诊断；在 direct baseline 中，则由相同 backbone/service 直接基于原始输入生成最终输出。因而，主比较中的 backbone 本体、服务地址、运行时模型配置与 prompt stack 需要在 evaluation manifest 中共同记录。

代码中也存在用于其它服务后端的脚本与运行目录，例如 MedGemma 相关启动脚本和状态目录；但根据本文的主问题设定，这类跨 backbone 对照不应替代 same-backbone direct baseline comparison。若论文最终版本包含不同服务后端的实验，应当在结果章节中明确区分为扩展性分析，而不是与主基线混写。具体服务超参数，例如 timeout、retry、served model name、base URL 与 prompt stack version，均可由 runtime manifest、run profiles 与 evaluation artifacts 自动补全。

#### 4.X.5 Baselines

本文的主 baseline 为 direct Qwen baseline，即在与 DermAgent 完全相同的 backbone/service、输入条件和病例列表下，不启用经验检索、skill selection、evidence package、reflection/writeback 等中间层，而直接由 Qwen 输出最终诊断。这一 baseline 与本文的研究问题严格对应，因为它回答的是：在底层模型保持不变时，引入结构化 agent reasoning 是否带来增益。

在此基础上，实验协议还支持若干 matched ablations 作为内部对照，包括移除 experience retrieval、移除 skill retrieval、去除 cognition bias、限制 skill 层级或移除 uncertainty/escalation 相关模块等。这些设置不构成独立 baseline，而是用于拆解 full-agent 中各个中间层的贡献。若后续引入 cross-backbone baseline 或外部 dermatology specialist models，其角色应明确标注为补充性对照，而非主公平基线。

#### 4.X.6 Implementation details

当前代码库已经提供了训练、评测与分析的主要运行入口，但并非所有实验超参数都适合在当前草稿中手动填写。可确认的实现细节包括：内置固定 split 定义、多个 end-to-end run profiles、frozen evaluation 入口、staged training pipeline、policy snapshot/manifest 机制以及 paper export 工具链。对于训练 epoch、不同 profile 的病例预算、是否运行 ablations、以及 controller/retrieval scorer 的 checkpoint 使用方式，代码中均已有明确配置项或脚本参数。

为避免手工转录错误，本文建议将以下具体实现细节在最终稿中由脚本或日志自动补全：各实验 profile 的名称与预算、所使用的 stable policy ID、controller 与 retrieval checkpoint 路径、服务端模型名、推理 timeout/retry 设置、以及每次正式评测对应的 output root 与 frozen state manifest 路径。对于目前尚未冻结确认的字段，本文在本节中统一采用 “to be filled from logs/configs” 的保守写法。

#### 4.X.7 Evaluation outputs and logging

DermAgent 的评测输出不仅包括聚合指标，还包括一组面向可审计性和后续训练的结构化日志。正式评测运行会生成 evaluation manifest、result manifest、各 target 的 summary 文件以及病例级 `case_execution_records.jsonl`。其中，evaluation manifest 记录数据源、病例列表、split、fairness constraints、model context、frozen state、run targets 与 execution config；result manifest 记录 protocol version、contamination check、各 target summary 与 matched comparisons；而病例级 execution records 则保存 input summary、初始 perception、retrieval bundle、planner 决策、skill outputs、evidence package、final diagnosis、reflection summary 以及 baseline delta 等细粒度信息。

这种输出设计对皮肤科 agent 评测尤为关键。首先，它支持逐病例 matched-case analysis，使研究者能够分析 agent 改善或退化发生在何种病例类型上。其次，它为 controller training、retrieval scorer training 与 error analysis 提供了统一数据来源。再次，它使论文中的表格、图和补充分析能够从相同的结果产物中自动生成，而无需依赖难以追踪的临时统计脚本。对于最终论文版本，建议将主要表格和图中涉及的数值全部回溯到 result manifest、summary 文件与 paper export manifest，以确保 setup、evaluation 与 reporting 三者在同一实验产物链条上闭环。
