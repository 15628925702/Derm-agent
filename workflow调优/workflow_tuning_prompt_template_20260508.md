# Workflow 调参提示词模板与执行规范 (2026-05-08)

## 文档目的

这份文档用于后续逐个 `model x dataset workflow` 做定向调参时，作为统一的：

- 调参提示词模板
- 跑后分析清单
- 可调参数/策略清单
- 实验节奏规范
- 记录与 git 规范

目标不是盲目“多试几次”，而是让每次 workflow 调整都：

1. 有明确问题定义
2. 有基于既有产物的追溯分析
3. 有小规模到中规模的验证节奏
4. 有清晰的版本记录
5. 有可回溯的 git 分支与结果说明

---

## 总原则

### 1. 逐个 workflow 单独调

一次只调一个明确的 `model x dataset workflow cell`，不要混着动多个组合。

例子：

- `llama / isic2019`
- `medgemma / ham10000`
- `qwen / scin`

### 2. 调参时不改评测协议

调参时：

- 不改最终 frozen evaluation 协议
- 不打开 test writeback
- 不让 agent final 污染 test split
- 不改数据划分原则

允许改的是该 workflow 对应的策略与路由行为，不是大实验协议本身。

### 3. 先小样本，再中样本，先不要直接上大规模

每次调 workflow 的验证节奏固定为：

1. 少 case 快速验证，8case
2. 中等 case 稳定验证  40case
3. 先不要跑大规模的测评

不要一上来就跑 300 case 或全矩阵。

### 4. 每次都用满 8 卡

每轮调试/验证都要尽量把 `8` 张卡全部利用起来。

推荐方式：

- 同一 workflow 的 compare 分 shard 并行
- 或同一 workflow 的不同 ablation / candidate 版本并行

不要让 GPU 空着。

### 5. 每次修改后都要留下记录

每次 workflow 调好一个阶段，都要：

- 在调参清单 md 中追加记录
- 保存 git commit
- 新建或更新对应分支
- 推到 GitHub

---

## 调参前必须先看什么

在动一个 workflow 之前，必须先回看该 workflow 已经跑完的大规模结果，不要凭印象改。

### 必看产物

对于目标 workflow，先看这些：

1. 该组合最终 compare report
2. case-level export
3. baseline summary 与 full_dermagent summary
4. 典型 helped / unchanged / hurt case
5. 该 workflow 对应的 routing / profile / overrides
6. 如果有 bootstrap / frozen state 历史异常，也要看

### 典型路径

先找：

- `paper_data/.../compare_reports/<model>/<dataset>/compare_agent_vs_qwen_*.json`
- `paper_data/case_level_exports/...__<model>__<dataset>__.../`

以及代码：

- `agent/model_workflow_router.py`
- `agent/workflow_profiles.py`
- `agent/paper_exports.py`
- 相关 skill 实现
- 相关 dataset label space / environment override

---

## 调参前的追溯分析：必须分析什么

调一个 workflow 前，先把问题分清楚。至少要回答下面这些问题。

### A. 这个 workflow 当前属于哪类问题

先判断它属于：

1. `Top-1 负向，Top-k 正向`
2. `Top-1 持平，Top-k 正向`
3. `Top-1 持平，Top-k 持平`
4. `Top-1 正向但很弱`
5. `Top-1 和 Top-k 都强`

不同类型，调参方向完全不同。

### B. 帮助 case 和伤害 case 是什么样

至少抽样看：

- `10-20` 个 helped case
- `10-20` 个 hurt case
- `10-20` 个 unchanged case

重点看：

1. agent 为什么能帮到
2. agent 为什么会误伤
3. 哪些 case evidence 明明对了，但 final 没吃进去
4. 哪些 case retrieval 很强，但最后还是跟 baseline 一样

### C. 是 retrieval 问题，还是 final decision 问题

这是最关键的诊断分叉。

#### 如果 Top-k 涨了但 Top-1 没涨

通常说明：

- evidence/retrieval 已经有帮助
- 但 final diagnosis/fusion 没把帮助转成 Top-1

这类优先调：

- final fusion
- override gate
- conservative fusion 程度
- evidence 权重

#### 如果 Top-1 和 Top-k 都没动

通常说明：

- skill 没起作用
- retrieval 没召回有效证据
- workflow 路由几乎没真正介入

这类优先调：

- skill 开关
- planner / selection
- retrieval coverage
- workflow profile 的触发方式

#### 如果 Top-1 负向但 Top-k 仍涨

通常说明：

- 候选集比 baseline 更好了
- 但 final 输出变得更容易“过度干预”或“错误推翻”

这类优先调：

- 推翻 baseline 的门槛
- override 触发条件
- contradiction/uncertainty 压制逻辑
- 风险分层后的 final output policy

### D. 分 label 看问题

一定要按 label / disease category 看：

1. 哪些类提升
2. 哪些类下降
3. 是否集中在少数混淆对
4. 是否是 grouped/coarse taxonomy 映射导致

尤其看：

- dataset label space 是否匹配
- 是否某类 specialist 对某类有帮助但对别类误伤

### E. 看 evidence package 内容是否真的高质量

要分析这些输出本身是不是好：

1. initial perception 是否稳定
2. selected skills 是否合理
3. selected evidence 是否相关
4. contradiction / uncertainty 是否过强或过弱
5. risk layer 是否压错方向
6. final diagnosis prompt 是否被无关 evidence 挤占

核心问题：

- 证据包是“没找到对的证据”
- 还是“找到了但没有被 final 采纳”

---

## 调一个 workflow 时可以改哪些内容

下面是建议系统性考虑的“可调旋钮”。

---

### 1. Skill 开关层

可以改：

- 新增某个 skill
- 关闭某个 skill
- 只在某 dataset 打开某 skill
- 只在某 model/dataset 组合关闭某 skill

适用场景：

- evidence 太弱，说明缺 skill
- evidence 太乱，说明 skill 太多或不相关
- 某 specialist 经常误导

重点可考虑：

- `benign_mimic_specialist`
- `mel_nev_specialist`
- `ack_scc_specialist`
- `differential_compare`
- `contradiction_check`
- `information_gap_detection`
- `uncertainty_assessment`
- `escalation_recommendation`

---

### 2. Skill 优先级 / 权重层

可以改：

- 某些 skill 更容易被选中
- 某些 skill 输出更容易进 selected evidence
- 某些 skill 在 final diagnosis 中占更大权重

适用场景：

- Top-k 已涨但 Top-1 没涨
- 某些正确信号被淹没
- 某些错误 specialist 话语权过高

---

### 3. Retrieval / evidence selection 层

可以改：

- 检索证据的范围
- 检索数量
- specialist evidence 的阈值
- 是否更强调 prototype / confusion memory
- 是否更强调 exclusion evidence

适用场景：

- 完全没起效
- evidence 方向常常偏
- retrieval 命中不稳定

---

### 4. Conservative fusion / override gate 层

可以改：

- agent 推翻 baseline 的门槛
- supporting vs opposing margin 阈值
- subtype override 的阈值
- malignancy override 的阈值
- 当 evidence 不足时是否更保守跟 baseline

适用场景：

- Top-1 负向但 Top-k 正向
- 明显是“会看了，但改错了”

这是最值得调的一层之一。

---

### 5. Uncertainty / contradiction 层

可以改：

- contradiction 触发敏感度
- uncertainty 对 final 的压制强度
- 是否允许低不确定下更积极 override
- 是否在高不确定下更保守退回 baseline

适用场景：

- 明明 evidence 强，却总不敢改
- 或反过来，明明 evidence 不稳，却乱改

---

### 6. Risk layer / malignancy policy 层

可以改：

- risk flag 的阈值
- malignancy evidence 的优先级
- risk layer 是否优先压过 benign mimic 解释
- final 输出是否过度受风险语言影响

适用场景：

- 恶性召回变化不理想
- 某些高风险类别被误判或不敢判

---

### 7. Final diagnosis prompt / final selection 层

可以改：

- final diagnosis 提示词中强调什么证据
- 如何要求模型在 evidence 和 baseline 间做选择
- 是否要求更显式比较 baseline vs evidence override
- 是否限制输出必须更贴合 selected evidence

适用场景：

- Top-k 明显变好，但 Top-1 不涨
- 说明最后一步决策没把候选优势转化出来

---

### 8. Dataset label-space / mapping 层

可以改：

- grouped label space
- dataset-specific label normalization
- coarse taxonomy / fine taxonomy 映射

适用场景：

- dataset 某些类总是被映射错
- 某模型在某 dataset 上需要 grouped label space 才稳定

---

### 9. Workflow profile 层

可以改：

- workflow profile 选择
- skip_specialist_skills
- skip_experience_retrieval
- baseline_anchored_final
- sparse lesion reasoning
- coarse taxonomy workflow

适用场景：

- 某模型在该 dataset 上整体 reasoning style 不对
- 当前 profile 跟数据集特征不匹配

---

### 10. Ablation 与新增 workflow cell

如果已有 workflow 很难通过小改修好，可以：

- 新增一个明确命名的 workflow cell
- 保留旧 cell，不直接覆盖
- 通过新 cell 做 A/B

推荐命名：

- `llama__isic2019__v2_conservative_final`
- `medgemma__ham10000__v2_evidence_promote`
- `qwen__scin__v2_skill_rebalance`

---

## 每轮调参的标准实验节奏

每个 workflow 的调试节奏固定建议如下：

### Step 1. Small-case smoke test

先跑少量：

- `24` 或 `32` case

目的：

- 验证没跑崩
- 看方向对不对
- 快速看 helped/hurt case

### Step 2. Medium-case validation

再跑中等量：

- `80` 或 `120` case

目的：

- 看收益是否稳定
- 看是否只是少数 case 偶然涨

### Step 3. Promotion to candidate

如果中等 case 结果满足：

- Top-1 不负
- Top-k 不负
- hurt case 可解释

再进入更强验证，仍然先不要直接上全量大规模 final。

### Step 4. Large-scale final candidate

只有经过前两步稳定后，才允许上：

- `300` frozen eval case

---

## 每轮调试必须 8 卡全用上

### 允许的 8 卡使用方式

1. 同一 workflow 分 shard 并行
2. 同一 workflow 的多个候选版本并行
3. small-case / medium-case 多组并行

### 不推荐

- 只跑单卡串行，其他卡空着
- 一边分析一边闲置 GPU 很久

---

## 每次调参要记录什么

每次 workflow 结束一轮，都要在调参清单里补一条记录。

建议记录字段：

- 日期
- workflow
- 版本名
- 改了什么
- small-case 结果
- medium-case 结果
- 是否进入下一轮
- 主要 helped/hurt 模式
- 下一步准备改什么

建议追加到：

- `paper_data/workflow_tuning_checklist_20260508.md`

可以新增一个小节，比如：

`## Tuning Log`

---

## Git 与分支规范

每次 workflow 达到一个“阶段性 OK”的版本，都要保存 git。

### 分支命名

按这个格式新建分支：

- `<model>-<dataset>-v2OK`
- `<model>-<dataset>-v3OK`
- `<model>-<dataset>-v4OK`

例子：

- `llama-isic2019-v2OK`
- `medgemma-ham10000-v2OK`
- `qwen-scin-v3OK`

### Git 操作规范

每次阶段完成后：

1. `git status` 确认改动范围
2. 提交清晰 commit message
3. 推送对应分支到 GitHub

commit message 建议：

- `tune workflow for llama isic2019 v2`
- `adjust medgemma ham10000 workflow evidence fusion`

### 不要做的事

- 不要把多个 workflow 混在一个分支里调
- 不要调完不 commit
- 不要跑完有结果但代码状态不可回溯

---

## 每轮调参的标准提示词模板

下面这个模板可以直接给代理/自己下一轮工作时用。

---

### Prompt Template

你现在在 `/data/gh/DermAgent` 项目中。  
我要继续调一个已经完成大规模实验的单个 workflow。  

目标 workflow：`<model> / <dataset>`  
当前版本：`<current_workflow_or_branch>`  

这次调参目标：

1. 基于已经跑完的大规模结果，先做追溯分析，不要直接盲改。
2. 必须先阅读并分析该 workflow 已有产物：
   - compare report JSON
   - case-level export
   - baseline / full_dermagent summary
   - helped / hurt / unchanged case
   - 对应 workflow routing / profile / overrides
3. 必须先判断该 workflow 属于哪类问题：
   - Top-1 负向 / Top-k 正向
   - Top-1 持平 / Top-k 正向
   - Top-1 持平 / Top-k 持平
   - Top-1 弱正向
4. 必须明确分析：
   - 是 retrieval/evidence 问题，还是 final decision/fusion 问题
   - 哪些 label / disease category 在掉点
   - evidence package 是否真的提供了有用证据
   - final diagnosis 是否吃进了这些证据
5. 给出本轮拟修改点，修改范围只限该 workflow 相关内容，必要时可：
   - 新增或关闭 skill
   - 调 skill 优先级
   - 调 retrieval / evidence selection
   - 调 conservative fusion / override gate
   - 调 uncertainty / contradiction 抑制
   - 调 risk layer
   - 调 final diagnosis prompt/fusion
   - 调 dataset-specific label space / profile / workflow cell
6. 不改大实验协议：
   - 不污染 test split
   - 不打开 writeback
   - 不改 frozen evaluation 原则
7. 每次验证都必须用满 8 卡。
8. 验证节奏固定：
   - 先少 case（24~32）
   - 再中等 case（80~120）
   - 先不要直接上大规模 300
9. 每轮结束后要：
   - 更新 `paper_data/workflow_tuning_checklist_20260508.md`
   - git commit
   - 新建或更新分支 `<model>-<dataset>-v2OK` / `v3OK`
   - 推到 GitHub

输出要求：

1. 先给出该 workflow 的问题诊断
2. 再给出本轮拟修改点
3. 再实施代码修改
4. 再跑 small-case
5. 若 small-case 方向正确，再跑 medium-case
6. 最后汇报：
   - Top-1 / Top-k 变化
   - helped / hurt 模式
   - 是否建议进入下一轮

---

## 每次结束时的标准汇报格式

建议每轮结束按这个格式汇报：

- Workflow: `<model> / <dataset>`
- Version: `<workflow version / branch>`
- 本轮修改点:
  - `...`
  - `...`
- Small-case:
  - Top-1 `...`
  - Top-k `...`
- Medium-case:
  - Top-1 `...`
  - Top-k `...`
- 主要 helped 模式:
  - `...`
- 主要 hurt 模式:
  - `...`
- 结论:
  - `继续推进 / 回退 / 再调一轮`

---

## 推荐实践

### 强 workflow 先当模板，不先乱动

优先参考这些强组合的行为模式：

- `medgemma / pad20`
- `skinvl / scin`
- `llama / scin`
- `skinvl / pad20`

### 优先处理这些问题类型

最先该花时间的是：

1. `Top-1 负向`
2. `Top-1 持平但 Top-k 已涨`
3. `Top-1 和 Top-k 都不动`

### 最后才碰最强组合

因为最强组合最容易：

- 一改就回退
- 引入副作用
- 破坏已经稳定的大规模收益

所以它们更适合最后当“精修对象”，不是开局优先项。
