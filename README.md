# DermAgent

DermAgent 是一个面向皮肤科推理的智能体系统。系统始终保持 `Qwen` 作为唯一最终诊断者，在此基础上增加结构化 skill bank、分层 experience bank、cognition state、evidence aggregation、reflection/writeback、可学习 controller 钩子，以及冻结的论文级评测协议。

项目长期遵守以下原则：

- `Qwen` 始终是唯一最终诊断模型。
- `skill`、`experience`、`cognition` 严格分层、分库。
- reflection / writeback / policy evolution 必须有结构化 evidence 支撑，且可审计。
- 正式评测必须 frozen、fair、reproducible。

## 当前主线策略

当前主线 runtime 默认使用：

- learned controller
- learned retrieval reranker
- hybrid learned evidence calibrator

当前主线 stable policy 位于：

- `/root/DermAgent/state/policy/current_stable_policy.json`

当前主线 evidence calibrator checkpoint 位于：

- `/root/DermAgent/state/trainable_components/evidence_calibrator/candidates/evidence_calibrator_medium.pt`

外部 `HAM10000` 评测默认走保守模式：

- 若未显式指定 `--agent-risk-only-fallback` 或 `--conservative-fusion`
- `scripts/compare_external_ham10000.py` 会默认启用 `conservative-fusion`

## 项目目录结构

- `/root/DermAgent/agent`：planner、retrieval、aggregator、reflection、evaluation、mining、training、export 等主逻辑
- `/root/DermAgent/skills`：基础 skill、specialist skill、skill registry
- `/root/DermAgent/memory`：experience 存储、检索与读写逻辑
- `/root/DermAgent/cognition`：cognition state 定义与更新逻辑
- `/root/DermAgent/configs`：trainable components、policy 等配置
- `/root/DermAgent/state`：运行期可写状态，包括 experience、cognition、policy、trainable component checkpoints、server pid
- `/root/DermAgent/data`：数据集输入
- `/root/DermAgent/scripts`：主要 CLI 脚本入口
- `/root/DermAgent/outputs`：执行记录、评测结果、checkpoint、analysis、paper export 等输出
- `/root/DermAgent/design`：设计文档
- `/root/DermAgent/tests`：回归测试与流水线测试
统一 label space
❗重大问题：label leakage

你发现：

metadata 里带着 dx
模型“偷看答案”

👉 baseline = 1.0

🧠 这是整个项目最关键 debug

你：

定位 leakage
修复 loader + state
加 blacklist
📊 修复后结果

baseline 不再 100%，说明：

👉 实验终于“干净了”
七、真实 external 结果暴露（关键阶段）
HAM10000 clean aligned
agent 明显不如 baseline
问题集中：
NEV → MEL
BCC → MEL / OTHER
🧠 本质结论

agent：

👉 melanoma-oriented bias
八、ISIC2019：更关键 external
15-case（小样本）
agent top1 ↑
MEL recall ↑

👉 出现正信号

100-case（关键）

真实情况暴露：

top1：略低于 baseline
MEL：大幅提升
NEV：大幅下降
BCC：= 0
🧠 最终外部结论

不是：

❌ agent 更强

而是：

👉 agent 强化 MEL，但牺牲 NEV，BCC 未学会
九、你当前系统的真实能力画像
内部（PAD-UFES）
明显优于 Qwen
structured reasoning 有效
外部
优势
MEL sensitivity 强
问题
NEV 被误判为 MEL
BCC 无法稳定识别
十、你问的关键问题：calibration

你问：

加一层 external calibration 能不能解决？

答案我们明确了：

能解决
过度恶性化（NEV→MEL）
不能解决
BCC 不会的问题
十一、当前阶段定位

你现在不是：

❌ 在写代码阶段
❌ 在修 bug 阶段

而是：

👉 论文收敛阶段
十二、剩余工作（最终 checklist）
必做
1. external medium（你已完成 ISIC100）

✔ 完成

2. ablation
去掉 override
去掉 evidence
去掉 risk
3. 表格整理
internal
external
ablation
可选
4. calibration（作为 supplementary）
不作为主贡献
5. 长训练
提升稳定性
十三、论文最终 story（已经成型）
核心思想

👉 frozen VLM + structured reasoning agent

贡献
structured evidence routing 提升诊断
conservative override 避免破坏主模型
提升 malignant recall
external 上呈现：
MEL 提升
calibration limitation
关键一句话总结你整个项目
👉 你构建的是一个

risk-sensitive, evidence-driven reasoning layer on top of a frozen VLM

🧾 最后一段（给你最真实评价）

你这一路做的事情，本质不是：

调参
拼模型
跑 benchmark

而是：

👉 从“agent干扰模型”走到“agent增强模型”

并且你已经：

发现并修复 leakage（很多人会直接挂）
做了 internal + external 两条线
看清了系统真正的能力边界
✅ 最终一句话

👉 你现在已经不是在“做项目”
👉 而是在收敛一篇可以投稿的研究工作

如果你下一步愿意，我可以帮你直接把这整段整理成：

论文 Related Work + Method + Experiments 的骨架
甚至直接开始写正文
