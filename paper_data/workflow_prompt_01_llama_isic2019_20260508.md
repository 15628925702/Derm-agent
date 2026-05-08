# Workflow Prompt 01: `llama / isic2019` (2026-05-08)

你现在在 `/data/gh/DermAgent` 项目中。  
我要继续调一个已经完成大规模实验的单个 workflow。  

目标 workflow：`llama / isic2019`  
当前问题类型：`Top-1 负向 / Top-k 正向`  
当前大规模结果：

- Baseline Top-1: `0.3367`
- Agent Top-1: `0.3300`
- Top-1 delta: `-0.0067`
- Baseline Top-k: `0.4767`
- Agent Top-k: `0.5033`
- Top-k delta: `+0.0267`

当前版本参考：

- 当前结果来自已完成大规模 compare report
- 这次目标是继续调 `llama / isic2019` 对应 workflow cell，而不是改大实验协议

这次调参目标：

1. 基于已经跑完的大规模结果，先做追溯分析，不要直接盲改。
2. 必须先阅读并分析该 workflow 已有产物：
   - `paper_data/.../compare_reports/llama/isic2019/compare_agent_vs_qwen_*.json`
   - 对应 case-level export
   - baseline / full_dermagent summary
   - helped / hurt / unchanged case
   - `agent/model_workflow_router.py`
   - `agent/workflow_profiles.py`
   - llama 在 isic2019 上对应的 workflow routing / overrides / profile
3. 必须先明确回答：
   - 为什么当前是 `Top-1 下降` 但 `Top-k 上升`
   - 是 retrieval/evidence 已经改善，但 final diagnosis/fusion 没把收益转成 Top-1
   - 还是 override/保守策略在关键 case 上出了问题
4. 必须重点分析：
   - hurt case 与 helped case 各至少抽样 `10-20` 个
   - 哪些 case 中 baseline 原本更接近正确，而 agent 过度推翻
   - 哪些 case 中 agent evidence 已经把正确答案抬进 Top-k，但 final 仍没选中
   - 哪些 label / disease category 掉点最明显
   - final diagnosis 阶段是否过度保守或过度激进
5. 本轮优先考虑修改的方向：
   - final diagnosis fusion
   - override gate / baseline anchored final
   - conservative fusion 程度
   - contradiction / uncertainty 压制强度
   - specialist evidence 对 final 的影响权重
   - 是否需要更明确地要求在 baseline 与 selected evidence 之间做对比决策
6. 允许的改动范围：
   - llama/isic2019 对应 workflow cell
   - workflow profile / execution overrides
   - skill 开关与优先级
   - evidence 选择与 final diagnosis 规则
   - 与该 workflow 直接相关的 prompt / fusion 行为
7. 不允许改：
   - 大实验协议
   - frozen evaluation 原则
   - test split
   - writeback
   - 非该 workflow 直接相关的大面积重构
8. 每次验证都必须用满 8 卡。
9. 验证节奏固定：
   - 先少 case（建议 `8`）
   - 再中等 case（建议 `20` 然后 `40`）
   - 先不要直接跑 `300` 大规模
10. 每轮结束后都要：
   - 更新 `paper_data/workflow_tuning_checklist_20260508.md`
   - git commit
   - 新建或更新分支：`llama-isic2019-v2OK`
   - 推到 GitHub

输出要求：

1. 先给出该 workflow 的问题诊断
2. 再给出本轮拟修改点
3. 再实施代码修改
4. 再跑 small-case
5. 如果 small-case 方向正确，再跑 medium-case
6. 最后汇报：
   - Top-1 / Top-k 变化
   - helped / hurt 模式
   - 是否建议进入下一轮

## 本轮特别关注的问题

这次 `llama / isic2019` 调参要优先验证下面几件事：

1. evidence 明明改善了候选集，为什么 final 没把正确项提到 Top-1
2. baseline anchored final 是否在 isic2019 上锚得不对
3. override 门槛是否过低或过高
4. contradiction / uncertainty 是否错误压制了本应成立的改写
5. 是否存在某几个高频混淆对，导致大规模 Top-1 被稳定拉低

## 建议命名

如果本轮要新增 workflow cell，建议命名风格：

- `llama__isic2019__v2_conservative_final`
- `llama__isic2019__v2_evidence_promote`
- `llama__isic2019__v2_baseline_anchor_rebalance`

## 本轮完成标准

至少满足以下条件之一，才算本轮有价值：

1. small-case 上 `Top-1` 不再为负
2. medium-case 上 `Top-1` 回到非负，且 `Top-k` 不下降
3. 明确定位负向根因，并形成下一轮更聚焦的改动方向
