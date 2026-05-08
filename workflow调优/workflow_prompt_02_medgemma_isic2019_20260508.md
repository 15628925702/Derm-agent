# Workflow Prompt 02: `medgemma / isic2019` (2026-05-08)

你现在在 `/data/gh/DermAgent` 项目中。  
我要继续调一个已经完成大规模实验的单个 workflow。

目标 workflow：`medgemma / isic2019`  
当前问题类型：`Top-1 负向 / Top-k 正向`  
当前大规模结果：

- Baseline Top-1: `0.2433` (`73/300`)
- Agent Top-1: `0.2400` (`72/300`)
- Top-1 delta: `-0.0033`
- Baseline Top-k: `0.4800` (`144/300`)
- Agent Top-k: `0.5000` (`150/300`)
- Top-k delta: `+0.0200`
- helped: `0`
- hurt: `1`
- Top-k gain: `6`
- Top-k hurt: `0`

当前版本参考：

- 当前结果来自已完成大规模 compare report：
  - `paper_data/final_3x3_delta_pilot_20260507/compare_reports/medgemma/isic2019/compare_agent_vs_qwen_20260507T085330Z.json`
- 对应 case-level export：
  - `paper_data/case_level_exports/final_3x3_delta_pilot_20260507__medgemma__isic2019__eval300/`
- 本轮目标是继续调 `medgemma / isic2019` 对应 workflow cell，而不是改大实验协议。

## 已知初步追溯线索

1. 当前 Top-1 负向主要来自单个 hurt case：
   - `ISIC_0060096`
   - GT: `MEL`
   - baseline final: `Malignant Melanoma`
   - agent final: `Basal Cell Carcinoma`
   - agent differential: `[Basal Cell Carcinoma, Squamous Cell Carcinoma, Nevus, Malignant Melanoma]`
   - 说明 final/fusion 把 baseline 原本正确的 melanoma 推翻了，但正确项仍在 differential 中。

2. Top-k 正向来自 6 个新增命中，且没有 Top-k hurt：
   - `ISIC_0068276`: GT `MEL`, final `NV`, differential includes `MEL`
   - `ISIC_0070484`: GT `AK`, final `SCC`, differential includes `AK`
   - `ISIC_0053804`: GT `MEL`, final `NV`, differential includes `MEL`
   - `ISIC_0058074`: GT `SCC`, final `NV`, differential includes `SCC`
   - `ISIC_0066085`: GT `MEL`, final `NV`, differential includes `MEL`
   - `ISIC_0066929`: GT `BCC`, final `SCC`, differential includes `BCC`

3. Label 层面：
   - Top-1 掉点集中在 `MEL`: baseline TP `27` -> agent TP `26`
   - 其他 label Top-1 基本持平
   - 这更像 final diagnosis / fusion / override gate 问题，而不是 retrieval 全局变差。

## 本轮调参目标

1. 基于已经跑完的大规模结果，先做追溯分析，不要直接盲改。
2. 必须先阅读并分析该 workflow 已有产物：
   - compare report
   - 对应 case-level export
   - baseline / full_dermagent summary
   - helped / hurt / unchanged case
   - Top-k gain case
   - `agent/model_workflow_router.py`
   - `agent/workflow_profiles.py`
   - medgemma 在 isic2019 上对应的 workflow routing / overrides / profile
3. 必须先明确回答：
   - 为什么当前是 `Top-1 下降` 但 `Top-k 上升`
   - 是 evidence 已经改善候选集，但 final diagnosis/fusion 没把收益转成 Top-1
   - 还是 override/保守策略在关键 MEL case 上过度推翻 baseline
4. 必须重点分析：
   - hurt case `ISIC_0060096` 的完整 evidence / final fusion 过程
   - Top-k gain case 至少全部 6 个
   - unchanged case 抽样 `10-20` 个
   - baseline 原本正确但 agent 过度推翻的模式
   - agent evidence 已把正确答案抬进 Top-k 但 final 没选中的模式
   - 哪些 label / disease category 掉点最明显
   - final diagnosis 阶段是否对 melanoma / malignant label 保护不足
5. 本轮优先考虑修改方向：
   - final diagnosis fusion
   - melanoma baseline protection / malignant protection
   - override gate / baseline anchored final
   - conservative fusion 程度
   - contradiction / uncertainty 压制强度
   - specialist evidence 对 final 的影响权重
   - 是否需要更明确地要求在 baseline 与 selected evidence 之间做对比决策
6. 允许的改动范围：
   - `medgemma/isic2019` 对应 workflow cell
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
   - 先少 case：`8`
   - 再中等 case：`20`，然后 `40`
   - 如方向稳定，再考虑 `80`
   - 先不要直接跑 `300`
10. 每轮结束后都要：
   - 更新 `workflow调优/workflow_tuning_checklist_20260508.md`
   - git commit
   - 新建或更新分支：`medgemma-isic2019-v2OK`
   - 推到 GitHub

## 本轮特别关注的问题

1. `ISIC_0060096` 为什么从正确 `MEL` 被推到 `BCC`
2. melanoma baseline anchor 是否应该更强，尤其当 baseline 已是 `MEL` 且 agent differential 仍包含 `MEL`
3. Top-k gain 的 6 个 case 是否存在可安全 promotion 的共同门槛
4. malignant differential 中 `MEL/BCC/SCC/AK` 的最终排序是否过度保守或过度激进
5. contradiction / uncertainty 是否错误压制了本应成立的 malignant promotion
6. 是否存在高频混淆对：
   - `NV -> MEL`
   - `NV -> SCC`
   - `SCC -> AK`
   - `SCC -> BCC`

## 建议命名

如果本轮要新增 workflow cell，建议命名风格：

- `medgemma__isic2019__v2_melanoma_anchor`
- `medgemma__isic2019__v2_malignant_promote`
- `medgemma__isic2019__v2_evidence_rank_fusion`

## 本轮完成标准

至少满足以下条件之一，才算本轮有价值：

1. small-case 上 `Top-1` 不再为负
2. medium-case 上 `Top-1` 回到非负，且 `Top-k` 不下降
3. 明确定位负向根因，并形成下一轮更聚焦的改动方向

## 输出要求

1. 先给出该 workflow 的问题诊断
2. 再给出本轮拟修改点
3. 再实施代码修改
4. 再跑 small-case
5. 如果 small-case 方向正确，再跑 medium-case
6. 最后汇报：
   - Top-1 / Top-k 变化
   - helped / hurt 模式
   - 是否建议进入下一轮
