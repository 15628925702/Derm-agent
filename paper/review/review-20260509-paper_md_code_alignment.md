# paper.md 与 DermAgent 代码一致性审查

审查对象：`/data/gh/paper.md`，主要对照当前 `/data/gh/DermAgent` 代码实现。
日期：2026-05-09。

更新说明：按你的要求，本轮不审查数据集数量、病例数、病种数是否最终正确；这些属于后续 benchmark 扩展与结果填充问题。下面只看方法模块是否符合真实代码和设计思想。

二次更新：按你的补充要求，本版不再把 `paper.md:96-99`, `paper.md:153-157` 中的 clinician-guided skill evolution 表述，以及 `paper.md:398-424` 中 specialist skill / benign mimic guard 的细节作为需要处理的问题；workflow routing 也不作为主卖点，只保留为实验配置层面的轻量备注。

## 总体判断

`paper.md` 的大方向与代码设计思想基本一致：DermAgent 确实是围绕 frozen backbone 的 test-time structured evidence scaffold，而不是重新训练皮肤科 backbone；skill 也确实被约束为结构化证据生产动作，不承担最终诊断；代码中也真实存在 skill registry、layered experience bank、cognition state、policy snapshot、reflection/writeback、frozen writeback guard 和 physician evidence summary。

但 `paper.md` 当前写法比代码更“理想化”和“抽象化”，有几处会让读者误解实际系统：

1. 方法流程漏掉了几个真实关键模块：skill retrieval、evidence calibrator、split-aware contamination control、label-space/metadata leakage handling；conservative fusion/guard 如果在正式实验启用，应放在实验配置或附录里交代。
2. 公式化流程把 final diagnosis 前的 baseline diagnosis anchor 和二次 retrieval 隐去了；代码不是简单的 `h0 -> seek -> synthesize -> evolve`。
3. `Experience-enhanced skill`、`hierarchical experience memory`、`cognition state`、`policy` 这些概念方向正确，但需要更清楚地区分“知识内容层”“行为统计层”“版本化控制层”，否则容易显得像堆模块。
4. evidence package 的概念对，但需要补上真实代码中的 evidence calibrator、selected evidence、evidence decision policy 和 serialized evidence text。

## 逐条批注

### 暂不审查：数据集与 benchmark 数量

位置：`paper.md:27-29`, `paper.md:100-103`, `paper.md:130-132`

本轮按方法审查处理，不判断这里的数量是否最终正确。后面新增数据集后，这一段只需要统一从正式 benchmark manifest / paper export 回填即可。

### 1. 核心执行流程基本正确，但漏掉真实的 baseline anchor 与二次 retrieval

位置：`paper.md:220-240`, `paper.md:321-351`

当前写法：`initial hypothesis -> evidence seeking -> final synthesis -> evolution`。

代码事实：

- `run_agent()` 的主路径是：
  `initial_perception` -> `baseline_diagnosis` -> first experience retrieval -> skill retrieval -> planner -> skill execution -> second experience retrieval -> evidence package/evidence calibrator -> optional physician summary -> final diagnosis -> conservative fusion wrapper -> reflection/writeback。
- 证据：`agent/run_agent.py:88-230`。
- 初始阶段不是只有 `h0`，还真实生成了 `baseline_diagnosis`，并在 final prompt 中作为 `risk_layer.baseline_preview` / image-metadata anchor 使用。
- 证据：`agent/run_agent.py:90-95`, `integrations/openai_client.py:1255-1268`。

建议改法：

- 把公式中的 `h0` 扩展为 `initial perception + baseline anchor`。
- 在 `Aseek` 中明确包含 “pre-skill retrieval” 和 “post-skill retrieval/update of retrieval bundle”；代码在技能执行前后各检索一次。
- 最终 synthesis 应写成 “backbone reads curated evidence package under a conservative baseline-anchor prompt”，而不是像无条件用全部证据直接生成。

### 2. “skills do not produce final disease labels” 基本正确，但需要加边界措辞

位置：`paper.md:336-341`, `paper.md:399-402`, `paper.md:419-424`

代码事实：

- 系统 prompt 明确禁止 skill 输出 final diagnosis / disease classification / treatment decision。
- 证据：`integrations/openai_client.py:1118-1124`。
- 但 specialist skill 的 clinical purpose 是处理高混淆 disease pairs，输出中可能会出现候选病名相关的 supporting/opposing evidence。因此论文里最好写 “do not emit authoritative final labels or vote on final diagnosis”，不要写成 “never mention labels”。

建议改法：

- “Skills are not final classifiers and do not have diagnostic authority; disease names may appear only as candidate-specific evidence targets.”

### 3. Evidence package 写法方向正确，但字段不完全对应真实对象

位置：`paper.md:425-451`

当前写法列出了 initial observation, observation evidence, metadata consistency, differential support/conflict, risk cues, uncertainty, gaps, retrieved experience, final diagnostic context。

代码事实：

- 真实 `EvidencePackage` 字段包括：
  `perception`, `retrieved_experience`, `skill_outputs`, `risk_flags`, `uncertainty`, `notes`,
  `initial_perception_summary`,
  `retrieved_raw_cases_summary`,
  `retrieved_tactical_experiences_summary`,
  `retrieved_abstract_experiences_summary`,
  `uncertainty_summary`,
  `contradiction_summary`,
  `information_gap_summary`,
  `escalation_summary`,
  `planner_rationale`,
  `confusion_cluster_summary`,
  `selected_evidence`,
  `evidence_decision_policy`,
  `serialized_evidence_text`,
  `evidence_calibration_debug`。
- 证据：`agent/evidence_package.py:10-35`。
- 真实关键不是简单 route/filter/compress，而是由 `EvidenceCalibrator` 对 skill outputs + retrieval summaries 做打分、去重、配额与 section plan，再生成 `selected_evidence` 和 `serialized_evidence_text`。
- 证据：`agent/aggregator.py:33-133`, `agent/evidence_calibrator.py:20-50`, `agent/evidence_calibrator.py:205-340`。

建议改法：

- 4.5 需要单独加一个 “Evidence calibration and selection” 小节。
- 把 `selected_evidence`、`evidence_decision_policy`、`serialized_evidence_text` 作为最终进入 backbone 的核心接口写清楚。
- “support/conflict/risk/gaps” 可以作为语义分组，不要暗示它们是完全一一对应的顶层 JSON 字段。

### 4. Hierarchical experience memory 方向正确，但代码不只是三层存储，还包含 transform/writeback/consolidation/retrieval scoring

位置：`paper.md:452-457` 及后续待写部分

代码事实：

- experience schema 确实有三层：`RawCaseMemory`, `TacticalExperience`, `AbstractExperience`。
- 证据：`memory/experience_schema.py:66-127`。
- `ExperienceBank.retrieve_bundle()` 返回 raw/tactical/abstract/merged/planner_summary/skill_summary/aggregator_summary，不是单一 nearest-neighbor 列表。
- 证据：`memory/experience_bank.py:119-158`, `memory/experience_retriever.py:96-162`。
- retrieval scoring 使用 ddx、morphology clues、metadata patterns、risk patterns、uncertainty、confusion pair/cluster 等启发式；还支持 optional learned reranker。
- 证据：`memory/experience_retriever.py:41-94`, `memory/experience_retriever.py:181-325`, `agent/run_agent.py:81-118`。

建议补写：

- 经验库不是 “flat memory retrieval baseline” 的反面而已；它在代码里同时服务 planner、skill execution context、aggregator/evidence calibrator。
- formal evaluation 中写回关闭，training/accumulation 中 writeback 打开。这个边界需要紧跟 memory 章节写，防止 reviewer 质疑 test leakage。

### 5. Cognition state 与 policy 的表达匹配，但应加上 version/frozen split 细节

位置：`paper.md:365-370`

代码事实：

- Cognition state 真实字段包括 confusion patterns、preferred skills、retrieval preferences、failure statistics、skill statistics、workflow preferences、state split/version、evolution generation。
- 证据：`cognition/cognition_state.py:14-46`。
- Policy layer 真实分成 planner_policy、retrieval_policy、evidence_policy、evaluation_gate，并带 policy_id/version/status/state_version。
- 证据：`agent/policy_config.py:72-90`, `agent/policy_config.py:97-220`。
- Frozen guard 禁止 frozen mode 下 writeback。
- 证据：`agent/contamination_guard.py:93-114`。

建议改法：

- 在 Methods 中明确 `Ω = (S, M, c, π)` 不是一个抽象状态，而是 split-aware and versioned state。
- 加一句 “formal evaluation binds experience, cognition and policy snapshots to a split-aware state version”。

### 6. 实现备注：workflow routing 不作为主卖点

位置：当前 `paper.md` 方法部分未覆盖

代码事实：

- 系统有 `workflow_context`，会根据 dataset/label space/presentation mode 推断 workflow profile。
- 证据：`agent/workflow_profiles.py:6-28`, `agent/workflow_profiles.py:81-173`。
- 还有 model-dataset workflow profiles 与 cell-level overrides。
- 证据：`agent/model_workflow_router.py:43-185`, `agent/model_workflow_router.py:715-807`。

建议处理：

- 不建议把 workflow routing 放进方法主线，也不建议把它写成 DermAgent 的主要贡献。
- 如果某组正式实验依赖 dataset/model-specific overlay，只需在 Experimental Setup 或 Appendix 写一句：runtime profile/policy overlays were versioned and frozen during evaluation。
- 主文方法仍应强调共享机制：structured evidence scaffold、adaptive skill selection、experience retrieval、evidence calibration、frozen evaluation。

### 7. Conservative fusion / guard 作为实验配置备注即可

位置：当前 `paper.md` 方法部分未覆盖

代码事实：

- `run_agent()` 在非 legacy path 下会调用 `apply_conservative_agent_fusion()`。
- 证据：`agent/run_agent.py:202-211`。
- 当前 global stable policy 中 `conservative_fusion_mode` 是 `off`，但 dataset adaptation policies 与 model workflow routing 可打开 soft/graded conservative behavior。
- 证据：`state/policy/current_stable_policy.json`, `state/dataset_adaptation/*/policy/current_stable_policy.json`, `agent/model_workflow_router.py:640-707`。
- Fusion decision 使用 baseline label、agent label、selected evidence、risk layer、override layer、uncertainty、support margin、workflow context 等决定是否 anchor baseline 或接受 agent。
- 证据：`skills/workflow_fusion_decision.py:27-70`, `skills/workflow_fusion_decision.py:73-137`。

建议改法：

- 如果正式实验使用了 fusion/guard，需要在 Experimental Setup 或 Appendix 说明，否则 reviewer 会认为 agent 不只是 evidence scaffold，而是多了一层 hidden decision rule。
- 如果正式主实验 fusion mode 是 off，可以一句话说明 “fusion wrapper was present but configured as pass-through for the reported main comparison”，并报告 policy id。
- 不建议把 fusion/guard 写成方法主卖点。

### 8. Physician-readable evidence summary 描述基本真实，但应强调 optional / doctor-support-only

位置：`paper.md:372-380`

代码事实：

- 该模块通过 execution override/env 开启，默认不是必然运行。
- 证据：`agent/run_agent.py:193-201`, `agent/run_agent.py:418-455`。
- 输出 intended_use 明确是 `doctor_support_only_not_final_diagnosis`。
- 证据：`agent/run_agent.py:487-523`。

建议改法：

- 保留该段，但写 “optional module disabled unless explicitly enabled”。
- 不要把它作为所有实验默认输出，除非 manifest 证明已开启。

### 9. Results 里的方法性 claim 需要谨慎

位置：`paper.md:115-126`, `paper.md:170-180`

当前写法已经声称 “consistently improves across evaluated benchmarks”，且 PAD-UFES-20 指标为 29.6 -> 39.7、48.2 -> 82.2。

风险：

- 当前 repo 中较新的 `docs/experiments/当前实验结果与大规模运行建议_20260424.md` 记录过一组 80-case PAD 结果：top1 28.75 -> 35.00，但 malignant recall 74.19 -> 61.29，明确提示不能只写全面提升。
- `paper/main.tex` 里的 345-case 数字可能是另一套更晚的主结果，但 `paper.md` 没有给 manifest/source path。

建议改法：

- 所有结果数字后面必须绑定 run id / result manifest / case count / split / policy snapshot。
- “consistently delivers superior performance” 改成更保守的 “improves primary top-1 and malignant-risk metrics in the reported controlled comparison; external generalization is evaluated separately under label-space constraints”。
- 如果 top-k 下降，应在 abstract/results 主文承认。`paper/main.tex:238-257` 这一点写得比 `paper.md` 更稳。

## 建议补进 Methods 的关键小节

1. `Runtime execution path`
   - initial perception
   - baseline diagnosis anchor
   - experience retrieval before planning
   - skill retrieval
   - planner/controller
   - skill execution
   - second retrieval
   - evidence calibration/package
   - final diagnosis
   - reflection writeback boundary

2. `Skill retrieval and adaptive budget`
   - skill candidate retrieval 与 planner 是两个阶段。
   - controller 的设计思想不是多调用，而是按 uncertainty/risk/confusion 分配 test-time compute。

3. `Evidence calibration`
   - selected evidence, section plan, quotas, dedup, critical skills, cluster-aware ordering。

4. `Governance and frozen evaluation`
   - writeback allowed only in accumulation/training, forbidden in frozen evaluation。
   - split-aware versions for experience/cognition/policy。

## 一句话结论

`paper.md` 的设计主线是对的，但现在更像“理想框架描述”；真实代码是一套更复杂、更保守、更工程化的 frozen-backbone evidence-routing system。最终稿要把 claim 收到真实实现上：强调 structured evidence、skill/experience/cognition/policy 分层、adaptive skill budget、evidence calibration 与 frozen evaluation；workflow/fusion 只作为实验配置边界轻量交代即可。
