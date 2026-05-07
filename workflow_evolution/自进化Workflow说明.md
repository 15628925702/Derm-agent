# DermAgent 自进化 Workflow 说明

这份文档用于描述 DermAgent 在新模型、新医院数据或新数据集上如何通过“离线反思 + 医生经验输入 + 人工审核启用”的方式生成新的专属 workflow。

核心定位：

- 自进化 workflow 不是默认自动上线的黑箱策略。
- 它是一个离线 proposal 机制：先分析本地运行结果，再生成候选 workflow。
- 候选 workflow 默认隔离保存，不影响正式运行。
- 只有医生或工程人员审核后，显式批准并打开运行时开关，候选 workflow 才会生效。

这套机制可以把逐条调 `model x dataset` workflow 的过程，包装成可审计、可复现、可人工把关的“workflow evolution”能力。

## 适用场景

当 DermAgent 被部署到一个新环境时，可能遇到：

- 新医院图像风格和原数据集不同。
- 新模型 backbone 的错误模式不同。
- 某些病种或病种 family 容易被系统性误判。
- 默认 workflow 太保守，导致 agent 帮不上 baseline。
- 默认 workflow 太宽松，导致 agent 改坏 baseline。
- 某些 skill 在该模型上经常产生噪声。
- 医生知道本地真实临床经验，但不想改代码。

此时可以使用自进化 workflow 流程：

1. 在 frozen evaluation 上跑 direct baseline 和 agent。
2. 系统分析 helped/hurt/unchanged、top1/topk、malignant recall、错例分布、skill 分布、fusion reason。
3. 医生用自然语言写入本地临床经验。
4. 系统生成一个候选 workflow proposal。
5. 系统也可以生成候选 skill refinement / new skill / composite skill proposal。
6. 人工审核。
7. 一键批准。
8. 显式打开运行时开关后启用 workflow；新 skill 则必须从 pending 区人工批准后才进入正式 `skills/` registry。

## 不推荐的表述

不要说：

- 系统会自动改线上诊断策略。
- 系统会无审核地学习 test split。
- 系统会自动部署新 skill。
- 每个 disease-specific tweak 都是临时手写规则。

推荐说：

- DermAgent 支持离线 workflow evolution。
- 系统会根据新模型、新数据集上的运行结果反思错误模式。
- 系统会结合医生输入的真实经验，生成可审核的 workflow 候选。
- 候选 workflow 包括 skill 使用、workflow profile、label space、retrieval、specialist、conservative fusion 松紧和病种 family refinement。
- 候选默认不启用，需人工审核后一键批准，并显式设置运行时开关。

## 医生如何输入经验

医生不需要改代码，只需要填写：

```text
workflow_evolution/doctor_experience_template.md
```

可以写自然语言规则，例如：

```text
- 如果口角红斑伴脱屑，优先考虑 angular cheilitis，而不是普通 contact dermatitis。
- 如果甲板绿色变色，优先考虑感染或 green nail，不要按普通湿疹处理。
- 如果病灶位于口腔黏膜，BCC 类改判要更保守。
- 如果图像为 ISIC 风格色素痣，只有颜色不均但结构对称时，不要轻易从 nevus 改 melanoma。
- 如果皮损为多发毛囊性丘疹并位于头皮或项部，应提高 acne/folliculitis family 权重。
```

医生也可以写安全约束：

```text
- 恶性风险相关病例不能因为单个 benign cue 就降级。
- 如果存在溃疡、快速增大、多色不规则边界，保持 malignant guard。
- 如果病种 family 改判会降低 malignant recall，需要标注风险并二次审核。
```

系统会把医生经验作为 proposal 的一个输入，而不是直接变成运行时代码。

## 系统会分析什么

生成 proposal 时，系统会读取 compare report，统计：

- baseline top1
- agent top1
- baseline topk
- agent topk
- malignant recall
- helped case
- hurt case
- unchanged case
- malformed final
- timeout
- empty final
- workflow distribution
- workflow cell distribution
- label space distribution
- selected skill distribution
- fusion reason distribution
- baseline 错误方向，例如 `MEL -> NV`
- agent 错误方向，例如 `DERMATITIS_ECZEMA -> MUCOSAL_GENITAL_ORAL`

然后生成候选调整项：

- 是否需要固定新的 `model x dataset` cell。
- 是否需要强制 grouped label space。
- 是否需要替换或继承 `workflow_profile`。
- 是否需要增加 `workflow_capabilities`。
- 是否需要禁用某些重 skill。
- 是否需要启用或禁用 retrieval。
- 是否需要关闭 legacy final path。
- 是否需要更强或更弱的 conservative fusion。
- 是否需要增加窄的病种 family override。
- 是否需要从医生经验生成新 skill proposal。
- 是否需要根据错误经验演变现有 skill。
- 是否需要把重复出现的多 skill 序列合并成一个复合 skill proposal。

## Skill 自进化、创建和合并机制

当前机制里有“自动进化创建新 skill，但必须人工审核才能上线”的隔离流程。它和 workflow evolution 是并行关系：

- Workflow proposal 调整 model x dataset 的运行旋钮。
- Skill proposal 调整或新增可被 planner 选择的技能。
- 二者都默认离线生成、默认 pending、默认不进入运行时。

新 skill 不会直接写入正式技能池。自动生成的 skill 草稿先放在：

```text
skills/pending/
```

正式运行只读取 `skills/` 和 `skills/registry.py` 中注册过的 skill。`skills/pending/` 里的文件只是审核草稿，不会自动上线。

### Skill proposal 的来源

Skill 自进化可以综合三类输入：

1. 错误经验：hard case、helped/hurt/unchanged、skill helpfulness、malformed/timeout/empty final、常见 failure mode。
2. 医生经验：`workflow_evolution/doctor_experience_template.md` 里写的本地临床规则、安全约束和特定病种经验。
3. Skill 演变种子：已有 abstract experience 里的 rule、confusion memory、composite skill seed。

系统会生成以下类型的候选：

- `workflow_update`：改现有 skill 的步骤顺序或推理流程。
- `trigger_update`：改 skill 触发条件，避免该用时没选中。
- `watchout_update`：加入医生式注意事项和反例约束。
- `output_schema_update`：改输出结构，让证据更可审计。
- `split_skill`：把过宽 specialist 拆成更窄 skill。
- `merge_skill`：把反复一起出现且有效的 skill 序列合并成复合 skill。
- new skill / composite skill：从 confusion 或 composite seed 生成一个全新的 pending skill 草稿。

### 生成 skill refinement 候选

先从运行记录生成 hard case 和 skill helpfulness 证据，然后运行：

```bash
python scripts/generate_skill_refinement_candidates.py \
  --records-root outputs/<run> \
  --dataset <dataset_name> \
  --output-dir outputs/skill_refinement_candidates/<run_name>
```

生成结果包括：

```text
outputs/skill_refinement_candidates/<run_name>/skill_refinement_candidates.jsonl
outputs/skill_refinement_candidates/<run_name>/summary.json
```

这些候选只说明“应该怎么改 skill”，不会自动改代码。

### 生成复合 skill proposal

如果错误经验和成功经验里反复出现稳定的多 skill 序列，可以生成复合 skill proposal：

```bash
python scripts/generate_composite_skill_proposals.py \
  --batch-critique-path outputs/<run>/batch_critique.json \
  --refinement-candidates-path outputs/skill_refinement_candidates/<run_name>/skill_refinement_candidates.jsonl \
  --output-dir proposals/composite_skills
```

生成结果会放在：

```text
proposals/composite_skills/<proposal_id>.json
proposals/composite_skills/composite_skill_proposals.jsonl
proposals/composite_skills/summary.json
```

proposal 默认：

- `review_status = pending_review`
- 不进入 `skills/`
- 不 patch registry
- 不影响正式运行

### 从 proposal 生成 pending skill 草稿

如果人工认为某个 composite/confusion proposal 值得变成代码草稿，运行：

```bash
python scripts/generate_skill_from_proposal.py \
  --proposal-id <proposal_id>
```

生成的文件会进入：

```text
skills/pending/<skill_id>.py
skills/pending/<skill_id>.meta.json
```

这一步仍然不启用 skill。审核人员需要检查：

- skill 名称和触发条件是否清楚。
- 是否只组织证据，不替代 final diagnosis。
- 是否存在 label leakage 或 test split 经验污染。
- 是否会降低 malignant recall。
- 是否有足够 supporting cases 和 counter cases。
- 输出 schema 是否适合 Excel/论文审计和后续调试。

### 一键批准 pending skill

人工审核通过后，才运行：

```bash
python scripts/approve_skill.py \
  --proposal-id <proposal_id>
```

批准动作会做三件事：

1. 把 `skills/pending/<skill_id>.py` 复制到 `skills/<skill_id>.py`。
2. 修改 `skills/registry.py`，把新 skill 注册进默认 registry。
3. 把 proposal 的 `review_status` 标记为 `approved`，并删除 pending 草稿文件。

从这一步开始，新 skill 才可能被 planner/registry 看到。没有人工批准时，新 skill 永远停留在 pending 文件夹，不会上线。

### Skill 上线后的验证要求

新 skill 或复合 skill 批准后，还不能直接声称有效，必须重新跑 frozen evaluation：

- smoke 3-8 case，确认不会报错或污染输出。
- 8 卡 40-case frozen compare，记录 top1/topk/malignant recall/helped/hurt/unchanged。
- 如果 top1 不超过 direct baseline，不能作为调好结论。
- 如果 top1 赢但 malignant recall 明显下降，必须标风险并二次审核。
- test split 不开 writeback。

## 生成候选 Workflow

先跑一个 frozen compare，得到 report JSON。然后运行：

```bash
python workflow_evolution/generate_proposal.py \
  --reports 'outputs/<run>/shard_*/compare_agent_vs_qwen_*.json' \
  --model Hulu-Med-7B \
  --dataset isic2019 \
  --doctor-experience workflow_evolution/doctor_experience_template.md
```

生成结果会放在：

```text
proposals/workflow_evolution/<proposal_id>/workflow_evolution_proposal.json
proposals/workflow_evolution/<proposal_id>/README.md
```

注意：

- 这里生成的只是 proposal。
- proposal 的 `review_status` 默认是 `pending_review`。
- proposal 的 `default_enabled` 默认是 `false`。
- proposal 不会自动进入运行。

## 一键批准候选 Workflow

人工审核 proposal 后，如果决定启用，运行：

```bash
python workflow_evolution/apply_proposal.py \
  --proposal proposals/workflow_evolution/<proposal_id>/workflow_evolution_proposal.json \
  --approve \
  --reviewer doctor_or_engineer_name
```

批准后会复制到：

```text
state/workflow_evolution/approved/<proposal_id>.json
```

此时 proposal 已经是 approved，但仍然不会默认影响运行。

## 真正启用运行时自进化 Workflow

运行 compare 或正式推理时，必须显式加：

```bash
DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1 \
python scripts/compare_agent_vs_qwen.py ...
```

没有这个环境变量时：

- `state/workflow_evolution/approved/` 里的文件会被忽略。
- 当前手写固定 workflow 不受影响。
- 默认行为完全保持原样。

## 运行时读取优先级

当前整体 workflow 优先级仍然是：

1. approved workflow evolution proposal，且 `DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1`
2. `model x dataset` 专属 cell override
3. model overlay
4. dataset workflow
5. default workflow

这意味着自进化 workflow 是一个显式启用的最高优先级候选层。

## Proposal 里会包含什么

一个 proposal 会包含：

- `proposal_id`
- `model_name`
- `dataset_name`
- `review_status`
- `activation`
- `source_evidence`
- `doctor_experience`
- `proposed_workflow_cell`
- `proposed_skill_adjustments`
- `proposed_conservative_fusion_adjustments`
- `proposed_runtime_refinements`
- `safety_gates`

其中 `proposed_workflow_cell` 会长得像手工调好的 cell：

```json
{
  "workflow_cell_id": "hulumed_7b__isic2019__evolved_candidate_v1",
  "label_space_id": "isic2019_full",
  "workflow_profile": "hulumed_7b_isic2019_archive_guard_workflow",
  "workflow_capabilities": [
    "baseline_anchored_final",
    "melanocytic_guard_reasoning"
  ],
  "inherit_dataset_workflow": true,
  "force_conservative_fusion": true,
  "disable_legacy_final_path": true
}
```

这和当前逐条调 workflow 时实际修改的内容一致，只是从人工直接改代码，变成系统先生成候选、人工审核后再启用。

如果 proposal 同时包含 skill evolution 信息，还会引用：

- `proposed_skill_adjustments`
- `skill_refinement_candidate_ids`
- `composite_skill_proposal_ids`
- `pending_skill_artifacts`
- `skill_review_status`

这些字段只作为审核线索；workflow proposal 被 approve 不等于 pending skill 自动上线。skill 仍需单独走 `generate_skill_from_proposal.py` 和 `approve_skill.py`。

## 和逐条调参的对应关系

手工逐条调 workflow 时，主要改过这些旋钮：

- `workflow_cell_id`
- `label_space_id`
- `workflow_profile`
- `workflow_profile_mode`
- `workflow_capabilities`
- `replace_workflow_capabilities`
- `inherit_dataset_workflow`
- `allowed_skills`
- `force_enable_skills`
- `force_disable_skills`
- `skip_specialist_skills`
- `disabled_specialist_skills`
- `skip_experience_retrieval`
- `enable_experience_retrieval`
- `enable_skill_retrieval`
- `force_conservative_fusion`
- `conservative_fusion_weight`
- `fallback_on_malformed_final`
- `disable_legacy_final_path`
- disease-family specific override guard
- runtime grouped-family refinement

自进化 proposal 的目标就是根据错例和医生经验，生成这些同类型的候选改动。

## 审核标准

批准一个 proposal 前，至少检查：

- 是否使用 frozen evaluation。
- 是否没有打开 test writeback。
- 是否没有污染 test split。
- 同批 case 上 agent top1 是否超过 direct baseline。
- topk 是否至少不明显下降。
- malignant recall 是否没有明显下降。
- helped/hurt/unchanged 是否可接受。
- workflow distribution 是否符合预期。
- label space 是否正确。
- malformed/timeout/empty final 是否可接受。
- 新增或放松的 override 是否有明确 evidence 条件。
- 医生经验是否只影响对应 workflow，而不是全局污染。
- new skill 是否仍在 `skills/pending/`，没有绕过人工审核直接进入 `skills/`。
- composite skill 是否只是组织证据，不接管 final diagnosis。
- registry patch 是否只在 `approve_skill.py` 人工批准后发生。

## 推荐对外口径

可以这样描述：

> DermAgent 支持面向新医院和新模型的离线 workflow evolution。系统在本地 frozen evaluation 上分析错例、skill 使用、fusion 决策和病种 family 混淆，再结合医生输入的真实临床经验，生成新的 model×dataset workflow 候选。候选 workflow 包括 skill selection、workflow profile、label space、conservative fusion 强度和病种 family refinement。所有候选默认隔离保存，必须经过人工审核后一键批准，并显式打开运行时开关才会生效。

更短一点：

> DermAgent can reflect on local deployment errors and physician feedback to generate auditable workflow-evolution proposals. Proposals are disabled by default and require human approval before activation.

## 当前状态

当前实现已经具备：

- 独立文件夹：`workflow_evolution/`
- 医生经验模板：`workflow_evolution/doctor_experience_template.md`
- proposal 生成入口：`workflow_evolution/generate_proposal.py`
- proposal 批准入口：`workflow_evolution/apply_proposal.py`
- 运行时开关：`DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1`
- approved proposal 目录：`state/workflow_evolution/approved/`
- 默认不启用的安全机制
- skill refinement 候选入口：`scripts/generate_skill_refinement_candidates.py`
- composite skill proposal 入口：`scripts/generate_composite_skill_proposals.py`
- pending skill 生成入口：`scripts/generate_skill_from_proposal.py`
- pending skill 人工批准入口：`scripts/approve_skill.py`
- pending skill 隔离目录：`skills/pending/`

后续还可以继续增强：

- 把医生经验拆成结构化 rule。
- 对 proposal 自动跑 smoke/40/80 case 验证。
- 生成一页医生可读的审核报告。
- 给每个 proposal 生成风险等级。
