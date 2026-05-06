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
5. 人工审核。
6. 一键批准。
7. 显式打开运行时开关后启用。

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

后续还可以继续增强：

- 自动生成更细的新 skill 草稿。
- 把医生经验拆成结构化 rule。
- 对 proposal 自动跑 smoke/40/80 case 验证。
- 生成一页医生可读的审核报告。
- 给每个 proposal 生成风险等级。
