# DermAgent Trainable Components Boundary (Stage 4 Step 1)

## Purpose
This document defines what can be optimized in DermAgent and what must stay frozen during experiments.

The boundary follows these invariants:
- Qwen remains the only final diagnosis maker.
- No Qwen weight update, no end-to-end training with Qwen as a trainable module.
- Skill bank semantics are fixed within a single experiment.
- Any learned policy must be auditable and rollback-safe.

## Frozen Components
| Component | Freeze Scope | Why Frozen | Enforcement |
| --- | --- | --- | --- |
| `qwen_backbone` | Model weights and final diagnosis role | Preserve "Qwen-only final diagnosis" and fair baseline | Do not run any training job against Qwen params; reject configs that set trainable=true for Qwen |
| `qwen_service_runtime` | Model name, serving endpoint, runtime settings in formal evaluation | Keep same-model fair comparison | Record in evaluation manifest and fail evaluation if changed inside one run |
| `skill_semantic_definitions` | Skill workflow text, output contract, clinical semantics during one experiment | Avoid moving target in ablation/comparison | Freeze by skill bank version snapshot in evaluation manifest |
| `clinical_reasoning_contract` | Evidence-only agent behavior and no final disease prediction in skills | Prevent agent collapse into classifier/voter | Validate output contracts and planner policy constraints |
| `dataset_case_list_in_eval` | Case list used by baseline/full/ablation in one run | Ensure reproducibility and fairness | Shared case ids in evaluation manifest |

## Trainable Components
### 1) `controller_planner_scorer`
- Scope: learn skill scoring/ranking weights for planner decisions.
- Input:
  - `CaseExecutionRecord.controller_training_example.state_features`
  - `available_skill_candidates`
  - cognition and retrieval summaries.
- Output:
  - per-skill score
  - selected skill ranking proposal
  - confidence.
- Training signal source:
  - execution record outcome (`final_correct`, `delta_vs_baseline`)
  - helpful/harmful skill annotations from reflection.
- Versioning:
  - candidate artifacts under `state/trainable_components/controller_planner_scorer/candidates`
  - stable pointer JSON in `state/trainable_components/controller_planner_scorer/stable.json`
  - promote only after frozen eval gate; rollback to stable pointer.

### 2) `retrieval_reranker`
- Scope: rerank raw/tactical/abstract retrieval outputs.
- Input:
  - retrieval query (`ddx`, morphology, metadata, uncertainty, confusion pair)
  - retrieved candidates with original heuristic scores.
- Output:
  - reranked ids and calibrated relevance score.
- Training signal source:
  - downstream case correctness delta
  - uncertainty reduction / contradiction detection improvements
  - skill helpfulness conditioned on retrieved evidence.
- Versioning:
  - `state/trainable_components/retrieval_reranker/{candidates,stable.json}`
  - version id bound to feature schema hash + training data snapshot.

### 3) `skill_selection_policy_optimizer`
- Scope: optimize planner policy parameters (`thresholds`, `bonuses`, `signal enable/disable`) without changing skill semantics.
- Input:
  - current policy config
  - execution summaries
  - hard case clusters.
- Output:
  - candidate policy delta patch.
- Training signal source:
  - policy evaluation metrics (`top1`, `topk`, `malignant_recall`, `error_rate`, confusion subsets).
- Versioning:
  - existing policy store (`state/policy/versions/*.json`)
  - gated promotion + rollback via policy evaluation records.

### 4) `evidence_calibrator`
- Scope: calibrate evidence strength/recommendation type aggregation weights (not diagnosis label generation).
- Input:
  - skill outputs (`evidence_strength`, `recommendation_type`, referenced experiences)
  - evidence bundle fields.
- Output:
  - calibrated evidence scores and uncertainty/risk weighting hints.
- Training signal source:
  - reflection outcome labels (`success/failure/partially_helpful`)
  - contradiction and uncertainty outcomes.
- Versioning:
  - `state/trainable_components/evidence_calibrator/{candidates,stable.json}`
  - include calibration schema version in artifact manifest.

### 5) `skill_helpfulness_predictor`
- Scope: predict whether a skill is likely helpful/partially_helpful/harmful under current case state.
- Input:
  - state features + selected skill + retrieval summary + cognition stats.
- Output:
  - class probabilities `{helpful, partially_helpful, harmful}`
  - expected utility score.
- Training signal source:
  - `skill_helpfulness_reports.jsonl`
  - reflection skill assessments in execution records.
- Versioning:
  - `state/trainable_components/skill_helpfulness_predictor/{candidates,stable.json}`
  - artifact includes label mapping + threshold config.

### 6) `hard_case_prioritization_scorer`
- Scope: prioritize hard cases for refinement/consolidation loops.
- Input:
  - hard case candidate fields (`failure_type`, contradiction count, uncertainty, confusion tags, repeated failures).
- Output:
  - priority score and review queue tier.
- Training signal source:
  - retrospective refinement yield (did candidate produce useful refinement/prototype/rule)
  - repeated failure persistence.
- Versioning:
  - `state/trainable_components/hard_case_prioritization_scorer/{candidates,stable.json}`
  - store scorer version with mining report manifest.

## Shared Constraints for All Trainable Components
- Must not emit final diagnosis labels.
- Must not bypass Qwen final diagnosis stage.
- Must support offline training and frozen evaluation replay.
- Must log:
  - training data snapshot refs
  - feature schema version
  - objective and metric definition
  - parent version
  - gate result
  - rollback target.

## Training Signal Sources (Canonical)
- `outputs/**/records/case_execution_records.jsonl`
- `outputs/hard_case_mining/hard_cases.jsonl`
- `outputs/skill_helpfulness/skill_helpfulness_reports.jsonl`
- `outputs/skill_refinement_candidates/skill_refinement_candidates.jsonl`
- `outputs/experience_consolidation/consolidated_abstract_experiences.jsonl`
- policy evaluation manifests under `state/policy/evaluations`.

## Version Management Contract
- Each trainable component must maintain:
  - `stable` version pointer
  - immutable `candidate` artifacts
  - `evaluation` records linked to frozen eval run ids
  - explicit rollback target.
- Promotion rule:
  - only promote candidate when frozen compare/ablation gates pass.
- Rollback rule:
  - if key metrics degrade against stable config, revert to previous stable immediately.
