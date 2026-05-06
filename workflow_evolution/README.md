# Workflow Evolution

`workflow_evolution/` is a disabled-by-default layer for generating and
reviewing model x dataset workflow candidates.

It is meant to describe the same process used during manual cell tuning:

1. Run a new model on a new hospital/dataset under frozen evaluation.
2. Analyze helped/hurt cases, common confusion pairs, selected skills, fusion
   reasons, malformed finals, workflow distribution, and label-space routing.
3. Optionally add physician experience in plain text.
4. Generate a candidate workflow proposal.
5. Human-review the proposal.
6. Approve it into `state/workflow_evolution/approved`.
7. Enable it at runtime only with `DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1`.

Default behavior is unchanged. Proposals under `proposals/workflow_evolution/`
are not imported by runtime. Approved proposals are also ignored unless the
runtime environment switch is explicitly set.

## Generate Candidate

```bash
python workflow_evolution/generate_proposal.py \
  --reports 'outputs/<run>/shard_*/compare_agent_vs_qwen_*.json' \
  --model Hulu-Med-7B \
  --dataset isic2019 \
  --doctor-experience workflow_evolution/doctor_experience_template.md
```

The output goes to:

```text
proposals/workflow_evolution/<proposal_id>/workflow_evolution_proposal.json
```

## Approve Candidate

```bash
python workflow_evolution/apply_proposal.py \
  --proposal proposals/workflow_evolution/<proposal_id>/workflow_evolution_proposal.json \
  --approve \
  --reviewer doctor_or_engineer_name
```

The approved file is copied to:

```text
state/workflow_evolution/approved/<proposal_id>.json
```

## Run With Approved Evolution

```bash
DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1 \
python scripts/compare_agent_vs_qwen.py ...
```

Without `DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1`, approved candidates remain
inactive.

## What The Proposal Can Adjust

- `workflow_profile`
- `workflow_capabilities`
- `label_space_id`
- `allowed_skills`
- `force_enable_skills`
- `force_disable_skills`
- specialist enable/disable choices
- retrieval enable/disable choices
- `force_conservative_fusion`
- `fallback_on_malformed_final`
- `disable_legacy_final_path`
- narrow disease-family override proposals
- candidate physician-experience rules

## Safety Position

This layer does not claim uncontrolled autonomous deployment. It is an offline
proposal generator plus a manual approval gate. That is the intended product
story: the agent reflects on local errors, proposes workflow evolution, accepts
physician knowledge, and produces an auditable workflow candidate that can be
enabled after review.
