# Fusion Experience Memory

`memory/fusion_experience/` stores workflow-specific conservative fusion
experience. These rules are learned from model x dataset confusion patterns, so
they live with accumulated memory instead of the callable skill registry.

Runtime rules are implemented in `workflow_fusion_decision.py`. The legacy
`skills/workflow_fusion_decision.py` path remains as a compatibility wrapper.

## Accumulation Switch

Fusion experience accumulation is disabled by default. The runtime will not
write proposal observations unless this exact environment variable is set:

```bash
DERMAGENT_ENABLE_FUSION_EXPERIENCE_ACCUMULATION=1
```

When enabled, observations are appended to:

```text
memory/fusion_experience/proposals/pending_fusion_experience.jsonl
```

or to `DERMAGENT_FUSION_EXPERIENCE_PROPOSAL_PATH` if that path is explicitly
provided.

Pending observations have no runtime effect. A human reviewer must turn them
into a tested, `workflow_cell_id`-bound rule before they can affect inference.
