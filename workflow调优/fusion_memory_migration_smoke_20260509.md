# Fusion Memory Migration Smoke Validation - 2026-05-09

## Scope

- Branch: `merge-dermatollama-qwen-remaining-workflow-tuning`
- Migration commit: `9a7b8080 Move fusion rules into memory experience`
- Main implementation path: `memory/fusion_experience/workflow_fusion_decision.py`
- Compatibility wrapper: `skills/workflow_fusion_decision.py`
- Accumulation switch: `DERMAGENT_ENABLE_FUSION_EXPERIENCE_ACCUMULATION`
- Default accumulation state: off. The smoke run did not enable the switch.

## Validation Run

Smoke output root:

`paper_data/workflow_tuning_runs_20260509/merge_memory_migration_smoke8_8gpu_r3/`

The run used 8 GPUs with 8 shards, one case per shard. This is a migration equivalence smoke check, not a new 20-case or 300-case tuning result.

| workflow | cases | baseline Top-1 | baseline Top-k | agent Top-1 | agent Top-k | comparison with pre-migration live20 same cases |
|---|---:|---:|---:|---:|---:|---|
| `dermatollama / isic2019` | 2 | `1/2 = 50.0%` | `2/2 = 100.0%` | `1/2 = 50.0%` | `2/2 = 100.0%` | eval flags and final diagnosis matched |
| `dermatollama / ham10000` | 2 | `0/2 = 0.0%` | `0/2 = 0.0%` | `0/2 = 0.0%` | `0/2 = 0.0%` | eval flags and final diagnosis matched |
| `qwen / scin` | 2 | `0/2 = 0.0%` | `0/2 = 0.0%` | `0/2 = 0.0%` | `0/2 = 0.0%` | eval flags and final diagnosis matched |
| `dermatollama / sd198` | 2 | `1/2 = 50.0%` | `1/2 = 50.0%` | `1/2 = 50.0%` | `1/2 = 50.0%` | eval flags and final diagnosis matched |

## Equivalence Check

Compared against the same case ids from the pre-migration merge live20 runs:

- Final diagnosis matched: `16/16`
- Evaluation flags matched: `16/16`
- Observed Top-1 / Top-k regression on this smoke set: none

This smoke set did not demonstrate new tuning gains; it only confirms that moving fusion rules from `skills` into `memory/fusion_experience` preserved runtime behavior on the checked cases.

## Artifact Policy

Large smoke output directories remain untracked local artifacts and are not committed:

- `paper_data/workflow_tuning_runs_20260509/merge_memory_migration_smoke8_8gpu_r3/`
- earlier failed/intermediate smoke directories under `paper_data/workflow_tuning_runs_20260509/merge_memory_migration_smoke16_8gpu_r2/`

