# SCIN HuluMed Agent Run Interrupted 2026-05-14

## Stop Reason

The SCIN HuluMed+Agent full 70/30 run was stopped manually because the partial
metrics did not match the expected workflow gain, especially for malignant
recall.

## Run

- Run id: `scin_hulumed_agent_70train_30test_20260513T190318Z`
- Run root: `paper_data/final_30_70_large_runs/scin_hulumed_agent_70train_30test_20260513T190318Z/machine_0`
- Dataset: `scin`
- Model: `hulumed`
- Train/test split: final 70/30 split
- Bootstrap state: train experience was built and promoted to test state
- Physician evidence summary flag: enabled in the compare+doctor command, but
  the doctor stage had not started before interruption

## Completed Before Stop

- Bootstrap merge completed after supplemental shard 24 recovery.
- Bootstrap merge manifest reported:
  - raw case records: 2142
  - tactical records: 21832
  - abstract records: 919
  - shard count: 42
- One train bootstrap case remained failed: `case_index=161`.
- Compare completed: 16 of 29 shards.
- Completed compare shards: `shard_00000` through `shard_00015`.
- Completed compare cases: 512 of 918 test cases.

## Partial Metrics At Stop

HuluMed+Agent on completed 512 SCIN test cases:

- Top1: 151 / 512 = 29.49%
- Top3: 363 / 512 = 70.90%
- Malignant recall: 0 / 16 = 0.00%

Same first 512 test cases for direct baselines:

| Model | Top1 | Top3 | Malignant Recall |
| --- | ---: | ---: | ---: |
| dermatollama direct | 28.91% | 53.91% | 0 / 16 = 0.00% |
| hulumed direct | 27.34% | 72.07% | 0 / 16 = 0.00% |
| llama direct | 6.64% | 40.82% | 0 / 16 = 0.00% |
| medgemma direct | 42.97% | 71.29% | 0 / 16 = 0.00% |
| qwen direct | 41.41% | 68.95% | 0 / 16 = 0.00% |
| skinvl direct | 23.44% | 32.23% | 0 / 16 = 0.00% |

## Current Diagnosis

The workflow code is active, but the SCIN HuluMed fusion is too conservative.
Observed case exports include fusion notes such as:

- `hulumed_scin_*_grouped_promotion`
- `hulumed_scin_grouped_guarded_override`
- `malignancy_override_not_allowed`
- `hulumed_scin_baseline_anchor_guard`
- `fallback_to_baseline`

This means the current workflow can slightly improve HuluMed direct Top1, but it
does not produce the expected strong gains against qwen/medgemma direct and it
does not yet raise malignant recall.

## Resume Notes

The SCIN run was intentionally stopped. Do not resume it unchanged unless the
goal is only to finish the failed baseline run for record keeping.

If resuming after workflow fixes, use `--resume` and the same run id only if it
is acceptable to mix old and new compare shards. For clean evaluation of the new
workflow, start a fresh run id and reuse the completed train experience only if
the workflow change is inference/fusion-only.
