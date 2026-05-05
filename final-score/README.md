# Final Score Directory

This directory centralizes the outputs of the final DermAgent experiments.

## Intended contents

- `final_runs/`: all run-time experiment outputs
- `paper_exports/`: table and figure exports used for manuscript writing

## Standard structure

- `final_runs/main_qwen_vs_agent_qwen/`: internal main comparison for Direct Qwen vs Agent+Qwen
- `final_runs/main_medgemma_vs_agent_medgemma/`: internal main comparison for Direct MedGemma vs Agent+MedGemma
- `final_runs/ablations_qwen/`: Qwen-agent ablation outputs
- `final_runs/external_qwen/`: external-dataset Direct Qwen vs Agent+Qwen outputs
- `final_runs/paired_stats/`: paired statistical summaries and CSV tables
- `final_runs/qual_case_study/`: copied qualitative case bundles grouped by bucket
- `paper_exports/tables/`: manuscript-ready tables
- `paper_exports/figures/`: manuscript-ready figure data and exports

## Source of truth

All wrappers under `final-script/` now write their outputs here by default.

## Suggested usage

- Run experiments via `final-script/runs/*.sh`
- Read final numbers and case bundles from this directory only during paper writing

## Storage split

- `final_runs/` is treated as a large-results store and is excluded from the main GitHub repo.
- Keep the project source, scripts, and small reproducibility assets in GitHub.
- Sync `final_runs/` to a private Hugging Face dataset repo with `scripts/upload_final_runs_to_hf.sh`.
