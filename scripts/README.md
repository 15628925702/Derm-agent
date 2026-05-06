# Scripts Layout

`scripts/` remains the active command surface for DermAgent. It is intentionally kept at the repository root because many runbooks and wrapper scripts call paths such as `scripts/compare_agent_vs_qwen.py`.

## Main current entry points

- `compare_agent_vs_qwen.py`: frozen baseline-vs-agent evaluation runner.
- `run_qwen_final_round_asset_rerun.sh`: qwen final-round frozen-asset rerun helper.
- `run_6x6_matrix_parallel.sh`: 6 model x 6 dataset matrix runner.
- `summarize_6x6_matrix.py`: matrix result summarizer.
- `start_*_server.sh`: model server launchers.
- `switch_model_server.sh`: model server switch helper.

## Legacy and specialized scripts

Older stage-2, medium-run, metadata, export, and training scripts are kept here for reproducibility. Prefer adding new workflow-tuning commands as small wrappers around `compare_agent_vs_qwen.py` or `run_6x6_matrix_parallel.sh`, rather than creating a parallel scripts tree.
