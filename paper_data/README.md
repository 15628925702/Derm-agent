# Paper Data Workspace

This directory is reserved for paper-facing exports that should be easy to inspect in Excel or audit case by case.

- `case_level_exports/`: per-case compare exports generated from `compare_agent_vs_qwen.py` reports.
  These exports are paper-facing audit tables and intentionally omit internal workflow/profile/routing fields.
- `physician_evidence_package_examples/`: real doctor-readable evidence-package examples for paper supplements.
  Do not place fallback/template examples here; examples must come from actual runs with the separate Qwen physician-summary service.
  Current examples include both `brief` and `detailed` outputs from the same real HAM10000 cases.

Generated CSV/JSON/JSONL/XLSX files are ignored by Git so local paper tables can be refreshed without bloating the repository.

## Export Excel During Compare Runs

Run experiments in the `dermagent-6x6` environment and add `--export-paper-case-data` to `scripts/compare_agent_vs_qwen.py`:

```bash
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/compare_agent_vs_qwen.py \
  ...existing compare arguments... \
  --export-paper-case-data \
  --paper-case-data-dir paper_data/case_level_exports/<run_name>
```

The same switch can be enabled for existing launch scripts with:

```bash
DERMAGENT_EXPORT_PAPER_CASE_DATA=1
```

Each compare run writes a manifest plus case-level `csv`, `json`, `jsonl`, and `xlsx` files. The Excel table is designed for paper audit/source-data use and excludes internal workflow, routing, policy, label-space, and fusion implementation fields.

## Export From Existing Reports

To backfill Excel files from already-finished compare reports:

```bash
/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/export_paper_case_data.py \
  --compare-report-glob 'outputs/<run_dir>/**/compare_agent_vs_qwen_*.json' \
  --output-dir paper_data/case_level_exports/<run_name> \
  --export-stem <run_name>_case_level
```

Use one `--compare-report-glob` per model/dataset group when combining multiple shards or cells into one workbook.

## Physician Evidence Package Examples

Start the separate Qwen summary service before generating examples:

```bash
cd /data/gh/DermAgent
bash scripts/start_qwen_physician_summary_server.sh
```

Then run compare or direct `run_agent` calls with `--enable-physician-evidence-summary`. Use `--physician-evidence-summary-detail brief` for compact examples, or `--physician-evidence-summary-detail detailed` for the fuller doctor-facing package with `structured_evidence_appendix`.

Valid examples should have `physician_evidence_summary.status: "ok"`. Empty or failed summaries are useful for engineering audits but should not be copied into the paper example folder as doctor-readable examples.
