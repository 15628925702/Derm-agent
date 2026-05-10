from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from agent.paper_exports import export_compare_case_data
from agent.policy_evaluation import build_policy_summary, compare_policy_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge final 30/70 compare shard reports for one model/dataset.")
    parser.add_argument("--compare-report", type=Path, action="append", default=[])
    parser.add_argument("--compare-report-glob", type=str, action="append", default=[])
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--paper-case-data-dir", type=Path, default=None)
    parser.add_argument("--export-stem", type=str, default="case_level_compare_export_merged")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    reports = list(args.compare_report)
    for pattern in args.compare_report_glob:
        reports.extend(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    reports = sorted(set(path for path in reports if path.is_file()))
    if not reports:
        raise ValueError("No compare reports were provided.")

    cases: list[dict[str, Any]] = []
    source_reports: list[str] = []
    first_payload: dict[str, Any] | None = None
    for report in reports:
        payload = read_json(report)
        if first_payload is None:
            first_payload = payload
        source_reports.append(str(report))
        cases.extend(payload.get("cases", []) or [])

    cases.sort(key=lambda item: (int((item.get("evaluation_context") or {}).get("case_offset", 0) or 0), str(item.get("case_id", ""))))
    baseline_records = []
    for case in cases:
        baseline = dict(case)
        baseline["qwen_final"] = case.get("baseline_qwen", {})
        baseline["evaluation"] = {
            "correct": (case.get("evaluation") or {}).get("baseline_correct"),
            "topk_hit": (case.get("evaluation") or {}).get("baseline_topk_hit"),
            "malignant_recall_hit": (case.get("evaluation") or {}).get("baseline_malignant_recall_hit"),
        }
        baseline_records.append(baseline)

    dataset_name = str((first_payload or {}).get("run_config", {}).get("inferred_dataset_name", "")).strip() or None
    baseline_summary = build_policy_summary(baseline_records, dataset_name=dataset_name)
    agent_summary = build_policy_summary(cases, dataset_name=dataset_name)
    comparison = compare_policy_summaries(baseline_summary, agent_summary, dataset_name=dataset_name)
    report = {
        "run_config": dict((first_payload or {}).get("run_config", {}) or {}),
        "summary": {
            "baseline": baseline_summary,
            "agent": agent_summary,
            "agent_vs_baseline": comparison,
        },
        "artifacts": {
            "merge_source_reports": source_reports,
        },
        "cases": cases,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.paper_case_data_dir is not None:
        export_compare_case_data(
            compare_report_paths=[args.output_report],
            output_dir=args.paper_case_data_dir,
            export_stem=args.export_stem,
        )
    print(json.dumps({"output_report": str(args.output_report), "cases": len(cases)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
