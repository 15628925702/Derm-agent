from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.paper_exports import export_compare_case_data
from agent.policy_evaluation import build_policy_summary, compare_policy_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a partial compare run with one or more shard compare reports into a final compare report."
    )
    parser.add_argument("--partial-run-root", type=Path, required=True, help="Original compare_agent_vs_qwen_* run root.")
    parser.add_argument(
        "--shard-report",
        type=Path,
        action="append",
        default=[],
        help="Completed shard compare report JSON. Can be provided multiple times.",
    )
    parser.add_argument("--output-report", type=Path, required=True, help="Final merged compare report JSON path.")
    parser.add_argument("--paper-case-data-dir", type=Path, default=None, help="Optional export dir for merged case-level files.")
    parser.add_argument(
        "--ignore-partial-records",
        action="store_true",
        help="Ignore any baseline/agent records already present in the partial run root and rebuild entirely from shard artifacts.",
    )
    parser.add_argument(
        "--export-stem",
        type=str,
        default="case_level_compare_export_merged",
        help="Export stem when --paper-case-data-dir is provided.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_partial_context(partial_run_root: Path) -> dict[str, Any]:
    evaluation_manifest = _read_json(partial_run_root / "evaluation_manifest.json")
    result_manifest_path = partial_run_root / "result_manifest.json"
    if result_manifest_path.exists():
        result_manifest = _read_json(result_manifest_path)
        baseline_target = next(
            item for item in result_manifest["target_results"] if item["target"]["target_id"] == "direct_baseline"
        )
        agent_target = next(
            item for item in result_manifest["target_results"] if item["target"]["target_id"] == "full_dermagent"
        )
        baseline_records = _read_jsonl(Path(baseline_target["artifacts"]["records_jsonl_path"]))
        partial_agent_records = _read_jsonl(Path(agent_target["artifacts"]["records_jsonl_path"]))
    else:
        baseline_records_path = partial_run_root / "targets" / "direct_baseline" / "records" / "case_execution_records.jsonl"
        baseline_summary_path = partial_run_root / "targets" / "direct_baseline" / "summary.json"
        agent_records_path = partial_run_root / "targets" / "full_dermagent" / "records" / "case_execution_records.jsonl"
        agent_summary_path = partial_run_root / "targets" / "full_dermagent" / "summary.json"
        baseline_records = _read_jsonl(baseline_records_path)
        partial_agent_records = _read_jsonl(agent_records_path)
        result_manifest = {
            "eval_id": evaluation_manifest.get("eval_id", partial_run_root.name),
            "protocol_version": evaluation_manifest.get("protocol_version", ""),
            "contamination_check": evaluation_manifest.get("contamination_check", {}),
            "target_results": [
                {
                    "target": {
                        "target_id": "direct_baseline",
                        "label": "Direct Baseline",
                        "target_type": "baseline",
                        "mode": "baseline",
                    },
                    "artifacts": {
                        "records_jsonl_path": str(baseline_records_path),
                        "summary_path": str(baseline_summary_path),
                    },
                },
                {
                    "target": {
                        "target_id": "full_dermagent",
                        "label": "Full DermAgent",
                        "target_type": "agent",
                        "mode": "agent",
                    },
                    "artifacts": {
                        "records_jsonl_path": str(agent_records_path),
                        "summary_path": str(agent_summary_path),
                    },
                },
            ],
        }
        baseline_target = result_manifest["target_results"][0]
        agent_target = result_manifest["target_results"][1]
    return {
        "evaluation_manifest": evaluation_manifest,
        "result_manifest": result_manifest,
        "baseline_target": baseline_target,
        "agent_target": agent_target,
        "baseline_records": baseline_records,
        "partial_agent_records": partial_agent_records,
    }


def _collect_agent_records(partial_records: list[dict[str, Any]], shard_reports: list[Path]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in partial_records:
        case_id = str(record.get("case_id", "")).strip()
        if case_id:
            merged[case_id] = record
    for shard_report_path in shard_reports:
        shard_payload = _read_json(shard_report_path)
        for record in shard_payload.get("cases", []) or []:
            case_id = str(record.get("case_id", "")).strip()
            if case_id:
                merged[case_id] = record
    return list(merged.values())


def _collect_baseline_records(partial_records: list[dict[str, Any]], shard_reports: list[Path]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in partial_records:
        case_id = str(record.get("case_id", "")).strip()
        if case_id:
            merged[case_id] = record
    for shard_report_path in shard_reports:
        shard_result_manifest = _read_json(shard_report_path.parent / shard_report_path.stem / "result_manifest.json")
        baseline_target = next(
            item for item in shard_result_manifest["target_results"] if item["target"]["target_id"] == "direct_baseline"
        )
        for record in _read_jsonl(Path(baseline_target["artifacts"]["records_jsonl_path"])):
            case_id = str(record.get("case_id", "")).strip()
            if case_id:
                merged[case_id] = record
    return list(merged.values())


def _order_records(case_ids: list[str], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_id = {str(record.get("case_id", "")).strip(): record for record in records}
    ordered: list[dict[str, Any]] = []
    for case_id in case_ids:
        record = by_case_id.get(case_id)
        if record is None:
            raise ValueError(f"Missing merged agent record for case_id={case_id}")
        ordered.append(record)
    return ordered


def main() -> int:
    args = parse_args()
    partial = _load_partial_context(args.partial_run_root)
    case_ids = list(partial["evaluation_manifest"]["dataset"]["case_ids"])
    baseline_seed_records = [] if args.ignore_partial_records else partial["baseline_records"]
    agent_seed_records = [] if args.ignore_partial_records else partial["partial_agent_records"]
    baseline_records = _collect_baseline_records(baseline_seed_records, list(args.shard_report))
    agent_records = _collect_agent_records(agent_seed_records, list(args.shard_report))
    ordered_baseline_records = _order_records(case_ids, baseline_records)
    ordered_agent_records = _order_records(case_ids, agent_records)

    baseline_summary = build_policy_summary(ordered_baseline_records)
    agent_summary = build_policy_summary(ordered_agent_records)
    comparisons = {
        "direct_baseline": {
            "vs_baseline": compare_policy_summaries(baseline_summary, baseline_summary),
        },
        "full_dermagent": {
            "vs_baseline": compare_policy_summaries(baseline_summary, agent_summary),
        },
    }

    baseline_target = dict(partial["baseline_target"])
    agent_target = dict(partial["agent_target"])
    baseline_target["summary"] = baseline_summary
    agent_target["summary"] = agent_summary

    result_manifest = dict(partial["result_manifest"])
    result_manifest["target_results"] = [baseline_target, agent_target]
    result_manifest["comparisons"] = comparisons
    _write_json(args.partial_run_root / "result_manifest.json", result_manifest)

    baseline_records_path = Path(baseline_target["artifacts"]["records_jsonl_path"])
    agent_records_path = Path(agent_target["artifacts"]["records_jsonl_path"])
    _write_jsonl(baseline_records_path, ordered_baseline_records)
    _write_jsonl(agent_records_path, ordered_agent_records)

    agent_summary_path = Path(agent_target["artifacts"]["summary_path"])
    baseline_summary_path = Path(baseline_target["artifacts"]["summary_path"])
    _write_json(baseline_summary_path, baseline_summary)
    _write_json(agent_summary_path, agent_summary)

    eval_manifest = partial["evaluation_manifest"]
    model_context = dict(eval_manifest.get("model_context", {}) or {})
    dataset_block = dict(eval_manifest.get("dataset", {}) or {})
    experiment_state = dict(eval_manifest.get("experiment_state", {}) or {})
    source_shards = [str(path) for path in args.shard_report]

    report = {
        "run_config": {
            "data_root": dataset_block.get("data_root", ""),
            "limit": len(case_ids),
            "seed": dataset_block.get("seed", 0),
            "case_offset": dataset_block.get("case_offset", 0),
            "base_url": model_context.get("agent_client", {}).get("base_url", ""),
            "model": model_context.get("agent_client", {}).get("model", ""),
            "agent_base_url": model_context.get("agent_client", {}).get("base_url", ""),
            "agent_model": model_context.get("agent_client", {}).get("model", ""),
            "baseline_base_url": model_context.get("baseline_client", {}).get("base_url", ""),
            "baseline_model": model_context.get("baseline_client", {}).get("model", ""),
            "policy_id": experiment_state.get("stable_policy_id", ""),
            "policy_source_path": experiment_state.get("stable_policy_path", ""),
            "policy_label": "",
            "data_split": dataset_block.get("data_split", ""),
            "split_json": dataset_block.get("split_json_path", ""),
            "phase": "full",
            "strict_frozen_eval": bool(result_manifest.get("contamination_check", {}).get("strict_frozen_eval", True)),
            "evaluation_protocol_version": result_manifest.get("protocol_version", ""),
            "agent_model_name": model_context.get("agent_client", {}).get("model", ""),
            "inferred_dataset_name": dataset_block.get("dataset_name", ""),
            "disable_model_workflow_routing": False,
            "model_workflow_overrides": {},
            "agent_execution_overrides": {},
            "merge_source_shards": source_shards,
        },
        "summary": {
            "baseline": baseline_summary,
            "agent": agent_summary,
            "agent_vs_baseline": comparisons["full_dermagent"]["vs_baseline"],
        },
        "artifacts": {
            "evaluation_manifest_path": str(args.partial_run_root / "evaluation_manifest.json"),
            "result_manifest_path": str(args.partial_run_root / "result_manifest.json"),
            "run_root": str(args.partial_run_root),
            "merge_source_shards": source_shards,
        },
        "cases": ordered_agent_records,
    }
    _write_json(args.output_report, report)

    if args.paper_case_data_dir is not None:
        export_compare_case_data(
            compare_report_paths=[args.output_report],
            output_dir=args.paper_case_data_dir,
            export_stem=args.export_stem,
        )

    print(
        json.dumps(
            {
                "merged_report_path": str(args.output_report),
                "baseline_cases": len(baseline_records),
                "agent_cases": len(ordered_agent_records),
                "source_shards": source_shards,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
