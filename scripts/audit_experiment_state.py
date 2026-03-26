from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.contamination_guard import find_run_agent_calls_without_explicit_writeback, is_frozen_mode, normalize_split_name
from agent.hard_case_miner import load_execution_records


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation_protocol"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "contamination_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit contamination/leakage risks across execution records, eval manifests, and scripts.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing execution records.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT, help="Root directory containing evaluation manifests.")
    parser.add_argument("--split-json", type=Path, default=None, help="Optional split definition JSON (train/val/test lists).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for audit report artifacts.")
    parser.add_argument("--fail-on-issues", action="store_true", help="Return non-zero exit code when any blocking issue is found.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    execution_records = load_execution_records(args.records_root)
    split_map = load_split_map(args.split_json) if args.split_json else {}
    record_checks = audit_execution_records(execution_records, split_map=split_map)
    eval_checks = audit_evaluation_manifests(args.eval_root)
    script_checks = audit_script_writeback_explicitness(PROJECT_ROOT / "scripts")

    issues = record_checks["issues"] + eval_checks["issues"] + script_checks["issues"]
    severity_counts: dict[str, int] = defaultdict(int)
    for issue in issues:
        severity_counts[str(issue.get("severity", "warning"))] += 1

    report = {
        "schema_version": "contamination_audit_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "records_root": str(args.records_root),
            "eval_root": str(args.eval_root),
            "split_json": str(args.split_json) if args.split_json else "",
        },
        "summary": {
            "execution_record_count": len(execution_records),
            "evaluation_manifest_count": eval_checks["evaluation_manifest_count"],
            "script_file_count": script_checks["script_file_count"],
            "issue_count": len(issues),
            "severity_counts": dict(severity_counts),
            "passed": not any(issue.get("severity") == "error" for issue in issues),
        },
        "checks": {
            "execution_records": record_checks,
            "evaluation_manifests": eval_checks,
            "scripts": script_checks,
        },
        "issues": issues,
    }
    output_path = args.output_dir / f"contamination_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "audit_report_path": str(output_path),
                "issue_count": report["summary"]["issue_count"],
                "severity_counts": report["summary"]["severity_counts"],
                "passed": report["summary"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.fail_on_issues and not report["summary"]["passed"]:
        return 1
    return 0


def load_split_map(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    split_map: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        values = payload.get(split_name)
        if values is None:
            values = payload.get(f"{split_name}_case_ids", [])
        if not isinstance(values, list):
            split_map[split_name] = set()
            continue
        split_map[split_name] = {str(item).strip() for item in values if str(item).strip()}
    return split_map


def audit_execution_records(records: list[dict[str, Any]], *, split_map: dict[str, set[str]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    split_cases: dict[str, set[str]] = {"train": set(), "val": set(), "test": set(), "unknown": set()}

    for record in records:
        case_id = str(record.get("case_id", "")).strip()
        state_versions = dict(record.get("state_versions", {}))
        run_mode = str(state_versions.get("run_mode", "")).strip().lower()
        data_split = normalize_split_name(str(state_versions.get("data_split", "")).strip(), default="unknown")
        writeback_ops = dict(record.get("writeback_ops", {}))
        writeback_enabled = bool(writeback_ops.get("writeback_enabled", False))
        split_cases.setdefault(data_split, set()).add(case_id)

        if not state_versions:
            issues.append(
                _issue(
                    "error",
                    "missing_state_versions",
                    f"Execution record `{case_id}` missing state_versions.",
                    {"case_id": case_id},
                )
            )
            continue
        for component in ("experience_state", "cognition_state", "policy_state"):
            comp_payload = dict(state_versions.get(component, {}))
            if not comp_payload:
                issues.append(
                    _issue(
                        "warning",
                        "missing_component_state_version",
                        f"Execution record `{case_id}` missing `{component}` split-aware metadata.",
                        {"case_id": case_id, "component": component},
                    )
                )
                continue
            if not str(comp_payload.get("split_name", "")).strip():
                issues.append(
                    _issue(
                        "warning",
                        "missing_component_split_name",
                        f"Execution record `{case_id}` component `{component}` missing split_name.",
                        {"case_id": case_id, "component": component},
                    )
                )
            if not str(comp_payload.get("split_aware_version", "")).strip():
                issues.append(
                    _issue(
                        "warning",
                        "missing_component_split_aware_version",
                        f"Execution record `{case_id}` component `{component}` missing split_aware_version.",
                        {"case_id": case_id, "component": component},
                    )
                )

        if is_frozen_mode(run_mode) and writeback_enabled:
            issues.append(
                _issue(
                    "error",
                    "frozen_mode_writeback_enabled",
                    f"Frozen-mode record `{case_id}` has writeback_enabled=true.",
                    {"case_id": case_id, "run_mode": run_mode},
                )
            )
        if is_frozen_mode(run_mode):
            persisted_count = int(writeback_ops.get("tactical_experience_count", 0) or 0) + int(
                writeback_ops.get("abstract_experience_count", 0) or 0
            )
            if persisted_count > 0 or bool(writeback_ops.get("raw_case_memory_written")):
                issues.append(
                    _issue(
                        "error",
                        "frozen_mode_persisted_writeback",
                        f"Frozen-mode record `{case_id}` persisted writeback artifacts.",
                        {
                            "case_id": case_id,
                            "run_mode": run_mode,
                            "tactical_experience_count": writeback_ops.get("tactical_experience_count", 0),
                            "abstract_experience_count": writeback_ops.get("abstract_experience_count", 0),
                            "raw_case_memory_written": bool(writeback_ops.get("raw_case_memory_written")),
                        },
                    )
                )
        if data_split in {"val", "test"} and writeback_enabled:
            issues.append(
                _issue(
                    "error",
                    "non_train_writeback_enabled",
                    f"Record `{case_id}` uses split `{data_split}` but writeback_enabled=true.",
                    {"case_id": case_id, "data_split": data_split, "run_mode": run_mode},
                )
            )
        if split_map and case_id:
            expected = expected_split_for_case(case_id, split_map)
            if expected and data_split != "unknown" and data_split != expected:
                issues.append(
                    _issue(
                        "error",
                        "record_split_mismatch_split_json",
                        f"Record `{case_id}` labeled `{data_split}` but split_json expects `{expected}`.",
                        {"case_id": case_id, "record_split": data_split, "expected_split": expected},
                    )
                )

    for case_id in split_cases.get("train", set()):
        if case_id in split_cases.get("val", set()) or case_id in split_cases.get("test", set()):
            issues.append(
                _issue(
                    "warning",
                    "case_seen_across_train_and_eval_splits",
                    f"Case `{case_id}` appears in both train and val/test execution records.",
                    {"case_id": case_id},
                )
            )

    return {
        "record_count": len(records),
        "split_case_counts": {key: len(value) for key, value in split_cases.items()},
        "issues": issues,
    }


def expected_split_for_case(case_id: str, split_map: dict[str, set[str]]) -> str:
    for split_name in ("train", "val", "test"):
        if case_id in split_map.get(split_name, set()):
            return split_name
    return ""


def audit_evaluation_manifests(eval_root: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    manifests = sorted(eval_root.rglob("evaluation_manifest.json")) if eval_root.exists() else []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        fairness = dict(payload.get("fairness_constraints", {}))
        execution_config = dict(payload.get("execution_config", {}))
        frozen_state = dict(payload.get("frozen_state", {}))
        data_split = normalize_split_name(str(payload.get("dataset", {}).get("data_split", "")).strip(), default="unknown")

        if fairness.get("frozen_evaluation_mode") is not True:
            issues.append(
                _issue(
                    "error",
                    "manifest_not_frozen_mode",
                    f"Evaluation manifest `{manifest_path}` missing frozen_evaluation_mode=true.",
                    {"manifest_path": str(manifest_path)},
                )
            )
        if execution_config.get("enable_writeback") is not False:
            issues.append(
                _issue(
                    "error",
                    "manifest_writeback_not_disabled",
                    f"Evaluation manifest `{manifest_path}` has enable_writeback != false.",
                    {"manifest_path": str(manifest_path)},
                )
            )
        if data_split not in {"val", "test"}:
            issues.append(
                _issue(
                    "warning",
                    "manifest_data_split_not_val_or_test",
                    f"Evaluation manifest `{manifest_path}` has data_split `{data_split}`.",
                    {"manifest_path": str(manifest_path), "data_split": data_split},
                )
            )
        for field_name in (
            "experience_split_aware_version",
            "cognition_split_aware_version",
            "policy_version",
        ):
            if not str(frozen_state.get(field_name, "")).strip():
                issues.append(
                    _issue(
                        "warning",
                        "missing_frozen_split_aware_field",
                        f"Evaluation manifest `{manifest_path}` missing frozen_state.{field_name}.",
                        {"manifest_path": str(manifest_path), "field_name": field_name},
                    )
                )

    return {
        "evaluation_manifest_count": len(manifests),
        "issues": issues,
    }


def audit_script_writeback_explicitness(scripts_root: Path) -> dict[str, Any]:
    implicit_calls = find_run_agent_calls_without_explicit_writeback(scripts_root)
    issues = [
        _issue(
            "warning",
            "implicit_run_agent_writeback",
            f"Script call to run_agent has no explicit enable_writeback at {item['file']}:{item['line']}.",
            item,
        )
        for item in implicit_calls
    ]
    script_files = list(scripts_root.glob("*.py")) if scripts_root.exists() else []
    return {
        "script_file_count": len(script_files),
        "implicit_run_agent_calls_without_enable_writeback": len(implicit_calls),
        "issues": issues,
    }


def _issue(severity: str, issue_type: str, message: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": severity,
        "issue_type": issue_type,
        "message": message,
        "context": context,
    }


if __name__ == "__main__":
    raise SystemExit(main())
