from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.aggregator import build_evidence_bundle
from agent.hard_case_miner import load_execution_records
from agent.state import CaseInput, CaseState


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "evidence_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evidence serialization quality before/after evidence calibration.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-filter", type=str, default="")
    parser.add_argument("--mode", type=str, default="heuristic", choices=("heuristic", "learned", "hybrid"))
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional learned calibrator checkpoint for learned/hybrid mode.")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--early-window", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_execution_records(args.records_root)
    if args.dataset_filter.strip():
        records = [record for record in records if str(record.get("dataset_name", "")).strip() == args.dataset_filter.strip()]
    if int(args.max_cases) > 0:
        records = records[: int(args.max_cases)]
    if not records:
        raise ValueError("No execution records found for evidence calibration evaluation.")

    per_case_rows: list[dict[str, Any]] = []
    for record in records:
        baseline_bundle = build_evidence_bundle(
            _state_from_record(
                record,
                evidence_policy={
                    "enable_evidence_calibrator": False,
                    "calibrator_mode": "off",
                    "debug_output": False,
                },
            )
        )
        calibrated_bundle = build_evidence_bundle(
            _state_from_record(
                record,
                evidence_policy={
                    "enable_evidence_calibrator": True,
                    "calibrator_mode": args.mode,
                    "calibrator_checkpoint_path": args.checkpoint_path.strip(),
                    "debug_output": True,
                },
            )
        )
        baseline_metrics = _text_quality_metrics(
            serialized_text=str(baseline_bundle.get("serialized_evidence_text", "")),
            record=record,
            early_window=max(4, int(args.early_window)),
        )
        calibrated_metrics = _text_quality_metrics(
            serialized_text=str(calibrated_bundle.get("serialized_evidence_text", "")),
            record=record,
            early_window=max(4, int(args.early_window)),
        )
        per_case_rows.append(
            {
                "case_id": str(record.get("case_id", "")),
                "dataset_name": str(record.get("dataset_name", "")),
                "baseline_metrics": baseline_metrics,
                "calibrated_metrics": calibrated_metrics,
                "delta": _metric_delta(calibrated_metrics, baseline_metrics),
                "calibrator_type": calibrated_bundle.get("evidence_calibration_debug", {}).get("calibrator_type"),
            }
        )

    summary = {
        "num_cases": len(per_case_rows),
        "dataset_filter": args.dataset_filter.strip(),
        "mode": args.mode,
        "checkpoint_path": args.checkpoint_path.strip(),
        "baseline": _aggregate_metrics([row["baseline_metrics"] for row in per_case_rows]),
        "calibrated": _aggregate_metrics([row["calibrated_metrics"] for row in per_case_rows]),
    }
    summary["delta"] = _metric_delta(summary["calibrated"], summary["baseline"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evidence_calibration_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_path = args.output_dir / "evidence_calibration_case_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in per_case_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "report_path": str(report_path),
        "rows_path": str(rows_path),
        "summary": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _state_from_record(record: dict[str, Any], *, evidence_policy: dict[str, Any]) -> CaseState:
    input_summary = dict(record.get("input_summary", {}))
    clinical_metadata = dict(input_summary.get("clinical_metadata", {}))
    case_input = CaseInput(
        case_id=str(record.get("case_id", "")),
        image_path=str(input_summary.get("image_path", "/tmp/unknown_image.png")),
        metadata=clinical_metadata,
        dataset_name=str(record.get("dataset_name", "") or None) or None,
    )
    state = CaseState(case_input=case_input)
    state.perception = dict(record.get("qwen_initial", {}))
    retrieval_bundle = dict(record.get("retrieval_bundle", {}).get("after_skills", {}))
    if not retrieval_bundle:
        retrieval_bundle = dict(record.get("retrieval_bundle", {}).get("before_skills", {}))
    state.retrieval_bundle = retrieval_bundle
    state.skill_outputs = dict(record.get("skill_outputs", {}))
    evidence_bundle = dict(record.get("evidence_bundle", {}))
    state.risk_flags = [
        str(item).strip()
        for item in (
            evidence_bundle.get("risk_flags")
            or record.get("reflection_summary", {}).get("case_outcome", {}).get("risk_flags", [])
        )
        if str(item).strip()
    ]
    uncertainty_summary = dict(evidence_bundle.get("uncertainty_summary", {}))
    state.uncertainty = {
        "uncertainty_level": str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower(),
        "reasons": [str(item).strip() for item in uncertainty_summary.get("reasons", []) if str(item).strip()],
        "missing_information": [
            str(item).strip()
            for item in uncertainty_summary.get("missing_information", [])
            if str(item).strip()
        ],
        "referenced_experiences": [
            str(item).strip()
            for item in uncertainty_summary.get("referenced_experiences", [])
            if str(item).strip()
        ],
    }
    state.planner_output = dict(record.get("planner_decision", {}))
    state.skill_retrieval_bundle = dict(record.get("skill_retrieval", {}))
    state.retrieved_experience = list(retrieval_bundle.get("aggregator_summary", []))
    state.notes = [str(item).strip() for item in evidence_bundle.get("notes", []) if str(item).strip()]
    state.policy_snapshot = {"evidence_policy": dict(evidence_policy)}
    return state


def _text_quality_metrics(*, serialized_text: str, record: dict[str, Any], early_window: int) -> dict[str, float]:
    lines = [line.strip() for line in serialized_text.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]
    skill_positions = _skill_positions(serialized_text)
    helpful_skills, harmful_skills = _helpfulness_sets(record)
    helpful_mrr = _mean_reciprocal_rank(helpful_skills, skill_positions)
    harmful_early_ratio = _harmful_early_ratio(harmful_skills, skill_positions, early_window=early_window)
    priority_coverage = _priority_signal_coverage(
        bullet_lines=bullet_lines,
        record=record,
        early_window=early_window,
    )
    return {
        "char_len": float(len(serialized_text)),
        "line_count": float(len(lines)),
        "bullet_count": float(len(bullet_lines)),
        "helpful_mrr": helpful_mrr,
        "harmful_early_ratio": harmful_early_ratio,
        "priority_signal_coverage": priority_coverage,
    }


def _skill_positions(serialized_text: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    lines = [line.strip() for line in serialized_text.splitlines() if line.strip()]
    known_skills = [
        "lesion_description_structuring_skill",
        "morphology_analysis_skill",
        "color_pattern_analysis_skill",
        "border_surface_analysis_skill",
        "distribution_analysis_skill",
        "temporal_evolution_skill",
        "metadata_consistency_skill",
        "differential_compare_skill",
        "exclusion_reasoning_skill",
        "mel_nev_specialist_skill",
        "ack_scc_specialist_skill",
        "malignancy_risk_assessment_skill",
        "uncertainty_assessment_skill",
        "information_gap_detection_skill",
        "contradiction_check_skill",
        "escalation_recommendation_skill",
    ]
    for index, line in enumerate(lines):
        for skill_name in known_skills:
            if skill_name in line and skill_name not in positions:
                positions[skill_name] = index
    return positions


def _helpfulness_sets(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    helpful: list[str] = []
    harmful: list[str] = []
    for item in record.get("reflection_summary", {}).get("skill_assessments", []):
        if not isinstance(item, dict):
            continue
        skill_name = str(item.get("skill_name", "")).strip()
        if not skill_name:
            continue
        impact = str(item.get("impact", "")).strip().lower()
        if impact in {"helpful", "partially_helpful"}:
            helpful.append(skill_name)
        elif impact == "harmful":
            harmful.append(skill_name)
    return list(dict.fromkeys(helpful)), list(dict.fromkeys(harmful))


def _mean_reciprocal_rank(skills: list[str], positions: dict[str, int]) -> float:
    if not skills:
        return 0.0
    reciprocal_values = []
    for skill_name in skills:
        position = positions.get(skill_name)
        if position is None:
            reciprocal_values.append(0.0)
        else:
            reciprocal_values.append(1.0 / (position + 1.0))
    return round(sum(reciprocal_values) / len(reciprocal_values), 6)


def _harmful_early_ratio(skills: list[str], positions: dict[str, int], *, early_window: int) -> float:
    if not skills:
        return 0.0
    hits = 0.0
    for skill_name in skills:
        position = positions.get(skill_name)
        if position is not None and position < int(early_window):
            hits += 1.0
    return round(hits / len(skills), 6)


def _priority_signal_coverage(*, bullet_lines: list[str], record: dict[str, Any], early_window: int) -> float:
    early_text = " ".join(bullet_lines[: max(1, int(early_window))]).lower()
    checks: list[bool] = []
    risk_flags = record.get("evidence_bundle", {}).get("risk_flags", []) or record.get("reflection_summary", {}).get("case_outcome", {}).get("risk_flags", [])
    if risk_flags:
        checks.append(any(term in early_text for term in ("risk", "malignan", "alarm")))
    contradiction_summary = record.get("evidence_bundle", {}).get("contradiction_summary", {})
    contradiction_count = 0
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts", "suspicious_points"):
        contradiction_count += len(contradiction_summary.get(field_name, []) or [])
    if contradiction_count > 0:
        checks.append(any(term in early_text for term in ("contradiction", "conflict", "inconsisten")))
    uncertainty_level = str(record.get("evidence_bundle", {}).get("uncertainty_summary", {}).get("uncertainty_level", "unknown")).lower()
    missing_information = record.get("evidence_bundle", {}).get("uncertainty_summary", {}).get("missing_information", [])
    if uncertainty_level in {"medium", "high"} or missing_information:
        checks.append(any(term in early_text for term in ("uncertainty", "missing information", "information gap")))
    if not checks:
        return 0.0
    return round(sum(1.0 for item in checks if item) / len(checks), 6)


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    result: dict[str, float] = {}
    keys = sorted(rows[0].keys())
    for key in keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        result[key] = round(statistics.mean(values), 6)
    return result


def _metric_delta(newer: dict[str, Any], older: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in set(newer.keys()).intersection(older.keys()):
        if isinstance(newer.get(key), (int, float)) and isinstance(older.get(key), (int, float)):
            result[f"{key}_delta"] = round(float(newer.get(key, 0.0)) - float(older.get(key, 0.0)), 6)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

