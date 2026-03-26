from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.hard_case_miner import load_execution_records


DEFAULT_MIN_CALLS = 1
DEFAULT_TOP_K = 5


@dataclass
class SkillHelpfulnessSummary:
    skill_name: str
    call_count: int
    selected_count: int
    helpful_count: int
    partially_helpful_count: int
    harmful_count: int
    uncertainty_reduction_count: int
    contradiction_detection_count: int
    malignant_flag_support_count: int
    average_evidence_strength: float
    common_failure_modes: list[dict[str, Any]] = field(default_factory=list)
    common_applicable_scenarios: list[dict[str, Any]] = field(default_factory=list)
    common_failure_scenarios: list[dict[str, Any]] = field(default_factory=list)
    evidence_strength_distribution: dict[str, int] = field(default_factory=dict)
    recommendation_type_distribution: dict[str, int] = field(default_factory=dict)
    helpful_rate: float = 0.0
    harmful_rate: float = 0.0
    source_case_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_skill_helpfulness(
    execution_records: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
    skill_name: str | None = None,
    label: str | None = None,
    confusion_pair: str | None = None,
    min_calls: int = DEFAULT_MIN_CALLS,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered_records = [
        record
        for record in execution_records
        if _record_passes_filters(
            record,
            dataset_name=dataset_name,
            skill_name=skill_name,
            label=label,
            confusion_pair=confusion_pair,
        )
    ]

    aggregated: dict[str, dict[str, Any]] = {}
    for record in filtered_records:
        for assessment in _iter_normalized_skill_assessments(record):
            name = assessment["skill_name"]
            if skill_name and name != skill_name:
                continue
            bucket = aggregated.setdefault(name, _new_bucket(name))
            _accumulate_skill_assessment(bucket, record, assessment)

    reports: list[dict[str, Any]] = []
    for name, bucket in aggregated.items():
        if int(bucket["call_count"]) < int(min_calls):
            continue
        reports.append(_finalize_bucket(name, bucket, top_k=top_k))

    reports.sort(
        key=lambda item: (
            float(item.get("helpful_rate", 0.0)),
            -float(item.get("harmful_rate", 0.0)),
            int(item.get("call_count", 0)),
            str(item.get("skill_name", "")),
        ),
        reverse=True,
    )
    summary = _build_analysis_summary(
        execution_record_count=len(execution_records),
        filtered_record_count=len(filtered_records),
        reports=reports,
        filters={
            "dataset_name": dataset_name,
            "skill_name": skill_name,
            "label": label,
            "confusion_pair": confusion_pair,
            "min_calls": min_calls,
            "top_k": top_k,
        },
    )
    return reports, summary


def save_skill_helpfulness_outputs(
    reports: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    reports_path = root / "skill_helpfulness_reports.jsonl"
    with reports_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")

    summary_path = root / "skill_helpfulness_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "reports_path": str(reports_path),
        "summary_path": str(summary_path),
    }


def _iter_normalized_skill_assessments(record: dict[str, Any]) -> list[dict[str, Any]]:
    reflection = record.get("reflection_summary", {})
    raw_assessments = reflection.get("skill_assessments", [])
    selected_skills = [
        str(item).strip()
        for item in (
            record.get("selected_skills")
            or record.get("planner_decision", {}).get("selected_skills", [])
            or record.get("planner_decision", {}).get("ordering", [])
        )
        if str(item).strip()
    ]
    skill_outputs = record.get("skill_outputs", {})
    assessment_map = {
        str(item.get("skill_name", "")).strip(): deepcopy(item)
        for item in raw_assessments
        if isinstance(item, dict) and str(item.get("skill_name", "")).strip()
    }
    ordered_names = list(dict.fromkeys(selected_skills + list(skill_outputs.keys()) + list(assessment_map.keys())))
    case_outcome = reflection.get("case_outcome", {})
    normalized: list[dict[str, Any]] = []
    for name in ordered_names:
        output = skill_outputs.get(name, {})
        base = assessment_map.get(name, {})
        output_present = bool(base.get("output_present")) if "output_present" in base else _output_present(output)
        evidence_strength = _normalize_evidence_strength(base.get("evidence_strength") or output.get("evidence_strength"))
        evidence_strength_score = float(base.get("evidence_strength_score", _evidence_strength_score(evidence_strength)))
        contradiction_detected = bool(base.get("contradiction_detected")) if "contradiction_detected" in base else _detect_contradiction_signal(name, output)
        malignant_flag_support = bool(base.get("malignant_flag_support")) if "malignant_flag_support" in base else _detect_malignant_flag_support(record, name, output)
        impact = str(base.get("impact", "")).strip().lower() or _infer_impact(
            record=record,
            skill_name=name,
            output=output,
            output_present=output_present,
            evidence_strength_score=evidence_strength_score,
            contradiction_detected=contradiction_detected,
            malignant_flag_support=malignant_flag_support,
        )
        helpfulness = str(base.get("helpfulness", "")).strip().lower() or _infer_helpfulness(record, impact, output_present)
        uncertainty_reduction = (
            bool(base.get("uncertainty_reduction"))
            if "uncertainty_reduction" in base
            else _detect_uncertainty_reduction(record, name, output, impact)
        )
        applicable_scenarios = [
            str(item).strip()
            for item in base.get("applicable_scenarios", _derive_scenarios(record, name))
            if str(item).strip()
        ]
        failure_modes = [
            str(item).strip()
            for item in base.get(
                "failure_modes",
                _derive_failure_modes(
                    record=record,
                    output_present=output_present,
                    evidence_strength=evidence_strength,
                    impact=impact,
                    contradiction_detected=contradiction_detected,
                ),
            )
            if str(item).strip()
        ]
        normalized.append(
            {
                "skill_name": name,
                "selected": bool(base.get("selected", name in selected_skills)),
                "triggered": bool(base.get("triggered", name in skill_outputs)),
                "output_present": output_present,
                "impact": impact,
                "helpfulness": helpfulness,
                "evidence_strength": evidence_strength,
                "evidence_strength_score": evidence_strength_score,
                "recommendation_type": str(base.get("recommendation_type") or output.get("recommendation_type") or "unknown").strip().lower(),
                "referenced_experiences": [
                    str(item).strip()
                    for item in base.get("referenced_experiences", output.get("referenced_experiences", []))
                    if str(item).strip()
                ],
                "uncertainty_reduction": uncertainty_reduction,
                "contradiction_detected": contradiction_detected,
                "malignant_flag_support": malignant_flag_support,
                "applicable_scenarios": applicable_scenarios,
                "failure_modes": failure_modes,
                "case_outcome": deepcopy(case_outcome),
            }
        )
    return normalized


def _record_passes_filters(
    record: dict[str, Any],
    *,
    dataset_name: str | None,
    skill_name: str | None,
    label: str | None,
    confusion_pair: str | None,
) -> bool:
    if dataset_name and str(record.get("dataset_name", "")).strip() != dataset_name:
        return False
    if label:
        ground_truth = record.get("ground_truth", {})
        candidate_labels = {
            str(ground_truth.get("raw_label", "")).strip().lower(),
            str(ground_truth.get("canonical_label", "")).strip().lower(),
        }
        if label.strip().lower() not in candidate_labels:
            return False
    if confusion_pair:
        case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
        confusion_text = str(case_outcome.get("confusion_pair", "")).strip().lower()
        if confusion_pair.strip().lower() not in confusion_text:
            return False
    if skill_name:
        available = {
            str(item).strip()
            for item in (
                record.get("selected_skills")
                or record.get("planner_decision", {}).get("selected_skills", [])
                or list(record.get("skill_outputs", {}).keys())
            )
            if str(item).strip()
        }
        if skill_name not in available:
            return False
    return True


def _new_bucket(skill_name: str) -> dict[str, Any]:
    return {
        "skill_name": skill_name,
        "call_count": 0,
        "selected_count": 0,
        "helpful_count": 0,
        "partially_helpful_count": 0,
        "harmful_count": 0,
        "uncertainty_reduction_count": 0,
        "contradiction_detection_count": 0,
        "malignant_flag_support_count": 0,
        "evidence_strength_sum": 0.0,
        "evidence_strength_count": 0,
        "evidence_strength_distribution": Counter(),
        "recommendation_type_distribution": Counter(),
        "failure_modes": Counter(),
        "applicable_scenarios": Counter(),
        "failure_scenarios": Counter(),
        "case_ids": set(),
    }


def _accumulate_skill_assessment(bucket: dict[str, Any], record: dict[str, Any], assessment: dict[str, Any]) -> None:
    bucket["call_count"] += 1 if assessment.get("triggered", True) else 0
    bucket["selected_count"] += 1 if assessment.get("selected", assessment.get("triggered", True)) else 0
    impact = str(assessment.get("impact", "")).strip().lower()
    if impact == "helpful":
        bucket["helpful_count"] += 1
    elif impact == "partially_helpful":
        bucket["partially_helpful_count"] += 1
    elif impact == "harmful":
        bucket["harmful_count"] += 1
    if assessment.get("uncertainty_reduction"):
        bucket["uncertainty_reduction_count"] += 1
    if assessment.get("contradiction_detected"):
        bucket["contradiction_detection_count"] += 1
    if assessment.get("malignant_flag_support"):
        bucket["malignant_flag_support_count"] += 1

    evidence_strength = str(assessment.get("evidence_strength", "unknown")).strip().lower()
    bucket["evidence_strength_distribution"][evidence_strength] += 1
    if assessment.get("output_present"):
        bucket["evidence_strength_sum"] += float(assessment.get("evidence_strength_score", 0.0))
        bucket["evidence_strength_count"] += 1

    recommendation_type = str(assessment.get("recommendation_type", "unknown")).strip().lower() or "unknown"
    bucket["recommendation_type_distribution"][recommendation_type] += 1

    applicable_scenarios = assessment.get("applicable_scenarios", [])
    for scenario in applicable_scenarios:
        bucket["applicable_scenarios"][str(scenario).strip()] += 1

    if impact == "harmful":
        for scenario in applicable_scenarios:
            bucket["failure_scenarios"][str(scenario).strip()] += 1
    for failure_mode in assessment.get("failure_modes", []):
        bucket["failure_modes"][str(failure_mode).strip()] += 1
    bucket["case_ids"].add(str(record.get("case_id", "")))


def _finalize_bucket(skill_name: str, bucket: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    call_count = int(bucket["call_count"])
    helpful_rate = (bucket["helpful_count"] / call_count) if call_count else 0.0
    harmful_rate = (bucket["harmful_count"] / call_count) if call_count else 0.0
    average_evidence_strength = (
        bucket["evidence_strength_sum"] / bucket["evidence_strength_count"]
        if bucket["evidence_strength_count"]
        else 0.0
    )
    report = SkillHelpfulnessSummary(
        skill_name=skill_name,
        call_count=call_count,
        selected_count=int(bucket["selected_count"]),
        helpful_count=int(bucket["helpful_count"]),
        partially_helpful_count=int(bucket["partially_helpful_count"]),
        harmful_count=int(bucket["harmful_count"]),
        uncertainty_reduction_count=int(bucket["uncertainty_reduction_count"]),
        contradiction_detection_count=int(bucket["contradiction_detection_count"]),
        malignant_flag_support_count=int(bucket["malignant_flag_support_count"]),
        average_evidence_strength=round(average_evidence_strength, 4),
        common_failure_modes=_counter_to_ranked_list(bucket["failure_modes"], top_k=top_k),
        common_applicable_scenarios=_counter_to_ranked_list(bucket["applicable_scenarios"], top_k=top_k),
        common_failure_scenarios=_counter_to_ranked_list(bucket["failure_scenarios"], top_k=top_k),
        evidence_strength_distribution=dict(bucket["evidence_strength_distribution"]),
        recommendation_type_distribution=dict(bucket["recommendation_type_distribution"]),
        helpful_rate=round(helpful_rate, 4),
        harmful_rate=round(harmful_rate, 4),
        source_case_count=len(bucket["case_ids"]),
    )
    return report.to_dict()


def _build_analysis_summary(
    *,
    execution_record_count: int,
    filtered_record_count: int,
    reports: list[dict[str, Any]],
    filters: dict[str, Any],
) -> dict[str, Any]:
    total_calls = sum(int(item.get("call_count", 0)) for item in reports)
    total_helpful = sum(int(item.get("helpful_count", 0)) for item in reports)
    total_partially_helpful = sum(int(item.get("partially_helpful_count", 0)) for item in reports)
    total_harmful = sum(int(item.get("harmful_count", 0)) for item in reports)
    average_evidence_strength = (
        sum(float(item.get("average_evidence_strength", 0.0)) * int(item.get("call_count", 0)) for item in reports) / total_calls
        if total_calls
        else 0.0
    )
    top_helpful = sorted(
        reports,
        key=lambda item: (float(item.get("helpful_rate", 0.0)), int(item.get("call_count", 0))),
        reverse=True,
    )[:5]
    top_harmful = sorted(
        reports,
        key=lambda item: (float(item.get("harmful_rate", 0.0)), int(item.get("call_count", 0))),
        reverse=True,
    )[:5]
    return {
        "execution_record_count": execution_record_count,
        "filtered_record_count": filtered_record_count,
        "skill_count": len(reports),
        "totals": {
            "call_count": total_calls,
            "helpful_count": total_helpful,
            "partially_helpful_count": total_partially_helpful,
            "harmful_count": total_harmful,
            "average_evidence_strength": round(average_evidence_strength, 4),
        },
        "filters": deepcopy(filters),
        "top_helpful_skills": [
            {
                "skill_name": item.get("skill_name"),
                "helpful_rate": item.get("helpful_rate"),
                "call_count": item.get("call_count"),
            }
            for item in top_helpful
        ],
        "top_harmful_skills": [
            {
                "skill_name": item.get("skill_name"),
                "harmful_rate": item.get("harmful_rate"),
                "call_count": item.get("call_count"),
            }
            for item in top_harmful
        ],
    }


def _infer_impact(
    *,
    record: dict[str, Any],
    skill_name: str,
    output: dict[str, Any],
    output_present: bool,
    evidence_strength_score: float,
    contradiction_detected: bool,
    malignant_flag_support: bool,
) -> str:
    if not output_present:
        return "harmful"
    evaluation = record.get("evaluation", {})
    correct = evaluation.get("correct")
    baseline_correct = evaluation.get("baseline_correct")
    support_score = 0
    if evidence_strength_score >= 0.6:
        support_score += 2
    elif evidence_strength_score > 0.0:
        support_score += 1
    if contradiction_detected:
        support_score += 1
    if malignant_flag_support:
        support_score += 1
    if _detect_uncertainty_reduction(record, skill_name, output, impact="partially_helpful"):
        support_score += 1

    if correct is True and support_score >= 1:
        return "helpful"
    if baseline_correct is False and correct is True and support_score >= 1:
        return "helpful"
    if baseline_correct is True and correct is False and support_score <= 1:
        return "harmful"
    if correct is False and support_score == 0:
        return "harmful"
    if support_score >= 2:
        return "partially_helpful"
    return "harmful" if evidence_strength_score == 0.0 else "partially_helpful"


def _infer_helpfulness(record: dict[str, Any], impact: str, output_present: bool) -> str:
    if not output_present or impact == "harmful":
        return "failure"
    evaluation = record.get("evaluation", {})
    if evaluation.get("correct") is True:
        return "success"
    return "partially_helpful"


def _output_present(output: dict[str, Any]) -> bool:
    if not isinstance(output, dict) or not output:
        return False
    for key, value in output.items():
        if key in {"referenced_experiences", "evidence_strength", "recommendation_type"}:
            continue
        if _value_has_meaningful_content(value):
            return True
    return False


def _value_has_meaningful_content(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized and normalized not in {"unknown", "none", "null", "n/a"})
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return any(_value_has_meaningful_content(item) for item in value)
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"referenced_experiences", "evidence_strength", "recommendation_type"}:
                continue
            if _value_has_meaningful_content(nested):
                return True
    return False


def _normalize_evidence_strength(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return "unknown"


def _evidence_strength_score(evidence_strength: str) -> float:
    return {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.3,
    }.get(str(evidence_strength).strip().lower(), 0.0)


def _flatten_text(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
    elif isinstance(value, list):
        for item in value:
            texts.extend(_flatten_text(item))
    elif isinstance(value, dict):
        for nested in value.values():
            texts.extend(_flatten_text(nested))
    return texts


def _detect_contradiction_signal(skill_name: str, output: dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False
    if skill_name == "contradiction_check_skill":
        return True
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "conflicts", "suspicious_points"):
        if _value_has_meaningful_content(output.get(field_name)):
            return True
    return str(output.get("recommendation_type", "")).strip().lower() == "conflict_signal"


def _detect_malignant_flag_support(record: dict[str, Any], skill_name: str, output: dict[str, Any]) -> bool:
    if not isinstance(output, dict) or not _output_present(output):
        return False
    if skill_name == "malignancy_risk_assessment_skill":
        return True
    ground_truth = record.get("ground_truth", {})
    if bool(ground_truth.get("malignant_flag")) and str(output.get("recommendation_type", "")).strip().lower() == "risk_signal":
        return True
    text_blob = " ".join(_flatten_text(output)).lower()
    return any(pattern in text_blob for pattern in ("malignan", "high risk", "alarm", "melanom", "scc", "bcc"))


def _detect_uncertainty_reduction(record: dict[str, Any], skill_name: str, output: dict[str, Any], impact: str) -> bool:
    if not isinstance(output, dict) or not _output_present(output):
        return False
    if skill_name in {
        "uncertainty_assessment_skill",
        "information_gap_detection_skill",
        "lesion_description_structuring_skill",
        "differential_compare_skill",
        "exclusion_reasoning_skill",
    }:
        return True
    text_blob = " ".join(_flatten_text(output)).lower()
    recommendation_type = str(output.get("recommendation_type", "")).strip().lower()
    if recommendation_type == "uncertainty_signal":
        return True
    return any(pattern in text_blob for pattern in ("exclude", "narrow", "less likely", "missing", "uncertain", "compare"))


def _derive_scenarios(record: dict[str, Any], skill_name: str) -> list[str]:
    qwen_initial = record.get("qwen_initial", {})
    case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    input_summary = record.get("input_summary", {})
    clinical_metadata = input_summary.get("clinical_metadata", {})
    ddx_candidates = [str(item).strip() for item in qwen_initial.get("ddx_candidates", [])[:2] if str(item).strip()]
    scenarios = [skill_name]
    if ddx_candidates:
        scenarios.append(f"ddx:{'|'.join(ddx_candidates)}")
    region = str(clinical_metadata.get("region", "")).strip()
    if region:
        scenarios.append(f"region:{region.lower()}")
    for risk_flag in case_outcome.get("risk_flags", [])[:2]:
        scenarios.append(f"risk:{risk_flag}")
    uncertainty_level = str(case_outcome.get("uncertainty_level", "unknown")).strip().lower()
    if uncertainty_level and uncertainty_level != "unknown":
        scenarios.append(f"uncertainty:{uncertainty_level}")
    confusion_pair_value = str(case_outcome.get("confusion_pair", "")).strip()
    if confusion_pair_value:
        scenarios.append(f"confusion:{confusion_pair_value}")
    return list(dict.fromkeys(scenarios))


def _derive_failure_modes(
    *,
    record: dict[str, Any],
    output_present: bool,
    evidence_strength: str,
    impact: str,
    contradiction_detected: bool,
) -> list[str]:
    case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    failure_modes: list[str] = []
    if not output_present:
        failure_modes.append("no_structured_output")
    if impact == "harmful":
        failure_modes.append("low_net_reasoning_value")
    if case_outcome.get("confusion_pair"):
        failure_modes.append("active_confusion_unresolved")
    if case_outcome.get("uncertainty_level") == "high":
        failure_modes.append("uncertainty_remained_high")
    if record.get("evaluation", {}).get("correct") is False:
        failure_modes.append("final_reasoning_failed")
    if evidence_strength in {"low", "unknown"}:
        failure_modes.append("weak_evidence_strength")
    if contradiction_detected and impact == "harmful":
        failure_modes.append("contradiction_raised_without_resolution")
    return list(dict.fromkeys(failure_modes))


def _counter_to_ranked_list(counter: Counter[str], *, top_k: int) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": count}
        for name, count in counter.most_common(top_k)
        if str(name).strip() and int(count) > 0
    ]
