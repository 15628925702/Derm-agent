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
SKILL_JUDGEMENT_VERSION = "skill_helpfulness_judgement_v3"


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
    evidence_usage_count: int
    average_evidence_strength: float
    common_failure_modes: list[dict[str, Any]] = field(default_factory=list)
    common_applicable_scenarios: list[dict[str, Any]] = field(default_factory=list)
    common_failure_scenarios: list[dict[str, Any]] = field(default_factory=list)
    evidence_strength_distribution: dict[str, int] = field(default_factory=dict)
    recommendation_type_distribution: dict[str, int] = field(default_factory=dict)
    harmful_reason_distribution: dict[str, int] = field(default_factory=dict)
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
        for assessment in normalize_skill_assessments_for_record(record):
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


def normalize_skill_assessments_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
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
    total_selected = len(ordered_names)
    for skill_rank, name in enumerate(ordered_names):
        output = skill_outputs.get(name, {})
        base = assessment_map.get(name, {})
        base_version = str(base.get("judgement_version", reflection.get("skill_judgement_version", ""))).strip()
        has_skill_output = isinstance(output, dict) and bool(output)
        use_stored_judgement = bool(base) and (base_version == SKILL_JUDGEMENT_VERSION or not has_skill_output)
        output_present = bool(base.get("output_present")) if "output_present" in base else _output_present(output)
        evidence_strength = _normalize_evidence_strength(base.get("evidence_strength") or output.get("evidence_strength"))
        evidence_strength_score = float(base.get("evidence_strength_score", _evidence_strength_score(evidence_strength)))
        contradiction_detected = bool(base.get("contradiction_detected")) if "contradiction_detected" in base else _detect_contradiction_signal(name, output)
        malignant_flag_support = bool(base.get("malignant_flag_support")) if "malignant_flag_support" in base else _detect_malignant_flag_support(record, name, output)
        impact, judgement_reasons, harmful_reasons, evidence_usage_score = _infer_impact(
            record=record,
            skill_name=name,
            output=output,
            output_present=output_present,
            evidence_strength_score=evidence_strength_score,
            contradiction_detected=contradiction_detected,
            malignant_flag_support=malignant_flag_support,
            skill_rank=skill_rank,
            total_selected=total_selected,
        )
        if use_stored_judgement:
            impact = str(base.get("impact", "")).strip().lower() or impact
        helpfulness = (
            str(base.get("helpfulness", "")).strip().lower() or _infer_helpfulness(record, impact, output_present)
            if use_stored_judgement
            else _infer_helpfulness(record, impact, output_present)
        )
        if use_stored_judgement and "judgement_reasons" in base:
            judgement_reasons = [str(item).strip() for item in base.get("judgement_reasons", []) if str(item).strip()]
        if use_stored_judgement and "harmful_reasons" in base:
            harmful_reasons = [str(item).strip() for item in base.get("harmful_reasons", []) if str(item).strip()]
        if use_stored_judgement and "evidence_usage_score" in base:
            evidence_usage_score = float(base.get("evidence_usage_score", 0.0) or 0.0)
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
                "evidence_usage_score": evidence_usage_score,
                "judgement_reasons": judgement_reasons,
                "harmful_reasons": harmful_reasons,
                "applicable_scenarios": applicable_scenarios,
                "failure_modes": failure_modes,
                "case_outcome": deepcopy(case_outcome),
            }
        )
    return normalized


def _iter_normalized_skill_assessments(record: dict[str, Any]) -> list[dict[str, Any]]:
    return normalize_skill_assessments_for_record(record)


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
        lowered = confusion_pair.strip().lower()
        record_tags = _record_confusion_tags(record)
        case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
        confusion_text = str(case_outcome.get("confusion_pair", "")).strip().lower()
        if lowered not in confusion_text and not any(lowered in tag for tag in record_tags):
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


def _record_confusion_tags(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    confusion_pair = str(case_outcome.get("confusion_pair", "")).strip()
    if confusion_pair:
        tags.append(confusion_pair.lower())
        parts = confusion_pair.split("->")
        if len(parts) == 2:
            tags.append(f"{parts[0].strip().lower()}_vs_{parts[1].strip().lower()}")

    ground_truth = dict(record.get("ground_truth", {}) or {})
    gt_label = str(ground_truth.get("canonical_label") or ground_truth.get("raw_label") or "").strip().lower()
    baseline_output = dict(record.get("baseline_qwen", {}) or {})
    qwen_final = dict(record.get("qwen_final", {}) or {})
    fusion = dict(qwen_final.get("fusion_decision", {}) or {})
    baseline_label = str(
        baseline_output.get("final_diagnosis")
        or fusion.get("baseline_label", "")
        or qwen_final.get("final_diagnosis", "")
    ).strip().lower()
    agent_label = str(fusion.get("agent_label", "")).strip().lower()

    benign_mimic_terms = ("nev", "nevus", "bkl", "seborrheic keratosis", "df", "dermatofibroma", "vasc", "vascular")
    if ("bcc" in baseline_label or "basal cell" in baseline_label) and any(term in agent_label for term in benign_mimic_terms):
        tags.append("bcc_benign_mimic")
    if ("bcc" in baseline_label or "basal cell" in baseline_label) and gt_label in {"nv", "bkl", "df", "vasc"}:
        tags.append("bcc_benign_mimic")

    return _dedupe_strings(tags)


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
        "evidence_usage_count": 0,
        "evidence_strength_sum": 0.0,
        "evidence_strength_count": 0,
        "evidence_strength_distribution": Counter(),
        "recommendation_type_distribution": Counter(),
        "harmful_reason_distribution": Counter(),
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
    if float(assessment.get("evidence_usage_score", 0.0) or 0.0) > 0.0:
        bucket["evidence_usage_count"] += 1

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
    for reason in assessment.get("harmful_reasons", []):
        bucket["harmful_reason_distribution"][str(reason).strip()] += 1
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
        evidence_usage_count=int(bucket["evidence_usage_count"]),
        average_evidence_strength=round(average_evidence_strength, 4),
        common_failure_modes=_counter_to_ranked_list(bucket["failure_modes"], top_k=top_k),
        common_applicable_scenarios=_counter_to_ranked_list(bucket["applicable_scenarios"], top_k=top_k),
        common_failure_scenarios=_counter_to_ranked_list(bucket["failure_scenarios"], top_k=top_k),
        evidence_strength_distribution=dict(bucket["evidence_strength_distribution"]),
        recommendation_type_distribution=dict(bucket["recommendation_type_distribution"]),
        harmful_reason_distribution=dict(bucket["harmful_reason_distribution"]),
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
    total_evidence_usage = sum(int(item.get("evidence_usage_count", 0)) for item in reports)
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
            "evidence_usage_count": total_evidence_usage,
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
        "top_harmful_reasons": _counter_to_ranked_list(
            Counter(
                reason
                for item in reports
                for reason, count in dict(item.get("harmful_reason_distribution", {})).items()
                for _ in range(int(count))
            ),
            top_k=10,
        ),
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
    skill_rank: int,
    total_selected: int,
) -> tuple[str, list[str], list[str], float]:
    evaluation = record.get("evaluation", {})
    evidence_bundle = record.get("evidence_bundle", {})
    case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    delta = dict(evaluation.get("agent_vs_baseline_delta", {}) or {})
    correct = evaluation.get("correct")
    baseline_correct = evaluation.get("baseline_correct")
    recommendation_type = str(output.get("recommendation_type", "unknown")).strip().lower()
    contradiction_count = _contradiction_count(evidence_bundle.get("contradiction_summary", {}))
    uncertainty_reduction = _detect_uncertainty_reduction(record, skill_name, output, impact="partially_helpful")
    uncertainty_final = str(
        evidence_bundle.get("uncertainty_summary", {}).get("uncertainty_level", case_outcome.get("uncertainty_level", "unknown"))
    ).lower()
    uncertainty_initial = str(record.get("qwen_initial", {}).get("uncertainty", {}).get("level", "unknown")).lower()
    risk_flags = list(evidence_bundle.get("risk_flags", []) or case_outcome.get("risk_flags", []))
    case_failed = correct is False
    regression = baseline_correct is True and correct is False

    evidence_usage_score = _estimate_evidence_usage_score(
        record=record,
        skill_name=skill_name,
        output=output,
        recommendation_type=recommendation_type,
        contradiction_detected=contradiction_detected,
        malignant_flag_support=malignant_flag_support,
        uncertainty_reduction=uncertainty_reduction,
    )
    reasons: list[str] = []
    harmful_reasons: list[str] = []
    positive = 0.0
    negative = 0.0

    if output_present:
        positive += 0.8
        reasons.append("structured_output_present")
    else:
        negative += 2.5
        harmful_reasons.append("no_structured_output")

    if evidence_strength_score >= 0.95:
        positive += 1.2
        reasons.append("evidence_strength_high")
    elif evidence_strength_score >= 0.6:
        positive += 0.7
        reasons.append("evidence_strength_medium")
    elif evidence_strength_score > 0.0:
        positive += 0.2
        negative += 0.6
        harmful_reasons.append("weak_evidence_strength")
    else:
        negative += 1.5
        harmful_reasons.append("weak_or_unknown_evidence")

    if evidence_usage_score >= 1.5:
        positive += min(1.6, evidence_usage_score)
        reasons.append("evidence_used_in_final_bundle")
    elif evidence_usage_score > 0.0:
        positive += 0.4
    else:
        negative += 1.2
        harmful_reasons.append("evidence_not_used")

    if contradiction_detected:
        if contradiction_count > 0:
            positive += 0.8
            reasons.append("contradiction_signal_supported")
        else:
            negative += 1.4
            harmful_reasons.append("false_conflict_signal")

    if uncertainty_reduction:
        if _uncertainty_rank(uncertainty_final) < _uncertainty_rank(uncertainty_initial):
            positive += 0.9
            reasons.append("uncertainty_reduced")
        elif _uncertainty_rank(uncertainty_final) == _uncertainty_rank(uncertainty_initial) and uncertainty_final in {"high", "medium"}:
            positive += 0.4
            reasons.append("uncertainty_structure_preserved")
        else:
            negative += 0.8
            harmful_reasons.append("uncertainty_not_reduced")

    if malignant_flag_support:
        positive += 0.8
        reasons.append("malignancy_risk_support")

    if regression:
        negative += 2.5
        harmful_reasons.append("baseline_regression_low_support")
    if case_failed:
        negative += 0.9
    if float(delta.get("correct_delta", 0) or 0) > 0:
        positive += 1.8
        reasons.append("delta_correct_positive")
    if float(delta.get("topk_hit_delta", 0) or 0) > 0:
        positive += 0.7
        reasons.append("delta_topk_positive")
    if float(delta.get("malignant_recall_delta", 0) or 0) < 0:
        negative += 1.2
        harmful_reasons.append("malignant_recall_regression")

    if recommendation_type in {"risk_signal", "escalation_signal"}:
        if not risk_flags:
            negative += 1.2
            harmful_reasons.append("misleading_risk_or_escalation")
        if case_failed and evidence_usage_score < 1.0:
            negative += 0.6
    if recommendation_type == "conflict_signal" and contradiction_count == 0:
        negative += 0.8
        harmful_reasons.append("false_conflict_signal")
    if (
        case_failed
        and recommendation_type in {"descriptive_evidence", "comparative_support"}
        and evidence_usage_score < 0.3
        and not uncertainty_reduction
        and not malignant_flag_support
        and evidence_strength_score <= 0.6
    ):
        negative += 1.1
        harmful_reasons.append("low_signal_on_failed_case")

    if (
        total_selected >= 10
        and skill_rank >= max(7, int(total_selected * 0.6))
        and evidence_usage_score <= 0.5
        and not uncertainty_reduction
        and not malignant_flag_support
    ):
        negative += 1.2
        harmful_reasons.append("excess_redundancy_low_usage")
    if (
        total_selected >= 10
        and skill_rank >= max(7, int(total_selected * 0.6))
        and evidence_usage_score <= 0.3
        and case_failed
    ):
        negative += 0.8
        harmful_reasons.append("selection_tail_low_value")

    critical_harm = bool(
        regression
        or "false_conflict_signal" in harmful_reasons
        or ("misleading_risk_or_escalation" in harmful_reasons and case_failed)
        or (
            case_failed
            and evidence_usage_score <= 0.3
            and {"excess_redundancy_low_usage", "selection_tail_low_value"} <= set(harmful_reasons)
        )
    )

    if critical_harm or (negative - positive) >= 1.2 or (case_failed and evidence_usage_score <= 0.2 and positive < 1.6):
        return "harmful", _dedupe_strings(reasons), _dedupe_strings(harmful_reasons), round(evidence_usage_score, 3)
    if (positive - negative) >= 1.8 and (
        correct is True
        or float(delta.get("correct_delta", 0) or 0) > 0
        or float(delta.get("topk_hit_delta", 0) or 0) > 0
        or float(delta.get("malignant_recall_delta", 0) or 0) > 0
    ):
        return "helpful", _dedupe_strings(reasons), [], round(evidence_usage_score, 3)
    return "partially_helpful", _dedupe_strings(reasons), _dedupe_strings(harmful_reasons if negative >= 2.0 else []), round(evidence_usage_score, 3)


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


def _estimate_evidence_usage_score(
    *,
    record: dict[str, Any],
    skill_name: str,
    output: dict[str, Any],
    recommendation_type: str,
    contradiction_detected: bool,
    malignant_flag_support: bool,
    uncertainty_reduction: bool,
) -> float:
    score = 0.0
    referenced = [str(item).strip() for item in output.get("referenced_experiences", []) if str(item).strip()]
    if referenced:
        score += 0.3
    evidence_bundle = record.get("evidence_bundle", {})
    uncertainty_summary = dict(evidence_bundle.get("uncertainty_summary", {}))
    risk_flags = list(evidence_bundle.get("risk_flags", []) or record.get("reflection_summary", {}).get("case_outcome", {}).get("risk_flags", []))
    if recommendation_type == "risk_signal" and risk_flags:
        score += 1.0
    if recommendation_type in {"uncertainty_signal", "gap_signal"} and (
        uncertainty_summary.get("reasons") or uncertainty_summary.get("missing_information")
    ):
        score += 1.0
    if recommendation_type == "conflict_signal" and contradiction_detected and _contradiction_count(evidence_bundle.get("contradiction_summary", {})) > 0:
        score += 1.0
    if uncertainty_reduction and uncertainty_summary.get("uncertainty_level") in {"high", "medium"}:
        score += 1.0
    if malignant_flag_support:
        score += 0.5
    if skill_name in {"differential_compare_skill", "lesion_description_structuring_skill", "exclusion_reasoning_skill"}:
        score += 0.5
    return min(3.0, score)


def _uncertainty_rank(level: str) -> int:
    mapping = {"high": 3, "medium": 2, "low": 1}
    return mapping.get(str(level).strip().lower(), 2)


def _contradiction_count(summary: dict[str, Any]) -> int:
    total = 0
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts"):
        total += len(summary.get(field_name, []) or [])
    return total


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
