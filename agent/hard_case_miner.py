from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.label_space import canonicalize_label


DEFAULT_MIN_IMPORTANCE = 3.0
DEFAULT_MAX_PER_CLUSTER = 20


@dataclass
class HardCaseRecord:
    hard_case_id: str
    case_id: str
    dataset_name: str
    hardness_reason: list[str]
    failure_type: str
    involved_skills: list[str]
    confusion_tags: list[str]
    baseline_result: dict[str, Any]
    agent_result: dict[str, Any]
    ground_truth: dict[str, Any]
    importance_score: float
    cluster_key: str
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)
    source_record_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_execution_records(records_root: str | Path) -> list[dict[str, Any]]:
    root = Path(records_root)
    record_files = sorted(root.rglob("case_execution_record.json"))
    records: list[dict[str, Any]] = []
    if record_files:
        for path in record_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("case_id"):
                payload["_source_record_path"] = str(path)
                records.append(payload)
        return _deduplicate_records(records)

    for path in sorted(root.rglob("case_execution_records.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("case_id"):
                payload["_source_record_path"] = f"{path}:{line_number}"
                records.append(payload)
    return _deduplicate_records(records)


def mine_hard_cases(
    execution_records: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
    label: str | None = None,
    confusion_pair: str | None = None,
    skill_name: str | None = None,
    min_importance: float = DEFAULT_MIN_IMPORTANCE,
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = [build_hard_case_candidate(record) for record in execution_records]
    cluster_counts = Counter(item["cluster_key"] for item in prepared if _is_failure_like(item))
    case_failure_counts = Counter(item["case_id"] for item in prepared if _is_failure_like(item))

    scored: list[dict[str, Any]] = []
    for item in prepared:
        enriched = enrich_hard_case_candidate(
            item,
            repeated_cluster_count=int(cluster_counts.get(item["cluster_key"], 0)),
            repeated_case_failure_count=int(case_failure_counts.get(item["case_id"], 0)),
        )
        if not passes_filters(
            enriched,
            dataset_name=dataset_name,
            label=label,
            confusion_pair=confusion_pair,
            skill_name=skill_name,
        ):
            continue
        if float(enriched["importance_score"]) < float(min_importance):
            continue
        scored.append(enriched)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        grouped[item["cluster_key"]].append(item)

    limited_results: list[dict[str, Any]] = []
    for cluster_key, items in grouped.items():
        items.sort(
            key=lambda record: (
                float(record.get("importance_score", 0.0)),
                str(record.get("timestamp", "")),
                str(record.get("case_id", "")),
            ),
            reverse=True,
        )
        limited_results.extend(items[:max_per_cluster])

    limited_results.sort(
        key=lambda record: (
            float(record.get("importance_score", 0.0)),
            str(record.get("failure_type", "")),
            str(record.get("case_id", "")),
        ),
        reverse=True,
    )
    summary = build_hard_case_summary(
        execution_record_count=len(execution_records),
        hard_cases=limited_results,
        all_candidates=scored,
        filters={
            "dataset_name": dataset_name,
            "label": label,
            "confusion_pair": confusion_pair,
            "skill_name": skill_name,
            "min_importance": min_importance,
            "max_per_cluster": max_per_cluster,
        },
    )
    return limited_results, summary


def build_hard_case_candidate(record: dict[str, Any]) -> dict[str, Any]:
    ground_truth = record.get("ground_truth", {})
    evaluation = record.get("evaluation", {})
    qwen_initial = record.get("qwen_initial", {})
    qwen_final = record.get("qwen_final", {})
    evidence_bundle = record.get("evidence_bundle", {})
    reflection_summary = record.get("reflection_summary", {})
    contradiction_summary = evidence_bundle.get("contradiction_summary", {})
    uncertainty_summary = evidence_bundle.get("uncertainty_summary", {})
    planner_decision = record.get("planner_decision", {})
    case_outcome = reflection_summary.get("case_outcome", {})
    baseline_qwen = record.get("baseline_qwen") if isinstance(record.get("baseline_qwen"), dict) else {}

    dataset_name = str(record.get("dataset_name", "")).strip() or None
    agent_correct = evaluation.get("correct")
    baseline_correct = evaluation.get("baseline_correct")
    malignant_recall_hit = evaluation.get("malignant_recall_hit")
    baseline_malignant_recall_hit = evaluation.get("baseline_malignant_recall_hit")

    initial_ddx = [
        canonicalize_label(item, dataset_name=dataset_name)
        for item in qwen_initial.get("ddx_candidates", [])
        if canonicalize_label(item, dataset_name=dataset_name)
    ]
    final_label = canonicalize_label(qwen_final.get("final_diagnosis"), dataset_name=dataset_name)
    volatility = bool(final_label and initial_ddx and final_label not in initial_ddx)

    contradiction_count = _contradiction_count(contradiction_summary)
    uncertainty_level = str(
        uncertainty_summary.get("uncertainty_level", case_outcome.get("uncertainty_level", "unknown"))
    ).lower()
    confusion_tags = _extract_confusion_tags(record)

    failure_type = _primary_failure_type(
        agent_correct=agent_correct,
        baseline_correct=baseline_correct,
        uncertainty_level=uncertainty_level,
        contradiction_count=contradiction_count,
        confusion_tags=confusion_tags,
        volatility=volatility,
    )
    reasons = _base_hardness_reasons(
        agent_correct=agent_correct,
        baseline_correct=baseline_correct,
        uncertainty_level=uncertainty_level,
        contradiction_count=contradiction_count,
        confusion_tags=confusion_tags,
        volatility=volatility,
        malignant_recall_hit=malignant_recall_hit,
        baseline_malignant_recall_hit=baseline_malignant_recall_hit,
    )
    involved_skills = [
        str(item).strip()
        for item in (record.get("selected_skills") or planner_decision.get("selected_skills", []))
        if str(item).strip()
    ]
    if not involved_skills:
        involved_skills = [str(name).strip() for name in record.get("skill_outputs", {}).keys() if str(name).strip()]

    cluster_key = "|".join(
        [
            failure_type,
            confusion_tags[0] if confusion_tags else "no_confusion",
            str(ground_truth.get("canonical_label") or "unknown_gt"),
        ]
    )
    return {
        "case_id": record.get("case_id"),
        "dataset_name": record.get("dataset_name", "unknown_dataset"),
        "ground_truth": {
            "raw_label": ground_truth.get("raw_label"),
            "canonical_label": ground_truth.get("canonical_label"),
            "malignant_flag": ground_truth.get("malignant_flag"),
        },
        "baseline_result": {
            "correct": baseline_correct,
            "topk_hit": evaluation.get("baseline_topk_hit"),
            "malignant_recall_hit": baseline_malignant_recall_hit,
            "final_diagnosis": baseline_qwen.get("final_diagnosis"),
        },
        "agent_result": {
            "correct": agent_correct,
            "topk_hit": evaluation.get("topk_hit"),
            "malignant_recall_hit": malignant_recall_hit,
            "final_diagnosis": qwen_final.get("final_diagnosis"),
            "uncertainty_level": uncertainty_level,
        },
        "failure_type": failure_type,
        "hardness_reason": reasons,
        "involved_skills": involved_skills,
        "confusion_tags": confusion_tags,
        "cluster_key": cluster_key,
        "timestamp": record.get("timestamp", ""),
        "source_record_path": record.get("_source_record_path", ""),
        "evidence_snapshot": {
            "initial_ddx": [item for item in qwen_initial.get("ddx_candidates", []) if str(item).strip()],
            "final_diagnosis": qwen_final.get("final_diagnosis"),
            "contradiction_count": contradiction_count,
            "uncertainty_level": uncertainty_level,
            "selected_skills": involved_skills,
            "experience_type": reflection_summary.get("experience_type"),
        },
        "_volatility": volatility,
        "_contradiction_count": contradiction_count,
    }


def enrich_hard_case_candidate(
    candidate: dict[str, Any],
    *,
    repeated_cluster_count: int,
    repeated_case_failure_count: int,
) -> dict[str, Any]:
    enriched = deepcopy(candidate)
    reasons = list(enriched.get("hardness_reason", []))
    importance = 0.0
    failure_type = str(enriched.get("failure_type", "unknown"))

    if failure_type == "agent_regression_vs_baseline":
        importance += 5.0
    elif failure_type == "shared_failure":
        importance += 4.0
    elif failure_type == "agent_rescue_vs_baseline":
        importance += 3.0
    elif failure_type == "agent_failure_no_baseline":
        importance += 3.0
    elif failure_type == "high_uncertainty_case":
        importance += 1.5
    elif failure_type == "confusion_driven_case":
        importance += 2.0
    elif failure_type in {"diagnostic_volatility_case", "contradiction_heavy_case"}:
        importance += 1.5

    if enriched.get("_volatility"):
        importance += 1.5
    uncertainty_level = str(enriched.get("agent_result", {}).get("uncertainty_level", "unknown")).lower()
    if uncertainty_level == "high":
        importance += 1.5
    contradiction_count = int(enriched.get("_contradiction_count", 0))
    if contradiction_count >= 3:
        importance += 1.5
    elif contradiction_count >= 1:
        importance += 0.5

    if enriched.get("confusion_tags"):
        importance += 1.0

    malignant_recall_hit = enriched.get("agent_result", {}).get("malignant_recall_hit")
    ground_truth_malignant = enriched.get("ground_truth", {}).get("malignant_flag")
    if ground_truth_malignant is True and malignant_recall_hit is False:
        importance += 1.0
        reasons.append("malignant_recall_miss")

    if repeated_cluster_count >= 2:
        importance += min(2.0, 0.75 * (repeated_cluster_count - 1))
        reasons.append("repeated_failure_cluster")
    if repeated_case_failure_count >= 2:
        importance += min(1.5, 0.5 * (repeated_case_failure_count - 1))
        reasons.append("repeated_case_failure")

    enriched["hardness_reason"] = _dedupe_strings(reasons)
    enriched["importance_score"] = round(importance, 2)
    enriched["hard_case_id"] = _stable_id(
        enriched["case_id"],
        enriched["failure_type"],
        enriched["cluster_key"],
        enriched["timestamp"],
    )
    enriched.pop("_volatility", None)
    enriched.pop("_contradiction_count", None)
    return HardCaseRecord(
        hard_case_id=enriched["hard_case_id"],
        case_id=str(enriched["case_id"]),
        dataset_name=str(enriched.get("dataset_name", "unknown_dataset")),
        hardness_reason=list(enriched.get("hardness_reason", [])),
        failure_type=str(enriched.get("failure_type", "unknown")),
        involved_skills=list(enriched.get("involved_skills", [])),
        confusion_tags=list(enriched.get("confusion_tags", [])),
        baseline_result=dict(enriched.get("baseline_result", {})),
        agent_result=dict(enriched.get("agent_result", {})),
        ground_truth=dict(enriched.get("ground_truth", {})),
        importance_score=float(enriched.get("importance_score", 0.0)),
        cluster_key=str(enriched.get("cluster_key", "")),
        evidence_snapshot=dict(enriched.get("evidence_snapshot", {})),
        source_record_path=str(enriched.get("source_record_path", "")),
    ).to_dict()


def passes_filters(
    record: dict[str, Any],
    *,
    dataset_name: str | None,
    label: str | None,
    confusion_pair: str | None,
    skill_name: str | None,
) -> bool:
    if dataset_name and str(record.get("dataset_name", "")).strip() != str(dataset_name).strip():
        return False
    if label:
        target_label = canonicalize_label(label, dataset_name=str(record.get("dataset_name", "")).strip() or None) or str(label).strip()
        ground_truth_label = record.get("ground_truth", {}).get("canonical_label") or record.get("ground_truth", {}).get("raw_label")
        if str(ground_truth_label).strip() != str(target_label).strip():
            return False
    if confusion_pair:
        lowered = str(confusion_pair).strip().lower()
        if not any(lowered in str(tag).lower() for tag in record.get("confusion_tags", [])):
            return False
    if skill_name and str(skill_name).strip() not in record.get("involved_skills", []):
        return False
    return True


def build_hard_case_summary(
    *,
    execution_record_count: int,
    hard_cases: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    filters: dict[str, Any],
) -> dict[str, Any]:
    by_failure_type = Counter(str(item.get("failure_type", "unknown")) for item in hard_cases)
    by_confusion_tag = Counter(tag for item in hard_cases for tag in item.get("confusion_tags", []))
    by_skill = Counter(skill for item in hard_cases for skill in item.get("involved_skills", []))
    by_cluster = Counter(str(item.get("cluster_key", "")) for item in hard_cases)
    by_ground_truth_label = Counter(
        str(item.get("ground_truth", {}).get("canonical_label") or item.get("ground_truth", {}).get("raw_label") or "unknown")
        for item in hard_cases
    )

    cluster_examples: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in hard_cases:
        grouped[str(item.get("cluster_key", ""))].append(item)
    for cluster_key, items in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        items = sorted(items, key=lambda record: float(record.get("importance_score", 0.0)), reverse=True)
        cluster_examples.append(
            {
                "cluster_key": cluster_key,
                "count": len(items),
                "average_importance": round(
                    sum(float(record.get("importance_score", 0.0)) for record in items) / len(items),
                    2,
                ),
                "example_case_ids": [record.get("case_id") for record in items[:5]],
                "failure_type": items[0].get("failure_type"),
                "confusion_tags": items[0].get("confusion_tags", []),
            }
        )

    return {
        "total_execution_records_scanned": execution_record_count,
        "candidate_count_after_filters": len(all_candidates),
        "hard_case_count": len(hard_cases),
        "filters": dict(filters),
        "counts": {
            "by_failure_type": dict(by_failure_type),
            "by_confusion_tag": dict(by_confusion_tag),
            "by_skill": dict(by_skill.most_common(20)),
            "by_ground_truth_label": dict(by_ground_truth_label),
            "by_cluster": dict(by_cluster),
        },
        "clusters": cluster_examples,
        "top_hard_cases": [
            {
                "hard_case_id": item.get("hard_case_id"),
                "case_id": item.get("case_id"),
                "failure_type": item.get("failure_type"),
                "importance_score": item.get("importance_score"),
                "hardness_reason": item.get("hardness_reason", []),
            }
            for item in sorted(hard_cases, key=lambda record: float(record.get("importance_score", 0.0)), reverse=True)[:20]
        ],
    }


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_case: dict[str, dict[str, Any]] = {}
    for record in records:
        dedup_key = _record_dedup_key(record)
        if not dedup_key:
            continue
        existing = best_by_case.get(dedup_key)
        if existing is None or _record_priority(record) > _record_priority(existing):
            best_by_case[dedup_key] = record
    return list(best_by_case.values())


def _record_dedup_key(record: dict[str, Any]) -> str:
    case_id = str(record.get("case_id", "")).strip()
    if not case_id:
        return ""
    state_versions = dict(record.get("state_versions", {}) or {})
    run_mode = str(state_versions.get("run_mode", "")).strip()
    data_split = str(state_versions.get("data_split", "")).strip()
    return "||".join([case_id, run_mode, data_split])


def _record_priority(record: dict[str, Any]) -> tuple[int, int, str, int]:
    structure_score = _record_structure_score(record)
    has_baseline = 1 if record.get("baseline_qwen") else 0
    timestamp = str(record.get("timestamp", ""))
    writeback_enabled = 1 if record.get("writeback_ops", {}).get("writeback_enabled") else 0
    return (structure_score, has_baseline, timestamp, writeback_enabled)


def _record_structure_score(record: dict[str, Any]) -> int:
    planner = record.get("planner_decision", {})
    skill_retrieval = record.get("skill_retrieval", {})
    controller_training_example = record.get("controller_training_example", {})
    score = 0
    if str(record.get("record_version", "")).strip().lower() == "v2":
        score += 4
    if isinstance(controller_training_example, dict) and controller_training_example.get("selected_skills"):
        score += 4
    if isinstance(skill_retrieval, dict) and skill_retrieval.get("candidate_skill_names"):
        score += 3
    if isinstance(planner, dict) and planner.get("available_skill_candidates"):
        score += 2
    if isinstance(planner, dict) and planner.get("controller_training_ready") is True:
        score += 1
    return score


def _primary_failure_type(
    *,
    agent_correct: bool | None,
    baseline_correct: bool | None,
    uncertainty_level: str,
    contradiction_count: int,
    confusion_tags: list[str],
    volatility: bool,
) -> str:
    if baseline_correct is True and agent_correct is False:
        return "agent_regression_vs_baseline"
    if baseline_correct is False and agent_correct is True:
        return "agent_rescue_vs_baseline"
    if baseline_correct is False and agent_correct is False:
        return "shared_failure"
    if baseline_correct is None and agent_correct is False:
        return "agent_failure_no_baseline"
    if uncertainty_level == "high":
        return "high_uncertainty_case"
    if confusion_tags:
        return "confusion_driven_case"
    if contradiction_count >= 3:
        return "contradiction_heavy_case"
    if volatility:
        return "diagnostic_volatility_case"
    return "supporting_hard_case"


def _base_hardness_reasons(
    *,
    agent_correct: bool | None,
    baseline_correct: bool | None,
    uncertainty_level: str,
    contradiction_count: int,
    confusion_tags: list[str],
    volatility: bool,
    malignant_recall_hit: bool | None,
    baseline_malignant_recall_hit: bool | None,
) -> list[str]:
    reasons: list[str] = []
    if baseline_correct is False and agent_correct is True:
        reasons.append("baseline_wrong_agent_correct")
    if baseline_correct is True and agent_correct is False:
        reasons.append("baseline_correct_agent_wrong")
    if baseline_correct is False and agent_correct is False:
        reasons.append("baseline_and_agent_both_wrong")
    if volatility:
        reasons.append("initial_final_shift_large")
    if uncertainty_level == "high":
        reasons.append("high_uncertainty")
    if contradiction_count >= 3:
        reasons.append("contradiction_heavy")
    elif contradiction_count >= 1:
        reasons.append("contradiction_present")
    if confusion_tags:
        reasons.append("clear_confusion_pair")
    if baseline_malignant_recall_hit is False and malignant_recall_hit is True:
        reasons.append("agent_recovers_malignant_signal")
    return _dedupe_strings(reasons)


def _extract_confusion_tags(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    reflection_summary = record.get("reflection_summary", {})
    case_outcome = reflection_summary.get("case_outcome", {})
    confusion_pair = case_outcome.get("confusion_pair")
    if confusion_pair:
        tags.append(str(confusion_pair))
    experience_type = str(reflection_summary.get("experience_type", "")).strip()
    if experience_type == "confusion_experience" and confusion_pair:
        parts = str(confusion_pair).split("->")
        if len(parts) == 2:
            tags.append(f"{parts[0].strip().lower()}_vs_{parts[1].strip().lower()}")
    return _dedupe_strings(tags)


def _contradiction_count(summary: dict[str, Any]) -> int:
    count = 0
    for field_name in ("contradictions", "metadata_conflicts", "reasoning_gaps"):
        values = summary.get(field_name, [])
        if isinstance(values, list):
            count += len([item for item in values if str(item).strip()])
    return count


def _is_failure_like(candidate: dict[str, Any]) -> bool:
    failure_type = str(candidate.get("failure_type", ""))
    return failure_type in {
        "agent_regression_vs_baseline",
        "shared_failure",
        "agent_failure_no_baseline",
        "high_uncertainty_case",
        "confusion_driven_case",
        "contradiction_heavy_case",
        "diagnostic_volatility_case",
    }


def _stable_id(case_id: str, failure_type: str, cluster_key: str, timestamp: str) -> str:
    payload = f"{case_id}|{failure_type}|{cluster_key}|{timestamp}"
    return f"hard_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


def _dedupe_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
