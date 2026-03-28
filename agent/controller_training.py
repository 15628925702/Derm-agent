from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.skill_helpfulness_analyzer import normalize_skill_assessments_for_record


DEFAULT_OUTPUT_DIR = Path("/root/DermAgent/outputs/controller_training_data")


@dataclass
class ControllerTrainingExample:
    case_id: str
    dataset_name: str
    policy_id: str
    planner_type: str
    planner_version: str
    controller_family: str
    state_features: dict[str, Any]
    available_skill_candidates: list[str]
    selected_skills: list[str]
    selection_reasons: list[dict[str, Any]]
    outcome: dict[str, Any]
    future_training_views: dict[str, Any] = field(default_factory=dict)
    source_record_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_controller_training_example_from_record(
    record: dict[str, Any],
    *,
    source_record_path: str = "",
) -> dict[str, Any]:
    case_id = str(record.get("case_id", "")).strip()
    qwen_initial = record.get("qwen_initial", {})
    input_summary = record.get("input_summary", {})
    clinical_metadata = input_summary.get("clinical_metadata", {})
    skill_retrieval = record.get("skill_retrieval", {})
    planner = record.get("planner_decision", {})
    policy_snapshot = record.get("policy_snapshot", {})
    retrieval_bundle = record.get("retrieval_bundle", {})
    evidence_bundle = record.get("evidence_bundle", {})
    evaluation = record.get("evaluation", {})
    reflection = record.get("reflection_summary", {})
    skill_assessments = normalize_skill_assessments_for_record(record)
    uncertainty_summary = evidence_bundle.get("uncertainty_summary", {})
    contradiction_summary = evidence_bundle.get("contradiction_summary", {})
    planner_rationale = evidence_bundle.get("planner_rationale", {})

    helpful_skills = [
        str(item.get("skill_name", "")).strip()
        for item in skill_assessments
        if str(item.get("impact", "")).strip().lower() == "helpful"
    ]
    partially_helpful_skills = [
        str(item.get("skill_name", "")).strip()
        for item in skill_assessments
        if str(item.get("impact", "")).strip().lower() == "partially_helpful"
    ]
    harmful_skills = [
        str(item.get("skill_name", "")).strip()
        for item in skill_assessments
        if str(item.get("impact", "")).strip().lower() == "harmful"
    ]

    confusion_tags = _controller_confusion_tags(record)
    metadata_summary = {
        field_name: str(clinical_metadata.get(field_name, "")).strip()
        for field_name in ("region", "age", "diameter_1", "diameter_2", "grew", "changed", "bleed", "itch", "hurt", "elevation")
        if str(clinical_metadata.get(field_name, "")).strip()
    }
    retrieval_summary = {
        "experience_layers": {
            "raw_case_count": len(retrieval_bundle.get("after_skills", {}).get("raw_case_results", [])),
            "tactical_count": len(retrieval_bundle.get("after_skills", {}).get("tactical_results", [])),
            "abstract_count": len(retrieval_bundle.get("after_skills", {}).get("abstract_results", [])),
        },
        "top_experience_refs": _top_experience_refs(retrieval_bundle.get("after_skills", {}).get("aggregator_summary", [])),
        "skill_retrieval": {
            "candidate_skill_names": list(skill_retrieval.get("candidate_skill_names", [])),
            "trigger_hits": dict(skill_retrieval.get("trigger_hits", {})),
            "retrieval_scores": dict(skill_retrieval.get("retrieval_scores", {})),
        },
    }

    selection_reasons = [
        {
            "skill_name": skill_name,
            "reasons": [str(item).strip() for item in planner.get("selection_reasons", {}).get(skill_name, []) if str(item).strip()],
            "decision_trace": _planner_trace_for_skill(planner.get("decision_trace", []), skill_name),
        }
        for skill_name in planner.get("selected_skills", [])
        if str(skill_name).strip()
    ]
    available_skill_candidates = list(
        planner.get("available_skill_candidates")
        or skill_retrieval.get("candidate_skill_names", [])
    )
    selected_skills = [str(item).strip() for item in planner.get("selected_skills", []) if str(item).strip()]
    sparse_targets = build_sparse_controller_targets(
        available_skill_candidates=available_skill_candidates,
        selected_skills=selected_skills,
        helpful_skills=helpful_skills,
        partially_helpful_skills=partially_helpful_skills,
        harmful_skills=harmful_skills,
        evaluation=evaluation,
    )

    reward = _bandit_reward(evaluation, helpful_skills, harmful_skills)
    example = ControllerTrainingExample(
        case_id=case_id,
        dataset_name=str(record.get("dataset_name", "unknown_dataset")).strip() or "unknown_dataset",
        policy_id=str(planner.get("policy_id") or policy_snapshot.get("policy_id") or "unknown_policy").strip(),
        planner_type=str(planner.get("planner_type", "unknown")),
        planner_version=str(planner.get("planner_version", "unknown")),
        controller_family=str(planner.get("controller_family", "heuristic")),
        state_features={
            "initial_ddx": [str(item).strip() for item in qwen_initial.get("ddx_candidates", []) if str(item).strip()],
            "uncertainty": {
                "initial_level": str(qwen_initial.get("uncertainty", {}).get("level", "unknown")).lower(),
                "final_level": str(uncertainty_summary.get("uncertainty_level", "unknown")).lower(),
                "reasons": [str(item).strip() for item in uncertainty_summary.get("reasons", [])[:5] if str(item).strip()],
                "missing_information": [
                    str(item).strip() for item in uncertainty_summary.get("missing_information", [])[:5] if str(item).strip()
                ],
            },
            "metadata_summary": metadata_summary,
            "risk_flags": list(reflection.get("case_outcome", {}).get("risk_flags", [])),
            "confusion_tags": confusion_tags,
            "retrieval_summary": retrieval_summary,
            "planner_rationale": planner_rationale,
            "contradiction_summary": {
                "contradiction_count": _contradiction_count(contradiction_summary),
                "has_contradiction": _contradiction_count(contradiction_summary) > 0,
            },
        },
        available_skill_candidates=available_skill_candidates,
        selected_skills=selected_skills,
        selection_reasons=selection_reasons,
        outcome={
            "final_correct": evaluation.get("correct"),
            "delta_vs_baseline": dict(evaluation.get("agent_vs_baseline_delta", {})),
            "helpful_skills": helpful_skills,
            "partially_helpful_skills": partially_helpful_skills,
            "harmful_skills": harmful_skills,
            "primary_positive_skills": list(sparse_targets["primary_positive_skills"]),
            "weak_positive_skills": list(sparse_targets["weak_positive_skills"]),
            "explicit_negative_skills": list(sparse_targets["explicit_negative_skills"]),
            "target_skill_scores": dict(sparse_targets["target_skill_scores"]),
            "target_k": int(sparse_targets["target_k"]),
            "case_status": reflection.get("case_outcome", {}).get("status"),
            "reward": reward,
        },
        future_training_views={
            "supervised_controller": {
                "target_selected_skills": selected_skills,
                "target_helpful_skills": helpful_skills,
                "primary_positive_skills": list(sparse_targets["primary_positive_skills"]),
                "weak_positive_skills": list(sparse_targets["weak_positive_skills"]),
                "explicit_negative_skills": list(sparse_targets["explicit_negative_skills"]),
                "target_skill_scores": dict(sparse_targets["target_skill_scores"]),
                "target_k": int(sparse_targets["target_k"]),
            },
            "contextual_bandit": {
                "action_skills": selected_skills,
                "reward": reward,
            },
            "lightweight_policy_scorer": {
                "candidate_skill_scores": dict(skill_retrieval.get("retrieval_scores", {})),
                "planner_decision_trace": planner.get("decision_trace", []),
            },
        },
        source_record_path=source_record_path,
    )
    return example.to_dict()


def export_controller_training_data(
    records: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        if dataset_name and str(record.get("dataset_name", "")).strip() != dataset_name:
            continue
        examples.append(
            build_controller_training_example_from_record(
                record,
                source_record_path=str(record.get("_source_record_path", "")),
            )
        )
    summary = {
        "example_count": len(examples),
        "dataset_name": dataset_name,
        "dataset_counts": dict(Counter(str(item.get("dataset_name", "unknown_dataset")) for item in examples)),
        "planner_type_counts": dict(Counter(str(item.get("planner_type", "unknown")) for item in examples)),
        "controller_family_counts": dict(Counter(str(item.get("controller_family", "unknown")) for item in examples)),
        "correct_count": sum(1 for item in examples if item.get("outcome", {}).get("final_correct") is True),
        "avg_target_k": round(
            sum(int(item.get("outcome", {}).get("target_k", 0) or 0) for item in examples) / max(len(examples), 1),
            4,
        ),
        "avg_selected_count": round(
            sum(len(item.get("selected_skills", [])) for item in examples) / max(len(examples), 1),
            4,
        ),
        "avg_primary_positive_count": round(
            sum(len(item.get("outcome", {}).get("primary_positive_skills", [])) for item in examples) / max(len(examples), 1),
            4,
        ),
        "helpful_skill_frequency": dict(
            Counter(
                skill_name
                for item in examples
                for skill_name in item.get("outcome", {}).get("helpful_skills", [])
                if str(skill_name).strip()
            )
        ),
        "partially_helpful_skill_frequency": dict(
            Counter(
                skill_name
                for item in examples
                for skill_name in item.get("outcome", {}).get("partially_helpful_skills", [])
                if str(skill_name).strip()
            )
        ),
        "harmful_skill_frequency": dict(
            Counter(
                skill_name
                for item in examples
                for skill_name in item.get("outcome", {}).get("harmful_skills", [])
                if str(skill_name).strip()
            )
        ),
    }
    return examples, summary


def save_controller_training_data(
    examples: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    examples_path = root / "controller_training_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    summary_path = root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return {
        "examples_path": str(examples_path),
        "summary_path": str(summary_path),
    }


def _controller_confusion_tags(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    skill_retrieval = record.get("skill_retrieval", {})
    query_summary = skill_retrieval.get("query_summary", {})
    confusion_pair = str(query_summary.get("confusion_pair", "")).strip()
    if confusion_pair:
        tags.append(confusion_pair)
    reflection_confusion = str(record.get("reflection_summary", {}).get("case_outcome", {}).get("confusion_pair", "")).strip()
    if reflection_confusion:
        tags.append(reflection_confusion)
    return list(dict.fromkeys(tag for tag in tags if tag))


def _top_experience_refs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records[:5]:
        results.append(
            {
                "source_id": str(record.get("source_id", "")).strip(),
                "source_layer": str(record.get("source_layer", "")).strip(),
                "experience_type": str(record.get("experience_type", "")).strip(),
                "retrieval_score": int(record.get("retrieval_score", 0) or 0),
            }
        )
    return results


def _planner_trace_for_skill(decision_trace: list[dict[str, Any]], skill_name: str) -> dict[str, Any]:
    for item in decision_trace:
        if str(item.get("skill_name", "")).strip() == skill_name:
            return deepcopy(item)
    return {}


def _bandit_reward(
    evaluation: dict[str, Any],
    helpful_skills: list[str],
    harmful_skills: list[str],
) -> float:
    reward = 0.0
    if evaluation.get("correct") is True:
        reward += 1.0
    elif evaluation.get("correct") is False:
        reward -= 1.0
    delta = evaluation.get("agent_vs_baseline_delta", {})
    reward += 0.5 * float(delta.get("correct_delta", 0) or 0)
    reward += 0.25 * float(delta.get("topk_hit_delta", 0) or 0)
    reward += 0.25 * float(delta.get("malignant_recall_delta", 0) or 0)
    reward += 0.1 * len(helpful_skills)
    reward -= 0.1 * len(harmful_skills)
    return round(reward, 4)


def _contradiction_count(summary: dict[str, Any]) -> int:
    total = 0
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts"):
        total += len(summary.get(field_name, []) or [])
    return total


def build_sparse_controller_targets(
    *,
    available_skill_candidates: list[str],
    selected_skills: list[str],
    helpful_skills: list[str],
    partially_helpful_skills: list[str],
    harmful_skills: list[str],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    available = [str(item).strip() for item in available_skill_candidates if str(item).strip()]
    selected = [str(item).strip() for item in selected_skills if str(item).strip()]
    helpful = [str(item).strip() for item in helpful_skills if str(item).strip() and str(item).strip() in set(available)]
    partial = [
        str(item).strip()
        for item in partially_helpful_skills
        if str(item).strip() and str(item).strip() in set(available)
    ]
    harmful = [str(item).strip() for item in harmful_skills if str(item).strip() and str(item).strip() in set(available)]

    delta = dict(evaluation.get("agent_vs_baseline_delta", {}) or {})
    case_improved = any(float(delta.get(field, 0) or 0) > 0 for field in ("correct_delta", "topk_hit_delta", "malignant_recall_delta"))
    case_not_worse = all(float(delta.get(field, 0) or 0) >= 0 for field in ("correct_delta", "topk_hit_delta", "malignant_recall_delta"))
    final_correct = evaluation.get("correct") is True

    harmful_set = set(harmful)
    helpful_set = set(helpful)
    partial_set = set(partial)
    selected_neutral = [
        skill_name
        for skill_name in selected
        if skill_name in set(available) and skill_name not in harmful_set and skill_name not in helpful_set and skill_name not in partial_set
    ]

    partial_primary_budget = 2 if (final_correct or case_improved) else 1
    partial_weak_budget = 2 if case_not_worse else 1
    partial_primary = partial[:partial_primary_budget]
    remaining_partial = [skill_name for skill_name in partial if skill_name not in set(partial_primary)]
    weak_positive = remaining_partial[:partial_weak_budget]
    if not weak_positive and (final_correct or case_improved):
        weak_positive = selected_neutral[:1]

    primary_positive = list(dict.fromkeys(helpful + partial_primary))
    if not primary_positive:
        primary_positive = list(dict.fromkeys(partial[: min(2, len(partial))]))
    if not primary_positive:
        primary_positive = selected_neutral[: min(2, len(selected_neutral))]
    if not primary_positive and selected:
        primary_positive = [skill_name for skill_name in selected[:1] if skill_name in set(available) and skill_name not in harmful_set]

    target_skill_scores: dict[str, float] = {skill_name: 0.0 for skill_name in available}
    for skill_name in helpful:
        target_skill_scores[skill_name] = 1.0
    for skill_name in partial_primary:
        target_skill_scores[skill_name] = max(target_skill_scores.get(skill_name, 0.0), 0.7)
    for skill_name in weak_positive:
        target_skill_scores[skill_name] = max(target_skill_scores.get(skill_name, 0.0), 0.35)
    min_target_k = min(max(len(available), 1), 4)
    max_target_k = min(max(len(available), 1), 8)
    target_k = len(primary_positive) + (1 if weak_positive else 0)
    target_k = max(min_target_k, target_k)
    target_k = min(max_target_k, target_k)
    overflow_selected = [
        skill_name
        for skill_name in selected[target_k:]
        if skill_name in set(available)
        and skill_name not in helpful_set
        and skill_name not in set(primary_positive)
        and skill_name not in set(weak_positive)
    ]
    explicit_negative = list(dict.fromkeys(harmful + overflow_selected))

    return {
        "primary_positive_skills": list(dict.fromkeys(skill_name for skill_name in primary_positive if skill_name not in harmful_set)),
        "weak_positive_skills": list(dict.fromkeys(skill_name for skill_name in weak_positive if skill_name not in harmful_set)),
        "explicit_negative_skills": explicit_negative,
        "target_skill_scores": {key: round(float(value), 4) for key, value in target_skill_scores.items()},
        "target_k": int(target_k),
    }
