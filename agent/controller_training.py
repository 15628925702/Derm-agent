from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    skill_assessments = reflection.get("skill_assessments", [])
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
            "case_status": reflection.get("case_outcome", {}).get("status"),
            "reward": reward,
        },
        future_training_views={
            "supervised_controller": {
                "target_selected_skills": selected_skills,
                "target_helpful_skills": helpful_skills,
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
        "helpful_skill_frequency": dict(
            Counter(
                skill_name
                for item in examples
                for skill_name in item.get("outcome", {}).get("helpful_skills", [])
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
