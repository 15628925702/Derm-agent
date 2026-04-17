from __future__ import annotations

from typing import Any

from agent.labels import labels_match
from agent.state import CaseState
from cognition.cognition_state import CognitionState
from memory.experience_transform import (
    build_abstract_experiences,
    build_case_outcome,
    build_reflection_extract,
    build_skill_assessments,
    build_tactical_experiences,
    build_writeback_bundle,
)


def build_reflection(state: CaseState, cognition: CognitionState) -> dict[str, Any]:
    useful_skills = [name for name, output in state.skill_outputs.items() if output]
    uncertainty_level = str(
        state.uncertainty.get("uncertainty_level", state.perception.get("uncertainty", {}).get("level", "unknown"))
    ).lower()
    confidence_text = str(state.final_diagnosis.get("confidence", "unknown")).lower()
    predicted = state.final_diagnosis.get("final_diagnosis")
    reference = state.case_input.reference_label

    detected_errors: list[str] = []
    confusion_pair: str | None = None
    experience_type = "raw_case_experience"

    label_match = labels_match(
        predicted,
        reference,
        dataset_name=state.case_input.dataset_name,
        label_space_id=state.case_input.label_space_id,
        metadata=state.case_input.metadata,
    )
    if predicted and reference and label_match is False:
        detected_errors.append("final_diagnosis_mismatch_reference")
        confusion_pair = f"{predicted}->{reference}"
        experience_type = "confusion_experience"
    elif uncertainty_level == "high" or confidence_text in {"low", "moderate"}:
        experience_type = "hard_case_experience"

    summary_parts = [
        f"Perception summary: {_perception_summary(state)}",
        f"Useful skills: {', '.join(useful_skills) if useful_skills else 'none'}",
        f"Final answer: {predicted or 'unavailable'}",
        f"Uncertainty: {uncertainty_level}",
    ]
    if state.risk_flags:
        summary_parts.append(f"Risk flags: {', '.join(state.risk_flags)}")
    if detected_errors:
        summary_parts.append(f"Detected errors: {', '.join(detected_errors)}")
    reasoning_summary = " | ".join(summary_parts)

    case_outcome = build_case_outcome(
        state=state,
        detected_errors=detected_errors,
        confusion_pair=confusion_pair,
    )
    skill_assessments = build_skill_assessments(state, case_outcome)
    tactical_experiences = build_tactical_experiences(
        state=state,
        case_outcome=case_outcome,
        skill_assessments=skill_assessments,
    )
    abstract_experiences = build_abstract_experiences(
        state=state,
        case_outcome=case_outcome,
        skill_assessments=skill_assessments,
    )
    helpful_skills = [
        assessment["skill_name"]
        for assessment in skill_assessments
        if assessment.get("helpfulness") in {"success", "partially_helpful"}
    ]
    cognition_update = {
        "total_cases_increment": 1,
        "failed_cases_increment": 1 if detected_errors else 0,
        "confusion_cases_increment": 1 if experience_type == "confusion_experience" else 0,
        "hard_cases_increment": 1 if experience_type == "hard_case_experience" else 0,
        "confusion_pair": confusion_pair,
        "preferred_skills": helpful_skills or useful_skills,
    }
    reflection_extract = build_reflection_extract(
        case_outcome=case_outcome,
        skill_assessments=skill_assessments,
        tactical_experiences=tactical_experiences,
        abstract_experiences=abstract_experiences,
    )
    writeback_bundle = build_writeback_bundle(
        state=state,
        case_outcome=case_outcome,
        skill_assessments=skill_assessments,
        reflection_extract=reflection_extract,
        cognition_update=cognition_update,
    )

    return {
        "reasoning_summary": reasoning_summary,
        "useful_skills": useful_skills,
        "detected_errors": detected_errors,
        "experience_type": experience_type,
        "diagnosis_correct": label_match is True,
        "write_experience": _should_write_experience(experience_type, useful_skills),
        "case_outcome": case_outcome,
        "skill_assessments": skill_assessments,
        "reflection_extract": reflection_extract,
        "writeback_bundle": writeback_bundle,
        "cognition_update": cognition_update,
    }


def apply_cognition_update(cognition: CognitionState, reflection: dict[str, Any], state: CaseState | None = None) -> CognitionState:
    update = reflection.get("cognition_update", {})
    cognition.update_failure_statistics(
        total_cases_increment=int(update.get("total_cases_increment", 0)),
        failed_cases_increment=int(update.get("failed_cases_increment", 0)),
        confusion_cases_increment=int(update.get("confusion_cases_increment", 0)),
        hard_cases_increment=int(update.get("hard_cases_increment", 0)),
    )
    cognition.update_known_confusion_patterns(update.get("confusion_pair"))
    cognition.update_preferred_skills(update.get("preferred_skills") or [])
    cognition.update_skill_statistics(reflection.get("writeback_bundle", {}).get("skill_stats_update", []))
    if state is not None:
        cognition.update_workflow_preferences(
            workflow_context=state.case_input.workflow_context,
            skills_used=list(state.skill_outputs.keys()),
            correct=bool(reflection.get("diagnosis_correct", False)),
        )
    return cognition


def _perception_summary(state: CaseState) -> str:
    if isinstance(state.perception.get("image_summary"), str) and state.perception.get("image_summary"):
        return str(state.perception["image_summary"])
    return str(state.perception)


def _should_write_experience(experience_type: str, useful_skills: list[str]) -> bool:
    if experience_type in {"hard_case_experience", "confusion_experience"}:
        return True
    return bool(useful_skills)
