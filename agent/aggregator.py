from __future__ import annotations

from typing import Any

from agent.evidence_calibrator import build_default_evidence_calibrator
from agent.state import CaseState


OBSERVATION_SKILLS = (
    "morphology_analysis_skill",
    "color_pattern_analysis_skill",
    "border_surface_analysis_skill",
    "distribution_analysis_skill",
    "lesion_description_structuring_skill",
    "temporal_evolution_skill",
)

COMPARISON_SKILLS = (
    "metadata_consistency_skill",
    "differential_compare_skill",
    "exclusion_reasoning_skill",
    "mel_nev_specialist_skill",
    "ack_scc_specialist_skill",
)

RISK_SKILLS = ("malignancy_risk_assessment_skill",)
UNCERTAINTY_SKILLS = ("uncertainty_assessment_skill", "information_gap_detection_skill", "contradiction_check_skill")
ESCALATION_SKILLS = ("escalation_recommendation_skill",)


def build_evidence_bundle(state: CaseState) -> dict[str, Any]:
    initial_perception_summary = _build_initial_perception_summary(state)
    retrieved_raw_cases_summary = _summarize_retrieval_records(state.retrieval_bundle.get("raw_case_results", []), top_k=3)
    retrieved_tactical_experiences_summary = _summarize_retrieval_records(
        state.retrieval_bundle.get("tactical_results", []),
        top_k=4,
    )
    retrieved_abstract_experiences_summary = _summarize_retrieval_records(
        state.retrieval_bundle.get("abstract_results", []),
        top_k=4,
    )
    contradiction_summary = _build_contradiction_summary(state)
    uncertainty_summary = _build_uncertainty_summary(state)
    information_gap_summary = _build_information_gap_summary(state)
    escalation_summary = _build_escalation_summary(state)
    planner_rationale = _build_planner_rationale(state)
    evidence_policy = _extract_evidence_policy(state)
    calibrator = build_default_evidence_calibrator(policy=evidence_policy)
    calibration = calibrator.calibrate(
        {
            "initial_perception_summary": initial_perception_summary,
            "skill_outputs": state.skill_outputs,
            "retrieved_raw_cases_summary": retrieved_raw_cases_summary,
            "retrieved_tactical_experiences_summary": retrieved_tactical_experiences_summary,
            "retrieved_abstract_experiences_summary": retrieved_abstract_experiences_summary,
            "risk_flags": state.risk_flags,
            "uncertainty_summary": uncertainty_summary,
            "contradiction_summary": contradiction_summary,
            "information_gap_summary": information_gap_summary,
            "escalation_summary": escalation_summary,
            "planner_rationale": planner_rationale,
            "skill_retrieval_scores": dict(state.skill_retrieval_bundle.get("retrieval_scores", {})),
            "policy": evidence_policy,
        }
    )
    serialized_evidence_text = _serialize_evidence(
        initial_perception_summary=initial_perception_summary,
        retrieved_raw_cases_summary=retrieved_raw_cases_summary,
        retrieved_tactical_experiences_summary=retrieved_tactical_experiences_summary,
        retrieved_abstract_experiences_summary=retrieved_abstract_experiences_summary,
        skill_outputs=state.skill_outputs,
        risk_flags=state.risk_flags,
        uncertainty_summary=uncertainty_summary,
        contradiction_summary=contradiction_summary,
        information_gap_summary=information_gap_summary,
        escalation_summary=escalation_summary,
        planner_rationale=planner_rationale,
        calibration=calibration.to_dict(),
    )

    return {
        "perception": state.perception,
        "retrieved_experience": state.retrieved_experience,
        "skill_outputs": state.skill_outputs,
        "risk_flags": state.risk_flags,
        "uncertainty": state.uncertainty,
        "notes": state.notes,
        "initial_perception_summary": initial_perception_summary,
        "retrieved_raw_cases_summary": retrieved_raw_cases_summary,
        "retrieved_tactical_experiences_summary": retrieved_tactical_experiences_summary,
        "retrieved_abstract_experiences_summary": retrieved_abstract_experiences_summary,
        "uncertainty_summary": uncertainty_summary,
        "contradiction_summary": contradiction_summary,
        "information_gap_summary": information_gap_summary,
        "escalation_summary": escalation_summary,
        "planner_rationale": planner_rationale,
        "serialized_evidence_text": serialized_evidence_text,
        "evidence_calibration_debug": calibration.to_dict() if evidence_policy.get("debug_output", True) else {},
    }


def _extract_evidence_policy(state: CaseState) -> dict[str, Any]:
    snapshot = state.policy_snapshot or {}
    return dict(snapshot.get("evidence_policy", {}) or {})


def _build_initial_perception_summary(state: CaseState) -> dict[str, Any]:
    perception = state.perception or {}
    uncertainty = perception.get("uncertainty", {}) if isinstance(perception.get("uncertainty", {}), dict) else {}
    notes = perception.get("notes", [])
    if not isinstance(notes, list):
        notes = [notes] if notes else []
    return {
        "image_summary": _clip_text(perception.get("image_summary", "")),
        "ddx_candidates": [str(item).strip() for item in perception.get("ddx_candidates", [])[:4] if str(item).strip()],
        "uncertainty_level": str(uncertainty.get("level", "unknown")).lower(),
        "uncertainty_reasons": [str(item).strip() for item in uncertainty.get("reasons", [])[:3] if str(item).strip()],
        "observation_notes": [str(item).strip() for item in notes[:4] if str(item).strip()],
    }


def _summarize_retrieval_records(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in records[:top_k]:
        summaries.append(
            {
                "source_id": str(record.get("source_id", "")).strip(),
                "source_layer": str(record.get("source_layer", "")).strip(),
                "experience_type": str(record.get("experience_type", "")).strip(),
                "case_id": str(record.get("case_id", "")).strip(),
                "confusion_pair": str(record.get("confusion_pair", "")).strip() or None,
                "perception_summary": _clip_text(record.get("perception_summary", ""), max_length=260),
                "learning_points": [str(item).strip() for item in record.get("learning_points", [])[:3] if str(item).strip()],
                "retrieval_score": round(float(record.get("retrieval_score", 0) or 0.0), 6),
                "base_retrieval_score": round(float(record.get("base_retrieval_score", record.get("retrieval_score", 0)) or 0.0), 6),
                "rerank_score": round(float(record.get("rerank_score", 0.0) or 0.0), 6),
                "rerank_probability": round(float(record.get("rerank_probability", 0.0) or 0.0), 6),
            }
        )
    return summaries


def _build_uncertainty_summary(state: CaseState) -> dict[str, Any]:
    uncertainty = state.uncertainty or {}
    return {
        "uncertainty_level": str(uncertainty.get("uncertainty_level", "unknown")).lower(),
        "reasons": [str(item).strip() for item in uncertainty.get("reasons", [])[:5] if str(item).strip()],
        "missing_information": [
            str(item).strip() for item in uncertainty.get("missing_information", [])[:5] if str(item).strip()
        ],
        "referenced_experiences": [
            str(item).strip() for item in uncertainty.get("referenced_experiences", [])[:6] if str(item).strip()
        ],
    }


def _build_contradiction_summary(state: CaseState) -> dict[str, Any]:
    contradiction_output = state.skill_outputs.get("contradiction_check_skill", {})
    metadata_output = state.skill_outputs.get("metadata_consistency_skill", {})
    return {
        "contradictions": [
            str(item).strip() for item in contradiction_output.get("contradictions", [])[:5] if str(item).strip()
        ],
        "missing_links": [
            str(item).strip() for item in contradiction_output.get("missing_links", [])[:5] if str(item).strip()
        ],
        "reasoning_gaps": [
            str(item).strip() for item in contradiction_output.get("reasoning_gaps", [])[:5] if str(item).strip()
        ],
        "metadata_conflicts": [
            str(item).strip() for item in metadata_output.get("conflicts", [])[:5] if str(item).strip()
        ],
        "suspicious_points": [
            str(item).strip() for item in metadata_output.get("suspicious_points", [])[:5] if str(item).strip()
        ],
    }


def _build_information_gap_summary(state: CaseState) -> dict[str, Any]:
    gap_output = state.skill_outputs.get("information_gap_detection_skill", {})
    return {
        "missing_information": [
            str(item).strip() for item in gap_output.get("missing_information", [])[:5] if str(item).strip()
        ],
        "why_it_matters": [str(item).strip() for item in gap_output.get("why_it_matters", [])[:5] if str(item).strip()],
        "impact_on_differential": [
            str(item).strip() for item in gap_output.get("impact_on_differential", [])[:5] if str(item).strip()
        ],
        "uncertainty_if_missing": str(gap_output.get("uncertainty_if_missing", "unknown")).strip().lower(),
    }


def _build_escalation_summary(state: CaseState) -> dict[str, Any]:
    escalation_output = state.skill_outputs.get("escalation_recommendation_skill", {})
    return {
        "whether_escalation_needed": str(escalation_output.get("whether_escalation_needed", "unknown")).strip().lower(),
        "escalation_reason": [
            str(item).strip() for item in escalation_output.get("escalation_reason", [])[:5] if str(item).strip()
        ],
        "suggested_next_check_type": str(escalation_output.get("suggested_next_check_type", "")).strip(),
        "caution_flags": [str(item).strip() for item in escalation_output.get("caution_flags", [])[:5] if str(item).strip()],
    }


def _build_planner_rationale(state: CaseState) -> dict[str, Any]:
    planner_output = state.planner_output or {}
    selection_reasons = planner_output.get("selection_reasons", {})
    selected_skills = [str(item).strip() for item in planner_output.get("selected_skills", []) if str(item).strip()]
    compact_reasons = {
        skill_name: [str(item).strip() for item in reasons[:4] if str(item).strip()]
        for skill_name, reasons in selection_reasons.items()
        if skill_name in selected_skills
    }
    return {
        "planner_type": str(planner_output.get("planner_type", "")),
        "planner_version": str(planner_output.get("planner_version", "")),
        "selected_skills": selected_skills,
        "selection_reasons": compact_reasons,
    }


def _serialize_evidence(
    *,
    initial_perception_summary: dict[str, Any],
    retrieved_raw_cases_summary: list[dict[str, Any]],
    retrieved_tactical_experiences_summary: list[dict[str, Any]],
    retrieved_abstract_experiences_summary: list[dict[str, Any]],
    skill_outputs: dict[str, dict[str, Any]],
    risk_flags: list[str],
    uncertainty_summary: dict[str, Any],
    contradiction_summary: dict[str, Any],
    information_gap_summary: dict[str, Any],
    escalation_summary: dict[str, Any],
    planner_rationale: dict[str, Any],
    calibration: dict[str, Any],
) -> str:
    section_plan = dict(calibration.get("section_plan", {}))
    observation_plan = dict(section_plan.get("observation", {}))
    comparison_plan = dict(section_plan.get("comparison", {}))
    risk_plan = dict(section_plan.get("risk", {}))
    conflict_plan = dict(section_plan.get("conflict_uncertainty", {}))
    planner_plan = dict(section_plan.get("planner", {}))
    section_order = section_plan.get("section_order", ["observation", "comparison", "risk", "conflict_uncertainty", "planner"])
    section_renderers = {
        "observation": lambda: _serialize_observation_section(
            initial_perception_summary,
            skill_outputs,
            _select_retrieval_records_by_ids(
                retrieved_raw_cases_summary,
                [str(item) for item in observation_plan.get("raw_case_source_ids", []) if str(item).strip()],
            ),
            skill_order=[str(item) for item in observation_plan.get("skill_names", []) if str(item).strip()],
            merged_groups=list(observation_plan.get("merged_groups", [])),
        ),
        "comparison": lambda: _serialize_comparison_section(
            skill_outputs,
            _select_retrieval_records_by_ids(
                retrieved_tactical_experiences_summary,
                [str(item) for item in comparison_plan.get("tactical_source_ids", []) if str(item).strip()],
            ),
            skill_order=[str(item) for item in comparison_plan.get("skill_names", []) if str(item).strip()],
            merged_groups=list(comparison_plan.get("merged_groups", [])),
        ),
        "risk": lambda: _serialize_risk_section(
            skill_outputs,
            risk_flags,
            _select_retrieval_records_by_ids(
                retrieved_abstract_experiences_summary,
                [str(item) for item in risk_plan.get("abstract_source_ids", []) if str(item).strip()],
            ),
            skill_order=[str(item) for item in risk_plan.get("skill_names", []) if str(item).strip()],
        ),
        "conflict_uncertainty": lambda: _serialize_conflict_uncertainty_section(
            uncertainty_summary,
            contradiction_summary,
            information_gap_summary,
            escalation_summary,
            skill_outputs,
            selected_skill_names=[str(item) for item in conflict_plan.get("skill_names", []) if str(item).strip()],
        ),
        "planner": lambda: _serialize_planner_section(
            planner_rationale,
            max_reason_skills=int(planner_plan.get("max_reason_skills", 3)),
        ),
    }
    sections: list[str] = []
    for section_name in section_order:
        renderer = section_renderers.get(str(section_name))
        if renderer is None:
            continue
        section_text = renderer()
        if section_text.strip():
            sections.append(section_text)
    return "\n\n".join(sections)


def _serialize_observation_section(
    initial_perception_summary: dict[str, Any],
    skill_outputs: dict[str, dict[str, Any]],
    retrieved_raw_cases_summary: list[dict[str, Any]],
    *,
    skill_order: list[str] | None = None,
    merged_groups: list[dict[str, Any]] | None = None,
) -> str:
    lines = ["[Observation Evidence]"]
    image_summary = initial_perception_summary.get("image_summary")
    if image_summary:
        lines.append(f"- Initial image summary: {image_summary}")
    ddx_candidates = initial_perception_summary.get("ddx_candidates", [])
    if ddx_candidates:
        lines.append(f"- Early differential candidates: {', '.join(ddx_candidates)}")
    chosen_order = skill_order if skill_order else list(OBSERVATION_SKILLS)
    lines.extend(
        _render_skill_entries(
            skill_outputs=skill_outputs,
            skill_order=chosen_order,
            merged_groups=merged_groups or [],
        )
    )
    for record in retrieved_raw_cases_summary:
        summary = record.get("perception_summary")
        if summary:
            lines.append(f"- Raw case reference {record.get('source_id')}: {summary}")
    return "\n".join(lines)


def _serialize_comparison_section(
    skill_outputs: dict[str, dict[str, Any]],
    retrieved_tactical_experiences_summary: list[dict[str, Any]],
    *,
    skill_order: list[str] | None = None,
    merged_groups: list[dict[str, Any]] | None = None,
) -> str:
    lines = ["[Exclusion And Comparison Evidence]"]
    chosen_order = skill_order if skill_order else list(COMPARISON_SKILLS)
    lines.extend(
        _render_skill_entries(
            skill_outputs=skill_outputs,
            skill_order=chosen_order,
            merged_groups=merged_groups or [],
        )
    )
    for record in retrieved_tactical_experiences_summary:
        description = record.get("perception_summary")
        learning_points = record.get("learning_points", [])
        details = "; ".join([description] + learning_points) if learning_points else str(description)
        if details.strip():
            lines.append(f"- Tactical experience {record.get('source_id')}: {details}")
    return "\n".join(lines)


def _serialize_risk_section(
    skill_outputs: dict[str, dict[str, Any]],
    risk_flags: list[str],
    retrieved_abstract_experiences_summary: list[dict[str, Any]],
    *,
    skill_order: list[str] | None = None,
) -> str:
    lines = ["[Risk Evidence]"]
    if risk_flags:
        lines.append(f"- Risk flags: {', '.join(str(item) for item in risk_flags)}")
    chosen_order = skill_order if skill_order else list(RISK_SKILLS)
    for skill_name in chosen_order:
        output = skill_outputs.get(skill_name, {})
        if output:
            lines.append(f"- {skill_name}: {_format_skill_output(output)}")
    for record in retrieved_abstract_experiences_summary:
        description = record.get("perception_summary")
        learning_points = record.get("learning_points", [])
        details = "; ".join([description] + learning_points) if learning_points else str(description)
        if details.strip():
            lines.append(f"- Abstract experience {record.get('source_id')}: {details}")
    return "\n".join(lines)


def _serialize_conflict_uncertainty_section(
    uncertainty_summary: dict[str, Any],
    contradiction_summary: dict[str, Any],
    information_gap_summary: dict[str, Any],
    escalation_summary: dict[str, Any],
    skill_outputs: dict[str, dict[str, Any]],
    *,
    selected_skill_names: list[str] | None = None,
) -> str:
    lines = ["[Conflict And Uncertainty]"]
    uncertainty_level = uncertainty_summary.get("uncertainty_level")
    if uncertainty_level:
        lines.append(f"- Uncertainty level: {uncertainty_level}")
    if uncertainty_summary.get("reasons"):
        lines.append(f"- Uncertainty reasons: {'; '.join(uncertainty_summary.get('reasons', []))}")
    if uncertainty_summary.get("missing_information"):
        lines.append(f"- Missing information: {'; '.join(uncertainty_summary.get('missing_information', []))}")
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts", "suspicious_points"):
        values = contradiction_summary.get(field_name, [])
        if values:
            label = field_name.replace("_", " ")
            lines.append(f"- {label}: {'; '.join(values)}")
    selected = set(selected_skill_names or [])
    include_gap_skill = not selected or "information_gap_detection_skill" in selected
    include_uncertainty_skill = not selected or "uncertainty_assessment_skill" in selected
    include_contradiction_skill = not selected or "contradiction_check_skill" in selected
    include_escalation_skill = not selected or "escalation_recommendation_skill" in selected
    if include_contradiction_skill and skill_outputs.get("contradiction_check_skill"):
        lines.append(f"- contradiction_check_skill: {_format_skill_output(skill_outputs.get('contradiction_check_skill', {}))}")
    if include_uncertainty_skill and skill_outputs.get("uncertainty_assessment_skill"):
        lines.append(f"- uncertainty_assessment_skill: {_format_skill_output(skill_outputs.get('uncertainty_assessment_skill', {}))}")
    gap_output = skill_outputs.get("information_gap_detection_skill", {})
    if include_gap_skill and gap_output:
        lines.append(f"- information_gap_detection_skill: {_format_skill_output(gap_output)}")
    if information_gap_summary.get("missing_information"):
        lines.append(f"- explicit information gaps: {'; '.join(information_gap_summary.get('missing_information', []))}")
    if information_gap_summary.get("why_it_matters"):
        lines.append(f"- why gaps matter: {'; '.join(information_gap_summary.get('why_it_matters', []))}")
    if information_gap_summary.get("impact_on_differential"):
        lines.append(f"- impact on differential: {'; '.join(information_gap_summary.get('impact_on_differential', []))}")
    if information_gap_summary.get("uncertainty_if_missing"):
        lines.append(f"- residual uncertainty if still missing: {information_gap_summary.get('uncertainty_if_missing')}")
    escalation_output = skill_outputs.get("escalation_recommendation_skill", {})
    if include_escalation_skill and escalation_output:
        lines.append(f"- escalation_recommendation_skill: {_format_skill_output(escalation_output)}")
    if escalation_summary.get("whether_escalation_needed"):
        lines.append(f"- escalation needed: {escalation_summary.get('whether_escalation_needed')}")
    if escalation_summary.get("escalation_reason"):
        lines.append(f"- escalation reasons: {'; '.join(escalation_summary.get('escalation_reason', []))}")
    if escalation_summary.get("suggested_next_check_type"):
        lines.append(f"- suggested next check type: {escalation_summary.get('suggested_next_check_type')}")
    if escalation_summary.get("caution_flags"):
        lines.append(f"- caution flags: {'; '.join(escalation_summary.get('caution_flags', []))}")
    return "\n".join(lines)


def _serialize_planner_section(planner_rationale: dict[str, Any], *, max_reason_skills: int = 3) -> str:
    lines = ["[Planner Rationale]"]
    selected_skills = planner_rationale.get("selected_skills", [])
    if selected_skills:
        lines.append(f"- Selected skills: {', '.join(selected_skills)}")
    selection_reasons = planner_rationale.get("selection_reasons", {})
    for skill_name, reasons in list(selection_reasons.items())[: max(1, int(max_reason_skills))]:
        if reasons:
            lines.append(f"- {skill_name}: {'; '.join(reasons)}")
    return "\n".join(lines)


def _format_skill_output(output: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key, value in output.items():
        if key == "referenced_experiences":
            continue
        if value in (None, "", [], {}, "unknown"):
            continue
        text = _format_value(value)
        if text:
            chunks.append(f"{key}={_clip_text(text, max_length=160)}")
    return " | ".join(chunks[:6])


def _render_skill_entries(
    *,
    skill_outputs: dict[str, dict[str, Any]],
    skill_order: list[str],
    merged_groups: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    consumed: set[str] = set()
    for group in merged_groups:
        label = str(group.get("label", "merged_evidence")).replace("_", " ")
        group_skills = [str(item).strip() for item in group.get("skills", []) if str(item).strip()]
        group_skills = [skill for skill in group_skills if skill in skill_outputs and skill not in consumed]
        if len(group_skills) < 2:
            continue
        max_members = max(2, int(group.get("max_members", len(group_skills))))
        entries = []
        for skill_name in group_skills[:max_members]:
            text = _format_skill_output(skill_outputs.get(skill_name, {}))
            if text:
                entries.append(f"{skill_name}: {text}")
                consumed.add(skill_name)
        if entries:
            lines.append(f"- {label}: {' || '.join(entries)}")
    for skill_name in skill_order:
        if skill_name in consumed:
            continue
        output = skill_outputs.get(skill_name, {})
        if output:
            lines.append(f"- {skill_name}: {_format_skill_output(output)}")
    return lines


def _select_retrieval_records_by_ids(records: list[dict[str, Any]], selected_ids: list[str]) -> list[dict[str, Any]]:
    ids = [str(item).strip() for item in selected_ids if str(item).strip()]
    if not ids:
        return records
    allowed = set(ids)
    return [record for record in records if str(record.get("source_id", "")).strip() in allowed]


def _format_value(value: Any) -> str:
    if value in (None, "", [], {}, "unknown"):
        return ""
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in list(value.items())[:3]:
            if item in (None, "", [], {}, "unknown"):
                continue
            if isinstance(item, list):
                rendered = "; ".join(str(each).strip() for each in item[:2] if str(each).strip())
            elif isinstance(item, dict):
                rendered = "; ".join(
                    f"{sub_key}:{str(sub_value).strip()}"
                    for sub_key, sub_value in list(item.items())[:2]
                    if str(sub_value).strip()
                )
            else:
                rendered = str(item).strip()
            if rendered:
                parts.append(f"{key}:{rendered}")
        return "; ".join(parts[:3])
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value[:3] if str(item).strip())
    return str(value).strip()


def _clip_text(value: Any, *, max_length: int = 220) -> str:
    text = str(value).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
