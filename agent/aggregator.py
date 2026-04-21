from __future__ import annotations

from typing import Any

from agent.confusion_clusters import cluster_guidance_snapshot, detect_confusion_clusters
from agent.evidence_calibrator import build_default_evidence_calibrator
from agent.state import CaseState
from agent.workflow_profiles import is_family_routing_case, is_sparse_lesion_case


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
    workflow_context = state.case_input.workflow_context or {}
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
    confusion_cluster_summary = _build_confusion_cluster_summary(state)
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
            "confusion_clusters": list(confusion_cluster_summary.get("active_clusters", [])),
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
        confusion_cluster_summary=confusion_cluster_summary,
        calibration=calibration.to_dict(),
    )
    selected_evidence = _build_selected_evidence(
        calibration=calibration.to_dict(),
        skill_outputs=state.skill_outputs,
        retrieved_raw_cases_summary=retrieved_raw_cases_summary,
        retrieved_tactical_experiences_summary=retrieved_tactical_experiences_summary,
        retrieved_abstract_experiences_summary=retrieved_abstract_experiences_summary,
    )
    if not selected_evidence:
        selected_evidence = _sparse_lesion_fallback_selected_evidence(state)
    selected_evidence = _ensure_sparse_lesion_specialist_evidence(state, selected_evidence)

    # 根据 workflow_context 调整证据排序
    selected_evidence = _reorder_evidence_by_workflow(selected_evidence, workflow_context)
    selected_evidence = _filter_evidence_by_workflow(selected_evidence, workflow_context)

    evidence_decision_policy = _build_evidence_decision_policy(
        state=state,
        selected_evidence=selected_evidence,
        retrieved_tactical_experiences_summary=retrieved_tactical_experiences_summary,
        retrieved_abstract_experiences_summary=retrieved_abstract_experiences_summary,
        contradiction_summary=contradiction_summary,
        uncertainty_summary=uncertainty_summary,
        escalation_summary=escalation_summary,
        initial_perception_summary=initial_perception_summary,
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
        "confusion_cluster_summary": confusion_cluster_summary,
        "selected_evidence": selected_evidence,
        "evidence_decision_policy": evidence_decision_policy,
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


def _build_confusion_cluster_summary(state: CaseState) -> dict[str, Any]:
    perception = state.perception or {}
    retrieval_query = (state.retrieval_bundle or {}).get("query", {})
    dataset_name = getattr(state.case_input, "dataset_name", None)
    active_clusters = detect_confusion_clusters(
        ddx_candidates=[str(item) for item in perception.get("ddx_candidates", []) if str(item).strip()],
        confusion_pair=str(retrieval_query.get("confusion_pair", "")).strip() or None,
        image_summary=str(perception.get("image_summary", "")),
        notes=[str(item) for item in perception.get("notes", []) if str(item).strip()],
        dataset_name=dataset_name,
    )
    return {
        "active_clusters": active_clusters,
        "guidance": cluster_guidance_snapshot(active_clusters, max_items=2, dataset_name=dataset_name),
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
    confusion_cluster_summary: dict[str, Any],
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
            (
                _select_retrieval_records_by_ids(
                    retrieved_abstract_experiences_summary,
                    [str(item) for item in comparison_plan.get("abstract_source_ids", []) if str(item).strip()],
                )
                if [str(item) for item in comparison_plan.get("abstract_source_ids", []) if str(item).strip()]
                else []
            ),
            confusion_cluster_summary=confusion_cluster_summary,
            skill_order=[str(item) for item in comparison_plan.get("skill_names", []) if str(item).strip()],
            merged_groups=list(comparison_plan.get("merged_groups", [])),
        ),
        "risk": lambda: _serialize_risk_section(
            skill_outputs,
            risk_flags,
            (
                _select_retrieval_records_by_ids(
                    retrieved_abstract_experiences_summary,
                    [str(item) for item in risk_plan.get("abstract_source_ids", []) if str(item).strip()],
                )
                if [str(item) for item in risk_plan.get("abstract_source_ids", []) if str(item).strip()]
                else []
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
    retrieved_comparison_abstract_experiences_summary: list[dict[str, Any]],
    *,
    confusion_cluster_summary: dict[str, Any] | None = None,
    skill_order: list[str] | None = None,
    merged_groups: list[dict[str, Any]] | None = None,
) -> str:
    lines = ["[Exclusion And Comparison Evidence]"]
    active_clusters = list((confusion_cluster_summary or {}).get("active_clusters", []))
    if active_clusters:
        lines.append(f"- Active confusion clusters: {', '.join(active_clusters)}")
    for guidance in list((confusion_cluster_summary or {}).get("guidance", []))[:2]:
        label = str(guidance.get("label", "")).strip()
        watch_outs = [str(item).strip() for item in guidance.get("watch_outs", []) if str(item).strip()]
        if label and watch_outs:
            lines.append(f"- {label} watch-out: {'; '.join(watch_outs[:2])}")
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
    for record in retrieved_comparison_abstract_experiences_summary:
        description = record.get("perception_summary")
        learning_points = record.get("learning_points", [])
        details = "; ".join([description] + learning_points) if learning_points else str(description)
        if details.strip():
            lines.append(f"- Abstract comparison memory {record.get('source_id')}: {details}")
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


def _build_selected_evidence(
    *,
    calibration: dict[str, Any],
    skill_outputs: dict[str, dict[str, Any]],
    retrieved_raw_cases_summary: list[dict[str, Any]],
    retrieved_tactical_experiences_summary: list[dict[str, Any]],
    retrieved_abstract_experiences_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    item_scores = list(calibration.get("item_scores", []))
    score_by_key = {
        (str(item.get("item_id", "")).strip(), str(item.get("item_type", "")).strip()): item
        for item in item_scores
        if str(item.get("item_id", "")).strip()
    }
    kept_items = list(dict(calibration.get("debug", {})).get("kept_items", []))
    retrieval_by_id = {
        str(record.get("source_id", "")).strip(): record
        for record in [
            *retrieved_raw_cases_summary,
            *retrieved_tactical_experiences_summary,
            *retrieved_abstract_experiences_summary,
        ]
        if str(record.get("source_id", "")).strip()
    }

    selected: list[dict[str, Any]] = []
    for kept in kept_items:
        item_id = str(kept.get("item_id", "")).strip()
        item_type = str(kept.get("item_type", "")).strip()
        if not item_id or not item_type:
            continue
        score_row = score_by_key.get((item_id, item_type), {})
        section = str(kept.get("section", "")).strip()
        category = str(kept.get("category", "")).strip()
        rank = int(kept.get("rank", score_row.get("rank", 0)) or 0)
        score = float(score_row.get("final_score", kept.get("score", 0.0)) or 0.0)
        keep_reason = str(score_row.get("reason", "kept")).strip() or "kept"
        if item_type == "skill_output":
            output = dict(skill_outputs.get(item_id, {}))
            selected.append(
                {
                    "item_id": item_id,
                    "item_type": item_type,
                    "source_type": "skill_output",
                    "source_name": item_id,
                    "skill_name": item_id,
                    "retrieval_type": "",
                    "section": section,
                    "category": category,
                    "summary": _summarize_skill_evidence(item_id, output),
                    "score": round(score, 6),
                    "rank": rank,
                    "keep_reason": keep_reason,
                }
            )
            continue

        record = retrieval_by_id.get(item_id, {})
        source_layer = str(record.get("source_layer", section)).strip()
        source_name = (
            str(record.get("source_id", "")).strip()
            or str(record.get("case_id", "")).strip()
            or item_id
        )
        selected.append(
            {
                "item_id": item_id,
                "item_type": item_type,
                "source_type": "retrieval_record",
                "source_name": source_name,
                "skill_name": "",
                "retrieval_type": source_layer,
                "section": section,
                "category": category,
                "summary": _summarize_retrieval_evidence(record),
                "score": round(score, 6),
                "rank": rank,
                "keep_reason": keep_reason,
            }
        )
    return selected


def _build_evidence_decision_policy(
    *,
    state: CaseState,
    selected_evidence: list[dict[str, Any]],
    retrieved_tactical_experiences_summary: list[dict[str, Any]],
    retrieved_abstract_experiences_summary: list[dict[str, Any]],
    contradiction_summary: dict[str, Any],
    uncertainty_summary: dict[str, Any],
    escalation_summary: dict[str, Any],
    initial_perception_summary: dict[str, Any],
) -> dict[str, Any]:
    supporting_items, opposing_items = _split_selected_evidence(selected_evidence)
    supporting_score = round(sum(float(item.get("score", 0.0) or 0.0) for item in supporting_items), 6)
    opposing_score = round(sum(float(item.get("score", 0.0) or 0.0) for item in opposing_items), 6)
    sparse_lesion_case = is_sparse_lesion_case(workflow_context=state.case_input.workflow_context)
    subtype_supporting_items = _subtype_supporting_items(selected_evidence if sparse_lesion_case else supporting_items)
    selected_evidence_present = bool(selected_evidence)

    risk_output = dict(state.skill_outputs.get("malignancy_risk_assessment_skill", {}))
    risk_level = str(risk_output.get("risk_level", "unknown")).strip().lower() or "unknown"
    contradiction_count = _count_contradiction_items(contradiction_summary)
    uncertainty_level = str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower() or "unknown"
    specialist_items = [
        item
        for item in (selected_evidence if sparse_lesion_case else supporting_items)
        if str(item.get("skill_name", "")).strip()
        in {"ack_scc_specialist_skill", "mel_nev_specialist_skill", "benign_mimic_specialist_skill"}
    ]
    specialist_support = bool(specialist_items)
    consistent_retrieval_count = _count_consistent_retrieval_support(
        supporting_items=supporting_items,
        retrieved_tactical_experiences_summary=retrieved_tactical_experiences_summary,
        retrieved_abstract_experiences_summary=retrieved_abstract_experiences_summary,
    )
    support_margin = round(supporting_score - opposing_score, 6)
    subtype_support_score = round(sum(float(item.get("score", 0.0) or 0.0) for item in subtype_supporting_items), 6)
    opposing_quota_satisfied = len(opposing_items) >= 1
    subtype_support_quota_satisfied = len(subtype_supporting_items) >= 1
    subtype_support_margin = round(subtype_support_score - opposing_score, 6)
    family_override_allowed = (
        _family_override_allowed_for_state(state)
        and selected_evidence_present
        and support_margin >= 2.0
        and uncertainty_level not in {"high", "unknown"}
    )
    malignancy_override_allowed = (
        selected_evidence_present
        and
        risk_level == "high"
        and support_margin >= 6.0
        and uncertainty_level not in {"high", "unknown"}
        and contradiction_count <= 2
    )
    subtype_override_allowed = (
        malignancy_override_allowed
        and subtype_support_margin >= 4.0
        and specialist_support
        and consistent_retrieval_count >= 1
        and opposing_quota_satisfied
        and subtype_support_quota_satisfied
    )
    if sparse_lesion_case and not subtype_override_allowed:
        # Sparse HAM-style cases often surface subtype direction through
        # benign-mimic or keratinocytic comparison evidence before the global
        # malignancy gate becomes strong enough for a full override.
        subtype_override_allowed = (
            selected_evidence_present
            and subtype_support_margin >= 3.0
            and specialist_support
            and opposing_quota_satisfied
            and subtype_support_quota_satisfied
            and uncertainty_level not in {"high", "unknown"}
        )
    override_mode = "risk_only"
    if subtype_override_allowed:
        override_mode = "subtype_override"
    elif family_override_allowed:
        override_mode = "family_override"
    elif malignancy_override_allowed:
        override_mode = "suspicious_subtype"

    override_reasons: list[str] = []
    if risk_level == "high":
        override_reasons.append("high_malignancy_risk")
    if support_margin >= 6.0 and len(supporting_items) >= max(2, len(opposing_items)):
        override_reasons.append("supporting_evidence_outweighs_opposition")
    if subtype_support_margin >= 4.0 and subtype_support_quota_satisfied:
        override_reasons.append("subtype_specific_support_present")
    if family_override_allowed:
        override_reasons.append("family_level_override_allowed")
    if specialist_support:
        override_reasons.append("specialist_support_present")
    if uncertainty_level not in {"high", "unknown"}:
        override_reasons.append("uncertainty_not_high")
    if contradiction_count <= 2:
        override_reasons.append("contradictions_not_excessive")
    if consistent_retrieval_count >= 1:
        override_reasons.append("consistent_retrieval_support")
    if opposing_quota_satisfied:
        override_reasons.append("opposing_quota_satisfied")

    caution_flags: list[str] = []
    if risk_level == "high":
        caution_flags.append("malignancy_risk_high")
    if uncertainty_level == "high":
        caution_flags.append("uncertainty_high")
    if contradiction_count > 0:
        caution_flags.append("contradictions_present")
    if str(escalation_summary.get("whether_escalation_needed", "")).strip().lower() in {"yes", "consider"}:
        caution_flags.append("follow_up_or_biopsy_consideration")

    why_not_confident_enough: list[str] = []
    if risk_level != "high" and not sparse_lesion_case:
        why_not_confident_enough.append("malignancy risk is not high enough for a diagnosis override")
    if not selected_evidence_present:
        why_not_confident_enough.append("no curated selected evidence is available to justify overriding the baseline diagnosis")
    if support_margin < 6.0:
        why_not_confident_enough.append("supporting evidence does not sufficiently outweigh opposing or exclusion evidence")
    if not subtype_support_quota_satisfied:
        why_not_confident_enough.append("supporting evidence is not specific enough to justify a subtype override")
    if _family_override_allowed_for_state(state) and not family_override_allowed:
        why_not_confident_enough.append("family-level evidence support is still too weak for a grouped-label override")
    if not specialist_support:
        why_not_confident_enough.append("no specialist evidence strongly supports the override direction")
    if uncertainty_level in {"high", "unknown"}:
        why_not_confident_enough.append("uncertainty remains too high for a confident diagnosis override")
    if contradiction_count > 2:
        why_not_confident_enough.append("contradictions or reasoning gaps remain too numerous")
    if consistent_retrieval_count < 1:
        why_not_confident_enough.append("retrieval evidence does not provide a consistent prototype or confusion-memory anchor")
    if not opposing_quota_satisfied:
        why_not_confident_enough.append("opposing or exclusion evidence quota is not satisfied")

    risk_layer = {
        "risk_flag": "malignancy_risk_high" if risk_level == "high" else f"malignancy_risk_{risk_level}",
        "caution_flags": caution_flags,
        "follow_up_suggestion": _build_follow_up_suggestion(escalation_summary=escalation_summary, uncertainty_summary=uncertainty_summary),
        "selected_evidence": supporting_items[:4] + opposing_items[:2],
        "baseline_preview": _build_baseline_preview(
            initial_perception_summary=initial_perception_summary,
            baseline_diagnosis=state.baseline_diagnosis,
        ),
    }
    diagnosis_override_layer = {
        "workflow_context": dict(state.case_input.workflow_context or {}),
        "override_allowed": subtype_override_allowed or family_override_allowed,
        "malignancy_override_allowed": malignancy_override_allowed or family_override_allowed,
        "subtype_override_allowed": subtype_override_allowed,
        "family_override_allowed": family_override_allowed,
        "override_mode": override_mode,
        "override_reasons": override_reasons,
        "why_not_confident_enough_to_override": why_not_confident_enough,
        "supporting_evidence": supporting_items,
        "opposing_evidence": opposing_items,
        "supporting_score": supporting_score,
        "opposing_score": opposing_score,
        "support_margin": support_margin,
        "selected_evidence_present": selected_evidence_present,
        "subtype_support_score": subtype_support_score,
        "subtype_support_margin": subtype_support_margin,
        "specialist_support_present": specialist_support,
        "consistent_retrieval_count": consistent_retrieval_count,
        "contradiction_count": contradiction_count,
        "uncertainty_level": uncertainty_level,
        "opposing_quota_satisfied": opposing_quota_satisfied,
        "subtype_support_quota_satisfied": subtype_support_quota_satisfied,
    }
    return {
        "risk_layer": risk_layer,
        "diagnosis_override_layer": diagnosis_override_layer,
    }


def _split_selected_evidence(selected_evidence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supporting: list[dict[str, Any]] = []
    opposing: list[dict[str, Any]] = []
    for item in selected_evidence:
        category = str(item.get("category", "")).strip().lower()
        summary = str(item.get("summary", "")).strip()
        annotated = dict(item)
        if category == "exclusion_opposing" or "opposing_evidence" in summary or "exclusion_evidence" in summary:
            opposing.append(annotated)
        else:
            supporting.append(annotated)
    return supporting, opposing


def _subtype_supporting_items(selected_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subtype_items: list[dict[str, Any]] = []
    subtype_skill_names = {
        "ack_scc_specialist_skill",
        "mel_nev_specialist_skill",
        "benign_mimic_specialist_skill",
        "differential_compare_skill",
    }
    for item in selected_evidence:
        skill_name = str(item.get("skill_name", "")).strip()
        category = str(item.get("category", "")).strip().lower()
        summary = str(item.get("summary", "")).lower()
        if skill_name in subtype_skill_names:
            subtype_items.append(item)
            continue
        if category == "differential_support":
            subtype_items.append(item)
            continue
        if any(term in summary for term in ("basal cell", "squamous cell", "melanoma", "actinic keratos")):
            subtype_items.append(item)
    return subtype_items


def _family_override_allowed_for_state(state: CaseState) -> bool:
    label_space_id = str(getattr(state.case_input, "label_space_id", "")).strip().lower()
    if not is_family_routing_case(workflow_context=state.case_input.workflow_context, label_space_id=label_space_id):
        return False
    snapshot = state.policy_snapshot or {}
    evidence_policy = dict(snapshot.get("evidence_policy", {}) or {})
    return bool(evidence_policy.get("allow_family_override", False))


def _sparse_lesion_fallback_selected_evidence(state: CaseState) -> list[dict[str, Any]]:
    if not is_sparse_lesion_case(workflow_context=state.case_input.workflow_context):
        return []
    if state.skill_outputs.get("benign_mimic_specialist_skill"):
        return [
            {
                "item_id": "benign_mimic_specialist_skill",
                "item_type": "skill_output",
                "source_type": "skill_output",
                "source_name": "benign_mimic_specialist_skill",
                "skill_name": "benign_mimic_specialist_skill",
                "retrieval_type": "",
                "section": "comparison",
                "category": "differential_support",
                "summary": _summarize_skill_evidence(
                    "benign_mimic_specialist_skill",
                    state.skill_outputs.get("benign_mimic_specialist_skill", {}),
                ),
                "score": 6.8,
                "rank": 1,
                "keep_reason": "sparse_lesion_benign_mimic_fallback",
            }
        ]
    if state.skill_outputs.get("ack_scc_specialist_skill"):
        return [
            {
                "item_id": "ack_scc_specialist_skill",
                "item_type": "skill_output",
                "source_type": "skill_output",
                "source_name": "ack_scc_specialist_skill",
                "skill_name": "ack_scc_specialist_skill",
                "retrieval_type": "",
                "section": "comparison",
                "category": "differential_support",
                "summary": _summarize_skill_evidence(
                    "ack_scc_specialist_skill",
                    state.skill_outputs.get("ack_scc_specialist_skill", {}),
                ),
                "score": 6.5,
                "rank": 1,
                "keep_reason": "sparse_lesion_specialist_fallback",
            }
        ]
    if state.skill_outputs.get("mel_nev_specialist_skill"):
        return [
            {
                "item_id": "mel_nev_specialist_skill",
                "item_type": "skill_output",
                "source_type": "skill_output",
                "source_name": "mel_nev_specialist_skill",
                "skill_name": "mel_nev_specialist_skill",
                "retrieval_type": "",
                "section": "comparison",
                "category": "differential_support",
                "summary": _summarize_skill_evidence(
                    "mel_nev_specialist_skill",
                    state.skill_outputs.get("mel_nev_specialist_skill", {}),
                ),
                "score": 6.2,
                "rank": 1,
                "keep_reason": "sparse_lesion_specialist_fallback",
            }
        ]
    if state.skill_outputs.get("differential_compare_skill"):
        return [
            {
                "item_id": "differential_compare_skill",
                "item_type": "skill_output",
                "source_type": "skill_output",
                "source_name": "differential_compare_skill",
                "skill_name": "differential_compare_skill",
                "retrieval_type": "",
                "section": "comparison",
                "category": "differential_support",
                "summary": _summarize_skill_evidence(
                    "differential_compare_skill",
                    state.skill_outputs.get("differential_compare_skill", {}),
                ),
                "score": 5.8,
                "rank": 1,
                "keep_reason": "sparse_lesion_comparison_fallback",
            }
        ]
    return []


def _ensure_sparse_lesion_specialist_evidence(
    state: CaseState,
    selected_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not is_sparse_lesion_case(workflow_context=state.case_input.workflow_context):
        return selected_evidence
    skill_name = "benign_mimic_specialist_skill"
    if not state.skill_outputs.get(skill_name):
        return selected_evidence
    if any(str(item.get("skill_name", "")).strip() == skill_name for item in selected_evidence):
        return selected_evidence
    injected = {
        "item_id": skill_name,
        "item_type": "skill_output",
        "source_type": "skill_output",
        "source_name": skill_name,
        "skill_name": skill_name,
        "retrieval_type": "",
        "section": "comparison",
        "category": "differential_support",
        "summary": _summarize_skill_evidence(skill_name, state.skill_outputs.get(skill_name, {})),
        "score": 6.6,
        "rank": max([int(item.get("rank", 0) or 0) for item in selected_evidence] + [0]) + 1,
        "keep_reason": "sparse_lesion_specialist_injected",
    }
    return [*selected_evidence, injected]


def _count_contradiction_items(summary: dict[str, Any]) -> int:
    total = 0
    for key in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts", "suspicious_points"):
        total += len(summary.get(key, []) or [])
    return total


def _count_consistent_retrieval_support(
    *,
    supporting_items: list[dict[str, Any]],
    retrieved_tactical_experiences_summary: list[dict[str, Any]],
    retrieved_abstract_experiences_summary: list[dict[str, Any]],
) -> int:
    selected_retrieval_ids = {
        str(item.get("source_name", "")).strip()
        for item in supporting_items
        if str(item.get("source_type", "")).strip() == "retrieval_record"
    }
    count = 0
    for record in [*retrieved_tactical_experiences_summary, *retrieved_abstract_experiences_summary]:
        source_id = str(record.get("source_id", "")).strip()
        exp_type = str(record.get("experience_type", "")).strip().lower()
        if source_id and source_id in selected_retrieval_ids and exp_type in {"prototype", "confusion_memory", "rule"}:
            count += 1
    return count


def _build_follow_up_suggestion(*, escalation_summary: dict[str, Any], uncertainty_summary: dict[str, Any]) -> str:
    next_check = str(escalation_summary.get("suggested_next_check_type", "")).strip()
    if next_check:
        return next_check
    uncertainty_level = str(uncertainty_summary.get("uncertainty_level", "")).strip().lower()
    if uncertainty_level == "high":
        return "closer dermatologic review or biopsy consideration"
    return "clinical follow-up and closer inspection if concern persists"


def _build_baseline_preview(
    *,
    initial_perception_summary: dict[str, Any],
    baseline_diagnosis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ddx_candidates = [str(item).strip() for item in initial_perception_summary.get("ddx_candidates", []) if str(item).strip()]
    preview = {
        "early_ddx_candidates": ddx_candidates[:3],
        "image_summary": str(initial_perception_summary.get("image_summary", "")).strip()[:220],
    }
    if isinstance(baseline_diagnosis, dict) and baseline_diagnosis:
        preview["baseline_final_diagnosis"] = str(baseline_diagnosis.get("final_diagnosis", "")).strip()[:120]
        preview["baseline_differential_diagnoses"] = [
            str(item).strip()[:120]
            for item in baseline_diagnosis.get("differential_diagnoses", [])[:4]
            if str(item).strip()
        ]
        preview["baseline_confidence"] = str(baseline_diagnosis.get("confidence", "")).strip()[:60]
    return preview


def _summarize_skill_evidence(skill_name: str, output: dict[str, Any]) -> str:
    body = _format_skill_output(output)
    if not body:
        return skill_name
    return _clip_text(f"{skill_name}: {body}", max_length=360)


def _summarize_retrieval_evidence(record: dict[str, Any]) -> str:
    parts: list[str] = []
    perception_summary = _clip_text(record.get("perception_summary", ""), max_length=180)
    if perception_summary:
        parts.append(perception_summary)
    learning_points = [
        _clip_text(item, max_length=120)
        for item in record.get("learning_points", [])[:2]
        if str(item).strip()
    ]
    if learning_points:
        parts.append("; ".join(learning_points))
    confusion_pair = str(record.get("confusion_pair", "")).strip()
    if confusion_pair:
        parts.append(f"confusion_pair={confusion_pair}")
    if not parts:
        parts.append(str(record.get("source_id", "")).strip() or str(record.get("case_id", "")).strip())
    return _clip_text(" | ".join(part for part in parts if part), max_length=360)


def _reorder_evidence_by_workflow(
    evidence_items: list[dict[str, Any]],
    workflow_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """根据 workflow_context 调整证据排序"""
    if not workflow_context or not evidence_items:
        return evidence_items

    preference = str(workflow_context.get("workflow_preference", "")).strip()
    metadata_completeness = str(workflow_context.get("metadata_completeness", "")).strip()

    # 场景1: 信息缺失时，把 information_gap 和相关证据提到最前
    if metadata_completeness == "minimal":
        gap_items = [
            e for e in evidence_items
            if "information_gap" in e.get("skill_name", "").lower()
            or "information_gap" in e.get("category", "").lower()
        ]
        other_items = [
            e for e in evidence_items
            if "information_gap" not in e.get("skill_name", "").lower()
            and "information_gap" not in e.get("category", "").lower()
        ]
        return gap_items + other_items

    # 场景2: 高风险场景时，把 malignancy_risk 相关证据提到最前
    if preference == "risk_first":
        risk_items = [
            e for e in evidence_items
            if any(kw in e.get("skill_name", "").lower() for kw in ["malignancy", "risk"])
            or any(kw in e.get("category", "").lower() for kw in ["malignancy", "risk"])
            or e.get("section", "") == "risk"
        ]
        other_items = [
            e for e in evidence_items
            if not any(kw in e.get("skill_name", "").lower() for kw in ["malignancy", "risk"])
            and not any(kw in e.get("category", "").lower() for kw in ["malignancy", "risk"])
            and e.get("section", "") != "risk"
        ]
        return risk_items + other_items

    # 场景3: metadata_first 时，把 metadata_consistency 相关证据提到最前
    if preference == "metadata_first":
        metadata_items = [
            e for e in evidence_items
            if "metadata" in e.get("skill_name", "").lower()
            or "metadata" in e.get("category", "").lower()
        ]
        other_items = [
            e for e in evidence_items
            if "metadata" not in e.get("skill_name", "").lower()
            and "metadata" not in e.get("category", "").lower()
        ]
        return metadata_items + other_items

    return evidence_items


def _filter_evidence_by_workflow(
    evidence_items: list[dict[str, Any]],
    workflow_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not workflow_context or not evidence_items:
        return evidence_items

    time_budget = str(workflow_context.get("time_budget", "")).strip()
    metadata_completeness = str(workflow_context.get("metadata_completeness", "")).strip()

    result = list(evidence_items)

    if time_budget == "screening":
        high_conf = [e for e in result if float(e.get("calibration_score", 1.0)) >= 0.5]
        result = high_conf[:6] if high_conf else result[:6]

    if metadata_completeness == "minimal":
        metadata_dependent = {"metadata_consistency_skill", "distribution_analysis_skill"}
        result = [
            e for e in result
            if e.get("skill_name", "") not in metadata_dependent
            or bool(str(e.get("content", "")).strip())
        ]

    return result
