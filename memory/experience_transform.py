from __future__ import annotations

from typing import Any

from agent.labels import canonicalize_label, labels_match
from agent.state import CaseState
from memory.experience_schema import (
    AbstractExperience,
    CompositeSkillSeed,
    ExperienceRecord,
    RawCaseMemory,
    TacticalExperience,
    dedupe_strings,
    is_malignant_label,
    stable_hash,
)

SKILL_HELP_MAP: dict[str, list[str]] = {
    "morphology_analysis_skill": ["stabilized lesion structure", "made morphology explicit for downstream reasoning"],
    "color_pattern_analysis_skill": ["made pigment variation explicit", "surfaced color asymmetry as auditable evidence"],
    "border_surface_analysis_skill": ["separated edge evidence from surface evidence", "highlighted margin quality explicitly"],
    "distribution_analysis_skill": ["made location and extent assumptions explicit", "captured distribution context for later comparison"],
    "temporal_evolution_skill": ["converted history into structured temporal evidence", "made change over time inspectable"],
    "metadata_consistency_skill": ["checked image-metadata coherence", "flagged reliability issues before integration"],
    "differential_compare_skill": ["kept the differential open", "made pairwise reasoning explicit"],
    "contradiction_check_skill": ["surfaced missing links in the reasoning chain", "audited internal consistency"],
    "malignancy_risk_assessment_skill": ["made risk framing explicit", "highlighted alarm signals without forcing diagnosis"],
    "uncertainty_assessment_skill": ["made evidence limits explicit", "prevented overconfident downstream use"],
    "mel_nev_specialist_skill": ["focused a high-value confusion pair", "sharpened specialist comparison clues"],
    "ack_scc_specialist_skill": ["focused a high-value confusion pair", "sharpened specialist keratinization clues"],
}

SKILL_KEY_STEP_MAP: dict[str, list[str]] = {
    "morphology_analysis_skill": ["inspect dominant lesion silhouette", "judge elevation and lesion count"],
    "color_pattern_analysis_skill": ["inspect dominant color", "compare internal color asymmetry"],
    "border_surface_analysis_skill": ["judge border definition", "inspect surface change and scale"],
    "distribution_analysis_skill": ["identify likely anatomic site", "judge extent and clustering"],
    "temporal_evolution_skill": ["collect growth and symptom history", "translate history into stability and tempo"],
    "metadata_consistency_skill": ["compare visible cues against metadata", "separate direct conflicts from soft concerns"],
    "differential_compare_skill": ["select the most relevant candidate pair", "preserve unresolved comparative evidence"],
    "contradiction_check_skill": ["scan for conflicts across evidence sources", "surface missing links in the reasoning chain"],
    "malignancy_risk_assessment_skill": ["collect alarm signals", "balance concerning and reassuring evidence"],
    "uncertainty_assessment_skill": ["scan ambiguity across evidence streams", "name missing information explicitly"],
    "mel_nev_specialist_skill": ["focus the confusion pair", "contrast differentiating clues without forcing closure"],
    "ack_scc_specialist_skill": ["focus the confusion pair", "contrast malignant versus keratotic clues"],
}


def build_writeback_bundle(
    state: CaseState,
    *,
    case_outcome: dict[str, Any],
    skill_assessments: list[dict[str, Any]],
    reflection_extract: dict[str, Any],
    cognition_update: dict[str, Any],
) -> dict[str, Any]:
    raw_case_memory = build_raw_case_memory(state=state, case_outcome=case_outcome, skill_assessments=skill_assessments)
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
    return {
        "raw_case_memory": raw_case_memory.to_dict(),
        "tactical_experiences": [record.to_dict() for record in tactical_experiences],
        "abstract_experiences": [record.to_dict() for record in abstract_experiences],
        "cognition_update": dict(cognition_update),
        "skill_stats_update": build_skill_stats_update(skill_assessments),
        "reflection_extract": dict(reflection_extract),
    }


def build_case_outcome(
    state: CaseState,
    *,
    detected_errors: list[str],
    confusion_pair: str | None,
) -> dict[str, Any]:
    true_label = state.case_input.reference_label or state.case_input.label
    predicted_label = state.final_diagnosis.get("final_diagnosis")
    has_ground_truth = bool(true_label)
    label_match = labels_match(predicted_label, true_label)
    is_correct = bool(label_match) if label_match is not None else None
    uncertainty_level = _uncertainty_level(state)
    confidence_level = str(state.final_diagnosis.get("confidence", "unknown")).lower()

    if has_ground_truth and is_correct is True:
        status = "success"
    elif has_ground_truth and is_correct is False:
        status = "partially_helpful" if state.skill_outputs else "failure"
    elif detected_errors:
        status = "failure"
    elif uncertainty_level == "high" or confidence_level in {"low", "moderate"}:
        status = "partially_helpful" if state.skill_outputs else "failure"
    else:
        status = "success"

    return {
        "status": status,
        "has_ground_truth": has_ground_truth,
        "is_correct": is_correct,
        "error_type": detected_errors[0] if detected_errors else None,
        "predicted_label": predicted_label,
        "reference_label": true_label,
        "predicted_canonical_label": canonicalize_label(predicted_label),
        "reference_canonical_label": canonicalize_label(true_label),
        "confidence_level": confidence_level,
        "uncertainty_level": uncertainty_level,
        "malignant_flag": {
            "ground_truth": is_malignant_label(true_label),
            "predicted": is_malignant_label(predicted_label),
        },
        "confusion_pair": confusion_pair,
        "risk_flags": list(state.risk_flags),
    }


def build_skill_assessments(state: CaseState, case_outcome: dict[str, Any]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    ddx_candidates = [str(item) for item in state.perception.get("ddx_candidates", [])[:3]]
    trigger_context_base = {
        "ddx_candidates": ddx_candidates,
        "risk_flags": list(state.risk_flags),
        "uncertainty_level": case_outcome.get("uncertainty_level", "unknown"),
        "region": state.case_input.metadata.get("region", ""),
    }
    for skill_name, output in state.skill_outputs.items():
        output_present = _has_meaningful_output(output)
        evidence_strength = _normalize_evidence_strength(output)
        evidence_strength_score = _evidence_strength_score(evidence_strength)
        impact = _skill_impact(skill_name, output, output_present, case_outcome, evidence_strength_score)
        helpfulness = _skill_helpfulness(skill_name, output, case_outcome)
        contradiction_detected = _detect_contradiction_signal(skill_name, output)
        malignant_flag_support = _detect_malignant_flag_support(skill_name, output, case_outcome)
        uncertainty_reduction = _detect_uncertainty_reduction(skill_name, output, case_outcome, impact)
        helped_aspects = list(SKILL_HELP_MAP.get(skill_name, [])) if impact != "harmful" else []
        failed_aspects = _skill_failed_aspects(skill_name, output_present, case_outcome, impact, evidence_strength)
        key_steps = list(SKILL_KEY_STEP_MAP.get(skill_name, ["produce structured evidence", "support downstream reasoning"]))
        reusable = output_present and impact in {"helpful", "partially_helpful"}
        failure_modes = _skill_failure_modes(
            skill_name=skill_name,
            output_present=output_present,
            case_outcome=case_outcome,
            evidence_strength=evidence_strength,
            impact=impact,
            contradiction_detected=contradiction_detected,
        )
        assessments.append(
            {
                "skill_name": skill_name,
                "triggered": True,
                "selected": True,
                "output_present": output_present,
                "helpfulness": helpfulness,
                "impact": impact,
                "trigger_context": dict(trigger_context_base),
                "helped_aspects": helped_aspects,
                "failed_aspects": failed_aspects,
                "key_steps": key_steps,
                "referenced_experiences": _referenced_experiences(output),
                "evidence_strength": evidence_strength,
                "evidence_strength_score": evidence_strength_score,
                "recommendation_type": str(output.get("recommendation_type", "unknown")).strip().lower() or "unknown",
                "uncertainty_reduction": uncertainty_reduction,
                "contradiction_detected": contradiction_detected,
                "malignant_flag_support": malignant_flag_support,
                "applicable_scenarios": _skill_applicable_scenarios(skill_name, case_outcome, trigger_context_base),
                "failure_modes": failure_modes,
                "reusable": reusable,
                "reuse_reason": _reuse_reason(skill_name, helpfulness, case_outcome) if reusable else "",
            }
        )
    return assessments


def build_reflection_extract(
    case_outcome: dict[str, Any],
    skill_assessments: list[dict[str, Any]],
    tactical_experiences: list[TacticalExperience],
    abstract_experiences: list[AbstractExperience],
) -> dict[str, Any]:
    abstract_types = {record.type for record in abstract_experiences}
    return {
        "raw_case_memory_ready": True,
        "tactical_experience_count": len(tactical_experiences),
        "abstract_experience_candidate_count": len(abstract_experiences),
        "reusable_tactical_count": sum(1 for item in skill_assessments if item.get("reusable")),
        "composite_skill_seed_count": sum(1 for item in abstract_experiences if item.type == "composite_skill_seed"),
        "composite_skill_seed_ids": [item.seed_id for item in abstract_experiences if item.type == "composite_skill_seed"],
        "abstract_flags": {
            "confusion_memory": "confusion_memory" in abstract_types,
            "prototype": "prototype" in abstract_types,
            "rule": "rule" in abstract_types,
            "rule_candidate": "rule_candidate" in abstract_types,
            "composite_skill_seed": "composite_skill_seed" in abstract_types,
        },
        "case_status": case_outcome.get("status", "unknown"),
    }


def build_skill_stats_update(skill_assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for assessment in skill_assessments:
        helpfulness = assessment.get("helpfulness")
        impact = assessment.get("impact")
        evidence_strength_score = float(assessment.get("evidence_strength_score", 0.0))
        updates.append(
            {
                "skill_name": assessment.get("skill_name"),
                "call_increment": 1 if assessment.get("triggered") else 0,
                "selected_increment": 1 if assessment.get("selected", assessment.get("triggered")) else 0,
                "success_increment": 1 if helpfulness == "success" else 0,
                "partially_helpful_increment": 1 if helpfulness == "partially_helpful" else 0,
                "harmful_increment": 1 if impact == "harmful" else 0,
                "failure_increment": 1 if helpfulness == "failure" or impact == "harmful" else 0,
                "reusable_increment": 1 if assessment.get("reusable") else 0,
                "uncertainty_reduction_increment": 1 if assessment.get("uncertainty_reduction") else 0,
                "contradiction_detection_increment": 1 if assessment.get("contradiction_detected") else 0,
                "malignant_flag_support_increment": 1 if assessment.get("malignant_flag_support") else 0,
                "evidence_strength_sum": evidence_strength_score,
                "evidence_strength_count": 1 if assessment.get("output_present") else 0,
            }
        )
    return updates


def build_raw_case_memory(
    state: CaseState,
    *,
    case_outcome: dict[str, Any],
    skill_assessments: list[dict[str, Any]],
) -> RawCaseMemory:
    true_label = state.case_input.reference_label or state.case_input.label
    predicted_label = state.final_diagnosis.get("final_diagnosis")
    agent_output = {
        "perception": state.perception,
        "skill_outputs": state.skill_outputs,
        "risk_flags": state.risk_flags,
        "uncertainty": state.uncertainty,
        "retrieved_experience_refs": [
            record.get("source_id") or record.get("abs_id") or record.get("exp_id") or record.get("case_id")
            for record in state.retrieved_experience
        ],
        "notes": state.notes,
        "skill_assessments": skill_assessments,
    }
    return RawCaseMemory(
        case_id=state.case_input.case_id,
        image_paths=[state.case_input.image_path],
        metadata=dict(state.case_input.metadata),
        true_label=true_label,
        qwen_output=dict(state.final_diagnosis),
        agent_output=agent_output,
        final_decision={
            "label": predicted_label,
            "source": "qwen_final_diagnosis",
            "canonical_label": predicted_label,
        },
        correctness={
            "has_ground_truth": case_outcome.get("has_ground_truth"),
            "is_correct": case_outcome.get("is_correct"),
            "error_type": case_outcome.get("error_type"),
        },
        malignant_flag=dict(case_outcome.get("malignant_flag", {})),
    )


def build_tactical_experiences(
    state: CaseState,
    *,
    case_outcome: dict[str, Any],
    skill_assessments: list[dict[str, Any]],
) -> list[TacticalExperience]:
    records: list[TacticalExperience] = []
    missing_information = state.uncertainty.get("missing_information", [])
    constraints = ["single_image"]
    if not state.case_input.metadata.get("region"):
        constraints.append("missing_region_metadata")
    constraints.extend(f"missing:{item}" for item in missing_information[:3])
    for assessment in skill_assessments:
        skill_name = str(assessment.get("skill_name", "")).strip()
        if not skill_name:
            continue
        exp_id = f"tac_{state.case_input.case_id}_{stable_hash({'skill': skill_name, 'status': assessment.get('helpfulness')})}"
        records.append(
            TacticalExperience(
                exp_id=exp_id,
                case_id=state.case_input.case_id,
                condition={
                    "trigger_type": _trigger_type(case_outcome),
                    "confusion_pair": case_outcome.get("confusion_pair"),
                    "observed_state": {
                        "ddx_candidates": assessment.get("trigger_context", {}).get("ddx_candidates", []),
                        "risk_flags": assessment.get("trigger_context", {}).get("risk_flags", []),
                        "uncertainty_level": assessment.get("trigger_context", {}).get("uncertainty_level", "unknown"),
                        "region": assessment.get("trigger_context", {}).get("region", ""),
                    },
                    "constraints": dedupe_strings(constraints),
                },
                action={
                    "skill_name": skill_name,
                    "skills_used": [skill_name],
                    "decision_pattern": _decision_pattern([skill_name]),
                },
                rationale=dedupe_strings(assessment.get("helped_aspects", []) + assessment.get("failed_aspects", [])),
                step_trace=dedupe_strings(assessment.get("key_steps", [])),
                outcome={
                    "result": assessment.get("helpfulness"),
                    "case_status": case_outcome.get("status"),
                    "error_type": case_outcome.get("error_type"),
                    "confidence_state": case_outcome.get("confidence_level", "unknown"),
                },
                reusable_scope={
                    "applies_to": _skill_reusable_scope(state, skill_name, case_outcome),
                    "priority": "high" if assessment.get("reusable") else "low",
                },
            )
        )
    return records


def build_abstract_experiences(
    state: CaseState,
    *,
    case_outcome: dict[str, Any],
    skill_assessments: list[dict[str, Any]],
) -> list[AbstractExperience]:
    records: list[AbstractExperience] = []
    uncertainty_level = str(case_outcome.get("uncertainty_level", "unknown")).lower()
    top_ddx = [str(item) for item in state.perception.get("ddx_candidates", [])][:2]
    confusion_pair = case_outcome.get("confusion_pair")
    helpful_skills = [item["skill_name"] for item in skill_assessments if item.get("helpfulness") != "failure"]
    supporting_pattern = dedupe_strings(
        [
            str(state.perception.get("image_summary", "")),
            *top_ddx,
            *state.risk_flags[:2],
        ]
    )

    if confusion_pair:
        concept = confusion_pair.lower().replace(" ", "_")
        records.append(
            AbstractExperience(
                abs_id=f"abs_confusion_{stable_hash({'type': 'confusion_memory', 'concept': concept})}",
                type="confusion_memory",
                concept=concept,
                pattern_summary={
                    "confusion_pair": confusion_pair,
                    "supporting_pattern": supporting_pattern,
                    "failure_pattern": [case_outcome["error_type"]] if case_outcome.get("error_type") else [],
                },
                supporting_cases=[state.case_input.case_id],
                counter_cases=[],
                derived_rule={
                    "rule_text": (
                        "When a confusion pair remains active, preserve differential openness and avoid forcing an early label."
                    ),
                    "action_hint": _decision_pattern(helpful_skills),
                },
                version="v1",
            )
        )

    if helpful_skills and (
        case_outcome.get("status") in {"failure", "partially_helpful"}
        or uncertainty_level == "high"
        or state.risk_flags
    ):
        concept = _decision_pattern(helpful_skills)
        records.append(
            AbstractExperience(
                abs_id=f"abs_rule_{stable_hash({'type': 'rule', 'concept': concept})}",
                type="rule",
                concept=concept,
                pattern_summary={
                    "supporting_pattern": supporting_pattern,
                    "failure_pattern": [case_outcome["error_type"]] if case_outcome.get("error_type") else [],
                },
                supporting_cases=[state.case_input.case_id],
                counter_cases=[],
                derived_rule={
                    "rule_text": (
                        "When ambiguity or risk remains elevated, explicitly structure observation, comparison, and uncertainty before final Qwen integration."
                    ),
                    "action_hint": _decision_pattern(helpful_skills),
                },
                version="v1",
            )
        )

    composite_seed_record = build_composite_skill_seed(
        state=state,
        case_outcome=case_outcome,
        skill_assessments=skill_assessments,
        supporting_pattern=supporting_pattern,
    )
    if composite_seed_record is not None:
        records.append(composite_seed_record)

    if (
        case_outcome.get("status") == "success"
        and not case_outcome.get("error_type")
        and uncertainty_level in {"low", "medium"}
        and state.final_diagnosis.get("final_diagnosis")
    ):
        concept = str(state.final_diagnosis.get("final_diagnosis", "")).strip().lower().replace(" ", "_")
        if concept:
            records.append(
                AbstractExperience(
                    abs_id=f"abs_prototype_{stable_hash({'type': 'prototype', 'concept': concept})}",
                    type="prototype",
                    concept=concept,
                    pattern_summary={
                        "supporting_pattern": supporting_pattern,
                        "failure_pattern": [],
                    },
                    supporting_cases=[state.case_input.case_id],
                    counter_cases=[],
                    derived_rule={
                        "rule_text": "Prototype memories capture recurring structured observation patterns linked to stable final outcomes.",
                        "action_hint": "retrieve_as_reference_pattern",
                    },
                    version="v1",
                )
            )
    return records


def build_composite_skill_seed(
    state: CaseState,
    *,
    case_outcome: dict[str, Any],
    skill_assessments: list[dict[str, Any]],
    supporting_pattern: list[str],
) -> AbstractExperience | None:
    if case_outcome.get("status") != "success" or case_outcome.get("error_type"):
        return None

    successful_skills = {
        str(item.get("skill_name", "")).strip()
        for item in skill_assessments
        if item.get("helpfulness") == "success" and str(item.get("skill_name", "")).strip()
    }
    if len(successful_skills) < 3:
        return None

    planner_order = [str(item).strip() for item in state.planner_output.get("ordering", []) if str(item).strip()]
    executed_order = [str(item).strip() for item in state.skill_outputs.keys() if str(item).strip()]
    ordered_candidates = planner_order or executed_order
    skill_sequence = [skill_name for skill_name in ordered_candidates if skill_name in successful_skills]
    if len(skill_sequence) < 3:
        skill_sequence = [skill_name for skill_name in executed_order if skill_name in successful_skills]
    if len(skill_sequence) < 3:
        return None

    trigger_pattern = {
        "decision_pattern": _decision_pattern(skill_sequence),
        "ddx_candidates": [str(item) for item in state.perception.get("ddx_candidates", [])[:3]],
        "risk_flags": list(state.risk_flags[:3]),
        "uncertainty_level": str(case_outcome.get("uncertainty_level", "unknown")).lower(),
        "confusion_pair": case_outcome.get("confusion_pair"),
        "region": str(state.case_input.metadata.get("region", "")).strip(),
    }
    seed_hash = stable_hash(
        {
            "type": "composite_skill_seed",
            "trigger_pattern": trigger_pattern,
            "skill_sequence": skill_sequence,
        }
    )
    seed_id = f"seed_composite_{seed_hash}"
    promotion_interface = {
        "builder": "promote_composite_skill_seed",
        "future_skill_id": f"composite_workflow_{seed_hash}",
        "future_skill_type": "composite_workflow_skill",
        "required_min_success_count": 2,
        "component_skills": list(skill_sequence),
        "trigger_pattern_source": "composite_skill_seed.trigger_pattern",
        "output_contract_source": "component_skill_union",
    }
    seed_notes = dedupe_strings(
        [
            "Candidate only; do not auto-materialize a composite skill in the current stage.",
            "Promote only after repeated successful supporting cases under a stable trigger pattern.",
            f"Observed decision pattern: {trigger_pattern['decision_pattern']}",
        ]
    )
    composite_seed = CompositeSkillSeed(
        seed_id=seed_id,
        trigger_pattern=trigger_pattern,
        skill_sequence=skill_sequence,
        supporting_cases=[state.case_input.case_id],
        success_count=1,
        notes=seed_notes,
        promotion_interface=promotion_interface,
    )
    return AbstractExperience(
        abs_id=f"abs_composite_{seed_hash}",
        seed_id=seed_id,
        type="composite_skill_seed",
        concept=trigger_pattern["decision_pattern"],
        pattern_summary={
            "supporting_pattern": supporting_pattern,
            "trigger_pattern": trigger_pattern,
            "seed_skills": list(skill_sequence),
        },
        supporting_cases=[state.case_input.case_id],
        counter_cases=[],
        derived_rule={
            "rule_text": (
                "Repeated successful skill sequences under a stable trigger pattern can later be promoted into a composite workflow skill."
            ),
            "action_hint": f"compose_candidate:{'+'.join(skill_sequence[:4])}",
            "promotion_interface": promotion_interface,
        },
        version="v2",
        composite_skill_seed=composite_seed.to_dict(),
    )


def legacy_record_to_bundle(record: dict[str, Any]) -> dict[str, Any]:
    case_id = str(record.get("case_id", "")).strip() or f"legacy_{stable_hash(record)}"
    perception_summary = str(record.get("perception_summary", "")).strip()
    skills_used = [str(item) for item in record.get("skills_used", []) if str(item).strip()]
    learning_points = [str(item) for item in record.get("learning_points", []) if str(item).strip()]
    confusion_pair = record.get("confusion_pair")
    experience_type = str(record.get("experience_type", "raw_case_experience")).strip() or "raw_case_experience"

    raw_case_memory = RawCaseMemory(
        case_id=case_id,
        image_paths=[],
        metadata={},
        true_label=None,
        qwen_output={},
        agent_output={
            "perception_summary": perception_summary,
            "legacy_key_evidence": record.get("key_evidence", {}),
            "notes": [],
        },
        final_decision={"label": None, "source": "legacy_experience_bank", "canonical_label": None},
        correctness={
            "has_ground_truth": False,
            "is_correct": None,
            "error_type": record.get("error_type") or None,
        },
        malignant_flag={"ground_truth": None, "predicted": None},
    )

    tactical = TacticalExperience(
        exp_id=f"tac_{case_id}_{stable_hash({'skills': skills_used, 'pair': confusion_pair, 'summary': perception_summary})}",
        case_id=case_id,
        condition={
            "trigger_type": experience_type,
            "confusion_pair": confusion_pair,
            "observed_state": {
                "perception_summary": perception_summary,
                "risk_flags": record.get("key_evidence", {}).get("risk_flags", []),
                "uncertainty_level": record.get("key_evidence", {}).get("uncertainty", {}).get(
                    "uncertainty_level", "unknown"
                ),
            },
            "constraints": [],
        },
        action={
            "skills_used": skills_used,
            "decision_pattern": _decision_pattern(skills_used),
        },
        rationale=learning_points[:4],
        step_trace=[f"legacy_record -> {experience_type}"],
        outcome={
            "result": "legacy_import",
            "error_type": record.get("error_type") or None,
            "confidence_state": "unknown",
        },
        reusable_scope={
            "applies_to": dedupe_strings([confusion_pair] if confusion_pair else []),
            "priority": "high" if confusion_pair else "medium",
        },
    )

    abstract_records: list[AbstractExperience] = []
    if confusion_pair:
        concept = str(confusion_pair).lower().replace(" ", "_")
        abstract_records.append(
            AbstractExperience(
                abs_id=f"abs_confusion_{stable_hash({'type': 'confusion_memory', 'concept': concept})}",
                type="confusion_memory",
                concept=concept,
                pattern_summary={
                    "confusion_pair": confusion_pair,
                    "supporting_pattern": dedupe_strings([perception_summary]),
                    "failure_pattern": [record.get("error_type")] if record.get("error_type") else [],
                },
                supporting_cases=[case_id],
                counter_cases=[],
                derived_rule={
                    "rule_text": "Imported legacy confusion memory.",
                    "action_hint": _decision_pattern(skills_used),
                },
                version="v1",
            )
        )
    if learning_points:
        abstract_records.append(
            AbstractExperience(
                abs_id=f"abs_rule_{stable_hash({'type': 'rule', 'concept': _decision_pattern(skills_used)})}",
                type="rule",
                concept=_decision_pattern(skills_used),
                pattern_summary={
                    "supporting_pattern": dedupe_strings([perception_summary]),
                    "failure_pattern": [record.get("error_type")] if record.get("error_type") else [],
                },
                supporting_cases=[case_id],
                counter_cases=[],
                derived_rule={
                    "rule_text": learning_points[0],
                    "action_hint": _decision_pattern(skills_used),
                },
                version="v1",
            )
        )

    return {
        "raw_case_memory": raw_case_memory.to_dict(),
        "tactical_experiences": [tactical.to_dict()],
        "abstract_experiences": [record.to_dict() for record in abstract_records],
        "cognition_update": {},
        "skill_stats_update": [],
        "reflection_extract": {
            "raw_case_memory_ready": True,
            "tactical_experience_count": 1,
            "abstract_experience_candidate_count": len(abstract_records),
            "reusable_tactical_count": 1,
            "abstract_flags": {
                "confusion_memory": any(item.type == "confusion_memory" for item in abstract_records),
                "prototype": any(item.type == "prototype" for item in abstract_records),
                "rule": any(item.type == "rule" for item in abstract_records),
                "rule_candidate": any(item.type == "rule_candidate" for item in abstract_records),
                "composite_skill_seed": any(item.type == "composite_skill_seed" for item in abstract_records),
            },
            "case_status": "legacy_import",
        },
    }


def tactical_to_retrieval_packet(record: dict[str, Any]) -> dict[str, Any]:
    condition = record.get("condition", {})
    observed_state = condition.get("observed_state", {})
    ddx_candidates = observed_state.get("ddx_candidates", [])
    perception_summary = "; ".join(
        [
            f"ddx={', '.join(ddx_candidates)}" if ddx_candidates else "",
            f"risk={', '.join(observed_state.get('risk_flags', []))}" if observed_state.get("risk_flags") else "",
            f"uncertainty={observed_state.get('uncertainty_level')}" if observed_state.get("uncertainty_level") else "",
            observed_state.get("perception_summary", ""),
        ]
    ).strip("; ")
    decision_pattern = record.get("action", {}).get("decision_pattern")
    learning_points = dedupe_strings(record.get("rationale", []) + ([f"Action pattern: {decision_pattern}"] if decision_pattern else []))
    return {
        "case_id": record.get("case_id"),
        "source_layer": "tactical_experience",
        "source_subtype": record.get("condition", {}).get("trigger_type"),
        "experience_type": "tactical_experience",
        "perception_summary": perception_summary or f"trigger={condition.get('trigger_type', 'unknown')}",
        "confusion_pair": condition.get("confusion_pair"),
        "learning_points": learning_points[:4],
        "source_id": record.get("exp_id"),
    }


def abstract_to_retrieval_packet(record: dict[str, Any]) -> dict[str, Any]:
    pattern_summary = record.get("pattern_summary", {})
    supporting_pattern = pattern_summary.get("supporting_pattern", [])
    composite_seed = record.get("composite_skill_seed", {})
    trigger_pattern = composite_seed.get("trigger_pattern", pattern_summary.get("trigger_pattern", {}))
    if str(record.get("type")) == "composite_skill_seed" and isinstance(trigger_pattern, dict):
        trigger_summary_parts = []
        ddx_candidates = trigger_pattern.get("ddx_candidates", [])
        if ddx_candidates:
            trigger_summary_parts.append(f"ddx={', '.join(str(item) for item in ddx_candidates[:3])}")
        if trigger_pattern.get("decision_pattern"):
            trigger_summary_parts.append(f"decision_pattern={trigger_pattern['decision_pattern']}")
        if trigger_pattern.get("uncertainty_level"):
            trigger_summary_parts.append(f"uncertainty={trigger_pattern['uncertainty_level']}")
        if trigger_pattern.get("risk_flags"):
            trigger_summary_parts.append(
                f"risk={', '.join(str(item) for item in trigger_pattern.get('risk_flags', [])[:3])}"
            )
        perception_summary = "; ".join(trigger_summary_parts) or str(record.get("concept", ""))
    else:
        perception_summary = "; ".join(str(item) for item in supporting_pattern[:3] if str(item).strip()) or str(
            record.get("concept", "")
        )
    derived_rule = record.get("derived_rule", {})
    learning_points = dedupe_strings(
        [
            derived_rule.get("rule_text", ""),
            derived_rule.get("action_hint", ""),
            *(composite_seed.get("notes", [])[:2] if isinstance(composite_seed.get("notes", []), list) else []),
        ]
    )
    supporting_cases = record.get("supporting_cases", [])
    return {
        "case_id": supporting_cases[0] if supporting_cases else None,
        "source_layer": "abstract_experience",
        "source_subtype": record.get("type"),
        "experience_type": record.get("type"),
        "perception_summary": perception_summary,
        "confusion_pair": pattern_summary.get("confusion_pair"),
        "learning_points": learning_points[:4],
        "source_id": record.get("abs_id"),
    }


def raw_case_to_retrieval_packet(record: dict[str, Any]) -> dict[str, Any]:
    qwen_output = record.get("qwen_output", {})
    perception = record.get("agent_output", {}).get("perception", {})
    summary = str(perception.get("image_summary", "")).strip() or str(qwen_output.get("final_diagnosis", "")).strip()
    correctness = record.get("correctness", {})
    learning_points = dedupe_strings(
        [
            f"Final decision: {record.get('final_decision', {}).get('label', 'unknown')}",
            f"Correctness: {correctness.get('is_correct')}",
        ]
    )
    return {
        "case_id": record.get("case_id"),
        "source_layer": "raw_case_memory",
        "source_subtype": "case_reference",
        "experience_type": "raw_case_memory",
        "perception_summary": summary,
        "confusion_pair": None,
        "learning_points": learning_points[:2],
        "source_id": record.get("case_id"),
    }


def legacy_record_to_retrieval_packet(record: ExperienceRecord | dict[str, Any]) -> dict[str, Any]:
    payload = record.to_dict() if isinstance(record, ExperienceRecord) else dict(record)
    return {
        "case_id": payload.get("case_id"),
        "source_layer": "legacy_experience",
        "source_subtype": payload.get("experience_type"),
        "experience_type": payload.get("experience_type"),
        "perception_summary": payload.get("perception_summary"),
        "confusion_pair": payload.get("confusion_pair"),
        "learning_points": payload.get("learning_points", [])[:4],
        "source_id": payload.get("case_id"),
    }


def _uncertainty_level(state: CaseState) -> str:
    return str(
        state.uncertainty.get("uncertainty_level", state.perception.get("uncertainty", {}).get("level", "unknown"))
    ).lower()


def _decision_pattern(useful_skills: list[str]) -> str:
    skill_set = set(useful_skills)
    if {"differential_compare_skill", "uncertainty_assessment_skill"}.issubset(skill_set):
        return "compare_then_audit_uncertainty"
    if {"malignancy_risk_assessment_skill", "uncertainty_assessment_skill"}.issubset(skill_set):
        return "risk_then_uncertainty_audit"
    if {"morphology_analysis_skill", "color_pattern_analysis_skill", "border_surface_analysis_skill"}.issubset(skill_set):
        return "observe_then_structure"
    if useful_skills:
        return "structured_skill_sequence"
    return "minimal_reasoning"


def _outcome_label(state: CaseState, detected_errors: list[str]) -> str:
    if detected_errors:
        return "helpful_but_incorrect" if state.skill_outputs else "incorrect"
    if state.case_input.reference_label and state.final_diagnosis.get("final_diagnosis") == state.case_input.reference_label:
        return "correct"
    if _uncertainty_level(state) == "high":
        return "high_uncertainty"
    return "recorded"


def _reusable_scope(state: CaseState, confusion_pair: str | None) -> list[str]:
    scope = []
    if confusion_pair:
        scope.append(confusion_pair)
    scope.extend(str(item) for item in state.perception.get("ddx_candidates", [])[:2])
    if state.risk_flags:
        scope.extend(state.risk_flags[:2])
    if _uncertainty_level(state) == "high":
        scope.append("high_uncertainty_case")
    return dedupe_strings(scope)


def _scope_priority(state: CaseState, confusion_pair: str | None) -> str:
    if confusion_pair:
        return "high"
    if _uncertainty_level(state) == "high" or state.risk_flags:
        return "medium"
    return "low"


def _has_meaningful_output(output: dict[str, Any]) -> bool:
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


def _normalize_evidence_strength(output: dict[str, Any]) -> str:
    if not isinstance(output, dict):
        return "unknown"
    normalized = str(output.get("evidence_strength", "unknown")).strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return "unknown"


def _evidence_strength_score(evidence_strength: str) -> float:
    return {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.3,
    }.get(str(evidence_strength).strip().lower(), 0.0)


def _referenced_experiences(output: dict[str, Any]) -> list[str]:
    if not isinstance(output, dict):
        return []
    return [str(item).strip() for item in output.get("referenced_experiences", []) if str(item).strip()]


def _flatten_output_text(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
    elif isinstance(value, list):
        for item in value:
            texts.extend(_flatten_output_text(item))
    elif isinstance(value, dict):
        for nested in value.values():
            texts.extend(_flatten_output_text(nested))
    return texts


def _skill_impact(
    skill_name: str,
    output: dict[str, Any],
    output_present: bool,
    case_outcome: dict[str, Any],
    evidence_strength_score: float,
) -> str:
    if not output_present:
        return "harmful"
    case_status = str(case_outcome.get("status", "unknown")).lower()
    contradiction_detected = _detect_contradiction_signal(skill_name, output)
    uncertainty_reduction = _detect_uncertainty_reduction(skill_name, output, case_outcome, impact="partially_helpful")
    malignant_flag_support = _detect_malignant_flag_support(skill_name, output, case_outcome)
    support_score = 0
    if evidence_strength_score >= 0.6:
        support_score += 2
    elif evidence_strength_score > 0.0:
        support_score += 1
    if contradiction_detected:
        support_score += 1
    if uncertainty_reduction:
        support_score += 1
    if malignant_flag_support:
        support_score += 1

    if case_status == "success" and support_score >= 1:
        return "helpful"
    if case_status == "success":
        return "partially_helpful"
    if case_status == "failure" and support_score == 0:
        return "harmful"
    if support_score >= 2:
        return "partially_helpful"
    if case_outcome.get("confusion_pair") or case_outcome.get("uncertainty_level") == "high":
        return "partially_helpful"
    return "harmful" if evidence_strength_score == 0.0 else "partially_helpful"


def _skill_helpfulness(skill_name: str, output: dict[str, Any], case_outcome: dict[str, Any]) -> str:
    output_present = _has_meaningful_output(output)
    if not output_present:
        return "failure"
    impact = _skill_impact(
        skill_name=skill_name,
        output=output,
        output_present=output_present,
        case_outcome=case_outcome,
        evidence_strength_score=_evidence_strength_score(_normalize_evidence_strength(output)),
    )
    case_status = str(case_outcome.get("status", "unknown"))
    if case_status == "success":
        return "success"
    if impact == "harmful":
        return "failure"
    if case_outcome.get("confusion_pair") and skill_name in {
        "differential_compare_skill",
        "uncertainty_assessment_skill",
        "mel_nev_specialist_skill",
        "ack_scc_specialist_skill",
    }:
        return "partially_helpful"
    if case_outcome.get("uncertainty_level") == "high" and skill_name in {
        "uncertainty_assessment_skill",
        "contradiction_check_skill",
        "metadata_consistency_skill",
    }:
        return "partially_helpful"
    if case_outcome.get("risk_flags") and skill_name == "malignancy_risk_assessment_skill":
        return "partially_helpful"
    return "partially_helpful"


def _detect_contradiction_signal(skill_name: str, output: dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False
    if skill_name == "contradiction_check_skill":
        return True
    contradiction_like_fields = ("contradictions", "missing_links", "reasoning_gaps", "conflicts", "suspicious_points")
    for field_name in contradiction_like_fields:
        if _value_has_meaningful_content(output.get(field_name)):
            return True
    recommendation_type = str(output.get("recommendation_type", "")).strip().lower()
    return recommendation_type == "conflict_signal"


def _detect_malignant_flag_support(skill_name: str, output: dict[str, Any], case_outcome: dict[str, Any]) -> bool:
    if not isinstance(output, dict) or not _has_meaningful_output(output):
        return False
    if skill_name == "malignancy_risk_assessment_skill":
        return True
    if case_outcome.get("malignant_flag", {}).get("ground_truth") and str(output.get("recommendation_type", "")).strip().lower() == "risk_signal":
        return True
    text_blob = " ".join(_flatten_output_text(output)).lower()
    return any(pattern in text_blob for pattern in ("malignan", "high risk", "alarm", "melanom", "scc", "bcc"))


def _detect_uncertainty_reduction(
    skill_name: str,
    output: dict[str, Any],
    case_outcome: dict[str, Any],
    impact: str,
) -> bool:
    if not isinstance(output, dict) or not _has_meaningful_output(output):
        return False
    if skill_name in {
        "uncertainty_assessment_skill",
        "information_gap_detection_skill",
        "lesion_description_structuring_skill",
        "differential_compare_skill",
        "exclusion_reasoning_skill",
    }:
        return True
    text_blob = " ".join(_flatten_output_text(output)).lower()
    recommendation_type = str(output.get("recommendation_type", "")).strip().lower()
    if recommendation_type == "uncertainty_signal":
        return True
    return any(pattern in text_blob for pattern in ("exclude", "narrow", "less likely", "missing", "uncertain", "compare"))


def _skill_failed_aspects(
    skill_name: str,
    output_present: bool,
    case_outcome: dict[str, Any],
    impact: str,
    evidence_strength: str,
) -> list[str]:
    if not output_present:
        return ["produced little or no structured evidence"]
    issues: list[str] = []
    if case_outcome.get("confusion_pair"):
        issues.append("did not fully resolve the active confusion pattern")
    if case_outcome.get("uncertainty_level") == "high":
        issues.append("did not lower case uncertainty enough for confident narrowing")
    if case_outcome.get("status") == "failure":
        issues.append("did not prevent the final decision from failing")
    if evidence_strength in {"low", "unknown"}:
        issues.append("evidence remained weak for downstream integration")
    if impact == "harmful":
        issues.append(f"{skill_name} likely added little net value under the observed evidence state")
    if not issues:
        issues.append(f"required downstream integration beyond {skill_name}")
    return dedupe_strings(issues)


def _skill_failure_modes(
    *,
    skill_name: str,
    output_present: bool,
    case_outcome: dict[str, Any],
    evidence_strength: str,
    impact: str,
    contradiction_detected: bool,
) -> list[str]:
    failure_modes: list[str] = []
    if not output_present:
        failure_modes.append("no_structured_output")
    if impact == "harmful":
        failure_modes.append("low_net_reasoning_value")
    if case_outcome.get("confusion_pair"):
        failure_modes.append("active_confusion_unresolved")
    if case_outcome.get("uncertainty_level") == "high":
        failure_modes.append("uncertainty_remained_high")
    if case_outcome.get("status") == "failure":
        failure_modes.append("final_reasoning_failed")
    if evidence_strength in {"low", "unknown"}:
        failure_modes.append("weak_evidence_strength")
    if skill_name == "contradiction_check_skill" and not contradiction_detected:
        failure_modes.append("missed_contradiction_signal")
    return dedupe_strings(failure_modes)


def _skill_applicable_scenarios(
    skill_name: str,
    case_outcome: dict[str, Any],
    trigger_context: dict[str, Any],
) -> list[str]:
    scenarios = [skill_name]
    ddx_candidates = [str(item).strip() for item in trigger_context.get("ddx_candidates", []) if str(item).strip()]
    if ddx_candidates:
        scenarios.append(f"ddx:{'|'.join(ddx_candidates[:2])}")
    for risk_flag in trigger_context.get("risk_flags", [])[:2]:
        scenarios.append(f"risk:{risk_flag}")
    region = str(trigger_context.get("region", "")).strip()
    if region:
        scenarios.append(f"region:{region.lower()}")
    uncertainty_level = str(trigger_context.get("uncertainty_level", "unknown")).strip().lower()
    if uncertainty_level and uncertainty_level != "unknown":
        scenarios.append(f"uncertainty:{uncertainty_level}")
    confusion_pair = str(case_outcome.get("confusion_pair") or "").strip()
    if confusion_pair:
        scenarios.append(f"confusion:{confusion_pair}")
    return dedupe_strings(scenarios)


def _reuse_reason(skill_name: str, helpfulness: str, case_outcome: dict[str, Any]) -> str:
    if helpfulness == "success":
        return f"{skill_name} was useful under the observed case conditions"
    if case_outcome.get("confusion_pair"):
        return f"{skill_name} remains reusable for {case_outcome['confusion_pair']} style confusion"
    if case_outcome.get("uncertainty_level") == "high":
        return f"{skill_name} remains reusable for high-uncertainty cases"
    return f"{skill_name} provides reusable structured evidence"


def _skill_reusable_scope(state: CaseState, skill_name: str, case_outcome: dict[str, Any]) -> list[str]:
    scope = [skill_name]
    if case_outcome.get("confusion_pair"):
        scope.append(case_outcome["confusion_pair"])
    scope.extend(str(item) for item in state.perception.get("ddx_candidates", [])[:2])
    scope.extend(state.risk_flags[:2])
    if case_outcome.get("uncertainty_level") == "high":
        scope.append("high_uncertainty_case")
    return dedupe_strings(scope)


def _trigger_type(case_outcome: dict[str, Any]) -> str:
    if case_outcome.get("confusion_pair"):
        return "confusion_pair"
    if case_outcome.get("uncertainty_level") == "high":
        return "high_uncertainty"
    if case_outcome.get("risk_flags"):
        return "risk_review"
    return "routine_reasoning"
