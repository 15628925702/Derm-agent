from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.aggregator import build_evidence_bundle
from agent.evidence_calibrator import build_default_evidence_calibrator
from agent.state import CaseInput, CaseState


def _mock_skill_outputs() -> dict[str, dict]:
    return {
        "lesion_description_structuring_skill": {
            "primary_lesion_morphology": "flat macule",
            "color": "brown patchy",
            "border": "irregular",
            "surface": "smooth",
            "size_count": "small solitary",
            "distribution": "localized",
            "evidence_strength": "high",
            "recommendation_type": "descriptive_evidence",
        },
        "morphology_analysis_skill": {
            "lesion_type": "macule",
            "size_range": "small",
            "elevation": "flat",
            "count": "solitary",
            "evidence_strength": "medium",
            "recommendation_type": "descriptive_evidence",
        },
        "differential_compare_skill": {
            "candidate_pairs": ["A vs B"],
            "supporting_evidence": ["shape clue"],
            "conflicting_evidence": ["color conflict"],
            "evidence_strength": "medium",
            "recommendation_type": "comparative_support",
        },
        "exclusion_reasoning_skill": {
            "unlikely_candidates": ["B"],
            "exclusion_evidence": ["missing hallmark"],
            "required_missing_evidence": ["dermoscopy detail"],
            "exclusion_confidence": "medium",
            "evidence_strength": "medium",
            "recommendation_type": "descriptive_evidence",
        },
        "malignancy_risk_assessment_skill": {
            "risk_level": "medium",
            "risk_evidence": {"alarm": "irregular border"},
            "alarm_signals": ["irregular", "patchy"],
            "evidence_strength": "medium",
            "recommendation_type": "risk_signal",
        },
        "uncertainty_assessment_skill": {
            "uncertainty_level": "high",
            "reasons": ["missing details"],
            "missing_information": ["size metric"],
            "evidence_strength": "medium",
            "recommendation_type": "uncertainty_signal",
        },
        "information_gap_detection_skill": {
            "missing_information": ["dermoscopy"],
            "why_it_matters": ["key structure unknown"],
            "impact_on_differential": ["keeps two candidates open"],
            "uncertainty_if_missing": "high",
            "evidence_strength": "medium",
            "recommendation_type": "descriptive_evidence",
        },
    }


def test_evidence_calibrator_builds_section_plan_with_merges() -> None:
    calibrator = build_default_evidence_calibrator(
        {
            "enable_evidence_calibrator": True,
            "calibrator_mode": "heuristic",
            "max_observation_skills": 3,
            "max_comparison_skills": 2,
        }
    )
    output = calibrator.calibrate(
        {
            "skill_outputs": _mock_skill_outputs(),
            "retrieved_raw_cases_summary": [{"source_id": "raw_1", "retrieval_score": 3}],
            "retrieved_tactical_experiences_summary": [{"source_id": "tac_1", "retrieval_score": 5}],
            "retrieved_abstract_experiences_summary": [{"source_id": "abs_1", "retrieval_score": 6, "experience_type": "rule"}],
            "risk_flags": ["malignancy_risk:medium"],
            "uncertainty_summary": {"uncertainty_level": "high"},
            "contradiction_summary": {"contradictions": ["x"], "missing_links": [], "reasoning_gaps": [], "metadata_conflicts": [], "suspicious_points": []},
            "skill_retrieval_scores": {
                "lesion_description_structuring_skill": 8.0,
                "morphology_analysis_skill": 6.0,
                "exclusion_reasoning_skill": 5.0,
            },
        }
    ).to_dict()
    assert output["calibrator_type"] in {"heuristic", "heuristic_fallback"}
    observation = output["section_plan"]["observation"]
    assert "lesion_description_structuring_skill" in observation["skill_names"]
    assert isinstance(observation.get("merged_groups", []), list)
    comparison = output["section_plan"]["comparison"]
    assert any(name in comparison["skill_names"] for name in ("differential_compare_skill", "exclusion_reasoning_skill"))


def test_evidence_calibrator_routes_cluster_confusion_memory_to_comparison() -> None:
    calibrator = build_default_evidence_calibrator(
        {
            "enable_evidence_calibrator": True,
            "calibrator_mode": "heuristic",
            "max_abstract": 2,
        }
    )
    output = calibrator.calibrate(
        {
            "skill_outputs": _mock_skill_outputs(),
            "retrieved_raw_cases_summary": [],
            "retrieved_tactical_experiences_summary": [],
            "retrieved_abstract_experiences_summary": [
                {
                    "source_id": "abs_conf",
                    "retrieval_score": 7,
                    "experience_type": "confusion_memory",
                    "confusion_pair": "ack->bcc",
                    "perception_summary": "rough keratotic lesion where BCC remained plausible",
                    "learning_points": ["do not exclude BCC without explicit opposing clue"],
                },
                {
                    "source_id": "abs_risk",
                    "retrieval_score": 5,
                    "experience_type": "rule",
                    "confusion_pair": None,
                    "perception_summary": "risk framing",
                    "learning_points": ["audit uncertainty after risk framing"],
                },
            ],
            "risk_flags": ["malignancy_risk:high"],
            "uncertainty_summary": {"uncertainty_level": "high"},
            "contradiction_summary": {"contradictions": [], "missing_links": [], "reasoning_gaps": [], "metadata_conflicts": [], "suspicious_points": []},
            "skill_retrieval_scores": {},
            "confusion_clusters": ["ack_bcc_scc"],
        }
    ).to_dict()
    comparison_ids = output["section_plan"]["comparison"]["abstract_source_ids"]
    risk_ids = output["section_plan"]["risk"]["abstract_source_ids"]
    assert "abs_conf" in comparison_ids
    assert "abs_conf" not in risk_ids


def test_aggregator_includes_calibration_debug_and_serialized_text() -> None:
    state = CaseState(
        case_input=CaseInput(
            case_id="CASE_001",
            image_path="/tmp/nonexistent.png",
            metadata={"region": "ARM", "age": "62"},
        )
    )
    state.perception = {
        "image_summary": "brown irregular macule on arm",
        "ddx_candidates": ["Melanoma", "Nevus"],
        "uncertainty": {"level": "high", "reasons": ["ambiguous border"]},
        "notes": [],
    }
    state.skill_outputs = _mock_skill_outputs()
    state.retrieval_bundle = {
        "raw_case_results": [{"source_id": "raw_1", "perception_summary": "similar benign case", "retrieval_score": 2}],
        "tactical_results": [{"source_id": "tac_1", "perception_summary": "compare then exclude", "learning_points": ["keep ddx open"], "retrieval_score": 5}],
        "abstract_results": [{"source_id": "abs_1", "perception_summary": "risk framing", "learning_points": ["audit uncertainty"], "retrieval_score": 6}],
    }
    state.skill_retrieval_bundle = {
        "retrieval_scores": {
            "lesion_description_structuring_skill": 9.0,
            "exclusion_reasoning_skill": 6.0,
            "malignancy_risk_assessment_skill": 7.0,
        }
    }
    state.risk_flags = ["malignancy_risk:medium"]
    state.uncertainty = {"uncertainty_level": "high", "reasons": ["ambiguous border"], "missing_information": ["dermoscopy"]}
    state.planner_output = {"selected_skills": list(state.skill_outputs.keys()), "selection_reasons": {}}
    state.policy_snapshot = {"evidence_policy": {"enable_evidence_calibrator": True, "calibrator_mode": "heuristic"}}

    evidence_bundle = build_evidence_bundle(state)
    assert "serialized_evidence_text" in evidence_bundle
    assert "evidence_calibration_debug" in evidence_bundle
    assert evidence_bundle["evidence_calibration_debug"].get("calibrator_version") == "v1"
    assert "[Observation Evidence]" in evidence_bundle["serialized_evidence_text"]
    assert "Active confusion clusters" in evidence_bundle["serialized_evidence_text"]


def test_sparse_lesion_evidence_policy_counts_benign_mimic_specialist_as_subtype_support() -> None:
    state = CaseState(
        case_input=CaseInput(
            case_id="HAM_CASE_001",
            image_path="/tmp/nonexistent.png",
            metadata={
                "age": "70",
                "localization": "scalp",
                "diagnosis_confidence": "histopathology_confirmed",
                "has_histopathology": True,
                "label_space_id": "ham10000_full",
            },
            dataset_name="HAM10000",
            label_space_id="ham10000_full",
            workflow_context={
                "workflow_profile": "sparse_lesion_workflow",
                "workflow_capabilities": ["sparse_lesion_reasoning", "focal_lesion_reasoning"],
                "workflow_preference": "morphology_first",
                "metadata_completeness": "partial",
                "available_tests": ["clinical_photo_only", "lesion_photo", "histopathology_reference_hidden"],
                "hospital_type": "specialist_clinic",
                "time_budget": "standard",
            },
        )
    )
    state.perception = {
        "image_summary": "well-circumscribed scalp lesion with central hypopigmentation",
        "ddx_candidates": ["Atypical nevus", "Dermatofibroma", "Seborrheic keratosis"],
        "uncertainty": {"level": "medium", "reasons": ["benign mimic remains plausible"]},
        "notes": ["central hypopigmented area", "slight elevation"],
    }
    state.skill_outputs = {
        "lesion_description_structuring_skill": {
            "primary_lesion_morphology": "slightly elevated plaque",
            "color": "brown with central hypopigmented area",
            "border": "irregular, slightly raised",
            "surface": "smooth",
            "size_count": "small to medium, solitary",
            "distribution": "scalp, localized",
            "evidence_strength": "medium",
            "recommendation_type": "descriptive_evidence",
        },
        "color_pattern_analysis_skill": {
            "primary_color": "brown",
            "color_variation": "marked",
            "pigmentation_pattern": "reticular",
            "asymmetry_color": "present",
            "evidence_strength": "medium",
            "recommendation_type": "descriptive_evidence",
        },
        "malignancy_risk_assessment_skill": {
            "risk_level": "medium",
            "risk_evidence": ["irregular border", "central hypopigmented area"],
            "alarm_signals": ["irregular border", "central hypopigmented area"],
            "benign_reassuring_features": ["well-circumscribed plaque"],
            "evidence_strength": "medium",
            "recommendation_type": "risk_signal",
        },
        "benign_mimic_specialist_skill": {
            "differentiation_features": ["nevus-like symmetry", "absence of pearly translucency"],
            "supporting_evidence": ["central hypopigmented area can fit atypical nevus"],
            "opposing_evidence": ["lack of classic BCC translucency"],
            "required_missing_evidence": ["dermoscopic pigment network detail"],
            "uncertainty_under_current_evidence": "medium",
            "critical_supporting_evidence": ["nevus-like competing explanation remains active"],
            "counterexample_watchouts": ["do not over-read pigmentation alone as melanoma"],
            "evidence_strength": "medium",
            "recommendation_type": "comparative_support",
        },
        "exclusion_reasoning_skill": {
            "unlikely_candidates": ["Dermatofibroma"],
            "exclusion_evidence": ["lesion lacks classic dermatofibroma scar-like center"],
            "required_missing_evidence": ["close surface detail"],
            "exclusion_confidence": "medium",
            "evidence_strength": "medium",
            "recommendation_type": "descriptive_evidence",
        },
    }
    state.retrieval_bundle = {
        "raw_case_results": [],
        "tactical_results": [{"source_id": "tac_sparse_1", "source_layer": "tactical_experience", "retrieval_score": 4.0}],
        "abstract_results": [{"source_id": "abs_sparse_1", "source_layer": "abstract_experience", "retrieval_score": 5.0, "experience_type": "confusion_memory", "confusion_pair": "melanoma->nv"}],
        "query": {"confusion_pair": "melanoma->nv"},
    }
    state.skill_retrieval_bundle = {
        "retrieval_scores": {
            "lesion_description_structuring_skill": 8.0,
            "color_pattern_analysis_skill": 7.5,
            "malignancy_risk_assessment_skill": 7.0,
            "benign_mimic_specialist_skill": 9.0,
            "exclusion_reasoning_skill": 6.0,
        }
    }
    state.risk_flags = ["malignancy_risk:medium"]
    state.uncertainty = {
        "uncertainty_level": "medium",
        "reasons": ["benign mimic remains plausible"],
        "missing_information": ["dermoscopy"],
    }
    state.planner_output = {"selected_skills": list(state.skill_outputs.keys()), "selection_reasons": {}}
    state.policy_snapshot = {"evidence_policy": {"enable_evidence_calibrator": True, "calibrator_mode": "heuristic"}}
    state.baseline_diagnosis = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "High",
    }

    evidence_bundle = build_evidence_bundle(state)
    diagnosis_layer = evidence_bundle["evidence_decision_policy"]["diagnosis_override_layer"]

    assert diagnosis_layer["specialist_support_present"] is True
    assert diagnosis_layer["subtype_support_quota_satisfied"] is True
    assert diagnosis_layer["subtype_support_score"] > 0
