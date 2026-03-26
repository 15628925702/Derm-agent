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

