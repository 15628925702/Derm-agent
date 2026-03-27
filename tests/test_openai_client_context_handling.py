from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openai_client import DermOpenAIClient, FULL_CLINICAL_PROFILE_ID


def _mock_payload() -> dict:
    return {
        "initial_perception_summary": {
            "image_summary": "irregular brown lesion on forearm",
            "ddx_candidates": ["Melanoma", "Nevus", "Seborrheic Keratosis"],
        },
        "retrieved_raw_cases_summary": [
            {
                "source_id": "raw_1",
                "source_layer": "raw_case",
                "experience_type": "raw_case",
                "case_id": "CASE_A",
                "perception_summary": "similar benign brown lesion",
                "learning_points": ["stable pigment", "clear border"],
            }
        ],
        "retrieved_tactical_experiences_summary": [
            {
                "source_id": "tac_1",
                "source_layer": "tactical",
                "experience_type": "tactical_experience",
                "case_id": "CASE_B",
                "perception_summary": "keep exclusion explicit",
                "learning_points": ["negative evidence mattered"],
            }
        ],
        "retrieved_abstract_experiences_summary": [
            {
                "source_id": "abs_1",
                "source_layer": "abstract",
                "experience_type": "confusion_memory",
                "case_id": "",
                "perception_summary": "melanoma versus nevus confusion pair",
                "learning_points": ["preserve asymmetry and border complexity"],
            }
        ],
        "skill_outputs": {
            "lesion_description_structuring_skill": {
                "primary_lesion_morphology": "pigmented macule",
                "color": ["brown", "dark brown", "focal black"],
                "border": ["irregular", "poorly circumscribed"],
                "surface": ["smooth"],
                "size_count": ["solitary lesion", "approximately 6 mm"],
                "distribution": ["localized on forearm"],
                "associated_context": ["adult patient"],
                "referenced_experiences": ["abs_1", "raw_1"],
                "evidence_strength": "high",
                "recommendation_type": "descriptive_evidence",
            },
            "exclusion_reasoning_skill": {
                "unlikely_candidates": ["Nevus"],
                "exclusion_evidence": ["border complexity is greater than expected"],
                "required_missing_evidence": ["dermoscopic network detail"],
                "exclusion_confidence": "medium",
                "evidence_strength": "medium",
                "recommendation_type": "comparative_support",
            },
        },
        "risk_flags": ["malignancy_risk:medium"],
        "uncertainty_summary": {"uncertainty_level": "medium", "reasons": ["single image only"]},
        "contradiction_summary": {"contradictions": [], "missing_links": [], "reasoning_gaps": []},
        "information_gap_summary": {"missing_information": ["dermoscopy"]},
        "escalation_summary": {"whether_escalation_needed": "consider"},
        "planner_rationale": {
            "planner_type": "rule_controller",
            "planner_version": "v1",
            "selected_skills": ["lesion_description_structuring_skill", "exclusion_reasoning_skill"],
            "selection_reasons": {
                "lesion_description_structuring_skill": ["stabilize the description"],
                "exclusion_reasoning_skill": ["keep negative evidence explicit"],
            },
        },
        "notes": ["Agent generated structured evidence only."],
        "serialized_evidence_text": "\n\n".join(
            [
                "[Observation Evidence]\n- morphology line\n- color line\n- border line",
                "[Exclusion And Comparison Evidence]\n- comparison line\n- exclusion line",
                "[Risk Evidence]\n- risk line",
                "[Conflict And Uncertainty]\n- uncertainty line\n- gap line",
                "[Planner Rationale]\n- planner line",
            ]
        ),
        "evidence_calibration_debug": {"debug_only": True},
    }


def test_canonicalize_evidence_package_keeps_clinical_structure_and_drops_reference_ids() -> None:
    canonical = DermOpenAIClient._canonicalize_evidence_package(_mock_payload())

    assert canonical["compression_profile"] == FULL_CLINICAL_PROFILE_ID
    assert "evidence_calibration_debug" not in canonical
    skill_output = canonical["skill_outputs"]["lesion_description_structuring_skill"]
    assert "referenced_experiences" not in skill_output
    assert list(skill_output.keys())[:3] == ["primary_lesion_morphology", "color", "border"]


def test_compact_skill_output_uses_clinical_priority_order() -> None:
    compact = DermOpenAIClient._compact_skill_output(
        "exclusion_reasoning_skill",
        {
            "recommendation_type": "comparative_support",
            "evidence_strength": "medium",
            "required_missing_evidence": ["dermoscopy"],
            "unlikely_candidates": ["Nevus"],
            "exclusion_confidence": "medium",
            "exclusion_evidence": ["irregular border argues against nevus"],
        },
        max_fields=3,
    )

    assert list(compact.keys()) == ["unlikely_candidates", "exclusion_evidence", "required_missing_evidence"]


def test_prepare_full_profile_preserves_full_clinical_payload() -> None:
    canonical = DermOpenAIClient._canonicalize_evidence_package(_mock_payload())
    prepared = DermOpenAIClient._prepare_evidence_for_profile(canonical, {"profile_id": FULL_CLINICAL_PROFILE_ID})

    assert prepared["compression_profile"] == FULL_CLINICAL_PROFILE_ID
    serialized = json.dumps(prepared, ensure_ascii=False, separators=(",", ":"))
    assert "\"skill_outputs\"" in serialized
    assert "\"notes\"" in serialized


def test_compact_serialized_evidence_text_preserves_section_headers() -> None:
    compact_text = DermOpenAIClient._compact_serialized_evidence_text(
        _mock_payload()["serialized_evidence_text"],
        max_length=140,
    )

    assert "[Observation Evidence]" in compact_text
    assert "[Risk Evidence]" in compact_text
    assert len(compact_text) <= 140
