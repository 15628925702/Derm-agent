from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.image_read_audit import (
    MISSING_IMAGE_SENTINEL_PATH,
    build_agent_vs_text_only_summary,
    build_text_only_case_input,
    compare_diagnosis_outputs,
    compare_initial_perceptions,
)
from agent.state import CaseInput


def test_build_text_only_case_input_replaces_image_path() -> None:
    case_input = CaseInput(case_id="case_1", image_path="/tmp/example.png", metadata={"age": 42})

    masked = build_text_only_case_input(case_input)

    assert masked.image_path == MISSING_IMAGE_SENTINEL_PATH
    assert masked.case_id == case_input.case_id
    assert masked.metadata == case_input.metadata


def test_compare_initial_perceptions_detects_large_counterfactual_shift() -> None:
    result = compare_initial_perceptions(
        primary={
            "image_summary": "asymmetric dark brown papule with irregular border",
            "ddx_candidates": ["Malignant Melanoma", "Nevus"],
            "uncertainty": {"level": "medium"},
        },
        counterfactual={
            "image_summary": "pink scaly plaque with diffuse surface scale",
            "ddx_candidates": ["Psoriasis", "Contact Dermatitis"],
            "uncertainty": {"level": "high"},
        },
    )

    assert result["verdict"] == "evidence_of_image_use"
    assert result["metrics"]["summary_similarity"] < 0.45


def test_compare_diagnosis_outputs_detects_same_prediction_as_no_clear_evidence() -> None:
    result = compare_diagnosis_outputs(
        primary={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis"],
            "rationale": "Pigmented lesion with benign-leaning features.",
            "confidence": "medium",
        },
        counterfactual={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis"],
            "rationale": "Pigmented lesion with benign-leaning features.",
            "confidence": "medium",
        },
    )

    assert result["verdict"] == "no_clear_evidence_of_image_use"
    assert result["metrics"]["diagnosis_changed"] is False


def test_build_agent_vs_text_only_summary_tracks_correct_delta() -> None:
    result = build_agent_vs_text_only_summary(
        primary_output={
            "final_diagnosis": "Malignant Melanoma",
            "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
            "rationale": "Asymmetry and color variegation are concerning.",
            "confidence": "medium",
            "follow_up_considerations": [],
        },
        text_only_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus"],
            "rationale": "Without the image, benign prior seems plausible.",
            "confidence": "low",
            "follow_up_considerations": [],
        },
        ground_truth_label="Malignant Melanoma",
    )

    assert result["correct_delta"] == 1
    assert result["diagnosis_shift"]["verdict"] == "evidence_of_image_use"
    assert result["result_impact_verdict"] == "image_changes_outcome_quality"
