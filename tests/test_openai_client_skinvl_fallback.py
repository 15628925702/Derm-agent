from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.state import CaseInput
from integrations.openai_client import DermOpenAIClient


def test_skinvl_raw_text_baseline_fallback_extracts_diagnosis() -> None:
    payload = {"raw_text": "The final diagnosis is a basal cell carcinoma (BCC)."}

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="baseline_diagnosis:PAT_1",
    )

    assert normalized["final_diagnosis"] == "Basal Cell Carcinoma"
    assert normalized["differential_diagnoses"] == ["Basal Cell Carcinoma"]
    assert normalized["rationale"] == "The final diagnosis is a basal cell carcinoma (BCC)."
    assert normalized["confidence"] == "unknown"
    assert normalized["follow_up_considerations"] == []


def test_skinvl_raw_text_fallback_does_not_override_structured_payload() -> None:
    payload = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Actinic Keratosis"],
        "rationale": "Structured output already present.",
        "confidence": "medium",
        "follow_up_considerations": [],
        "raw_text": "The final diagnosis is a basal cell carcinoma (BCC).",
    }

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="final_diagnosis:PAT_2",
    )

    assert normalized == payload


def test_skinvl_label_normalizer_maps_descriptive_sentence_to_short_label() -> None:
    text = "image shows a raised, pearly bump with telangiectasia and ulceration, which is characteristic of basal cell carcinoma"
    assert DermOpenAIClient._normalize_diagnosis_label(text) == "Basal Cell Carcinoma"


def test_skinvl_allowed_label_coercion_maps_visual_description() -> None:
    text = "lesion has a well-defined border and is raised above the surrounding skin. It also has a dark center and a lighter periphery"
    assert DermOpenAIClient._coerce_skinvl_allowed_label(text) == "Squamous Cell Carcinoma"


def test_skinvl_selector_converts_descriptive_payload_to_allowed_label() -> None:
    payload = {
        "final_diagnosis": "lesion has a well-defined border and is raised above the surrounding skin. It also has a dark center and a lighter periphery",
        "differential_diagnoses": [
            "lesion has a well-defined border and is raised above the surrounding skin. It also has a dark center and a lighter periphery"
        ],
        "rationale": "The lesion has a dark center and a lighter periphery.",
        "confidence": "unknown",
        "follow_up_considerations": [],
    }

    normalized = DermOpenAIClient._apply_skinvl_selector(payload)

    assert normalized["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert normalized["differential_diagnoses"] == ["Squamous Cell Carcinoma"]


def test_skinvl_pad20_prompt_uses_derm_six_labels_only() -> None:
    case_input = CaseInput(
        case_id="PAT_SKINVL_PAD",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "derm_six"},
        dataset_name="pad20",
        label_space_id="derm_six",
    )

    labels_text = DermOpenAIClient._skinvl_allowed_labels_text(case_input)

    assert "Basal Cell Carcinoma" in labels_text
    assert "Actinic Keratosis" in labels_text
    assert "Seborrheic Keratosis" in labels_text
    assert "Contact Dermatitis" not in labels_text
    assert "Psoriasis" not in labels_text


def test_skinvl_pad20_refiner_converts_descriptive_cancer_sentence_to_valid_label() -> None:
    case_input = CaseInput(
        case_id="PAT_SKINVL_PAD",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "derm_six"},
        dataset_name="pad20",
        label_space_id="derm_six",
    )
    payload = {
        "final_diagnosis": "lesion appears to be raised and has a rough texture, which could indicate a more aggressive form of skin cancer",
        "differential_diagnoses": [
            "lesion appears to be raised and has a rough texture, which could indicate a more aggressive form of skin cancer",
            "Contact Dermatitis",
        ],
        "rationale": "The lesion appears raised and rough.",
        "confidence": "low",
        "follow_up_considerations": [],
    }

    normalized = DermOpenAIClient._refine_skinvl_payload_for_case(case_input, payload)

    assert normalized["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert normalized["differential_diagnoses"] == ["Squamous Cell Carcinoma"]
