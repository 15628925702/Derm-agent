from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openai_client import DermOpenAIClient
from agent.state import CaseInput


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


def test_skinvl_nonpad_uses_dataset_label_space_codes() -> None:
    case_input = CaseInput(
        case_id="ISIC_CASE",
        image_path="missing.jpg",
        metadata={"label_space_id": "isic2019_full"},
        dataset_name="isic2019",
        label_space_id="isic2019_full",
    )
    payload = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
        "rationale": "Short structured rationale.",
        "confidence": "medium",
        "follow_up_considerations": [],
    }

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="baseline_diagnosis:ISIC_CASE",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert normalized["final_diagnosis"] == "MEL"
    assert normalized["differential_diagnoses"] == ["MEL", "NV", "BCC"]


def test_skinvl_nonpad_uses_metadata_label_space_when_case_field_missing() -> None:
    case_input = CaseInput(
        case_id="ISIC_METADATA_SPACE",
        image_path="missing.jpg",
        metadata={"label_space_id": "isic2019_full"},
        dataset_name="isic2019",
    )

    assert DermOpenAIClient._skinvl_allowed_labels_for_case(case_input) == (
        "MEL",
        "NV",
        "BCC",
        "AK",
        "BKL",
        "DF",
        "VASC",
        "SCC",
        "UNK",
    )
    prompt_block = DermOpenAIClient._skinvl_label_prompt_block(case_input)
    assert "Allowed canonical label IDs: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK." in prompt_block
    assert "Do not output disease descriptions" in prompt_block


def test_skinvl_prefers_workflow_label_space_for_scin_grouped() -> None:
    case_input = CaseInput(
        case_id="SCIN_GROUPED_SPACE",
        image_path="missing.jpg",
        metadata={"label_space_id": "scin_full"},
        dataset_name="scin",
        label_space_id="scin_full",
        workflow_context={"label_space_id": "scin_grouped"},
    )
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Herpes Zoster"],
        "rationale": "Short structured rationale.",
        "confidence": "medium",
        "follow_up_considerations": [],
    }

    labels = DermOpenAIClient._skinvl_allowed_labels_for_case(case_input)
    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="baseline_diagnosis:SCIN_GROUPED_SPACE",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert labels == (
        "DERMATITIS_ECZEMA",
        "URTICARIA_BITE_FOLLICULITIS",
        "INFECTION_VIRAL_FUNGAL",
        "VASCULAR_PURPURIC",
        "ACNE_ROSACEA_FOLLICULAR",
        "PIGMENT_KERATOSIS_NEVUS",
        "MALIGNANT_PREMALIGNANT",
        "OTHER",
    )
    assert normalized["final_diagnosis"] == "DERMATITIS_ECZEMA"
    assert normalized["differential_diagnoses"] == ["DERMATITIS_ECZEMA", "INFECTION_VIRAL_FUNGAL"]


def test_skinvl_prefers_model_routing_label_space_for_sd198_grouped() -> None:
    case_input = CaseInput(
        case_id="SD198_GROUPED_SPACE",
        image_path="missing.jpg",
        metadata={"label_space_id": "sd198_full"},
        dataset_name="sd198",
        label_space_id="sd198_full",
        workflow_context={"model_workflow_routing": {"label_space_id": "sd198_grouped"}},
    )
    payload = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Acne Vulgaris"],
        "rationale": "Short structured rationale.",
        "confidence": "medium",
        "follow_up_considerations": [],
    }

    labels = DermOpenAIClient._skinvl_allowed_labels_for_case(case_input)
    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="baseline_diagnosis:SD198_GROUPED_SPACE",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert "MALIGNANT_SKIN_CANCER" in labels
    assert "BASAL CELL CARCINOMA" not in labels
    assert normalized["final_diagnosis"] == "MALIGNANT_SKIN_CANCER"
    assert normalized["differential_diagnoses"] == ["MALIGNANT_SKIN_CANCER", "ACNE_FOLLICULITIS_ROSACEA"]


def test_skinvl_truncated_json_raw_text_prefers_embedded_final_label() -> None:
    case_input = CaseInput(
        case_id="SCIN_TRUNCATED_JSON",
        image_path="missing.jpg",
        metadata={"label_space_id": "scin_full"},
        dataset_name="scin",
        label_space_id="scin_full",
        workflow_context={"label_space_id": "scin_grouped"},
    )
    raw_text = (
        '{"final_diagnosis":"DERMATITIS_ECZEMA",'
        '"differential_diagnoses":["URTICARIA_BITE_FOLLICULITIS","PIGMENT_KERATOSIS_NEVUS"],'
        '"rationale":"truncated'
    )

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        {"raw_text": raw_text},
        request_name="baseline_diagnosis:SCIN_TRUNCATED_JSON",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert normalized["final_diagnosis"] == "DERMATITIS_ECZEMA"
    assert normalized["differential_diagnoses"] == [
        "DERMATITIS_ECZEMA",
        "URTICARIA_BITE_FOLLICULITIS",
        "PIGMENT_KERATOSIS_NEVUS",
    ]


def test_skinvl_nonpad_repair_echo_is_not_coerced_to_scc() -> None:
    case_input = CaseInput(
        case_id="ISIC_BAD_JSON",
        image_path="missing.jpg",
        metadata={"label_space_id": "isic2019_full"},
        dataset_name="isic2019",
        label_space_id="isic2019_full",
    )
    raw_text = (
        "Your previous answer was not valid JSON.\n"
        "Rewrite it as valid JSON only.\n"
        "Previous answer:\n"
        '{"final_diagnosis":"Squamous Cell Carcinoma","rationale":"crater"}'
    )

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        {"raw_text": raw_text},
        request_name="baseline_diagnosis:ISIC_BAD_JSON",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert normalized["final_diagnosis"] == ""
    assert normalized["differential_diagnoses"] == []
    assert normalized["parse_warning"] == "skinvl_nonpad_json_repair_echo"


def test_skinvl_nonpad_prompt_echo_is_not_coerced_from_allowed_labels() -> None:
    case_input = CaseInput(
        case_id="ISIC_PROMPT_ECHO",
        image_path="missing.jpg",
        metadata={"label_space_id": "isic2019_full"},
        dataset_name="isic2019",
        label_space_id="isic2019_full",
    )
    raw_text = (
        "Return valid JSON only. Allowed canonical label IDs: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK. "
        "The field `final_diagnosis` must be exactly one canonical label ID."
    )

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        {"raw_text": raw_text},
        request_name="baseline_diagnosis:ISIC_PROMPT_ECHO",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert normalized["final_diagnosis"] == ""
    assert normalized["differential_diagnoses"] == []
    assert normalized["parse_warning"] == "skinvl_nonpad_json_repair_echo"


def test_skinvl_pad_keeps_legacy_selector() -> None:
    case_input = CaseInput(
        case_id="PAD_CASE",
        image_path="missing.jpg",
        metadata={},
        dataset_name="pad_ufes_20",
    )
    payload = {
        "final_diagnosis": "raised lesion with dark center and lighter periphery",
        "differential_diagnoses": [],
        "rationale": "The lesion has a dark center and a lighter periphery.",
        "confidence": "unknown",
        "follow_up_considerations": [],
    }

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="baseline_diagnosis:PAD_CASE",
        case_input=case_input,
        skinvl_mode=True,
    )

    assert normalized["final_diagnosis"] == "Squamous Cell Carcinoma"
