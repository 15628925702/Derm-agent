from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.state import CaseInput
from integrations.openai_client import (
    DermOpenAIClient,
    _build_label_space_prompt_hint,
    _build_scin_routing_hint,
    _build_xiangya_family_routing_hint,
    _refine_scin_full_label_payload,
    _refine_scin_payload_for_runtime,
)


def test_scin_label_hint_mentions_specific_scin_style_labels() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "scin_full"},
        dataset_name="scin",
        label_space_id="scin_full",
    )
    hint = _build_label_space_prompt_hint(case)
    assert "SCIN full-label note" in hint
    assert "Allergic Contact Dermatitis" in hint
    assert "Herpes Zoster" in hint


def test_normalize_diagnosis_label_prefers_more_specific_scin_labels() -> None:
    assert DermOpenAIClient._normalize_diagnosis_label("Allergic Contact Dermatitis") == "Allergic Contact Dermatitis"
    assert DermOpenAIClient._normalize_diagnosis_label("Acute dermatitis, NOS") == "Acute dermatitis, NOS"
    assert DermOpenAIClient._normalize_diagnosis_label("Herpes Zoster") == "Herpes Zoster"
    assert DermOpenAIClient._normalize_diagnosis_label("Leukocytoclastic Vasculitis") == "Leukocytoclastic Vasculitis"


def test_scin_label_hint_uses_related_category_context() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "scin_full", "related_category": "RASH"},
        dataset_name="scin",
        label_space_id="scin_full",
    )
    hint = _build_label_space_prompt_hint(case)
    assert "Given SCIN metadata category `RASH`" in hint
    assert "Acute dermatitis, NOS" in hint


def test_build_case_multimodal_content_handles_single_image_runtime_limit() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={
            "image_paths": ["/tmp/a.png", "/tmp/b.png"],
            "shot_types": ["CLOSE_UP", "AT_AN_ANGLE"],
            "label_space_id": "scin_full",
            "related_category": "RASH",
        },
        dataset_name="scin",
        label_space_id="scin_full",
    )
    client = object.__new__(DermOpenAIClient)
    client.max_images_per_prompt = 1
    content = client._build_case_multimodal_content(case, "PROMPT")
    assert content[0]["type"] == "text"
    assert "Additional image context" in content[0]["text"]
    assert "image_1_shot_type=CLOSE_UP" in content[0]["text"]


def test_refine_scin_full_label_payload_promotes_more_specific_label_from_rationale() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "scin_full", "related_category": "RASH"},
        dataset_name="scin",
        label_space_id="scin_full",
    )
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "rationale": "The lesions are most consistent with allergic contact dermatitis rather than irritant contact dermatitis.",
        "confidence": "Moderate",
        "follow_up_considerations": [],
    }
    refined = _refine_scin_full_label_payload(case, payload)
    assert refined["raw_final_diagnosis"] == "Contact Dermatitis"
    assert refined["final_diagnosis"] == "Allergic Contact Dermatitis"
    assert refined["differential_diagnoses"][0] == "Allergic Contact Dermatitis"


def test_scin_routing_hint_uses_metadata_to_build_shortlist() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={
            "label_space_id": "scin_full",
            "related_category": "RASH",
            "condition_duration": "ONE_DAY",
            "body_sites": ["leg"],
            "textures_present": ["flat"],
            "symptoms_present": ["itching"],
        },
        dataset_name="scin",
        label_space_id="scin_full",
    )
    hint = _build_scin_routing_hint(case)
    assert "Leukocytoclastic Vasculitis" in hint
    assert "Herpes Zoster" in hint


def test_refine_scin_full_label_payload_can_use_metadata_when_dermatitis_family_is_too_broad() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={
            "label_space_id": "scin_full",
            "related_category": "RASH",
            "condition_duration": "ONE_TO_THREE_MONTHS",
            "textures_present": ["rough_or_flaky"],
        },
        dataset_name="scin",
        label_space_id="scin_full",
    )
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "rationale": "The skin is rough and flaky with a subacute to chronic course.",
        "confidence": "Moderate",
        "follow_up_considerations": [],
    }
    refined = _refine_scin_full_label_payload(case, payload)
    assert refined["final_diagnosis"] == "Eczema"


def test_refine_scin_payload_for_runtime_only_routes_grouped_agent_not_grouped_baseline() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={
            "label_space_id": "scin_grouped",
            "related_category": "RASH",
            "condition_duration": "ONE_DAY",
            "body_sites": ["leg"],
            "textures_present": ["flat"],
        },
        dataset_name="scin",
        label_space_id="scin_grouped",
    )
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "rationale": "Flat red lesions on the leg.",
        "confidence": "Moderate",
        "follow_up_considerations": [],
    }
    evidence_package = {
        "selected_evidence": [
            {"summary": "flat red lesions on the leg without scaling"},
        ],
        "serialized_evidence_text": "flat red lesions on the leg with purpuric appearance",
    }

    baseline_refined = _refine_scin_payload_for_runtime(
        case_input=case,
        payload=payload,
        evidence_package=None,
        baseline_mode=True,
    )
    agent_refined = _refine_scin_payload_for_runtime(
        case_input=case,
        payload=payload,
        evidence_package=evidence_package,
        baseline_mode=False,
    )

    assert baseline_refined["final_diagnosis"] == "Contact Dermatitis"
    assert agent_refined["final_diagnosis"] == "VASCULAR_PURPURIC"


def test_refine_scin_payload_for_runtime_can_group_baseline_when_requested() -> None:
    case = CaseInput(
        case_id="scin_case",
        image_path="/tmp/missing.png",
        metadata={
            "label_space_id": "scin_grouped",
            "related_category": "RASH",
            "condition_duration": "ONE_DAY",
            "body_sites": ["leg"],
            "textures_present": ["flat"],
        },
        dataset_name="scin",
        label_space_id="scin_grouped",
    )
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "rationale": "Flat red lesions on the leg.",
        "confidence": "Moderate",
        "follow_up_considerations": [],
    }

    baseline_refined = _refine_scin_payload_for_runtime(
        case_input=case,
        payload=payload,
        evidence_package=None,
        baseline_mode=True,
        allow_grouped_baseline_refinement=True,
    )

    assert baseline_refined["final_diagnosis"] == "VASCULAR_PURPURIC"


def test_xiangya_label_hint_mentions_grouped_eczematous_family_distinctions() -> None:
    case = CaseInput(
        case_id="xiangya_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "xiangya_sft_grouped", "case_source": "xiangya_sft", "related_category": "RASH"},
        dataset_name="xiangya_sft",
        label_space_id="xiangya_sft_grouped",
        workflow_context={"workflow_profile": "eczematous_family_routing_workflow"},
    )
    hint = _build_label_space_prompt_hint(case)
    assert "Xiangya grouped-family note" in hint
    assert "ATOPIC_DERMATITIS" in hint
    assert "ECZEMA_DERMATITIS" in hint


def test_xiangya_routing_hint_is_emitted_for_specialized_workflow() -> None:
    case = CaseInput(
        case_id="xiangya_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "xiangya_sft_grouped", "case_source": "xiangya_sft", "related_category": "RASH"},
        dataset_name="xiangya_sft",
        label_space_id="xiangya_sft_grouped",
        workflow_context={"workflow_profile": "eczematous_family_routing_workflow"},
    )
    hint = _build_xiangya_family_routing_hint(case)
    assert "contact dermatitis vs atopic dermatitis vs non-specific eczema" in hint


def test_refine_xiangya_payload_prefers_atopic_over_contact_when_evidence_is_diffuse_and_recurrent() -> None:
    case = CaseInput(
        case_id="xiangya_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "xiangya_sft_grouped", "case_source": "xiangya_sft", "related_category": "RASH"},
        dataset_name="xiangya_sft",
        label_space_id="xiangya_sft_grouped",
        workflow_context={"workflow_profile": "eczematous_family_routing_workflow"},
    )
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis", "Eczema Dermatitis"],
        "rationale": "Diffuse recurrent symmetric eczematous eruption in a child with xerosis and flexural involvement.",
        "confidence": "Moderate",
        "follow_up_considerations": [],
    }
    evidence_package = {
        "selected_evidence": [
            {"summary": "generalized symmetric rash with chronic recurrent course and xerosis"},
            {"summary": "unlikely_candidates=CONTACT_DERMATITIS"},
        ],
        "serialized_evidence_text": "childhood recurrent diffuse eczematous dermatitis with xerosis and flexural pattern",
    }

    refined = _refine_scin_payload_for_runtime(
        case_input=case,
        payload=payload,
        evidence_package=evidence_package,
        baseline_mode=False,
    )

    assert refined["final_diagnosis"] == "ATOPIC_DERMATITIS"


def test_image_archive_label_hint_adds_archive_specific_guardrails() -> None:
    case = CaseInput(
        case_id="isic_case",
        image_path="/tmp/missing.png",
        metadata={"label_space_id": "isic2019_full", "anatom_site_general": "torso", "age_approx": "55"},
        dataset_name="ISIC2019",
        label_space_id="isic2019_full",
        workflow_context={"workflow_profile": "image_archive_full_taxonomy_lesion_workflow"},
    )
    hint = _build_label_space_prompt_hint(case)
    assert "Image-archive full-taxonomy note" in hint
    assert "NV" in hint
