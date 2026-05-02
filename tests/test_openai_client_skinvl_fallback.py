from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openai_client import DermOpenAIClient


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.message = type("Message", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.choices = [_FakeChoice(content, finish_reason)]


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


def test_diagnosis_payload_uses_last_valid_payload_when_length_retry_is_malformed() -> None:
    client = object.__new__(DermOpenAIClient)
    responses = [
        _FakeResponse(
            (
                '{"final_diagnosis":"Contact Dermatitis",'
                '"differential_diagnoses":["Contact Dermatitis"],'
                '"rationale":"Valid but length-stopped.",'
                '"confidence":"medium",'
                '"follow_up_considerations":"Recheck if persistent."}'
            ),
            finish_reason="length",
        ),
        _FakeResponse(
            (
                '{"final_diagnosis":"Contact Dermatitis",'
                '"differential_diagnoses":["Contact Dermatitis"],'
                '"rationale":'
            ),
            finish_reason="length",
        ),
        _FakeResponse(
            (
                '{"final_diagnosis":"Contact Dermatitis",'
                '"differential_diagnoses":["Contact Dermatitis"],'
                '"rationale":'
            ),
            finish_reason="length",
        ),
    ]
    seen_token_budgets: list[int] = []

    def fake_completion(messages, max_tokens, request_name):
        seen_token_budgets.append(max_tokens)
        return responses.pop(0)

    client._create_json_completion = fake_completion

    payload = client._create_json_payload(
        messages=[],
        max_tokens=100,
        request_name="final_diagnosis:SCIN_CASE",
    )

    assert payload["final_diagnosis"] == "Contact Dermatitis"
    assert payload["rationale"] == "Valid but length-stopped."
    assert payload["follow_up_considerations"] == ["Recheck if persistent."]
    assert seen_token_budgets == [100, 260, 420]


def test_diagnosis_payload_normalizes_string_follow_up_to_list() -> None:
    payload = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "rationale": "Structured output.",
        "confidence": "medium",
        "follow_up_considerations": "Recheck if persistent.",
    }

    normalized = DermOpenAIClient._normalize_diagnosis_payload(
        payload,
        request_name="baseline_diagnosis:SCIN_CASE",
    )

    assert normalized["follow_up_considerations"] == ["Recheck if persistent."]
