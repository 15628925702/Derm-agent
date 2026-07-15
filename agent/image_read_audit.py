from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from typing import Any

from agent.evaluation import evaluate_diagnosis_output
from agent.state import CaseInput, CaseState


IMAGE_READ_AUDIT_VERSION = "image_read_audit_v1"
MISSING_IMAGE_SENTINEL_PATH = "/tmp/dermagent_image_audit_missing.png"


def build_text_only_case_input(case_input: CaseInput) -> CaseInput:
    return replace(case_input, image_path=MISSING_IMAGE_SENTINEL_PATH)


def build_image_read_audit(
    *,
    case_input: CaseInput,
    primary_state: CaseState,
    counterfactual_state: CaseState,
) -> dict[str, Any]:
    perception_audit = compare_initial_perceptions(
        primary=primary_state.perception,
        counterfactual=counterfactual_state.perception,
    )
    final_audit = compare_diagnosis_outputs(
        primary=primary_state.final_diagnosis,
        counterfactual=counterfactual_state.final_diagnosis,
    )
    result_impact = build_agent_vs_text_only_summary(
        primary_output=primary_state.final_diagnosis,
        text_only_output=counterfactual_state.final_diagnosis,
        ground_truth_label=case_input.reference_label or case_input.label,
    )

    return {
        "audit_version": IMAGE_READ_AUDIT_VERSION,
        "counterfactual_type": "no_image",
        "primary_image_present": True,
        "counterfactual_image_present": False,
        "initial_perception": {
            "primary": deepcopy(primary_state.perception),
            "counterfactual": deepcopy(counterfactual_state.perception),
            **perception_audit,
        },
        "final_diagnosis": {
            "primary": deepcopy(primary_state.final_diagnosis),
            "counterfactual": deepcopy(counterfactual_state.final_diagnosis),
            **final_audit,
        },
        "result_impact": result_impact,
        "overall_verdict": _overall_verdict(
            perception_verdict=str(perception_audit.get("verdict", "")),
            final_verdict=str(final_audit.get("verdict", "")),
        ),
    }


def compare_initial_perceptions(*, primary: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    summary_similarity = text_similarity(
        str(primary.get("image_summary", "")),
        str(counterfactual.get("image_summary", "")),
    )
    primary_ddx = [str(item).strip() for item in primary.get("ddx_candidates", []) if str(item).strip()]
    counterfactual_ddx = [str(item).strip() for item in counterfactual.get("ddx_candidates", []) if str(item).strip()]
    ddx_overlap = list_overlap_ratio(primary_ddx, counterfactual_ddx)
    uncertainty_changed = normalize_text(primary.get("uncertainty", {}).get("level")) != normalize_text(
        counterfactual.get("uncertainty", {}).get("level")
    )

    if summary_similarity < 0.45 or ddx_overlap < 0.34:
        verdict = "evidence_of_image_use"
    elif summary_similarity < 0.75 or ddx_overlap < 0.67 or uncertainty_changed:
        verdict = "weak_evidence_of_image_use"
    else:
        verdict = "no_clear_evidence_of_image_use"

    return {
        "metrics": {
            "summary_similarity": round(summary_similarity, 4),
            "ddx_overlap": round(ddx_overlap, 4),
            "uncertainty_changed": uncertainty_changed,
        },
        "verdict": verdict,
    }


def compare_diagnosis_outputs(*, primary: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    primary_final = normalize_text(primary.get("final_diagnosis"))
    counterfactual_final = normalize_text(counterfactual.get("final_diagnosis"))
    diagnosis_changed = bool(primary_final) and bool(counterfactual_final) and primary_final != counterfactual_final
    differential_overlap = list_overlap_ratio(
        [str(item).strip() for item in primary.get("differential_diagnoses", []) if str(item).strip()],
        [str(item).strip() for item in counterfactual.get("differential_diagnoses", []) if str(item).strip()],
    )
    rationale_similarity = text_similarity(
        str(primary.get("rationale", "")),
        str(counterfactual.get("rationale", "")),
    )
    confidence_changed = normalize_text(primary.get("confidence")) != normalize_text(counterfactual.get("confidence"))

    if diagnosis_changed or differential_overlap < 0.34:
        verdict = "evidence_of_image_use"
    elif rationale_similarity < 0.75 or confidence_changed:
        verdict = "weak_evidence_of_image_use"
    else:
        verdict = "no_clear_evidence_of_image_use"

    return {
        "metrics": {
            "diagnosis_changed": diagnosis_changed,
            "differential_overlap": round(differential_overlap, 4),
            "rationale_similarity": round(rationale_similarity, 4),
            "confidence_changed": confidence_changed,
        },
        "verdict": verdict,
    }


def build_agent_vs_text_only_summary(
    *,
    primary_output: dict[str, Any],
    text_only_output: dict[str, Any],
    ground_truth_label: str | None,
) -> dict[str, Any]:
    primary_eval = evaluate_diagnosis_output(primary_output, ground_truth_label)
    text_only_eval = evaluate_diagnosis_output(text_only_output, ground_truth_label)
    diagnosis_shift = compare_diagnosis_outputs(primary=primary_output, counterfactual=text_only_output)
    return {
        "audit_version": IMAGE_READ_AUDIT_VERSION,
        "primary_final_diagnosis": primary_output.get("final_diagnosis"),
        "text_only_final_diagnosis": text_only_output.get("final_diagnosis"),
        "primary_correct": primary_eval.get("correct"),
        "text_only_correct": text_only_eval.get("correct"),
        "correct_delta": _delta_bool(primary_eval.get("correct"), text_only_eval.get("correct")),
        "malignant_recall_delta": _delta_bool(
            primary_eval.get("malignant_recall_hit"),
            text_only_eval.get("malignant_recall_hit"),
        ),
        "diagnosis_shift": diagnosis_shift,
        "result_changed": bool(diagnosis_shift.get("metrics", {}).get("diagnosis_changed")),
        "result_impact_verdict": _result_impact_verdict(
            diagnosis_changed=bool(diagnosis_shift.get("metrics", {}).get("diagnosis_changed")),
            correct_delta=_delta_bool(primary_eval.get("correct"), text_only_eval.get("correct")),
            malignant_recall_delta=_delta_bool(
                primary_eval.get("malignant_recall_hit"),
                text_only_eval.get("malignant_recall_hit"),
            ),
        ),
    }


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def tokenize_text(value: Any) -> set[str]:
    normalized = normalize_text(value)
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if token}


def text_similarity(left: Any, right: Any) -> float:
    left_tokens = tokenize_text(left)
    right_tokens = tokenize_text(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def list_overlap_ratio(left: list[str], right: list[str]) -> float:
    left_set = {normalize_text(item) for item in left if normalize_text(item)}
    right_set = {normalize_text(item) for item in right if normalize_text(item)}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _overall_verdict(*, perception_verdict: str, final_verdict: str) -> str:
    verdicts = {perception_verdict, final_verdict}
    if "evidence_of_image_use" in verdicts:
        return "evidence_of_image_use"
    if "weak_evidence_of_image_use" in verdicts:
        return "weak_evidence_of_image_use"
    return "no_clear_evidence_of_image_use"


def _result_impact_verdict(
    *,
    diagnosis_changed: bool,
    correct_delta: int | None,
    malignant_recall_delta: int | None,
) -> str:
    if correct_delta == 1 or malignant_recall_delta == 1:
        return "image_changes_outcome_quality"
    if diagnosis_changed:
        return "image_changes_final_result"
    return "image_does_not_change_final_result"


def _delta_bool(left: bool | None, right: bool | None) -> int | None:
    if left is None or right is None:
        return None
    return int(bool(left)) - int(bool(right))
