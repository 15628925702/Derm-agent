from __future__ import annotations

from typing import Any

from agent.labels import canonicalize_label


MALIGNANT_LABELS = {"BCC", "ACK", "SCC", "MEL"}


def is_malignant_label(label: str | None) -> bool | None:
    if label is None:
        return None
    return label in MALIGNANT_LABELS


def evaluate_diagnosis_output(
    diagnosis_output: dict[str, Any],
    ground_truth_label: str | None,
) -> dict[str, Any]:
    ground_truth_canonical = canonicalize_label(ground_truth_label)
    final_label = canonicalize_label(diagnosis_output.get("final_diagnosis"))
    differential_labels = [
        canonicalize_label(item)
        for item in diagnosis_output.get("differential_diagnoses", [])
        if canonicalize_label(item)
    ]
    topk_candidates: list[str] = []
    if final_label:
        topk_candidates.append(final_label)
    for label in differential_labels:
        if label and label not in topk_candidates:
            topk_candidates.append(label)

    malignant_truth = is_malignant_label(ground_truth_canonical)
    predicted_malignant = is_malignant_label(final_label)

    return {
        "ground_truth_canonical": ground_truth_canonical,
        "final_canonical_label": final_label,
        "differential_canonical_labels": differential_labels,
        "correct": final_label == ground_truth_canonical if ground_truth_canonical else None,
        "topk_hit": ground_truth_canonical in topk_candidates if ground_truth_canonical else None,
        "malignant_recall_hit": bool(malignant_truth and predicted_malignant) if ground_truth_canonical else None,
    }


def build_agent_vs_baseline_delta(
    agent_evaluation: dict[str, Any],
    baseline_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline_evaluation:
        return {
            "correct_delta": None,
            "topk_hit_delta": None,
            "malignant_recall_delta": None,
        }
    return {
        "correct_delta": _delta_bool(agent_evaluation.get("correct"), baseline_evaluation.get("correct")),
        "topk_hit_delta": _delta_bool(agent_evaluation.get("topk_hit"), baseline_evaluation.get("topk_hit")),
        "malignant_recall_delta": _delta_bool(
            agent_evaluation.get("malignant_recall_hit"),
            baseline_evaluation.get("malignant_recall_hit"),
        ),
    }


def _delta_bool(agent_value: bool | None, baseline_value: bool | None) -> int | None:
    if agent_value is None or baseline_value is None:
        return None
    return int(bool(agent_value)) - int(bool(baseline_value))
