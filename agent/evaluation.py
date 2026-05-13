from __future__ import annotations

from typing import Any

from agent.label_space import canonicalize_label, is_malignant_label, label_space_snapshot, labels_match


TOPK_EVAL_LIMIT = 3


def evaluate_diagnosis_output(
    diagnosis_output: dict[str, Any],
    ground_truth_label: str | None,
    *,
    dataset_name: str | None = None,
    label_space_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ground_truth_canonical = canonicalize_label(
        ground_truth_label,
        dataset_name=dataset_name,
        label_space_id=label_space_id,
        metadata=metadata,
    )
    final_label = canonicalize_label(
        diagnosis_output.get("final_diagnosis"),
        dataset_name=dataset_name,
        label_space_id=label_space_id,
        metadata=metadata,
    )
    differential_labels = [
        canonicalize_label(
            item,
            dataset_name=dataset_name,
            label_space_id=label_space_id,
            metadata=metadata,
        )
        for item in diagnosis_output.get("differential_diagnoses", [])
        if canonicalize_label(
            item,
            dataset_name=dataset_name,
            label_space_id=label_space_id,
            metadata=metadata,
        )
    ]
    topk_candidates: list[str] = []
    if final_label:
        topk_candidates.append(final_label)
    for label in differential_labels:
        if label and label not in topk_candidates:
            topk_candidates.append(label)
        if len(topk_candidates) >= TOPK_EVAL_LIMIT:
            break

    malignant_truth = is_malignant_label(
        ground_truth_canonical,
        dataset_name=dataset_name,
        label_space_id=label_space_id,
        metadata=metadata,
    )
    predicted_malignant = is_malignant_label(
        final_label,
        dataset_name=dataset_name,
        label_space_id=label_space_id,
        metadata=metadata,
    )

    return {
        "label_space": label_space_snapshot(
            dataset_name=dataset_name,
            label_space_id=label_space_id,
            metadata=metadata,
        ),
        "ground_truth_canonical": ground_truth_canonical,
        "final_canonical_label": final_label,
        "differential_canonical_labels": differential_labels,
        "topk_canonical_labels": topk_candidates[:TOPK_EVAL_LIMIT],
        "topk_k": TOPK_EVAL_LIMIT,
        "correct": (
            labels_match(
                final_label or diagnosis_output.get("final_diagnosis"),
                ground_truth_canonical or ground_truth_label,
                dataset_name=dataset_name,
                label_space_id=label_space_id,
                metadata=metadata,
            )
            if ground_truth_canonical
            else None
        ),
        "topk_hit": (
            any(
                labels_match(
                    candidate,
                    ground_truth_canonical or ground_truth_label,
                    dataset_name=dataset_name,
                    label_space_id=label_space_id,
                    metadata=metadata,
                )
                for candidate in (topk_candidates[:TOPK_EVAL_LIMIT] or [diagnosis_output.get("final_diagnosis")])
            )
            if ground_truth_canonical
            else None
        ),
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
