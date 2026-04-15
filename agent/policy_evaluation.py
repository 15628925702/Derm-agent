from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_KEY_CONFUSION_SUBSETS = ("melanoma->nev", "ack->scc")


def build_policy_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_names = sorted(
        {
            str(case.get("dataset_name", "")).strip()
            for case in case_results
            if str(case.get("dataset_name", "")).strip()
        }
    )
    cases_with_truth = [case for case in case_results if case.get("ground_truth", {}).get("canonical_label")]
    malignant_cases = [
        case for case in cases_with_truth if case.get("ground_truth", {}).get("malignant_flag") is True
    ]
    confusion_groups: dict[str, list[dict[str, Any]]] = {}
    for case in case_results:
        for tag in _case_confusion_tags(case):
            confusion_groups.setdefault(tag, []).append(case)

    summary = {
        "dataset_names": dataset_names,
        "num_cases": len(case_results),
        "num_with_ground_truth": len(cases_with_truth),
        "top1": _metric_block(cases_with_truth, "correct"),
        "topk": _metric_block(cases_with_truth, "topk_hit"),
        "malignant_recall": _metric_block(malignant_cases, "malignant_recall_hit"),
        "error_rate": _error_metric(cases_with_truth),
        "key_confusion_subsets": {
            tag: _subset_metrics(records)
            for tag, records in sorted(confusion_groups.items())
        },
    }
    return summary


def compare_policy_summaries(
    stable_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    key_confusions = set(DEFAULT_KEY_CONFUSION_SUBSETS)
    key_confusions.update(stable_summary.get("key_confusion_subsets", {}).keys())
    key_confusions.update(candidate_summary.get("key_confusion_subsets", {}).keys())
    return {
        "top1_delta": _rate_delta(candidate_summary, stable_summary, "top1"),
        "topk_delta": _rate_delta(candidate_summary, stable_summary, "topk"),
        "malignant_recall_delta": _rate_delta(candidate_summary, stable_summary, "malignant_recall"),
        "error_rate_delta": _rate_delta(candidate_summary, stable_summary, "error_rate"),
        "key_confusion_subset_deltas": {
            confusion: {
                "top1_delta": _nested_subset_rate_delta(candidate_summary, stable_summary, confusion, "top1"),
                "topk_delta": _nested_subset_rate_delta(candidate_summary, stable_summary, confusion, "topk"),
                "malignant_recall_delta": _nested_subset_rate_delta(candidate_summary, stable_summary, confusion, "malignant_recall"),
                "error_rate_delta": _nested_subset_rate_delta(candidate_summary, stable_summary, confusion, "error_rate"),
                "candidate_cases": candidate_summary.get("key_confusion_subsets", {}).get(confusion, {}).get("num_cases", 0),
                "stable_cases": stable_summary.get("key_confusion_subsets", {}).get(confusion, {}).get("num_cases", 0),
            }
            for confusion in sorted(key_confusions)
        },
    }


def gate_policy_candidate(
    *,
    stable_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    evaluation_gate: dict[str, Any],
) -> dict[str, Any]:
    gate = {
        "minimum_cases": 10,
        "max_top1_drop": 0.0,
        "max_topk_drop": 0.02,
        "max_error_rate_increase": 0.02,
        "max_key_confusion_drop": 0.0,
        "require_non_decreasing_malignant_recall": True,
        **dict(evaluation_gate or {}),
    }
    summary_delta = compare_policy_summaries(stable_summary, candidate_summary)
    reasons: list[str] = []
    rollback_required = False

    if int(candidate_summary.get("num_with_ground_truth", 0)) < int(gate.get("minimum_cases", 10)):
        return {
            "decision": "insufficient_data",
            "passed": False,
            "rollback_required": False,
            "reasons": [
                f"Candidate evaluated on only {candidate_summary.get('num_with_ground_truth', 0)} labeled cases, below minimum_cases={gate.get('minimum_cases')}.",
            ],
            "summary_delta": summary_delta,
        }

    top1_delta = summary_delta.get("top1_delta")
    topk_delta = summary_delta.get("topk_delta")
    malignant_delta = summary_delta.get("malignant_recall_delta")
    error_delta = summary_delta.get("error_rate_delta")

    if top1_delta is not None and top1_delta < -float(gate.get("max_top1_drop", 0.0)):
        reasons.append(f"top1 dropped by {top1_delta:.4f}.")
        rollback_required = True
    if topk_delta is not None and topk_delta < -float(gate.get("max_topk_drop", 0.02)):
        reasons.append(f"topk dropped by {topk_delta:.4f}.")
        rollback_required = True
    if bool(gate.get("require_non_decreasing_malignant_recall", True)) and malignant_delta is not None and malignant_delta < 0.0:
        reasons.append(f"malignant recall dropped by {malignant_delta:.4f}.")
        rollback_required = True
    if error_delta is not None and error_delta > float(gate.get("max_error_rate_increase", 0.02)):
        reasons.append(f"error rate increased by {error_delta:.4f}.")
        rollback_required = True

    max_key_confusion_drop = float(gate.get("max_key_confusion_drop", 0.0))
    for confusion, payload in summary_delta.get("key_confusion_subset_deltas", {}).items():
        candidate_cases = int(payload.get("candidate_cases", 0) or 0)
        stable_cases = int(payload.get("stable_cases", 0) or 0)
        if max(candidate_cases, stable_cases) <= 0:
            continue
        confusion_top1 = payload.get("top1_delta")
        confusion_malignant = payload.get("malignant_recall_delta")
        if confusion_top1 is not None and confusion_top1 < -max_key_confusion_drop:
            reasons.append(f"key confusion subset `{confusion}` top1 dropped by {confusion_top1:.4f}.")
            rollback_required = True
        if bool(gate.get("require_non_decreasing_malignant_recall", True)) and confusion_malignant is not None and confusion_malignant < 0.0:
            reasons.append(f"key confusion subset `{confusion}` malignant recall dropped by {confusion_malignant:.4f}.")
            rollback_required = True

    if rollback_required:
        return {
            "decision": "rollback",
            "passed": False,
            "rollback_required": True,
            "reasons": reasons,
            "summary_delta": summary_delta,
        }

    return {
        "decision": "promote",
        "passed": True,
        "rollback_required": False,
        "reasons": ["Candidate satisfies conservative policy gate."],
        "summary_delta": summary_delta,
    }


def _metric_block(cases: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    total = len(cases)
    hits = sum(1 for case in cases if case.get("evaluation", {}).get(field_name) is True)
    return {"hits": hits, "total": total, "rate": hits / total if total else None}


def _error_metric(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    errors = sum(1 for case in cases if case.get("evaluation", {}).get("correct") is False)
    return {"errors": errors, "total": total, "rate": errors / total if total else None}


def _subset_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    cases_with_truth = [case for case in cases if case.get("ground_truth", {}).get("canonical_label")]
    malignant_cases = [case for case in cases_with_truth if case.get("ground_truth", {}).get("malignant_flag") is True]
    return {
        "num_cases": len(cases),
        "num_with_ground_truth": len(cases_with_truth),
        "top1": _metric_block(cases_with_truth, "correct"),
        "topk": _metric_block(cases_with_truth, "topk_hit"),
        "malignant_recall": _metric_block(malignant_cases, "malignant_recall_hit"),
        "error_rate": _error_metric(cases_with_truth),
    }


def _case_confusion_tags(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    reflection_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    confusion_pair = str(reflection_outcome.get("confusion_pair", "")).strip()
    if confusion_pair:
        tags.append(confusion_pair.lower())
    retrieval_pair = str(record.get("skill_retrieval", {}).get("query_summary", {}).get("confusion_pair", "")).strip()
    if retrieval_pair:
        tags.append(retrieval_pair.lower())
    return list(dict.fromkeys(tag for tag in tags if tag))


def _rate_delta(candidate_summary: dict[str, Any], stable_summary: dict[str, Any], key: str) -> float | None:
    candidate_rate = candidate_summary.get(key, {}).get("rate")
    stable_rate = stable_summary.get(key, {}).get("rate")
    if candidate_rate is None or stable_rate is None:
        return None
    return round(float(candidate_rate) - float(stable_rate), 6)


def _nested_subset_rate_delta(
    candidate_summary: dict[str, Any],
    stable_summary: dict[str, Any],
    subset_key: str,
    metric_key: str,
) -> float | None:
    candidate_rate = candidate_summary.get("key_confusion_subsets", {}).get(subset_key, {}).get(metric_key, {}).get("rate")
    stable_rate = stable_summary.get("key_confusion_subsets", {}).get(subset_key, {}).get(metric_key, {}).get("rate")
    if candidate_rate is None or stable_rate is None:
        return None
    return round(float(candidate_rate) - float(stable_rate), 6)


def attach_policy_metadata(record: dict[str, Any], *, policy_id: str, policy_label: str) -> dict[str, Any]:
    enriched = deepcopy(record)
    enriched["policy_run"] = {
        "policy_id": policy_id,
        "policy_label": policy_label,
    }
    return enriched
