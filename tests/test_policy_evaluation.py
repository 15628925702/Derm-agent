from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import normalize_policy_payload
from agent.policy_evaluation import build_policy_summary, gate_policy_candidate


def test_normalize_policy_payload_preserves_defaults_and_overrides() -> None:
    payload = normalize_policy_payload(
        {
            "policy_id": "candidate_test",
            "planner_policy": {"score_threshold_default": 5},
            "retrieval_policy": {"top_k_skill_candidates": 10},
        }
    )

    assert payload["policy_id"] == "candidate_test"
    assert payload["planner_policy"]["score_threshold_default"] == 5
    assert payload["planner_policy"]["foundational_bonus"] == 6
    assert payload["retrieval_policy"]["top_k_skill_candidates"] == 10
    assert payload["evaluation_gate"]["require_non_decreasing_malignant_recall"] is True


def test_gate_policy_candidate_rolls_back_on_malignant_recall_drop() -> None:
    stable_summary = {
        "num_with_ground_truth": 2,
        "top1": {"rate": 1.0},
        "topk": {"rate": 1.0},
        "malignant_recall": {"rate": 1.0},
        "error_rate": {"rate": 0.0},
        "key_confusion_subsets": {},
    }
    candidate_summary = {
        "num_with_ground_truth": 2,
        "top1": {"rate": 1.0},
        "topk": {"rate": 1.0},
        "malignant_recall": {"rate": 0.0},
        "error_rate": {"rate": 0.0},
        "key_confusion_subsets": {},
    }

    decision = gate_policy_candidate(
        stable_summary=stable_summary,
        candidate_summary=candidate_summary,
        evaluation_gate={"minimum_cases": 1},
    )

    assert decision["decision"] == "rollback"
    assert decision["rollback_required"] is True
    assert any("malignant recall dropped" in reason for reason in decision["reasons"])


def test_build_policy_summary_tracks_key_confusion_subsets() -> None:
    case_results = [
        {
            "ground_truth": {"canonical_label": "MEL", "malignant_flag": True},
            "evaluation": {"correct": True, "topk_hit": True, "malignant_recall_hit": True},
            "reflection_summary": {"case_outcome": {"confusion_pair": "melanoma->nev"}},
            "skill_retrieval": {"query_summary": {"confusion_pair": "melanoma->nev"}},
        },
        {
            "ground_truth": {"canonical_label": "NEV", "malignant_flag": False},
            "evaluation": {"correct": False, "topk_hit": True, "malignant_recall_hit": False},
            "reflection_summary": {"case_outcome": {"confusion_pair": "melanoma->nev"}},
            "skill_retrieval": {"query_summary": {"confusion_pair": "melanoma->nev"}},
        },
    ]

    summary = build_policy_summary(case_results)

    assert summary["num_with_ground_truth"] == 2
    assert summary["key_confusion_subsets"]["melanoma->nev"]["num_cases"] == 2
    assert summary["key_confusion_subsets"]["melanoma->nev"]["top1"]["rate"] == 0.5
