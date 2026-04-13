from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.retrieval_scorer import (
    LearnedRetrievalScorer,
    RetrievalRerankerMLP,
    build_retrieval_scorer_samples_from_record,
    collect_retrieval_priors,
)
from agent.state import CaseInput, CaseState
from cognition.cognition_state import CognitionState


def _mock_record() -> dict:
    return {
        "case_id": "CASE_001",
        "dataset_name": "mock",
        "qwen_initial": {
            "ddx_candidates": ["Melanoma", "Nevus"],
            "uncertainty": {"level": "high"},
        },
        "input_summary": {
            "clinical_metadata": {"region": "ARM", "age": "62", "changed": "True"},
        },
        "retrieval_bundle": {
            "before_skills": {
                "tactical_results": [
                    {
                        "source_id": "tac_1",
                        "source_layer": "tactical_experience",
                        "source_subtype": "confusion_pair",
                        "experience_type": "tactical_experience",
                        "retrieval_score": 3.0,
                    },
                    {
                        "source_id": "tac_2",
                        "source_layer": "tactical_experience",
                        "source_subtype": "risk_review",
                        "experience_type": "tactical_experience",
                        "retrieval_score": 2.0,
                    },
                ],
                "abstract_results": [
                    {
                        "source_id": "abs_1",
                        "source_layer": "abstract_experience",
                        "source_subtype": "prototype",
                        "experience_type": "prototype",
                        "retrieval_score": 4.0,
                    }
                ],
            }
        },
        "skill_retrieval": {
            "candidate_skill_names": ["skill_a", "skill_b"],
            "retrieval_scores": {"skill_a": 1.0, "skill_b": 1.0},
            "trigger_hits": {"skill_a": ["high_uncertainty"]},
            "match_reasons": {"skill_a": ["matched uncertainty"]},
            "query_summary": {"confusion_pair": "melanoma->nev"},
        },
        "planner_decision": {"selected_skills": ["skill_a"]},
        "selected_skills": ["skill_a"],
        "skill_outputs": {
            "skill_a": {"referenced_experiences": ["tac_1", "abs_1"]},
            "skill_b": {"referenced_experiences": ["tac_2"]},
        },
        "reflection_summary": {
            "case_outcome": {"risk_flags": ["malignancy_risk:high"], "confusion_pair": "melanoma->nev"},
            "skill_assessments": [
                {"skill_name": "skill_a", "impact": "helpful"},
                {"skill_name": "skill_b", "impact": "harmful"},
            ],
        },
        "evaluation": {
            "correct": True,
            "agent_vs_baseline_delta": {"correct_delta": 1},
        },
        "cognition_snapshot": {
            "before": {
                "skill_statistics": {
                    "skill_a": {"helpful_rate": 0.7, "failure_rate": 0.1},
                    "skill_b": {"helpful_rate": 0.2, "failure_rate": 0.6},
                }
            }
        },
    }


def test_build_retrieval_samples_includes_three_object_types() -> None:
    record = _mock_record()
    priors = collect_retrieval_priors([record])
    samples = build_retrieval_scorer_samples_from_record(record, priors=priors)
    object_types = {sample["object_type"] for sample in samples}
    assert {"tactical_experience", "abstract_experience", "skill_candidate"}.issubset(object_types)
    labels = {sample["object_id"]: sample["label"] for sample in samples}
    assert labels["tac_1"] > labels["tac_2"]
    assert labels["skill_a"] > labels["skill_b"]


def test_learned_retrieval_scorer_reranks_skill_candidates_by_prior(tmp_path) -> None:
    model = RetrievalRerankerMLP(input_dim=1, hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.zero_()
    checkpoint_path = tmp_path / "retrieval_scorer.pt"
    torch.save(
        {
            "feature_vocab": {"base_retrieval_score": 0},
            "input_dim": 1,
            "hidden_dim": 4,
            "dropout": 0.0,
            "blend_weight": 2.0,
            "model_state_dict": model.state_dict(),
            "object_priors": {},
            "skill_priors": {
                "skill_helpful_rate": {"skill_a": 1.0, "skill_b": 0.0},
                "skill_failure_rate": {"skill_a": 0.0, "skill_b": 0.6},
            },
        },
        checkpoint_path,
    )
    scorer = LearnedRetrievalScorer(checkpoint_path=checkpoint_path)
    case_state = CaseState(
        case_input=CaseInput(
            case_id="CASE_001",
            image_path="/tmp/image.jpg",
            metadata={"region": "ARM"},
        )
    )
    case_state.perception = {"ddx_candidates": ["Melanoma"], "uncertainty": {"level": "high"}}
    bundle = {
        "candidate_skill_names": ["skill_b", "skill_a"],
        "candidate_skill_ids": ["id_b", "id_a"],
        "skill_id_to_name": {"id_a": "skill_a", "id_b": "skill_b"},
        "retrieval_scores": {"skill_a": 1.0, "skill_b": 1.0},
        "trigger_hits": {},
        "match_reasons": {},
        "decision_trace": [{"skill_name": "skill_a"}, {"skill_name": "skill_b"}],
    }
    cognition = CognitionState(skill_statistics={})
    reranked = scorer.rerank_skill_bundle(bundle, case_state=case_state, cognition=cognition)
    assert reranked["candidate_skill_names"][0] == "skill_a"
    assert reranked["retrieval_scores"]["skill_a"] > reranked["retrieval_scores"]["skill_b"]


def test_learned_retrieval_scorer_reranks_tactical_experiences(tmp_path) -> None:
    model = RetrievalRerankerMLP(input_dim=1, hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.zero_()
    checkpoint_path = tmp_path / "retrieval_scorer_exp.pt"
    torch.save(
        {
            "feature_vocab": {"base_retrieval_score": 0},
            "input_dim": 1,
            "hidden_dim": 4,
            "dropout": 0.0,
            "blend_weight": 1.0,
            "model_state_dict": model.state_dict(),
            "object_priors": {
                "object_helpful_rate": {"tac_2": 1.0, "tac_1": 0.0},
                "object_harmful_rate": {"tac_2": 0.0, "tac_1": 0.0},
            },
            "skill_priors": {},
        },
        checkpoint_path,
    )
    scorer = LearnedRetrievalScorer(checkpoint_path=checkpoint_path)
    case_state = CaseState(
        case_input=CaseInput(case_id="CASE_001", image_path="/tmp/image.jpg", metadata={"region": "ARM"})
    )
    case_state.perception = {"ddx_candidates": ["Melanoma"], "uncertainty": {"level": "high"}}
    bundle = {
        "query": {"top_k_tactical": 2, "top_k_abstract": 1, "top_k_raw": 0, "top_k_merged": 3},
        "tactical_results": [
            {
                "source_id": "tac_1",
                "source_layer": "tactical_experience",
                "source_subtype": "risk_review",
                "experience_type": "tactical_experience",
                "retrieval_score": 1.0,
            },
            {
                "source_id": "tac_2",
                "source_layer": "tactical_experience",
                "source_subtype": "confusion_pair",
                "experience_type": "tactical_experience",
                "retrieval_score": 1.0,
            },
        ],
        "abstract_results": [],
        "raw_case_results": [],
    }
    reranked = scorer.rerank_experience_bundle(bundle, case_state=case_state)
    assert reranked["tactical_results"][0]["source_id"] == "tac_2"
    assert "rerank_score" in reranked["tactical_results"][0]

