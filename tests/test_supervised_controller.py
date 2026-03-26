from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.planner import build_default_planner
from agent.supervised_controller import (
    ControllerMLP,
    LearnedControllerScorer,
    build_feature_vocab,
    flatten_training_example_features,
    vectorize_feature_maps,
)


def test_flatten_training_example_features_extracts_core_fields() -> None:
    example = {
        "available_skill_candidates": ["morphology_analysis_skill", "uncertainty_assessment_skill"],
        "state_features": {
            "initial_ddx": ["Melanoma", "Nevus"],
            "uncertainty": {"initial_level": "high"},
            "metadata_summary": {"region": "ARM", "age": "63", "changed": "True"},
            "confusion_tags": ["melanoma->nev"],
            "retrieval_summary": {
                "experience_layers": {"raw_case_count": 1, "tactical_count": 2, "abstract_count": 3},
                "skill_retrieval": {
                    "candidate_skill_names": ["morphology_analysis_skill", "uncertainty_assessment_skill"],
                    "retrieval_scores": {"morphology_analysis_skill": 6.2},
                },
            },
        },
    }

    features = flatten_training_example_features(example)

    assert features["ddx::melanoma"] == 1.0
    assert features["ddx_count"] == 2.0
    assert features["uncertainty_initial::high"] == 1.0
    assert features["metadata::region::arm"] == 1.0
    assert features["metadata_num::age"] == 63.0
    assert features["confusion::melanoma->nev"] == 1.0
    assert features["retrieval_raw_case_count"] == 1.0
    assert features["candidate::morphology_analysis_skill"] == 1.0
    assert features["retrieval_score::morphology_analysis_skill"] == 6.2


def test_vectorization_and_vocab_build() -> None:
    feature_maps = [
        {"a": 1.0, "b": 2.0},
        {"a": 1.0, "c": 1.0},
    ]
    vocab = build_feature_vocab(feature_maps, min_feature_count=1)
    matrix = vectorize_feature_maps(feature_maps, vocab)

    assert set(vocab.keys()) == {"a", "b", "c"}
    assert matrix.shape == (2, 3)


def test_learned_controller_scorer_load_and_predict(tmp_path) -> None:
    model = ControllerMLP(input_dim=1, hidden_dim=4, output_dim=2, dropout=0.0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.zero_()
        # Keep logits deterministic via output layer bias.
        model.layers[3].bias[0] = 2.0
        model.layers[3].bias[1] = -2.0

    checkpoint_path = tmp_path / "controller_test.pt"
    torch.save(
        {
            "feature_vocab": {"a": 0},
            "label_list": ["skill_1", "skill_2"],
            "input_dim": 1,
            "hidden_dim": 4,
            "output_dim": 2,
            "dropout": 0.0,
            "threshold": 0.5,
            "top_k": 1,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    scorer = LearnedControllerScorer(checkpoint_path=checkpoint_path)
    prediction = scorer.predict_from_feature_map({"a": 1.0}, available_skill_names=["skill_1", "skill_2"])

    assert prediction.ranked_skills[0] == "skill_1"
    assert "skill_1" in prediction.selected_skills
    assert prediction.skill_probabilities["skill_1"] > prediction.skill_probabilities["skill_2"]


def test_planner_falls_back_when_learned_checkpoint_missing() -> None:
    planner = build_default_planner(
        {
            "controller_family": "learned_supervised",
            "controller_checkpoint_path": "/tmp/not_exists_checkpoint.pt",
        }
    )
    assert planner is not None
