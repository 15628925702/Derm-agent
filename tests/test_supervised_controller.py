from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.planner import PlannerInput, build_default_planner
from cognition.cognition_state import CognitionState
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger
from agent.supervised_controller import (
    ControllerSelectionPolicy,
    ControllerMLP,
    LearnedControllerScorer,
    build_feature_vocab,
    flatten_training_example_features,
    select_skills_with_policy_details,
    select_skills_with_policy,
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


def test_select_skills_with_policy_enforces_sparse_bounds() -> None:
    policy = ControllerSelectionPolicy(
        threshold=0.6,
        target_top_k=2,
        min_select=2,
        max_select=3,
        top_k_buffer=0,
        relative_margin=0.15,
        floor_score=0.05,
    )
    selected = select_skills_with_policy(
        ranked_skills=["skill_1", "skill_2", "skill_3", "skill_4"],
        score_map={"skill_1": 0.9, "skill_2": 0.7, "skill_3": 0.6, "skill_4": 0.1},
        policy=policy,
    )

    assert selected == ["skill_1", "skill_2"]


def test_select_skills_with_policy_details_exposes_rejections() -> None:
    policy = ControllerSelectionPolicy(
        threshold=0.7,
        target_top_k=2,
        min_select=2,
        max_select=3,
        top_k_buffer=0,
        relative_margin=0.08,
        floor_score=0.05,
    )
    details = select_skills_with_policy_details(
        ranked_skills=["skill_1", "skill_2", "skill_3", "skill_4"],
        score_map={"skill_1": 0.92, "skill_2": 0.78, "skill_3": 0.69, "skill_4": 0.2},
        policy=policy,
    )

    assert details["selected_skills"] == ["skill_1", "skill_2"]
    assert "skill_3" in details["rejected_skills"]
    assert details["desired_k"] == 2


def test_planner_falls_back_when_learned_checkpoint_missing() -> None:
    planner = build_default_planner(
        {
            "controller_family": "learned_supervised",
            "controller_checkpoint_path": "/tmp/not_exists_checkpoint.pt",
        }
    )
    assert planner is not None


def test_planner_prioritizes_benign_mimic_guard_for_sparse_lesion_workflow() -> None:
    planner = build_default_planner({"controller_family": "heuristic"})
    available_skills = [
        _skill("morphology_analysis_skill", "Use morphology evidence."),
        _skill("color_pattern_analysis_skill", "Use color evidence."),
        _skill("lesion_description_structuring_skill", "Structure lesion description."),
        _skill("malignancy_risk_assessment_skill", "Assess malignancy risk."),
        _skill("differential_compare_skill", "Compare differential candidates."),
        _skill("exclusion_reasoning_skill", "Use exclusion and negative evidence."),
        _skill("mel_nev_specialist_skill", "Use when melanoma versus nevus confusion is active."),
        _skill("benign_mimic_specialist_skill", "Use when benign keratosis, dermatofibroma, vascular lesion, or nevus-like mimics compete with malignancy."),
    ]

    output = planner.plan(
        PlannerInput(
            perception={
                "image_summary": "A well-circumscribed slightly elevated plaque with central hypopigmentation and surrounding hyperpigmentation.",
                "ddx_candidates": ["BCC", "NV", "AKIEC"],
                "uncertainty": {"level": "low"},
                "notes": ["BCC concern remains, but benign mimic features are visible."],
            },
            metadata={"localization": "lower extremity", "age": "68"},
            cognition=CognitionState(),
            retrieved_experience_summary=[],
            available_skills=available_skills,
            workflow_context={
                "workflow_profile": "sparse_lesion_workflow",
                "workflow_capabilities": ["sparse_lesion_reasoning", "focal_lesion_reasoning"],
                "benign_mimic_like_signature": True,
                "available_tests": ["clinical_photo_only", "lesion_photo"],
                "metadata_completeness": "partial",
            },
            dataset_name="HAM10000",
        )
    )

    assert "benign_mimic_specialist_skill" in output.selected_skills
    assert "exclusion_reasoning_skill" in output.selected_skills
    assert "mel_nev_specialist_skill" not in output.selected_skills


def _skill(name: str, trigger_text: str):
    return make_skill_object(
        skill_id=f"skill.{name}.v1",
        name=name,
        description=trigger_text,
        skill_type="specialist" if "specialist" in name else "reasoning",
        triggers=[SkillTrigger(condition=trigger_text, rationale=trigger_text)],
        workflow_text=trigger_text,
        steps=[SkillStep(step_id="s1", title="Step", instruction=trigger_text)],
        watch_outs=["Do not provide final diagnosis."],
        output_schema=[SkillSchemaField(name="evidence", field_type="str", description="Evidence.")],
    )
