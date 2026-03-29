from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller_training import build_controller_training_example_from_record, build_sparse_controller_targets, export_controller_training_data
from agent.hard_case_miner import load_execution_records


def test_build_controller_training_example_from_record_captures_state_and_outcome() -> None:
    record = {
        "case_id": "case_001",
        "dataset_name": "toyset",
        "qwen_initial": {
            "ddx_candidates": ["Melanoma", "Nevus"],
            "uncertainty": {"level": "high"},
        },
        "input_summary": {
            "clinical_metadata": {
                "region": "BACK",
                "age": "63",
                "changed": "True",
            }
        },
        "skill_retrieval": {
            "candidate_skill_names": ["morphology_analysis_skill", "mel_nev_specialist_skill"],
            "trigger_hits": {"mel_nev_specialist_skill": ["mel_nev_confusion", "known_confusion_match"]},
            "retrieval_scores": {"morphology_analysis_skill": 4.0, "mel_nev_specialist_skill": 6.5},
            "query_summary": {"confusion_pair": "melanoma->nev"},
        },
        "planner_decision": {
            "planner_type": "rule_based",
            "planner_version": "v1",
            "controller_family": "heuristic",
            "available_skill_candidates": ["morphology_analysis_skill", "mel_nev_specialist_skill"],
            "selected_skills": ["morphology_analysis_skill", "mel_nev_specialist_skill"],
            "selection_reasons": {
                "morphology_analysis_skill": ["Foundational observation layer."],
                "mel_nev_specialist_skill": ["Melanoma-nevus confusion detected."],
            },
            "decision_trace": [
                {"skill_name": "morphology_analysis_skill", "selected": True, "score": 6},
                {"skill_name": "mel_nev_specialist_skill", "selected": True, "score": 8},
            ],
        },
        "retrieval_bundle": {
            "after_skills": {
                "raw_case_results": [{"source_id": "raw_1"}],
                "tactical_results": [{"source_id": "tac_1"}],
                "abstract_results": [{"source_id": "abs_1"}],
                "aggregator_summary": [
                    {
                        "source_id": "abs_1",
                        "source_layer": "abstract_experience",
                        "experience_type": "confusion_memory",
                        "retrieval_score": 9,
                    }
                ],
            }
        },
        "evidence_bundle": {
            "uncertainty_summary": {
                "uncertainty_level": "medium",
                "reasons": ["Pigmented lesion remains ambiguous."],
                "missing_information": ["dermoscopic structures"],
            },
            "contradiction_summary": {
                "contradictions": ["Metadata age and malignant suspicion require careful framing."],
            },
            "planner_rationale": {"policy": "heuristic"},
        },
        "evaluation": {
            "correct": True,
            "agent_vs_baseline_delta": {
                "correct_delta": 1,
                "topk_hit_delta": 1,
                "malignant_recall_delta": 0,
            },
        },
        "reflection_summary": {
            "case_outcome": {
                "status": "success",
                "risk_flags": ["malignancy_risk:high"],
                "confusion_pair": "melanoma->nev",
            },
            "skill_assessments": [
                {"skill_name": "morphology_analysis_skill", "impact": "helpful"},
                {"skill_name": "mel_nev_specialist_skill", "impact": "partially_helpful"},
            ],
        },
    }

    example = build_controller_training_example_from_record(record, source_record_path="/tmp/record.json")

    assert example["case_id"] == "case_001"
    assert example["dataset_name"] == "toyset"
    assert example["state_features"]["initial_ddx"] == ["Melanoma", "Nevus"]
    assert example["state_features"]["confusion_tags"] == ["melanoma->nev"]
    assert example["available_skill_candidates"] == ["morphology_analysis_skill", "mel_nev_specialist_skill"]
    assert example["selected_skills"] == ["morphology_analysis_skill", "mel_nev_specialist_skill"]
    assert example["outcome"]["final_correct"] is True
    assert example["outcome"]["helpful_skills"] == ["morphology_analysis_skill"]
    assert example["outcome"]["partially_helpful_skills"] == ["mel_nev_specialist_skill"]
    assert example["outcome"]["primary_positive_skills"] == ["morphology_analysis_skill", "mel_nev_specialist_skill"]
    assert example["outcome"]["target_k"] >= 1
    assert example["future_training_views"]["contextual_bandit"]["reward"] > 0
    assert example["source_record_path"] == "/tmp/record.json"


def test_export_controller_training_data_summarizes_dataset_counts() -> None:
    records = [
        {
            "case_id": "case_a",
            "dataset_name": "dataset_a",
            "planner_decision": {"planner_type": "rule_based", "planner_version": "v1"},
        },
        {
            "case_id": "case_b",
            "dataset_name": "dataset_b",
            "planner_decision": {"planner_type": "rule_based", "planner_version": "v1"},
        },
    ]

    examples, summary = export_controller_training_data(records)

    assert len(examples) == 2
    assert summary["dataset_counts"] == {"dataset_a": 1, "dataset_b": 1}
    assert summary["planner_type_counts"] == {"rule_based": 2}
    assert "avg_target_k" in summary


def test_build_sparse_controller_targets_does_not_treat_all_selected_skills_as_positive() -> None:
    targets = build_sparse_controller_targets(
        available_skill_candidates=[
            "morphology_analysis_skill",
            "differential_compare_skill",
            "uncertainty_assessment_skill",
            "contradiction_check_skill",
        ],
        selected_skills=[
            "morphology_analysis_skill",
            "differential_compare_skill",
            "uncertainty_assessment_skill",
        ],
        helpful_skills=["morphology_analysis_skill"],
        partially_helpful_skills=["differential_compare_skill"],
        harmful_skills=["uncertainty_assessment_skill"],
        evaluation={"correct": True, "agent_vs_baseline_delta": {"correct_delta": 1}},
    )

    assert targets["primary_positive_skills"] == ["morphology_analysis_skill", "differential_compare_skill"]
    assert "uncertainty_assessment_skill" in targets["explicit_negative_skills"]
    assert targets["target_skill_scores"]["morphology_analysis_skill"] == 1.0
    assert targets["target_skill_scores"]["differential_compare_skill"] >= 0.6
    assert targets["target_skill_scores"]["uncertainty_assessment_skill"] == 0.0
    assert targets["target_k"] <= 4


def test_build_sparse_controller_targets_keeps_target_k_conservative() -> None:
    targets = build_sparse_controller_targets(
        available_skill_candidates=[
            "morphology_analysis_skill",
            "color_pattern_analysis_skill",
            "border_surface_analysis_skill",
            "distribution_analysis_skill",
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
            "contradiction_check_skill",
        ],
        selected_skills=[
            "morphology_analysis_skill",
            "color_pattern_analysis_skill",
            "border_surface_analysis_skill",
            "distribution_analysis_skill",
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
        ],
        helpful_skills=["morphology_analysis_skill", "malignancy_risk_assessment_skill"],
        partially_helpful_skills=["uncertainty_assessment_skill", "contradiction_check_skill"],
        harmful_skills=["distribution_analysis_skill"],
        evaluation={
            "correct": True,
            "agent_vs_baseline_delta": {"correct_delta": 1, "malignant_recall_delta": 1},
        },
    )

    assert targets["target_k"] <= 5
    assert "distribution_analysis_skill" in targets["explicit_negative_skills"]
    assert targets["primary_positive_skills"][:2] == [
        "morphology_analysis_skill",
        "malignancy_risk_assessment_skill",
    ]


def test_load_execution_records_prefers_richer_v2_record_for_same_case(tmp_path) -> None:
    stale_dir = tmp_path / "comparison" / "records" / "case_001"
    fresh_dir = tmp_path / "outputs" / "case_001"
    stale_dir.mkdir(parents=True)
    fresh_dir.mkdir(parents=True)

    stale_record = {
        "record_version": "v1",
        "case_id": "case_001",
        "timestamp": "2026-03-26T01:00:00Z",
        "dataset_name": "toyset",
        "planner_decision": {
            "selected_skills": ["morphology_analysis_skill"],
            "planner_type": "rule_based",
            "planner_version": "v1",
        },
        "baseline_qwen": {"final_diagnosis": "nevus"},
    }
    fresh_record = {
        "record_version": "v2",
        "case_id": "case_001",
        "timestamp": "2026-03-26T02:00:00Z",
        "dataset_name": "toyset",
        "skill_retrieval": {"candidate_skill_names": ["morphology_analysis_skill", "differential_compare_skill"]},
        "planner_decision": {
            "available_skill_candidates": ["morphology_analysis_skill", "differential_compare_skill"],
            "selected_skills": ["morphology_analysis_skill", "differential_compare_skill"],
            "planner_type": "rule_based",
            "planner_version": "v1",
            "controller_training_ready": True,
        },
        "controller_training_example": {
            "selected_skills": ["morphology_analysis_skill", "differential_compare_skill"],
        },
    }

    (stale_dir / "case_execution_record.json").write_text(json.dumps(stale_record), encoding="utf-8")
    (fresh_dir / "case_execution_record.json").write_text(json.dumps(fresh_record), encoding="utf-8")

    records = load_execution_records(tmp_path)

    assert len(records) == 1
    assert records[0]["record_version"] == "v2"
    assert records[0]["planner_decision"]["available_skill_candidates"] == [
        "morphology_analysis_skill",
        "differential_compare_skill",
    ]
