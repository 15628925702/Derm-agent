from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.batch_reflection import run_batch_reflection


def test_run_batch_reflection_emits_success_failure_and_candidate_sections(tmp_path) -> None:
    outputs_root = tmp_path / "outputs"
    case_success = outputs_root / "case_success"
    case_failure = outputs_root / "case_failure"
    case_success.mkdir(parents=True)
    case_failure.mkdir(parents=True)

    success_record = _record(
        case_id="case_success",
        correct=True,
        status="success",
        selected_skills=["morphology_analysis_skill", "differential_compare_skill"],
        tactical_experiences=[
            _tactical(
                case_id="case_success",
                trigger_type="risk_review",
                decision_pattern="compare_then_audit_uncertainty",
                skills_used=["morphology_analysis_skill", "differential_compare_skill"],
            )
        ],
        aggregator_summary=[
            _experience("abs_rule_1", "abstract_experience", "rule"),
            _experience("tac_1", "tactical_experience", "tactical_experience"),
        ],
        confusion_pair="melanoma->nev",
    )
    failure_record = _record(
        case_id="case_failure",
        correct=False,
        status="failure",
        selected_skills=["morphology_analysis_skill", "contradiction_check_skill"],
        tactical_experiences=[
            _tactical(
                case_id="case_failure",
                trigger_type="risk_review",
                decision_pattern="compare_then_audit_uncertainty",
                skills_used=["morphology_analysis_skill", "contradiction_check_skill"],
            )
        ],
        aggregator_summary=[
            _experience("abs_conf_1", "abstract_experience", "confusion_memory"),
            _experience("tac_2", "tactical_experience", "tactical_experience"),
        ],
        confusion_pair="melanoma->nev",
        contradiction_count=4,
    )

    (case_success / "case_execution_record.json").write_text(json.dumps(success_record), encoding="utf-8")
    (case_failure / "case_execution_record.json").write_text(json.dumps(failure_record), encoding="utf-8")

    hard_cases_path = tmp_path / "hard_cases.jsonl"
    hard_cases_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case_failure",
                        "failure_type": "contradiction_heavy_case",
                        "hardness_reason": ["contradiction_heavy", "clear_confusion_pair"],
                        "involved_skills": ["contradiction_check_skill"],
                        "confusion_tags": ["melanoma->nev"],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    refinement_candidates_path = tmp_path / "skill_refinement_candidates.jsonl"
    refinement_candidates_path.write_text(
        json.dumps(
            {
                "target_skill_name": "contradiction_check_skill",
                "proposed_update_type": "workflow_update",
                "confidence": "medium",
                "supporting_cases": ["case_failure"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    critique = run_batch_reflection(
        records_root=outputs_root,
        hard_cases_path=hard_cases_path,
        refinement_candidates_path=refinement_candidates_path,
        min_support=1,
    ).to_dict()

    assert critique["source_summary"]["execution_record_count"] == 2
    assert critique["source_summary"]["success_case_count"] == 1
    assert critique["source_summary"]["failure_case_count"] == 1
    assert critique["success_clusters"]
    assert critique["failure_clusters"]
    assert critique["skill_sequence_analysis"]["successful_sequences"]
    assert critique["experience_helpfulness_analysis"]["most_helpful_abstract_experiences"]
    assert critique["emergent_confusion_memories"]
    assert critique["rule_candidates"]
    assert critique["composite_skill_seed_candidates"]
    assert critique["refinement_inputs"]["failure_patterns_for_skill_refinement"]
    assert critique["dependency_links"]["experience_consolidation"]["consumes"]


def _record(
    *,
    case_id: str,
    correct: bool,
    status: str,
    selected_skills: list[str],
    tactical_experiences: list[dict],
    aggregator_summary: list[dict],
    confusion_pair: str,
    contradiction_count: int = 0,
) -> dict:
    contradictions = ["conflict"] * contradiction_count
    return {
        "record_version": "v3",
        "case_id": case_id,
        "dataset_name": "toyset",
        "qwen_initial": {
            "ddx_candidates": ["Melanoma", "Nevus"],
        },
        "planner_decision": {
            "ordering": selected_skills,
            "selected_skills": selected_skills,
        },
        "selected_skills": selected_skills,
        "retrieval_bundle": {
            "after_skills": {
                "aggregator_summary": aggregator_summary,
            }
        },
        "evaluation": {
            "correct": correct,
            "topk_hit": True,
            "malignant_recall_hit": True if correct else False,
        },
        "ground_truth": {
            "canonical_label": "MEL" if correct else "NEV",
            "malignant_flag": True if correct else False,
        },
        "reflection_summary": {
            "case_outcome": {
                "status": status,
                "confusion_pair": confusion_pair,
                "risk_flags": ["malignancy_risk:high"],
            }
        },
        "evidence_bundle": {
            "uncertainty_summary": {"uncertainty_level": "medium" if correct else "high"},
            "contradiction_summary": {
                "contradictions": contradictions,
                "metadata_conflicts": [],
                "reasoning_gaps": [],
                "missing_links": [],
            },
        },
        "writeback_ops": {
            "planned_writeback_bundle": {
                "tactical_experiences": tactical_experiences,
            }
        },
    }


def _tactical(*, case_id: str, trigger_type: str, decision_pattern: str, skills_used: list[str]) -> dict:
    return {
        "case_id": case_id,
        "condition": {"trigger_type": trigger_type},
        "action": {
            "decision_pattern": decision_pattern,
            "skills_used": skills_used,
        },
    }


def _experience(source_id: str, source_layer: str, experience_type: str) -> dict:
    return {
        "source_id": source_id,
        "source_layer": source_layer,
        "experience_type": experience_type,
    }
