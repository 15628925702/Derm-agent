from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.hard_case_miner import build_hard_case_candidate, load_execution_records


def test_load_execution_records_keeps_distinct_split_scoped_records_for_same_case(tmp_path: Path) -> None:
    train_dir = tmp_path / "train_debug" / "case_001"
    eval_dir = tmp_path / "frozen_eval" / "case_001"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    train_record = {
        "record_version": "v2",
        "case_id": "case_001",
        "timestamp": "2026-03-26T01:00:00Z",
        "dataset_name": "toyset",
        "state_versions": {
            "run_mode": "train_debug_seed",
            "data_split": "train",
        },
        "planner_decision": {
            "available_skill_candidates": ["morphology_analysis_skill"],
            "selected_skills": ["morphology_analysis_skill"],
            "controller_training_ready": True,
        },
        "controller_training_example": {
            "selected_skills": ["morphology_analysis_skill"],
        },
    }
    eval_record = {
        "record_version": "v2",
        "case_id": "case_001",
        "timestamp": "2026-03-26T02:00:00Z",
        "dataset_name": "toyset",
        "state_versions": {
            "run_mode": "frozen_eval",
            "data_split": "val",
        },
        "baseline_qwen": {
            "final_diagnosis": "nevus",
        },
        "planner_decision": {
            "available_skill_candidates": ["morphology_analysis_skill", "differential_compare_skill"],
            "selected_skills": ["morphology_analysis_skill", "differential_compare_skill"],
        },
    }

    (train_dir / "case_execution_record.json").write_text(json.dumps(train_record), encoding="utf-8")
    (eval_dir / "case_execution_record.json").write_text(json.dumps(eval_record), encoding="utf-8")

    records = load_execution_records(tmp_path)

    assert len(records) == 2
    keyed = {
        (
            record["case_id"],
            record.get("state_versions", {}).get("run_mode"),
            record.get("state_versions", {}).get("data_split"),
        ): record
        for record in records
    }
    assert ("case_001", "train_debug_seed", "train") in keyed
    assert ("case_001", "frozen_eval", "val") in keyed


def test_build_hard_case_candidate_tolerates_null_baseline_qwen() -> None:
    record = {
        "case_id": "case_002",
        "dataset_name": "toyset",
        "baseline_qwen": None,
        "ground_truth": {
            "raw_label": "melanoma",
            "canonical_label": "melanoma",
            "malignant_flag": True,
        },
        "evaluation": {
            "correct": False,
            "baseline_correct": None,
            "topk_hit": False,
            "malignant_recall_hit": False,
            "baseline_malignant_recall_hit": None,
        },
        "qwen_initial": {
            "ddx_candidates": ["nevus", "melanoma"],
        },
        "qwen_final": {
            "final_diagnosis": "nevus",
        },
        "evidence_bundle": {
            "contradiction_summary": {
                "contradictions": ["metadata-image mismatch"],
            },
            "uncertainty_summary": {
                "uncertainty_level": "high",
            },
        },
        "reflection_summary": {
            "case_outcome": {
                "uncertainty_level": "high",
            }
        },
        "planner_decision": {
            "selected_skills": ["uncertainty_assessment_skill"],
        },
        "timestamp": "2026-03-26T03:00:00Z",
    }

    candidate = build_hard_case_candidate(record)

    assert candidate["baseline_result"]["final_diagnosis"] is None
    assert candidate["failure_type"] == "agent_failure_no_baseline"
    assert candidate["agent_result"]["uncertainty_level"] == "high"


def test_build_hard_case_candidate_tags_bcc_benign_mimic_confusion_family() -> None:
    record = {
        "case_id": "case_bcc_bkl",
        "dataset_name": "HAM10000",
        "baseline_qwen": {
            "final_diagnosis": "Basal Cell Carcinoma",
        },
        "ground_truth": {
            "raw_label": "bkl",
            "canonical_label": "BKL",
            "malignant_flag": False,
        },
        "evaluation": {
            "correct": False,
            "baseline_correct": False,
            "topk_hit": False,
            "malignant_recall_hit": None,
            "baseline_malignant_recall_hit": None,
        },
        "qwen_initial": {
            "ddx_candidates": ["BCC", "NV", "AKIEC"],
        },
        "qwen_final": {
            "final_diagnosis": "Basal Cell Carcinoma",
            "fusion_decision": {
                "baseline_label": "Basal Cell Carcinoma",
                "agent_label": "Seborrheic Keratosis",
            },
        },
        "evidence_bundle": {
            "contradiction_summary": {},
            "uncertainty_summary": {
                "uncertainty_level": "medium",
            },
        },
        "reflection_summary": {
            "case_outcome": {
                "uncertainty_level": "medium",
            }
        },
        "planner_decision": {
            "selected_skills": ["differential_compare_skill"],
        },
        "timestamp": "2026-03-26T03:30:00Z",
    }

    candidate = build_hard_case_candidate(record)

    assert "bcc_benign_mimic" in candidate["confusion_tags"]
