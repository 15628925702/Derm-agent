from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.skill_helpfulness_analyzer import analyze_skill_helpfulness


def test_analyze_skill_helpfulness_accepts_bcc_benign_mimic_family_filter() -> None:
    record = {
        "case_id": "ISIC_0030070",
        "dataset_name": "HAM10000",
        "ground_truth": {
            "raw_label": "vasc",
            "canonical_label": "VASC",
            "malignant_flag": False,
        },
        "baseline_qwen": {
            "final_diagnosis": "Basal Cell Carcinoma",
        },
        "qwen_initial": {
            "ddx_candidates": ["Seborrheic Keratosis", "Actinic Keratosis", "Basal Cell Carcinoma"],
            "uncertainty": {"level": "low"},
        },
        "qwen_final": {
            "final_diagnosis": "Basal Cell Carcinoma",
            "fusion_decision": {
                "baseline_label": "Basal Cell Carcinoma",
                "agent_label": "Seborrheic Keratosis",
            },
        },
        "evaluation": {
            "correct": False,
            "baseline_correct": False,
            "topk_hit": False,
            "malignant_recall_hit": False,
            "baseline_malignant_recall_hit": False,
            "agent_vs_baseline_delta": {},
        },
        "selected_skills": ["benign_mimic_specialist_skill"],
        "skill_outputs": {
            "benign_mimic_specialist_skill": {
                "supporting_evidence": ["vascular-looking benign mimic remained plausible"],
                "opposing_evidence": ["lack of classic BCC translucency"],
                "evidence_strength": "medium",
                "recommendation_type": "comparative_support",
            }
        },
        "evidence_bundle": {
            "risk_flags": [],
            "uncertainty_summary": {"uncertainty_level": "low"},
            "contradiction_summary": {},
        },
        "reflection_summary": {
            "case_outcome": {
                "confusion_pair": "Basal Cell Carcinoma->vasc",
                "uncertainty_level": "low",
                "risk_flags": [],
            },
            "skill_assessments": [
                {
                    "skill_name": "benign_mimic_specialist_skill",
                    "selected": True,
                    "triggered": True,
                    "output_present": True,
                    "impact": "partially_helpful",
                    "helpfulness": "partially_helpful",
                    "evidence_strength": "medium",
                    "evidence_strength_score": 0.6,
                    "recommendation_type": "comparative_support",
                    "referenced_experiences": [],
                    "uncertainty_reduction": False,
                    "contradiction_detected": False,
                    "malignant_flag_support": False,
                    "evidence_usage_score": 1.0,
                    "judgement_reasons": ["structured_output_present"],
                    "harmful_reasons": [],
                    "applicable_scenarios": ["confusion:bcc_benign_mimic"],
                    "failure_modes": ["active_confusion_unresolved"],
                }
            ],
        },
    }

    reports, summary = analyze_skill_helpfulness(
        [record],
        dataset_name="HAM10000",
        confusion_pair="bcc_benign_mimic",
        min_calls=1,
        top_k=5,
    )

    assert len(reports) == 1
    assert reports[0]["skill_name"] == "benign_mimic_specialist_skill"
    assert summary["filtered_record_count"] == 1
