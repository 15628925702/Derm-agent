from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.run_pre_medium_smoke_eval import (
    build_recommended_medium_config,
    run_compare_quality_check,
    run_evidence_bundle_check,
)


def test_compare_quality_passes_when_agent_is_not_worse() -> None:
    payload = {
        "baseline_summary": {
            "top1": {"rate": 0.4},
            "topk": {"rate": 0.7},
            "malignant_recall": {"rate": 0.5},
            "error_rate": {"rate": 0.6},
        },
        "agent_summary": {
            "top1": {"rate": 0.5},
            "topk": {"rate": 0.8},
            "malignant_recall": {"rate": 0.5},
            "error_rate": {"rate": 0.5},
        },
        "comparison": {
            "top1_delta": 0.1,
            "topk_delta": 0.1,
            "malignant_recall_delta": 0.0,
            "error_rate_delta": -0.1,
        },
    }
    item = run_compare_quality_check(compare_payload=payload)
    assert item["status"] == "PASS"


def test_compare_quality_fails_on_malignant_recall_drop() -> None:
    payload = {
        "baseline_summary": {},
        "agent_summary": {},
        "comparison": {
            "top1_delta": 0.0,
            "topk_delta": 0.0,
            "malignant_recall_delta": -0.1,
            "error_rate_delta": 0.0,
        },
    }
    item = run_compare_quality_check(compare_payload=payload)
    assert item["status"] == "FAIL"
    assert item["blocking"] is True


def test_evidence_bundle_quality_warns_on_bloat() -> None:
    record = {
        "evidence_bundle": {
            "initial_perception_summary": "summary",
            "retrieved_raw_cases_summary": ["raw"],
            "retrieved_tactical_experiences_summary": ["tactical"],
            "retrieved_abstract_experiences_summary": ["abstract"],
            "uncertainty_summary": {"uncertainty_level": "medium", "missing_information": ["dermoscopy"]},
            "contradiction_summary": {"contradictions": ["border mismatch"]},
            "planner_rationale": {"selected_skills": ["a"]},
            "serialized_evidence_text": "risk contradiction uncertainty " + ("x" * 9500),
            "risk_flags": ["high_risk"],
            "skill_outputs": {"morphology_analysis_skill": {"summary": "x"}},
        }
    }
    item = run_evidence_bundle_check(compare_payload={"agent_records": [record]})
    assert item["status"] == "WARNING"


def test_recommended_medium_config_omitted_on_fail() -> None:
    config = build_recommended_medium_config(
        policy_path=None,  # type: ignore[arg-type]
        split_json=None,
        overall_status="FAIL",
    )
    assert config is None
