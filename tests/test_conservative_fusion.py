from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.conservative_fusion import decide_conservative_agent_fusion


def test_soft_fusion_allows_scin_grouped_family_override() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "DERMATITIS_ECZEMA",
        "differential_diagnoses": ["DERMATITIS_ECZEMA"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Contact Dermatitis"],
                }
            },
            "diagnosis_override_layer": {
                "selected_evidence_present": True,
                "override_allowed": True,
                "malignancy_override_allowed": True,
                "subtype_override_allowed": False,
                "family_override_allowed": True,
                "override_mode": "family_override",
                "support_margin": 3.0,
                "subtype_support_margin": 0.0,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {
            "policy": {
                "conservative_fusion_mode": "soft",
            }
        },
    }

    decision = decide_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert decision["use_agent_output"] is True
    assert "family_override_allowed" in decision["reasons"]


def test_soft_fusion_allows_keratinocyte_subtype_override() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Actinic Keratosis", "Squamous Cell Carcinoma"],
                }
            },
            "diagnosis_override_layer": {
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 18.0,
                "subtype_support_margin": 3.2,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {
            "policy": {
                "conservative_fusion_mode": "soft",
                "allow_keratinocyte_subtype_override": True,
            }
        },
    }

    decision = decide_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert decision["use_agent_output"] is True
    assert "keratinocyte_subtype_override_allowed" in decision["reasons"]
