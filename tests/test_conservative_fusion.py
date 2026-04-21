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


def test_soft_fusion_does_not_override_without_selected_evidence() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Malignant Melanoma"],
                }
            },
            "diagnosis_override_layer": {
                "selected_evidence_present": False,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 0.0,
                "subtype_support_margin": 0.0,
                "uncertainty_level": "high",
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

    assert decision["use_agent_output"] is False
    assert "no_selected_evidence" in decision["reasons"]


def test_soft_fusion_allows_clinical_bcc_consensus_without_selected_evidence() -> None:
    baseline_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Basal Cell Carcinoma"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "clinical_full_taxonomy_lesion_workflow",
                    "workflow_capabilities": ["clinical_metadata_reasoning", "full_taxonomy_reasoning"],
                },
                "selected_evidence_present": False,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 0.0,
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
    assert decision["consensus_override_label"] == "Basal Cell Carcinoma"
    assert "clinical_bcc_consensus_override_without_selected_evidence" in decision["reasons"]


def test_soft_fusion_allows_image_archive_melanoma_subtype_consensus_without_selected_evidence() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Atypical nevus", "Superficial spreading melanoma", "Lentigo maligna"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "workflow_capabilities": ["image_archive_reasoning", "full_taxonomy_reasoning"],
                },
                "selected_evidence_present": False,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 0.0,
                "subtype_support_margin": 0.0,
                "uncertainty_level": "high",
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
    assert "image_archive_consensus_override_without_selected_evidence" in decision["reasons"]


def test_soft_fusion_blocks_generic_image_archive_melanoma_consensus_without_selected_evidence() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Atypical nevus", "Malignant melanoma", "Seborrheic keratosis"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "workflow_capabilities": ["image_archive_reasoning", "full_taxonomy_reasoning"],
                },
                "selected_evidence_present": False,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 0.0,
                "subtype_support_margin": 0.0,
                "uncertainty_level": "high",
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

    assert decision["use_agent_output"] is False
    assert "no_selected_evidence" in decision["reasons"]


def test_soft_fusion_allows_sparse_lesion_safe_override_for_bcc_to_actinic_keratosis() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Moderate",
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
                    "early_ddx_candidates": ["Rosacea", "Actinic keratosis"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "sparse_lesion_workflow",
                    "workflow_capabilities": ["sparse_lesion_reasoning", "focal_lesion_reasoning"],
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 12.0,
                "subtype_support_margin": 3.8,
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
    assert "sparse_lesion_safe_override" in decision["reasons"]


def test_soft_fusion_blocks_sparse_lesion_override_to_melanoma() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Atypical nevus", "Malignant melanoma"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "sparse_lesion_workflow",
                    "workflow_capabilities": ["sparse_lesion_reasoning", "focal_lesion_reasoning"],
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 12.0,
                "subtype_support_margin": 4.5,
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

    assert decision["use_agent_output"] is False
    assert "malignancy_override_not_allowed" in decision["reasons"] or "risk_only_mode_without_subtype_override" in decision["reasons"] or "subtype_support_margin_too_low" in decision["reasons"] or "agent_confidence_below_baseline" in decision["reasons"] or "fallback_to_baseline" in decision["reasons"]
