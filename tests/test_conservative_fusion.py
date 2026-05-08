from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.conservative_fusion import apply_conservative_agent_fusion, decide_conservative_agent_fusion


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


def test_medgemma_scin_grouped_uses_moderate_conservative_override() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "VASCULAR_PURPURIC",
        "differential_diagnoses": ["VASCULAR_PURPURIC", "DERMATITIS_ECZEMA"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["Contact Dermatitis"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__scin__grouped_core_v1",
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                },
                "selected_evidence_present": True,
                "support_margin": 52.0,
                "subtype_support_margin": 7.0,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "VASCULAR_PURPURIC"
    assert "medgemma_scin_grouped_moderate_override" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_grouped_allows_same_canonical_family_without_baseline_anchor() -> None:
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
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["Contact Dermatitis"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__scin__grouped_core_v1",
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                },
                "selected_evidence_present": True,
                "support_margin": 52.0,
                "subtype_support_margin": 7.0,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "DERMATITIS_ECZEMA"
    assert "fallback_to_baseline" not in result["fusion_decision"]["reasons"]


def test_medgemma_sd198_grouped_uses_moderate_benign_override() -> None:
    baseline_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "SUN_DAMAGE_ACTINIC",
        "differential_diagnoses": ["SUN_DAMAGE_ACTINIC", "PIGMENTARY_NEVUS_KERATOSIS"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["Actinic Keratosis"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__sd198__grouped_coarse_v1",
                    "workflow_profile": "coarse_taxonomy_workflow",
                    "label_space_id": "sd198_grouped",
                    "dataset_name": "sd198",
                },
                "selected_evidence_present": True,
                "support_margin": 52.0,
                "subtype_support_margin": 9.0,
                "uncertainty_level": "low",
                "contradiction_count": 0,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "SUN_DAMAGE_ACTINIC"
    assert "medgemma_sd198_grouped_moderate_override" in result["fusion_decision"]["reasons"]


def test_medgemma_sd198_grouped_blocks_malignant_demotion() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "PIGMENTARY_NEVUS_KERATOSIS",
        "differential_diagnoses": ["PIGMENTARY_NEVUS_KERATOSIS"],
        "confidence": "High",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["Basal Cell Carcinoma"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__sd198__grouped_coarse_v1",
                    "workflow_profile": "coarse_taxonomy_workflow",
                    "label_space_id": "sd198_grouped",
                    "dataset_name": "sd198",
                },
                "selected_evidence_present": True,
                "support_margin": 70.0,
                "subtype_support_margin": 30.0,
                "uncertainty_level": "low",
                "contradiction_count": 0,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_sd198_grouped_conservative_guard" in result["fusion_decision"]["reasons"]


def test_medgemma_sd198_grouped_allows_strong_cyst_family_malignant_demotion() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "BENIGN_TUMOR_CYST",
        "differential_diagnoses": ["BENIGN_TUMOR_CYST", "Epidermoid Cyst"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["Epidermoid Cyst"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__sd198__grouped_coarse_v1",
                    "workflow_profile": "coarse_taxonomy_workflow",
                    "label_space_id": "sd198_grouped",
                    "dataset_name": "sd198",
                },
                "selected_evidence_present": True,
                "support_margin": 45.0,
                "subtype_support_margin": 9.5,
                "uncertainty_level": "low",
                "contradiction_count": 0,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "BENIGN_TUMOR_CYST"
    assert "medgemma_sd198_grouped_moderate_override" in result["fusion_decision"]["reasons"]


def test_medgemma_sd198_grouped_allows_cheilitis_mucosal_override() -> None:
    baseline_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Actinic Cheilitis",
        "differential_diagnoses": ["Actinic Cheilitis", "Actinic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["Actinic Keratosis"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__sd198__grouped_coarse_v1",
                    "workflow_profile": "coarse_taxonomy_workflow",
                    "label_space_id": "sd198_grouped",
                    "dataset_name": "sd198",
                },
                "selected_evidence_present": True,
                "support_margin": 45.5,
                "subtype_support_margin": 7.0,
                "uncertainty_level": "low",
                "contradiction_count": 1,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Actinic Cheilitis"
    assert "medgemma_sd198_grouped_moderate_override" in result["fusion_decision"]["reasons"]


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


def test_soft_fusion_allows_sparse_lesion_safe_override_for_bcc_to_nevus() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Dermatofibroma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Atypical nevus", "Dermatofibroma", "Seborrheic keratosis"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "sparse_lesion_workflow",
                    "workflow_capabilities": ["sparse_lesion_reasoning", "focal_lesion_reasoning"],
                },
                "selected_evidence_present": True,
                "override_allowed": True,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": True,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 39.68,
                "subtype_support_margin": 7.54,
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


def test_skinvl_direct_baseline_route_always_falls_back_to_baseline() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "source_id raw_case_memory retrieval_score",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "clinical_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "clinical_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "direct_baseline_workflow",
                    "fallback_on_malformed_final": True,
                    "label_space_id": "derm_six",
                    "dataset_name": "pad_ufes_20",
                },
                "selected_evidence_present": True,
                "support_margin": 99.0,
                "subtype_support_margin": 99.0,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "model_route_malformed_final_fallback" in result["fusion_decision"]["reasons"]


def test_llama_archive_route_preserves_malignant_baseline_against_benign_drift() -> None:
    baseline_output = {
        "final_diagnosis": "MEL",
        "differential_diagnoses": ["MEL", "BKL"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "BKL",
        "differential_diagnoses": ["BKL"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": ["MEL", "BKL"]}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "conservative_archive_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": True,
                "support_margin": 12.0,
                "subtype_support_margin": 5.0,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "MEL"
    assert "llama_archive_malignant_recall_guard" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_anchors_melanoma_when_bcc_subtype_support_is_weak() -> None:
    baseline_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma"],
        "confidence": "0.95",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": [
            "Basal Cell Carcinoma",
            "Squamous Cell Carcinoma",
            "Nevus",
            "Malignant Melanoma",
        ],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["BCC", "SCC", "NV"],
                    "image_summary": "Dermoscopic image with central necrosis and surrounding inflammatory infiltrate.",
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "medgemma_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "support_margin": 19.144,
                "subtype_support_margin": -8.26,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "medgemma_isic2019_baseline_anchor_guard" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_allows_strong_bcc_override_of_melanoma_baseline() -> None:
    baseline_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["BCC", "MEL"],
                    "image_summary": "Dermoscopic image with pearly rolled border and telangiectatic structures.",
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "medgemma_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": True,
                "support_margin": 31.0,
                "subtype_support_margin": 4.0,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_isic_bcc_consensus_override" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_anterior_torso_nevus_scc_differential() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Medium",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Squamous Cell Carcinoma"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["NV", "BKL", "DF", "SCC"],
                    "image_summary": (
                        "Reddish, slightly raised lesion with a central area of increased "
                        "pigmentation and some irregular borders."
                    ),
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "medgemma_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 23.38,
                "subtype_support_margin": -3.22,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_nevus_scc_comparison_when_agent_says_bcc() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "0.95",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "skill_outputs": {
            "differential_compare_skill": {
                "candidate_pairs": ["NV vs BKL", "NV vs DF", "NV vs SCC"],
            }
        },
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["NV", "BKL", "DF"],
                    "image_summary": (
                        "Reddish, slightly raised lesion with a central area of increased "
                        "pigmentation and some irregular borders."
                    ),
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "medgemma_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 23.38,
                "subtype_support_margin": -3.22,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_does_not_promote_head_neck_nevus_scc_differential() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Medium",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Squamous Cell Carcinoma"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["NV", "BKL", "DF", "SCC"],
                    "image_summary": (
                        "Reddish, slightly raised lesion with a central area of increased "
                        "pigmentation and some irregular borders."
                    ),
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "medgemma_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "head/neck"},
                },
                "selected_evidence_present": True,
                "support_margin": 23.38,
                "subtype_support_margin": -3.22,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_lower_extremity_bcc_scc_residual() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Medium",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus", "Squamous Cell Carcinoma"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["NV", "BKL", "DF", "SCC"],
                    "image_summary": (
                        "Lesion with irregular borders, red and brown pigmentation, "
                        "and a central area of increased pigmentation."
                    ),
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 23.08,
                "subtype_support_margin": -3.22,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_lower_extremity_nevus_scc_residual() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Medium",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["NV", "BKL", "DF", "SCC"],
                    "image_summary": (
                        "Lesion with irregular borders, red and brown pigmentation, "
                        "and a central area of increased pigmentation."
                    ),
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 23.08,
                "subtype_support_margin": -3.22,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_circular_lower_extremity_melanoma_to_scc() -> None:
    baseline_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "0.95",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["NV", "BKL", "DF", "SCC"],
                    "image_summary": "Circular lesion with irregular borders and areas of pigmentation variation.",
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 23.04,
                "subtype_support_margin": -3.22,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_lower_extremity_scc_bcc_residual() -> None:
    baseline_output = {
        "final_diagnosis": "Squamous Cell Carcinoma",
        "differential_diagnoses": ["Squamous Cell Carcinoma", "Nevus"],
        "confidence": "Medium",
    }
    agent_output = {
        "final_diagnosis": "Squamous Cell Carcinoma",
        "differential_diagnoses": ["Squamous Cell Carcinoma", "Basal Cell Carcinoma", "Actinic Keratosis"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
                    "image_summary": "Lesion with irregular borders, red color, and some areas of crusting.",
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 17.964,
                "subtype_support_margin": -7.9,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_isic_route_promotes_lower_extremity_scc_bcc_crusty_vessels_residual() -> None:
    baseline_output = {
        "final_diagnosis": "Squamous Cell Carcinoma",
        "differential_diagnoses": ["Squamous Cell Carcinoma", "Nevus"],
        "confidence": "Medium",
    }
    agent_output = {
        "final_diagnosis": "Squamous Cell Carcinoma",
        "differential_diagnoses": ["Squamous Cell Carcinoma", "Nevus"],
        "confidence": "Medium",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
                    "image_summary": "Reddish, irregular lesion with visible blood vessels and a crusty surface.",
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 17.964,
                "subtype_support_margin": -7.9,
                "uncertainty_level": "low",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_isic_nv_scc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_llama_isic_route_blocks_unknown_low_margin_bcc_overwrite_of_nevus() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "Unknown",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": [],
                    "image_summary": "",
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "workflow_cell_id": "llama__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 36.52,
                "subtype_support_margin": 6.86,
                "uncertainty_level": "unknown",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "llama_isic2019_baseline_anchor_guard" in result["fusion_decision"]["reasons"]


def test_llama_isic_route_promotes_trunk_bkl_nevus_differential_to_nevus() -> None:
    baseline_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis"],
        "confidence": "Unknown",
    }
    agent_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis", "Nevus"],
        "confidence": "Unknown",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": [], "image_summary": ""}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "workflow_cell_id": "llama__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 44.44,
                "subtype_support_margin": 13.42,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "llama_isic2019_guarded_archive_override" in result["fusion_decision"]["reasons"]
    assert "Seborrheic Keratosis" in result["differential_diagnoses"]


def test_llama_isic_route_does_not_promote_head_neck_bkl_nevus_pair() -> None:
    baseline_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis"],
        "confidence": "Unknown",
    }
    agent_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis", "Nevus"],
        "confidence": "Unknown",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {"baseline_preview": {"early_ddx_candidates": [], "image_summary": ""}},
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "workflow_cell_id": "llama__isic2019__archive_guard_v1",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "head/neck"},
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": 45.46,
                "subtype_support_margin": 14.3,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert result["fusion_decision"]["consensus_override_label"] == ""


def test_qwen_isic_archive_guard_allows_melanoma_upgrade_without_benign_reassurance() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "High",
    }
    evidence_bundle = {
        "skill_outputs": {
            "malignancy_risk_assessment_skill": {
                "benign_reassuring_features": [],
            }
        },
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": False,
                "support_margin": 0.0,
                "subtype_support_margin": 0.0,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_guarded_archive_override" in result["fusion_decision"]["reasons"]


def test_qwen_isic_archive_guard_preserves_nevus_when_benign_reassurance_is_present() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "High",
    }
    evidence_bundle = {
        "skill_outputs": {
            "malignancy_risk_assessment_skill": {
                "benign_reassuring_features": ["Symmetry", "Regular border", "Uniform color"],
            }
        },
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": False,
                "support_margin": 0.0,
                "subtype_support_margin": 0.0,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_nevus_preservation_guard" in result["fusion_decision"]["reasons"]


def test_qwen_isic_high_uncertainty_melanoma_override_preserves_nevus_anchor() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "High",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: asymmetry and pigment complexity remain active, but dermoscopy and evolution history are missing."
            }
        ],
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Malignant Melanoma", "Atypical Nevus", "Dermal Nevus"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": True,
                "support_margin": 39.964,
                "subtype_support_margin": 5.36,
                "uncertainty_level": "high",
                "contradiction_count": 5,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_high_uncertainty_mel_nevus_anchor_guard" in result["fusion_decision"]["reasons"]


def test_qwen_isic_high_uncertainty_melanoma_guard_requires_target_workflow_cell() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "High",
    }
    evidence_bundle = {
        "selected_evidence": [{"summary": "Selected evidence remains melanoma-leaning."}],
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Malignant Melanoma", "Atypical Nevus", "Dermal Nevus"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "qwen__ham10000__dataset_best",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                },
                "selected_evidence_present": True,
                "support_margin": 39.964,
                "subtype_support_margin": 5.36,
                "uncertainty_level": "high",
                "contradiction_count": 5,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_high_uncertainty_mel_nevus_anchor_guard" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_high_uncertainty_guard_allows_anterior_torso_melanoma_pattern() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: irregular borders and asymmetry with a bluish hue and mixed pigment."
            }
        ],
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Malignant Melanoma", "Atypical Nevus"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 40.924,
                "subtype_support_margin": 6.04,
                "uncertainty_level": "high",
                "contradiction_count": 6,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_high_uncertainty_mel_nevus_anchor_guard" not in result["fusion_decision"]["reasons"]
    assert "qwen_isic_anterior_torso_high_uncertainty_mel_final_acceptance" in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_high_risk_bluish_pattern_accepts_negative_subtype_window() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "source_name": "mel_nev_specialist_skill",
                "summary": "mel_nev_specialist_skill: irregular borders and asymmetry; bluish hue is concerning.",
            },
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: risk_level=high | alarm_signals=Irregular borders; Asymmetry; Bluish hue | evidence_strength=high",
            },
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 42.624,
                "subtype_support_margin": -10.56,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_anterior_torso_high_uncertainty_mel_final_acceptance" in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_high_risk_bluish_pattern_blocks_central_depigmentation() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "source_name": "mel_nev_specialist_skill",
                "summary": "mel_nev_specialist_skill: irregular borders and asymmetry; bluish hue and central depigmentation.",
            },
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: risk_level=high | alarm_signals=Irregular borders; Asymmetry; Bluish hue | evidence_strength=high",
            },
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 43.524,
                "subtype_support_margin": -10.56,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_anterior_torso_high_uncertainty_mel_final_acceptance" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_lower_extremity_speckled_pattern_accepts_melanoma_final() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: speckled pattern with irregular border and asymmetry on the lower extremity."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 39.5,
                "subtype_support_margin": 4.06,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_lower_extremity_speckled_mel_final_acceptance" in result["fusion_decision"]["reasons"]


def test_qwen_isic_lower_extremity_speckled_pattern_accepts_negative_subtype_window() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: speckled pattern with irregular border and asymmetry on the lower extremity."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 40.384,
                "subtype_support_margin": -3.42,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_lower_extremity_speckled_mel_final_acceptance" in result["fusion_decision"]["reasons"]


def test_qwen_isic_lower_extremity_speckled_promotion_bypasses_nevus_preservation_guard() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "skill_outputs": {
            "malignancy_risk_assessment_skill": {
                "benign_reassuring_features": ["small size", "stable history"],
            }
        },
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: speckled pattern with irregular border and asymmetry on the lower extremity."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 40.384,
                "subtype_support_margin": -3.42,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_lower_extremity_speckled_mel_final_acceptance" in result["fusion_decision"]["reasons"]
    assert "qwen_isic_nevus_preservation_guard" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_lower_extremity_speckled_pattern_blocks_central_depression() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Malignant Melanoma",
        "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: speckled pattern with irregular border, asymmetry, and central depression."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "lower extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 39.5,
                "subtype_support_margin": 4.06,
                "uncertainty_level": "high",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_lower_extremity_speckled_mel_final_acceptance" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_headneck_umbilication_promotes_bcc_from_differential() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "malignancy_risk_assessment_skill: risk_level=medium | risk_evidence=central umbilication on a small head/neck papule with pigment variation"
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "head/neck"},
                },
                "selected_evidence_present": True,
                "support_margin": 64.0,
                "subtype_support_margin": 7.6,
                "uncertainty_level": "medium",
                "contradiction_count": 7,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert result["fusion_decision"]["consensus_override_label"] == "Basal Cell Carcinoma"
    assert "qwen_isic_headneck_bcc_umbilication_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_bcc_differential_promotion_requires_umbilication_evidence() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [{"summary": "malignancy_risk_assessment_skill: risk_level=medium | risk_evidence=irregular border only"}],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "head/neck"},
                },
                "selected_evidence_present": True,
                "support_margin": 64.0,
                "subtype_support_margin": 7.6,
                "uncertainty_level": "medium",
                "contradiction_count": 7,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_headneck_bcc_umbilication_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_headneck_ak_bcc_telangiectatic_pattern_promotes_bcc() -> None:
    baseline_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "source_name": "lesion_description_structuring_skill",
                "summary": "lesion_description_structuring_skill: color=Pink with darker pigmented areas | border=Irregular | surface=Elevated with fine telangiectasias | distribution=Head/neck",
            },
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: alarm_signals=Irregular border; Color variation with darker pigmented areas; Elevated surface with fine telangiectasias",
            },
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "head/neck"},
                },
                "selected_evidence_present": True,
                "support_margin": 44.592,
                "subtype_support_margin": 18.03,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_isic_headneck_ak_bcc_telangiectatic_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_headneck_ak_bcc_telangiectatic_pattern_blocks_central_depression() -> None:
    baseline_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "source_name": "lesion_description_structuring_skill",
                "summary": "lesion_description_structuring_skill: color=Pink with darker pigmented areas | surface=Elevated with fine telangiectasias and central depression",
            },
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "head/neck"},
                },
                "selected_evidence_present": True,
                "support_margin": 44.592,
                "subtype_support_margin": 18.03,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_isic_headneck_ak_bcc_telangiectatic_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_high_uncertainty_promotes_melanoma_from_differential() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "mel_nev_specialist_skill: irregular border, asymmetry, and bluish pigment complexity remain active."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 40.9,
                "subtype_support_margin": 6.04,
                "uncertainty_level": "high",
                "contradiction_count": 6,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_anterior_torso_high_uncertainty_mel_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_depigmented_pattern_promotes_melanoma_from_differential() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "lesion_description_structuring_skill: central depigmentation with an erythematous halo and marked color variation on the anterior torso."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 63.36,
                "subtype_support_margin": 13.56,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_anterior_torso_depigmented_mel_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_depigmented_pattern_requires_halo() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "lesion_description_structuring_skill: central depigmentation with marked color variation but no halo descriptor."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 63.36,
                "subtype_support_margin": 13.56,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_anterior_torso_depigmented_mel_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_melanoma_promotion_requires_high_uncertainty() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [{"summary": "irregular pigment pattern and color variation"}],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "anterior torso"},
                },
                "selected_evidence_present": True,
                "support_margin": 40.9,
                "subtype_support_margin": 5.7,
                "uncertainty_level": "medium",
                "contradiction_count": 6,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_anterior_torso_high_uncertainty_mel_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_upper_extremity_mottled_promotes_melanoma_from_differential() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [
            {
                "summary": "color_pattern_analysis_skill: mottled pigmentation with irregular border and marked color variation on the upper extremity."
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "upper extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 63.0,
                "subtype_support_margin": 21.5,
                "uncertainty_level": "medium",
                "contradiction_count": 5,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_upper_extremity_mottled_mel_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_upper_extremity_mottled_promotion_blocks_central_pattern() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "selected_evidence": [{"summary": "mottled pigmentation with a central dark area"}],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__isic2019__dataset_best",
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": "upper extremity"},
                },
                "selected_evidence_present": True,
                "support_margin": 63.0,
                "subtype_support_margin": 21.5,
                "uncertainty_level": "medium",
                "contradiction_count": 5,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_upper_extremity_mottled_mel_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_clinical_route_blocks_weak_benign_overwrite_when_malignant_is_in_baseline_topk() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Basal Cell Carcinoma", "Seborrheic Keratosis"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_profile": "clinical_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "clinical_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "clinical_malignant_guard_workflow",
                    "label_space_id": "derm_six",
                    "dataset_name": "pad_ufes_20",
                },
                "selected_evidence_present": True,
                "support_margin": 4.0,
                "subtype_support_margin": 2.0,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "clinical_malignant_recall_guard" in result["fusion_decision"]["reasons"]
