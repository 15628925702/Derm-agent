from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.conservative_fusion import apply_conservative_agent_fusion, decide_conservative_agent_fusion
from memory.fusion_experience.accumulation import FUSION_EXPERIENCE_ACCUMULATION_ENV
from memory.fusion_experience.workflow_fusion_decision import (
    apply_conservative_agent_fusion as memory_apply_conservative_agent_fusion,
)
from skills.workflow_fusion_decision import (
    apply_conservative_agent_fusion as legacy_apply_conservative_agent_fusion,
)


def test_legacy_skill_fusion_wrapper_points_to_memory_implementation() -> None:
    assert legacy_apply_conservative_agent_fusion is memory_apply_conservative_agent_fusion


def test_fusion_experience_accumulation_is_default_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proposal_path = tmp_path / "pending_fusion_experience.jsonl"
    monkeypatch.delenv(FUSION_EXPERIENCE_ACCUMULATION_ENV, raising=False)
    monkeypatch.setenv("DERMAGENT_FUSION_EXPERIENCE_PROPOSAL_PATH", str(proposal_path))

    apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Nevus", "differential_diagnoses": []},
        agent_output={"final_diagnosis": "Nevus", "differential_diagnoses": []},
        evidence_bundle={
            "evidence_decision_policy": {
                "diagnosis_override_layer": {
                    "workflow_context": {"workflow_cell_id": "test__cell"},
                    "selected_evidence_present": False,
                }
            },
            "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
        },
    )

    assert not proposal_path.exists()


def test_fusion_experience_accumulation_writes_pending_observation_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proposal_path = tmp_path / "pending_fusion_experience.jsonl"
    monkeypatch.setenv(FUSION_EXPERIENCE_ACCUMULATION_ENV, "1")
    monkeypatch.setenv("DERMAGENT_FUSION_EXPERIENCE_PROPOSAL_PATH", str(proposal_path))

    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Nevus", "differential_diagnoses": []},
        agent_output={"final_diagnosis": "Nevus", "differential_diagnoses": []},
        evidence_bundle={
            "evidence_decision_policy": {
                "diagnosis_override_layer": {
                    "workflow_context": {
                        "workflow_cell_id": "test__cell",
                        "dataset_name": "test_dataset",
                        "label_space_id": "test_label_space",
                    },
                    "selected_evidence_present": False,
                }
            },
            "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
        },
    )

    assert result["final_diagnosis"] == "Nevus"
    line = proposal_path.read_text(encoding="utf-8").strip()
    assert '"review_status": "pending_human_review"' in line
    assert '"runtime_effect": "none"' in line
    assert '"workflow_cell_id": "test__cell"' in line


def _qwen_isic_evidence_bundle(
    *,
    site: str,
    selected_evidence: list[dict[str, str]],
    support_margin: float = 42.0,
    subtype_support_margin: float = 16.0,
    uncertainty_level: str = "medium",
    contradiction_count: int = 0,
    workflow_cell_id: str = "qwen__isic2019__dataset_best",
) -> dict:
    return {
        "selected_evidence": selected_evidence,
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "model_workflow_profile": "qwen_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": site},
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": uncertainty_level,
                "contradiction_count": contradiction_count,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _qwen_ham10000_evidence_bundle(
    *,
    selected_evidence_text: str,
    support_margin: float = 42.0,
    subtype_support_margin: float = 18.0,
    workflow_cell_id: str = "qwen__ham10000__dataset_best",
) -> dict:
    return {
        "selected_evidence": [
            {
                "source_name": "lesion_description_structuring_skill",
                "summary": f"lesion_description_structuring_skill: {selected_evidence_text}",
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "sparse_lesion_workflow",
                    "label_space_id": "ham10000_full",
                    "dataset_name": "ham10000",
                    "clinical_metadata": {"localization": "back"},
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _dermatollama_isic2019_evidence_bundle(
    *,
    site: str,
    age: str = "70",
    primary_color: str = "brown",
    color_variation: str = "marked",
    asymmetry_color: str = "present",
    border: list[str] | None = None,
    surface: list[str] | None = None,
    supporting_evidence: list[str] | None = None,
    support_margin: float = 58.0,
    subtype_support_margin: float = 12.0,
    workflow_cell_id: str = "dermatollama__isic2019__archive_guard_v1",
) -> dict:
    return {
        "selected_evidence": [
            {
                "source_name": "differential_compare_skill",
                "summary": "differential_compare_skill: supporting_evidence=irregular border; color variation",
            }
        ],
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "model_workflow_profile": "dermatollama_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {
                        "anatom_site_general": site,
                        "age_approx": age,
                    },
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": "medium",
                "contradiction_count": 0,
            },
        },
        "skill_outputs": {
            "lesion_description_structuring_skill": {
                "border": list(border or ["irregular"]),
                "surface": list(surface or ["flat"]),
            },
            "color_pattern_analysis_skill": {
                "primary_color": primary_color,
                "color_variation": color_variation,
                "asymmetry_color": asymmetry_color,
            },
            "differential_compare_skill": {
                "supporting_evidence": list(supporting_evidence or ["irregular border", "color variation"]),
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _dermatollama_ham10000_evidence_bundle(
    *,
    localization: str,
    surface_texture: str = "smooth",
    border_clarity: str = "poor",
    border_irregularity: str = "irregular",
    clustering_pattern: str = "solitary",
    color: list[str] | None = None,
    primary_lesion_morphology: str = "macule",
    support_margin: float = 39.0,
    subtype_support_margin: float = 1.0,
    uncertainty_level: str = "low",
    workflow_cell_id: str = "dermatollama__ham10000__baseline_guard_v1",
) -> dict:
    return {
        "evidence_decision_policy": {
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "sparse_lesion_workflow",
                    "label_space_id": "ham10000_full",
                    "dataset_name": "ham10000",
                    "clinical_metadata": {
                        "localization": localization,
                        "region": localization,
                    },
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": uncertainty_level,
            },
        },
        "skill_outputs": {
            "border_surface_analysis_skill": {
                "border_clarity": border_clarity,
                "border_irregularity": border_irregularity,
                "surface_texture": surface_texture,
            },
            "distribution_analysis_skill": {
                "body_location": localization,
                "clustering_pattern": clustering_pattern,
            },
            "lesion_description_structuring_skill": {
                "primary_lesion_morphology": primary_lesion_morphology,
                "color": list(color or ["brown"]),
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _qwen_scin_evidence_bundle(
    *,
    early_ddx_candidates: list[str],
    clinical_metadata: dict | None = None,
    skill_outputs: dict | None = None,
    uncertainty_level: str = "medium",
    workflow_cell_id: str = "qwen__scin__grouped_best",
) -> dict:
    return {
        "skill_outputs": dict(skill_outputs or {}),
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": early_ddx_candidates,
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                    "clinical_metadata": dict(clinical_metadata or {}),
                },
                "selected_evidence_present": True,
                "support_margin": 60.0,
                "subtype_support_margin": 14.0,
                "uncertainty_level": uncertainty_level,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def test_dermatollama_isic_upper_extremity_melanoma_topk_promotion() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(site="upper extremity", age="65", support_margin=60.2)

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "dermatollama_isic_upper_extremity_melanoma_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_upper_extremity_melanoma_requires_target_cell_and_support() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="upper extremity",
        age="65",
        support_margin=59.9,
        workflow_cell_id="qwen__isic2019__dataset_best",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_isic_upper_extremity_melanoma_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_upper_extremity_melanoma_allows_bkl_prefusion_anchor() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Seborrheic Keratosis",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(site="upper extremity", age="60", support_margin=60.2)

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "dermatollama_isic_upper_extremity_melanoma_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_red_pink_melanoma_topk_promotion() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="lower extremity",
        age="75",
        primary_color="red",
        color_variation="marked",
        asymmetry_color="present",
        support_margin=44.8,
        subtype_support_margin=18.0,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "dermatollama_isic_red_pink_melanoma_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_red_pink_melanoma_requires_marked_asymmetry() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="lower extremity",
        age="75",
        primary_color="red",
        color_variation="mild",
        asymmetry_color="absent",
        support_margin=44.8,
        subtype_support_margin=18.0,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_isic_red_pink_melanoma_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_headneck_actinic_topk_promotion() -> None:
    baseline_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Basal Cell Carcinoma",
        "differential_diagnoses": ["Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Actinic Keratosis"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="head/neck",
        age="75",
        primary_color="red",
        color_variation="mild",
        asymmetry_color="absent",
        subtype_support_margin=10.6,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "dermatollama_isic_headneck_actinic_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_headneck_actinic_requires_bcc_anchor() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="head/neck",
        age="75",
        primary_color="red",
        color_variation="mild",
        asymmetry_color="absent",
        subtype_support_margin=10.6,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_isic_headneck_actinic_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_older_irregular_scc_topk_promotion() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="lower extremity",
        age="75",
        primary_color="red",
        color_variation="mild",
        asymmetry_color="absent",
        surface=["flat"],
        supporting_evidence=["irregular border", "color variation"],
        support_margin=42.6,
        subtype_support_margin=9.7,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "dermatollama_isic_older_irregular_scc_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_older_irregular_scc_does_not_rewrite_smooth_vascular_like_case() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="anterior torso",
        age="75",
        primary_color="red",
        color_variation="marked",
        asymmetry_color="present",
        surface=["smooth"],
        supporting_evidence=["irregular border", "color variation"],
        support_margin=43.8,
        subtype_support_margin=10.1,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_isic_older_irregular_scc_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_truncal_scaly_bkl_topk_promotion() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="anterior torso",
        age="80",
        primary_color="brown",
        color_variation="mild",
        asymmetry_color="absent",
        surface=["scaly"],
        support_margin=43.6,
        subtype_support_margin=10.3,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "dermatollama_isic_truncal_scaly_bkl_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_isic_truncal_scaly_bkl_requires_absent_asymmetry() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Malignant Melanoma"],
        "confidence": "High",
    }
    evidence = _dermatollama_isic2019_evidence_bundle(
        site="anterior torso",
        age="80",
        primary_color="brown",
        color_variation="mild",
        asymmetry_color="present",
        surface=["scaly"],
        support_margin=43.6,
        subtype_support_margin=10.3,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_isic_truncal_scaly_bkl_topk_promotion" not in result["fusion_decision"]["reasons"]


def _dermatollama_sd198_evidence_bundle(
    *,
    early_ddx_candidates: list[str],
    image_summary: str,
    selected_evidence_text: str,
    support_margin: float,
    subtype_support_margin: float,
    contradiction_count: int,
    uncertainty_level: str = "low",
    workflow_cell_id: str = "dermatollama__sd198__grouped_guard_v1",
) -> dict:
    return {
        "selected_evidence": [
            {
                "source_name": "visual_summary_skill",
                "summary": selected_evidence_text,
            }
        ],
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": early_ddx_candidates,
                    "image_summary": image_summary,
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "coarse_taxonomy_workflow",
                    "dataset_workflow_profile": "coarse_taxonomy_workflow",
                    "label_space_id": "sd198_grouped",
                    "dataset_name": "sd198",
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": uncertainty_level,
                "contradiction_count": contradiction_count,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _hulumed_isic_evidence_bundle(
    *,
    site: str = "head/neck",
    early_ddx_candidates: list[str],
    image_summary: str,
    support_margin: float = 39.0,
    subtype_support_margin: float = 16.0,
    uncertainty_level: str = "medium",
    workflow_cell_id: str = "hulumed__isic2019__archive_guard_v1",
) -> dict:
    return {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": early_ddx_candidates,
                    "image_summary": image_summary,
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "dataset_workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                    "model_workflow_profile": "hulumed_isic2019_archive_guard_workflow",
                    "label_space_id": "isic2019_full",
                    "dataset_name": "isic2019",
                    "clinical_metadata": {"anatom_site_general": site},
                },
                "selected_evidence_present": True,
                "override_allowed": False,
                "malignancy_override_allowed": False,
                "subtype_override_allowed": False,
                "family_override_allowed": False,
                "override_mode": "risk_only",
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": uncertainty_level,
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _hulumed_scin_evidence_bundle(
    *,
    early_ddx_candidates: list[str],
    image_summary: str = "",
    clinical_metadata: dict | None = None,
    selected_evidence_text: str = "",
    support_margin: float = 42.0,
    subtype_support_margin: float = 6.0,
    workflow_cell_id: str = "hulumed__scin__grouped_guard_v1",
) -> dict:
    return {
        "selected_evidence": [
            {
                "source_name": "visual_summary_skill",
                "summary": selected_evidence_text,
            }
        ],
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": early_ddx_candidates,
                    "image_summary": image_summary,
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                    "clinical_metadata": dict(clinical_metadata or {}),
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def _medgemma_scin_evidence_bundle(
    *,
    early_ddx_candidates: list[str],
    clinical_metadata: dict | None = None,
    skill_outputs: dict | None = None,
    support_margin: float = 39.0,
    subtype_support_margin: float = 6.92,
    uncertainty_level: str = "low",
    workflow_cell_id: str = "medgemma__scin__grouped_core_v1",
) -> dict:
    return {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": early_ddx_candidates,
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                    "clinical_metadata": dict(clinical_metadata or {}),
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": uncertainty_level,
            },
        },
        "skill_outputs": dict(skill_outputs or {}),
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


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


def test_medgemma_scin_grouped_blocks_broad_cross_label_moderate_override() -> None:
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

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_grouped_conservative_guard" in result["fusion_decision"]["reasons"]
    assert "fallback_to_baseline" in result["fusion_decision"]["reasons"]


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


def test_medgemma_scin_grouped_expands_differential_from_initial_candidates() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
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
                    "early_ddx_candidates": ["Eczema", "Urticaria", "Tinea", "Acne"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__scin__grouped_core_v1",
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                },
                "selected_evidence_present": True,
                "support_margin": 39.0,
                "subtype_support_margin": 6.9,
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

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "URTICARIA_BITE_FOLLICULITIS" in result["differential_diagnoses"]
    assert "INFECTION_VIRAL_FUNGAL" in result["differential_diagnoses"]
    assert "ACNE_ROSACEA_FOLLICULAR" in result["differential_diagnoses"]
    assert "medgemma_scin_initial_grouped_differential_expansion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_grouped_differential_expansion_is_cell_bound() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
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
                    "early_ddx_candidates": ["Eczema", "Urticaria", "Tinea", "Acne"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "qwen__scin__grouped_best",
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                },
                "selected_evidence_present": True,
                "support_margin": 39.0,
                "subtype_support_margin": 6.9,
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

    assert "URTICARIA_BITE_FOLLICULITIS" not in result["differential_diagnoses"]
    assert "INFECTION_VIRAL_FUNGAL" not in result["differential_diagnoses"]
    assert "medgemma_scin_initial_grouped_differential_expansion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_acne_follicular_promotion_is_narrow() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "ACNE_ROSACEA_FOLLICULAR",
        "differential_diagnoses": ["ACNE_ROSACEA_FOLLICULAR", "DERMATITIS_ECZEMA"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Acne", "Rosacea", "Eczema"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__scin__grouped_core_v1",
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                },
                "selected_evidence_present": True,
                "support_margin": 33.0,
                "subtype_support_margin": 6.96,
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

    assert result["final_diagnosis"] == "ACNE_ROSACEA_FOLLICULAR"
    assert "medgemma_scin_acne_follicular_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_acne_follicular_promotion_requires_initial_acne_signal() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "ACNE_ROSACEA_FOLLICULAR",
        "differential_diagnoses": ["ACNE_ROSACEA_FOLLICULAR", "DERMATITIS_ECZEMA"],
        "confidence": "Moderate",
    }
    evidence_bundle = {
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": ["Eczema", "Urticaria"],
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": "medgemma__scin__grouped_core_v1",
                    "workflow_profile": "family_routing_workflow",
                    "label_space_id": "scin_grouped",
                    "dataset_name": "scin",
                },
                "selected_evidence_present": True,
                "support_margin": 33.0,
                "subtype_support_margin": 6.96,
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

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_acne_follicular_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_face_acne_evidence_promotes_from_topk() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis", "ACNE_ROSACEA_FOLLICULAR"],
        "confidence": "Moderate",
    }
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Acne", "Eczema", "Rosacea"],
        clinical_metadata={"region": "head_or_neck", "body_sites": ["head_or_neck"]},
        skill_outputs={
            "morphology_analysis_skill": {"lesion_type": "papule", "count": "multiple"},
            "distribution_analysis_skill": {"body_location": "face", "clustering_pattern": "clustered"},
            "lesion_description_structuring_skill": {
                "associated_context": ["face", "small red dots", "papules/pustules"],
            },
            "differential_compare_skill": {
                "supporting_evidence": ["The small papules are consistent with acne."],
            },
        },
        subtype_support_margin=6.96,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "ACNE_ROSACEA_FOLLICULAR"
    assert "medgemma_scin_face_acne_evidence_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_face_acne_evidence_requires_workflow_cell() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis", "ACNE_ROSACEA_FOLLICULAR"],
        "confidence": "Moderate",
    }
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Acne", "Eczema", "Rosacea"],
        skill_outputs={
            "morphology_analysis_skill": {"lesion_type": "papule"},
            "distribution_analysis_skill": {"body_location": "face"},
            "differential_compare_skill": {"supporting_evidence": ["consistent with acne"]},
        },
        workflow_cell_id="medgemma__scin__other_cell",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_face_acne_evidence_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_leg_fluid_urticaria_promotes_from_topk() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "DERMATITIS_ECZEMA",
        "differential_diagnoses": ["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
        "confidence": "Moderate",
    }
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Urticaria/Bite Follliculitis", "Eczema", "Contact Dermatitis"],
        clinical_metadata={
            "region": "leg",
            "body_sites": ["leg"],
            "textures_present": ["raised_or_bumpy", "fluid_filled"],
        },
        skill_outputs={
            "distribution_analysis_skill": {"clustering_pattern": "clustered"},
            "lesion_description_structuring_skill": {"associated_context": ["raised borders"]},
            "differential_compare_skill": {"supporting_evidence": ["consistent with urticaria"]},
        },
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "URTICARIA_BITE_FOLLICULITIS"
    assert "medgemma_scin_leg_fluid_urticaria_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_leg_fluid_urticaria_blocks_hand_pattern() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "DERMATITIS_ECZEMA",
        "differential_diagnoses": ["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
        "confidence": "Moderate",
    }
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Urticaria/Bite Reaction", "Eczema", "Contact Dermatitis"],
        clinical_metadata={
            "region": "leg",
            "body_sites": ["leg", "palm", "back_of_hand"],
            "textures_present": ["raised_or_bumpy", "fluid_filled"],
        },
        skill_outputs={
            "distribution_analysis_skill": {"clustering_pattern": "clustered"},
            "differential_compare_skill": {"supporting_evidence": ["consistent with urticaria"]},
        },
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_leg_fluid_urticaria_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_arm_ulcer_herpes_promotes_from_topk() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "DERMATITIS_ECZEMA",
        "differential_diagnoses": ["DERMATITIS_ECZEMA", "INFECTION_VIRAL_FUNGAL"],
        "confidence": "Moderate",
    }
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Herpes simplex", "Warts", "Contact dermatitis"],
        clinical_metadata={"region": "arm", "body_sites": ["arm"], "other_symptoms_present": ["mouth_sores"]},
        skill_outputs={
            "lesion_description_structuring_skill": {"surface": ["ulcerated"]},
            "temporal_evolution_skill": {"onset_type": "acute", "progression_speed": "rapid"},
        },
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "INFECTION_VIRAL_FUNGAL"
    assert "medgemma_scin_arm_ulcer_herpes_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_arm_ulcer_herpes_requires_ulcer_signal() -> None:
    baseline_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis"],
        "confidence": "High",
    }
    agent_output = {
        "final_diagnosis": "DERMATITIS_ECZEMA",
        "differential_diagnoses": ["DERMATITIS_ECZEMA", "INFECTION_VIRAL_FUNGAL"],
        "confidence": "Moderate",
    }
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Herpes simplex", "Warts", "Contact dermatitis"],
        clinical_metadata={"region": "arm", "body_sites": ["arm"], "other_symptoms_present": ["mouth_sores"]},
        skill_outputs={
            "lesion_description_structuring_skill": {"surface": ["raised_or_bumpy"]},
            "temporal_evolution_skill": {"onset_type": "acute"},
        },
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_arm_ulcer_herpes_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_headneck_skin_cancer_promotion() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "High"}
    agent_output = {"final_diagnosis": "Basal Cell Carcinoma", "differential_diagnoses": ["Basal Cell Carcinoma"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=[
            "Angiofibroma",
            "Hemangioma",
            "Pyogenic Granuloma",
            "Skin Cancer (e.g., basal cell carcinoma, squamous cell carcinoma)",
        ],
        clinical_metadata={"region": "head_or_neck", "textures_present": ["raised_or_bumpy", "rough_or_flaky"]},
        subtype_support_margin=7.26,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_scin_headneck_skin_cancer_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_headneck_skin_cancer_promotion_requires_headneck_site() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "High"}
    agent_output = {"final_diagnosis": "Basal Cell Carcinoma", "differential_diagnoses": ["Basal Cell Carcinoma"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Skin Cancer (e.g., basal cell carcinoma, squamous cell carcinoma)"],
        clinical_metadata={"region": "arm", "condition_duration": "MORE_THAN_ONE_YEAR"},
        subtype_support_margin=7.26,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_headneck_skin_cancer_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_pigment_bcc_symptom_promotion() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {"final_diagnosis": "Basal Cell Carcinoma", "differential_diagnoses": ["Basal Cell Carcinoma"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Eczema", "Contact Dermatitis", "Psoriasis"],
        clinical_metadata={
            "related_category": "PIGMENTARY_PROBLEM",
            "symptoms_present": ["bothersome_appearance", "increasing_size", "itching", "burning", "pain"],
        },
        subtype_support_margin=6.96,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_scin_pigment_bcc_symptom_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_pigment_bcc_symptom_promotion_blocks_nonpigment_anchor() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "High"}
    agent_output = {"final_diagnosis": "Basal Cell Carcinoma", "differential_diagnoses": ["Basal Cell Carcinoma"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Eczema", "Contact Dermatitis", "Psoriasis"],
        clinical_metadata={"related_category": "RASH", "symptoms_present": ["increasing_size", "pain"]},
        subtype_support_margin=6.96,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "medgemma_scin_pigment_bcc_symptom_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_genital_herpes_promotion() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "High"}
    agent_output = {"final_diagnosis": "INFECTION_VIRAL_FUNGAL", "differential_diagnoses": ["INFECTION_VIRAL_FUNGAL"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Herpes simplex", "Folliculitis", "Contact dermatitis"],
        clinical_metadata={"region": "genitalia_or_groin", "textures_present": ["raised_or_bumpy", "fluid_filled"]},
        subtype_support_margin=6.92,
        uncertainty_level="medium",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "INFECTION_VIRAL_FUNGAL"
    assert "medgemma_scin_genital_herpes_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_genital_herpes_promotion_blocks_non_genital_site() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "High"}
    agent_output = {"final_diagnosis": "INFECTION_VIRAL_FUNGAL", "differential_diagnoses": ["INFECTION_VIRAL_FUNGAL"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Herpes simplex", "Folliculitis", "Contact dermatitis"],
        clinical_metadata={"region": "arm", "textures_present": ["raised_or_bumpy", "fluid_filled"]},
        subtype_support_margin=6.92,
        uncertainty_level="medium",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_genital_herpes_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_scin_lower_body_vascular_promotion() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "High"}
    agent_output = {"final_diagnosis": "VASCULAR_PURPURIC", "differential_diagnoses": ["VASCULAR_PURPURIC"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Eczema", "Urticaria", "Contact dermatitis", "Viral exanthem"],
        clinical_metadata={
            "region": "buttocks",
            "body_sites": ["buttocks", "leg", "foot_top_or_side"],
            "condition_duration": "ONE_TO_FOUR_WEEKS",
            "textures_present": ["flat"],
            "symptoms_present": ["bothersome_appearance", "increasing_size", "burning", "pain"],
        },
        uncertainty_level="medium",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "VASCULAR_PURPURIC"
    assert "medgemma_scin_lower_body_vascular_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_scin_lower_body_vascular_promotion_requires_lower_body_pattern() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "High"}
    agent_output = {"final_diagnosis": "VASCULAR_PURPURIC", "differential_diagnoses": ["VASCULAR_PURPURIC"], "confidence": "Moderate"}
    evidence_bundle = _medgemma_scin_evidence_bundle(
        early_ddx_candidates=["Eczema", "Urticaria", "Contact dermatitis", "Viral exanthem"],
        clinical_metadata={
            "region": "leg",
            "body_sites": ["leg"],
            "condition_duration": "ONE_TO_FOUR_WEEKS",
            "textures_present": ["flat"],
            "symptoms_present": ["bothersome_appearance", "increasing_size", "burning", "pain"],
        },
        uncertainty_level="medium",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "medgemma_scin_lower_body_vascular_promotion" not in result["fusion_decision"]["reasons"]


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


def test_hulumed_isic_promotes_headneck_ak_scale_topk_case() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis", "Psoriasis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        early_ddx_candidates=["actinic keratosis", "lichen planus", "psoriasis"],
        image_summary="Erythematous patch with subtle scaling and faint vascular structures",
        support_margin=35.62,
        subtype_support_margin=19.78,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "hulumed_isic_headneck_ak_scale_promotion" in result["fusion_decision"]["reasons"]
    assert "Nevus" in result["differential_diagnoses"]


def test_hulumed_isic_ak_scale_promotion_blocks_slightly_raised_nevus_anchor() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Actinic Keratosis", "Squamous Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        early_ddx_candidates=["actinic keratosis", "basal cell carcinoma", "squamous cell carcinoma", "nevus"],
        image_summary="Erythematous, slightly raised lesion with subtle scaling and pink-red discoloration",
        support_margin=39.0,
        subtype_support_margin=12.0,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "hulumed_isic_headneck_ak_scale_promotion" not in result["fusion_decision"]["reasons"]
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_hulumed_isic_promotes_headneck_scc_vascular_scale_topk_case() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        early_ddx_candidates=["nevus", "basal cell carcinoma", "squamous cell carcinoma"],
        image_summary=(
            "Irregularly shaped lesion with pinkish-red background, "
            "brownish pigmentation, and visible blood vessels."
        ),
        support_margin=39.13,
        subtype_support_margin=16.95,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Squamous Cell Carcinoma"
    assert "hulumed_isic_headneck_scc_vascular_scale_promotion" in result["fusion_decision"]["reasons"]
    assert "Nevus" in result["differential_diagnoses"]


def test_hulumed_isic_scc_promotion_requires_target_workflow_cell_and_marker() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        early_ddx_candidates=["nevus", "basal cell carcinoma", "squamous cell carcinoma"],
        image_summary="Irregularly shaped lesion with pink-red background and brownish pigmentation.",
        support_margin=39.13,
        subtype_support_margin=16.95,
        workflow_cell_id="hulumed__isic2019__alternate_cell",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "hulumed_isic_headneck_scc_vascular_scale_promotion" not in result["fusion_decision"]["reasons"]
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_hulumed_isic_promotes_headneck_bkl_keratotic_topk_case() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Actinic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        early_ddx_candidates=["nevus", "seborrheic keratosis", "actinic keratosis"],
        image_summary="Irregularly shaped, light brown patch with uneven borders and subtle variations in pigmentation.",
        support_margin=40.354,
        subtype_support_margin=19.354,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "hulumed_isic_headneck_bkl_keratotic_promotion" in result["fusion_decision"]["reasons"]
    assert "Nevus" in result["differential_diagnoses"]


def test_hulumed_isic_bkl_keratotic_promotion_blocks_high_margin_malignant_mimic() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        early_ddx_candidates=["nevus", "basal cell carcinoma", "seborrheic keratosis"],
        image_summary="Reddish-brown patch with irregular borders and scattered dark spots on a light background",
        support_margin=62.4,
        subtype_support_margin=27.88,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "hulumed_isic_headneck_bkl_keratotic_promotion" not in result["fusion_decision"]["reasons"]
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_hulumed_isic_promotes_upper_extremity_mel_dark_irregular_topk_case() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        site="upper extremity",
        early_ddx_candidates=["melanoma", "nevus", "seborrheic keratosis"],
        image_summary="Irregularly shaped, asymmetric brown lesion with varying shades and a central darker area.",
        support_margin=42.99,
        subtype_support_margin=18.75,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert (
        "hulumed_isic_upper_extremity_mel_dark_irregular_promotion"
        in result["fusion_decision"]["reasons"]
    )
    assert "Nevus" in result["differential_diagnoses"]


def test_hulumed_isic_mel_dark_irregular_promotion_blocks_blue_vascular_mimic() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        site="upper extremity",
        early_ddx_candidates=["melanoma", "vascular lesion", "nevus"],
        image_summary="Irregular dark blue lesion with asymmetric pigmentation and varying shades.",
        support_margin=42.0,
        subtype_support_margin=12.0,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert (
        "hulumed_isic_upper_extremity_mel_dark_irregular_promotion"
        not in result["fusion_decision"]["reasons"]
    )
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_hulumed_isic_promotes_blue_purple_vascular_topk_case() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        site="posterior torso",
        early_ddx_candidates=["Kaposi sarcoma", "Angiosarcoma", "Hemangioma"],
        image_summary=(
            "Reddish-purple lesion with central red area and surrounding blue-gray "
            "structureless region on a pinkish background."
        ),
        support_margin=44.58,
        subtype_support_margin=3.66,
        uncertainty_level="high",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Vascular Lesion"
    assert "hulumed_isic_blue_purple_vascular_promotion" in result["fusion_decision"]["reasons"]
    assert "Nevus" in result["differential_diagnoses"]


def test_hulumed_isic_vascular_promotion_blocks_generic_possible_vascular_language() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        site="head/neck",
        early_ddx_candidates=["nevus", "basal cell carcinoma", "seborrheic keratosis"],
        image_summary="Irregular pigmentation with brown and pink hues, possible vascular structures, and uneven texture.",
        support_margin=42.68,
        subtype_support_margin=9.82,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "hulumed_isic_blue_purple_vascular_promotion" not in result["fusion_decision"]["reasons"]
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_hulumed_isic_promotes_upper_anterior_bcc_inflammatory_topk_case() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Actinic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        site="upper extremity",
        early_ddx_candidates=["actinic keratosis", "lichen planus", "psoriasis"],
        image_summary="Erythematous patch with subtle scaling and a small central red dot, surrounded by fine hair.",
        support_margin=40.6,
        subtype_support_margin=2.6,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert (
        "hulumed_isic_upper_anterior_bcc_inflammatory_promotion"
        in result["fusion_decision"]["reasons"]
    )
    assert "Nevus" in result["differential_diagnoses"]


def test_hulumed_isic_bcc_inflammatory_promotion_blocks_crusted_petechial_mimic() -> None:
    baseline_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus"],
        "confidence": "Moderate",
    }
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _hulumed_isic_evidence_bundle(
        site="anterior torso",
        early_ddx_candidates=["keratoacanthoma", "inflammatory plaque", "lichenoid keratosis"],
        image_summary="Erythematous patch with subtle scaling, central crust, and scattered petechiae.",
        support_margin=36.54,
        subtype_support_margin=2.44,
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert (
        "hulumed_isic_upper_anterior_bcc_inflammatory_promotion"
        not in result["fusion_decision"]["reasons"]
    )
    assert "agent_matches_baseline" in result["fusion_decision"]["reasons"]


def test_hulumed_scin_promotes_face_chest_acne_topk_pattern() -> None:
    baseline_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS"],
    }
    agent_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS", "ACNE_ROSACEA_FOLLICULAR"],
    }
    evidence_bundle = _hulumed_scin_evidence_bundle(
        early_ddx_candidates=["URTICARIA_BITE_FOLLICULITIS", "ACNE_ROSACEA_FOLLICULAR"],
        image_summary="Multiple inflammatory papules and pustules on the cheeks and forehead.",
        clinical_metadata={"body_sites": ["face"], "textures_present": ["raised_or_bumpy"]},
        selected_evidence_text="visual_summary_skill: papules and pustules clustered on face, acne-like follicular pattern.",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "ACNE_ROSACEA_FOLLICULAR"
    assert "hulumed_scin_face_chest_acne_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_hulumed_scin_acne_promotion_blocks_dermatitis_neck_scaling_mimic() -> None:
    baseline_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS"],
    }
    agent_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS", "ACNE_ROSACEA_FOLLICULAR"],
    }
    evidence_bundle = _hulumed_scin_evidence_bundle(
        early_ddx_candidates=["URTICARIA_BITE_FOLLICULITIS", "ACNE_ROSACEA_FOLLICULAR"],
        image_summary="Scaly plaque on the lateral neck without clear pustules.",
        clinical_metadata={"body_sites": ["neck"], "textures_present": ["rough_or_flaky"]},
        selected_evidence_text="visual_summary_skill: scaly plaque, no visible lesion suggesting acne.",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "URTICARIA_BITE_FOLLICULITIS"
    assert "hulumed_scin_face_chest_acne_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_hulumed_scin_promotes_lower_body_vascular_topk_pattern() -> None:
    baseline_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS"],
    }
    agent_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS", "VASCULAR_PURPURIC"],
    }
    evidence_bundle = _hulumed_scin_evidence_bundle(
        early_ddx_candidates=["URTICARIA_BITE_FOLLICULITIS", "VASCULAR_PURPURIC"],
        image_summary="Confluent erythematous papules on the leg with purpuric vascular appearance.",
        clinical_metadata={"body_sites": ["leg"], "symptoms_present": ["pain", "burning", "increasing_size"]},
        selected_evidence_text="visual_summary_skill: red painful lower extremity papules; vasculitis remains plausible.",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "VASCULAR_PURPURIC"
    assert "hulumed_scin_lower_body_vascular_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_hulumed_scin_vascular_promotion_blocks_insect_bite_anchor() -> None:
    baseline_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS"],
    }
    agent_output = {
        "final_diagnosis": "URTICARIA_BITE_FOLLICULITIS",
        "differential_diagnoses": ["URTICARIA_BITE_FOLLICULITIS", "VASCULAR_PURPURIC"],
    }
    evidence_bundle = _hulumed_scin_evidence_bundle(
        early_ddx_candidates=["URTICARIA_BITE_FOLLICULITIS", "VASCULAR_PURPURIC"],
        image_summary="Single bite-like red papule on the ankle after suspected insect bite.",
        clinical_metadata={"body_sites": ["ankle"], "symptoms_present": ["itching"]},
        selected_evidence_text="visual_summary_skill: single bite lesion, no vascular clustering.",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "URTICARIA_BITE_FOLLICULITIS"
    assert "hulumed_scin_lower_body_vascular_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_hulumed_scin_does_not_promote_malignant_pattern_from_rash_anchor() -> None:
    baseline_output = {"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"]}
    agent_output = {
        "final_diagnosis": "Contact Dermatitis",
        "differential_diagnoses": ["Contact Dermatitis", "MALIGNANT_PREMALIGNANT"],
    }
    evidence_bundle = _hulumed_scin_evidence_bundle(
        early_ddx_candidates=["MALIGNANT_PREMALIGNANT", "DERMATITIS_ECZEMA"],
        image_summary="Rough flaky keratotic scale on the back of hand, sun-exposed site.",
        clinical_metadata={"body_sites": ["back_of_hand"], "textures_present": ["rough_or_flaky"]},
        selected_evidence_text="visual_summary_skill: actinic keratosis-like crust and scale on hand.",
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert not any(reason.startswith("hulumed_scin_sun_exposed") for reason in result["fusion_decision"]["reasons"])


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


def test_qwen_isic_archive_guard_requires_target_workflow_cell() -> None:
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
                    "workflow_cell_id": "qwen__isic2019__alternate_cell",
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
    assert "qwen_isic_guarded_archive_override" not in result["fusion_decision"]["reasons"]


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


def test_qwen_isic_low_margin_central_pattern_promotes_melanoma_from_differential() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="upper extremity",
        support_margin=41.8,
        selected_evidence=[{"summary": "asymmetry with irregular border and central dark crusted change"}],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_low_margin_central_mel_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_low_margin_central_pattern_blocks_smooth_nevus_pattern() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="upper extremity",
        support_margin=41.8,
        selected_evidence=[{"summary": "asymmetry with irregular border and central dark area but smooth surface"}],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_low_margin_central_mel_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_risk_irregular_promotes_melanoma_from_differential() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="anterior torso",
        uncertainty_level="high",
        selected_evidence=[
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: risk_level=high | risk_evidence=irregular border and asymmetric color",
            }
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_anterior_torso_risk_irregular_mel_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_risk_irregular_requires_high_uncertainty() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="anterior torso",
        uncertainty_level="medium",
        selected_evidence=[
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: risk_level=high | risk_evidence=irregular border and asymmetric color",
            }
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_anterior_torso_risk_irregular_mel_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_upper_extremity_crusted_pattern_promotes_melanoma_from_differential() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="upper extremity",
        selected_evidence=[
            {"summary": "marked reticular pigmentation with focal crust"},
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: risk_level=medium | risk_evidence=marked color variation",
            },
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_isic_upper_extremity_crusted_mel_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_upper_extremity_crusted_pattern_requires_crust_marker() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="upper extremity",
        selected_evidence=[
            {"summary": "marked reticular pigmentation without surface breakdown"},
            {
                "source_name": "malignancy_risk_assessment_skill",
                "summary": "malignancy_risk_assessment_skill: risk_level=medium | risk_evidence=marked color variation",
            },
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_upper_extremity_crusted_mel_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_uniform_ak_bcc_promotes_bcc_from_differential() -> None:
    baseline_output = {"final_diagnosis": "Actinic Keratosis", "differential_diagnoses": ["Actinic Keratosis"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Basal Cell Carcinoma"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="upper extremity",
        selected_evidence=[
            {
                "source_name": "lesion_description_structuring_skill",
                "summary": "lesion_description_structuring_skill: slightly elevated pink plaque with uniform color and smooth border",
            }
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_isic_uniform_ak_bcc_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_uniform_ak_bcc_requires_bcc_in_differential() -> None:
    baseline_output = {"final_diagnosis": "Actinic Keratosis", "differential_diagnoses": ["Actinic Keratosis"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="upper extremity",
        selected_evidence=[
            {
                "source_name": "lesion_description_structuring_skill",
                "summary": "lesion_description_structuring_skill: slightly elevated pink plaque with uniform color and smooth border",
            }
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_isic_uniform_ak_bcc_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_headneck_nv_bkl_promotes_bkl_from_differential() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="head/neck",
        support_margin=62.0,
        selected_evidence=[
            {
                "source_name": "color_pattern_analysis_skill",
                "summary": "color_pattern_analysis_skill: marked color variation and reticular pigmentation",
            }
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "qwen_isic_headneck_nv_bkl_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_headneck_nv_bkl_respects_support_margin_gate() -> None:
    baseline_output = {"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Nevus",
        "differential_diagnoses": ["Nevus", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="head/neck",
        support_margin=64.0,
        selected_evidence=[
            {
                "source_name": "color_pattern_analysis_skill",
                "summary": "color_pattern_analysis_skill: marked color variation and reticular pigmentation",
            }
        ],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_isic_headneck_nv_bkl_differential_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_ak_bkl_promotes_bkl_from_differential() -> None:
    baseline_output = {"final_diagnosis": "Actinic Keratosis", "differential_diagnoses": ["Actinic Keratosis"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="anterior torso",
        uncertainty_level="high",
        selected_evidence=[{"summary": "rough stuck-on waxy plaque with marked color variation"}],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "qwen_isic_anterior_torso_ak_bkl_differential_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_isic_anterior_torso_ak_bkl_requires_high_uncertainty() -> None:
    baseline_output = {"final_diagnosis": "Actinic Keratosis", "differential_diagnoses": ["Actinic Keratosis"], "confidence": "Moderate"}
    agent_output = {
        "final_diagnosis": "Actinic Keratosis",
        "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis"],
        "confidence": "Moderate",
    }
    evidence_bundle = _qwen_isic_evidence_bundle(
        site="anterior torso",
        uncertainty_level="medium",
        selected_evidence=[{"summary": "rough stuck-on waxy plaque with marked color variation"}],
    )

    result = apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_isic_anterior_torso_ak_bkl_differential_promotion" not in result["fusion_decision"]["reasons"]


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "bcc_fine_telangiectasia_evidence_only",
            "baseline": "Nevus",
            "agent": "Nevus",
            "ddx": ["Nevus", "Actinic Keratosis", "Seborrheic Keratosis"],
            "site": "upper extremity",
            "support": 42.458,
            "subtype": 16.056,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "ack_scc_specialist_skill",
                    "summary": "ack_scc_specialist_skill: differentiation_features=Central depression; Fine telangiectasias; Pinkish color",
                },
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: slightly elevated plaque | color=pinkish | border=irregular | surface=rough",
                },
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_bcc_fine_telangiectasia_evidence_promotion",
        },
        {
            "id": "bcc_headneck_umbilication_evidence_only",
            "baseline": "Nevus",
            "agent": "Nevus",
            "ddx": ["Nevus", "Actinic Keratosis"],
            "site": "head/neck",
            "support": 60.5,
            "subtype": 19.7,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: papule | color=hyperpigmented | border=umbilicated | surface=smooth",
                },
                {
                    "source_name": "malignancy_risk_assessment_skill",
                    "summary": "malignancy_risk_assessment_skill: alarm_signals=central umbilication; asymmetric border",
                },
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_headneck_bcc_umbilication_evidence_promotion",
        },
        {
            "id": "bcc_fine_telangiectasia",
            "baseline": "Actinic Keratosis",
            "agent": "Actinic Keratosis",
            "ddx": ["Actinic Keratosis", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "site": "upper extremity",
            "support": 59.82,
            "subtype": 20.18,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "ack_scc_specialist_skill",
                    "summary": "ack_scc_specialist_skill: differentiation_features=Presence of fine telangiectasias",
                }
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_bcc_fine_telangiectasia_topk_promotion",
        },
        {
            "id": "bcc_dark_pigmented_ak",
            "baseline": "Actinic Keratosis",
            "agent": "Actinic Keratosis",
            "ddx": ["Actinic Keratosis", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "site": "head/neck",
            "support": 62.68,
            "subtype": 20.9,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: irregular plaque | color=pink with darker pigmented areas | border=irregular",
                }
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_headneck_bcc_dark_pigmented_ak_promotion",
        },
        {
            "id": "bcc_hyperpigmented_macule",
            "baseline": "Nevus",
            "agent": "Nevus",
            "ddx": ["Nevus", "Actinic Keratosis", "Basal Cell Carcinoma"],
            "site": "anterior torso",
            "support": 39.638,
            "subtype": 21.396,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: diffuse macule | color=pink with hyperpigmentation and hypopigmentation | border=slightly irregular",
                }
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_anterior_torso_bcc_hyperpigmented_macule_promotion",
        },
        {
            "id": "bcc_reticular_vessel",
            "baseline": "Seborrheic Keratosis",
            "agent": "Seborrheic Keratosis",
            "ddx": ["Seborrheic Keratosis", "Actinic Keratosis", "Basal Cell Carcinoma", "Nevus"],
            "site": "anterior torso",
            "support": 60.74,
            "subtype": 20.16,
            "uncertainty": "medium",
            "evidence": [
                {"source_name": "color_pattern_analysis_skill", "summary": "color_pattern_analysis_skill: pigmentation_pattern=reticular"},
                {
                    "source_name": "malignancy_risk_assessment_skill",
                    "summary": "malignancy_risk_assessment_skill: alarm_signals=Irregularly shaped vessel",
                },
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_anterior_torso_bcc_reticular_vessel_promotion",
        },
        {
            "id": "bcc_scaling_crusting_nodule",
            "baseline": "Seborrheic Keratosis",
            "agent": "Seborrheic Keratosis",
            "ddx": ["Seborrheic Keratosis", "Actinic Keratosis", "Basal Cell Carcinoma"],
            "site": "head/neck",
            "support": 41.562,
            "subtype": 15.16,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: primary_lesion_morphology=small, raised, slightly erythematous nodule | surface=scaling and crusting",
                }
            ],
            "expected_label": "Basal Cell Carcinoma",
            "expected_reason": "qwen_isic_headneck_bcc_scaling_crusting_nodule_promotion",
        },
        {
            "id": "mel_upper_mottled_marked",
            "baseline": "Nevus",
            "agent": "Nevus",
            "ddx": ["Nevus", "Malignant Melanoma"],
            "site": "upper extremity",
            "support": 62.96,
            "subtype": 13.86,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: irregular plaque | color=mottled brown | size=large | symmetry=asymmetric",
                },
                {
                    "source_name": "malignancy_risk_assessment_skill",
                    "summary": "malignancy_risk_assessment_skill: risk_evidence=irregular border; marked color variation",
                },
            ],
            "expected_label": "Malignant Melanoma",
            "expected_reason": "qwen_isic_upper_extremity_mottled_marked_mel_topk_promotion",
        },
        {
            "id": "mel_upper_speckled_final",
            "baseline": "Nevus",
            "agent": "Malignant Melanoma",
            "ddx": ["Nevus", "Malignant Melanoma"],
            "site": "upper extremity",
            "support": 41.624,
            "subtype": -2.32,
            "uncertainty": "high",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: large, irregularly shaped plaque | color=dark brown/black with speckled pigmentation | border=irregular and speckled",
                }
            ],
            "expected_label": "Malignant Melanoma",
            "expected_reason": "qwen_isic_upper_extremity_speckled_mel_final_acceptance",
        },
        {
            "id": "mel_posterior_crusted_halo",
            "baseline": "Nevus",
            "agent": "Nevus",
            "ddx": ["Nevus", "Actinic Keratosis", "Seborrheic Keratosis", "Malignant Melanoma"],
            "site": "posterior torso",
            "support": 42.158,
            "subtype": 15.756,
            "uncertainty": "medium",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: lesion with a central depression | color=erythematous with a surrounding erythematous halo | surface=coarse and slightly crusted",
                }
            ],
            "expected_label": "Malignant Melanoma",
            "expected_reason": "qwen_isic_posterior_torso_crusted_halo_mel_topk_promotion",
        },
        {
            "id": "mel_lower_marked_variation_final",
            "baseline": "Nevus",
            "agent": "Malignant Melanoma",
            "ddx": ["Nevus", "Malignant Melanoma"],
            "site": "lower extremity",
            "support": 41.224,
            "subtype": -2.32,
            "uncertainty": "high",
            "evidence": [
                {
                    "source_name": "lesion_description_structuring_skill",
                    "summary": "lesion_description_structuring_skill: irregularly shaped plaque | color=dark brown with marked variation in pigmentation | border=irregular and asymmetrical | size=large",
                }
            ],
            "expected_label": "Malignant Melanoma",
            "expected_reason": "qwen_isic_lower_extremity_marked_variation_mel_final_acceptance",
        },
    ],
    ids=lambda case: case["id"],
)
def test_qwen_isic_topk_promotion_gates_have_traceable_reasons(case: dict) -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": case["baseline"],
            "differential_diagnoses": [case["baseline"]],
            "confidence": "Moderate",
        },
        agent_output={
            "final_diagnosis": case["agent"],
            "differential_diagnoses": case["ddx"],
            "confidence": "Moderate",
        },
        evidence_bundle=_qwen_isic_evidence_bundle(
            site=case["site"],
            support_margin=case["support"],
            subtype_support_margin=case["subtype"],
            uncertainty_level=case["uncertainty"],
            selected_evidence=case["evidence"],
        ),
    )

    assert result["final_diagnosis"] == case["expected_label"]
    assert result["fusion_decision"]["consensus_override_label"] == case["expected_label"]
    assert case["expected_reason"] in result["fusion_decision"]["reasons"]
    assert "qwen_isic_nevus_preservation_guard" not in result["fusion_decision"]["reasons"]


def test_qwen_isic_topk_promotion_gates_require_target_workflow_cell() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "Moderate",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis", "Basal Cell Carcinoma"],
            "confidence": "Moderate",
        },
        evidence_bundle=_qwen_isic_evidence_bundle(
            site="upper extremity",
            support_margin=59.82,
            subtype_support_margin=20.18,
            workflow_cell_id="qwen__ham10000__dataset_best",
            selected_evidence=[
                {
                    "source_name": "ack_scc_specialist_skill",
                    "summary": "ack_scc_specialist_skill: differentiation_features=Presence of fine telangiectasias",
                }
            ],
        ),
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_isic_bcc_fine_telangiectasia_topk_promotion" not in result["fusion_decision"]["reasons"]
    assert "qwen_isic_bcc_fine_telangiectasia_evidence_promotion" not in result["fusion_decision"]["reasons"]


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

def _medgemma_ham10000_evidence_bundle(
    *,
    site: str,
    image_summary: str,
    early_ddx_candidates: list[str],
    support_margin: float = 62.5,
    subtype_support_margin: float = 7.2,
    workflow_cell_id: str = "medgemma__ham10000__akiec_face_guard_v1",
) -> dict:
    return {
        "selected_evidence": [
            {
                "source_name": "lesion_description_structuring_skill",
                "summary": f"lesion_description_structuring_skill: {image_summary}",
            }
        ],
        "evidence_decision_policy": {
            "risk_layer": {
                "baseline_preview": {
                    "early_ddx_candidates": early_ddx_candidates,
                    "image_summary": image_summary,
                }
            },
            "diagnosis_override_layer": {
                "workflow_context": {
                    "workflow_cell_id": workflow_cell_id,
                    "workflow_profile": "sparse_lesion_workflow",
                    "dataset_workflow_profile": "sparse_lesion_workflow",
                    "label_space_id": "ham10000_full",
                    "dataset_name": "ham10000",
                    "clinical_metadata": {"localization": site},
                },
                "selected_evidence_present": True,
                "support_margin": support_margin,
                "subtype_support_margin": subtype_support_margin,
                "uncertainty_level": "medium",
            },
        },
        "evidence_calibration_debug": {"policy": {"conservative_fusion_mode": "soft"}},
    }


def test_medgemma_ham10000_face_akiec_surface_promotion() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="face",
            image_summary="A well-defined reddish-brown lesion with rough mottled irregular surface change.",
            early_ddx_candidates=["AKIEC", "BKL", "DF"],
            support_margin=62.7,
            subtype_support_margin=4.1,
        ),
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "medgemma_ham10000_face_akiec_surface_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_face_akiec_surface_promotion_blocks_bcc_anchor_margin() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="scalp",
            image_summary="A well-defined reddish-brown irregular raised lesion.",
            early_ddx_candidates=["AKIEC", "BCC", "MEL"],
            support_margin=48.8,
            subtype_support_margin=14.4,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_face_akiec_surface_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_face_akiec_surface_promotion_requires_workflow_cell() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="face",
            image_summary="A well-defined reddish-brown lesion with rough mottled irregular surface change.",
            early_ddx_candidates=["AKIEC", "BKL", "DF"],
            support_margin=62.7,
            subtype_support_margin=4.1,
            workflow_cell_id="qwen__ham10000__dataset_best",
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_face_akiec_surface_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_fallback_guard_handles_agent_disagreement() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="trunk",
            image_summary="A uniform brown lesion without the low-uncertainty AKIEC surface pattern.",
            early_ddx_candidates=["NV", "BCC", "BKL"],
            support_margin=42.0,
            subtype_support_margin=12.0,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_baseline_anchor_guard" in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_truncal_nevus_topk_promotion() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="chest",
            image_summary="A well-defined irregular brown lesion with central pigmentation.",
            early_ddx_candidates=["AKIEC", "BCC", "MEL"],
            support_margin=41.2,
            subtype_support_margin=21.4,
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "medgemma_ham10000_truncal_nevus_topk_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_truncal_nevus_topk_promotion_blocks_vascular_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="neck",
            image_summary="A solitary well-defined dark purple vascular lesion.",
            early_ddx_candidates=["MEL", "BCC", "NV"],
            support_margin=49.0,
            subtype_support_margin=22.5,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_truncal_nevus_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_truncal_nevus_topk_promotion_blocks_bcc_anchor_margin() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="back",
            image_summary="Sparse slightly irregular and somewhat heterogeneous pigmentation.",
            early_ddx_candidates=["BKL", "AKIEC", "NV"],
            support_margin=61.0,
            subtype_support_margin=14.4,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_truncal_nevus_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_extremity_melanoma_topk_promotion() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="upper extremity",
            image_summary="A dark irregular lesion with a central area of necrosis.",
            early_ddx_candidates=["MEL", "BCC", "AKIEC"],
            support_margin=48.2,
            subtype_support_margin=15.9,
        ),
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "medgemma_ham10000_extremity_melanoma_topk_promotion" in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_extremity_melanoma_topk_promotion_blocks_face_anchor() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="face",
            image_summary="A well-defined dark lesion with a central area of pigmentation.",
            early_ddx_candidates=["MEL", "BCC", "AKIEC"],
            support_margin=46.2,
            subtype_support_margin=2.3,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_extremity_melanoma_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_medgemma_ham10000_extremity_melanoma_topk_promotion_requires_workflow_cell() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_medgemma_ham10000_evidence_bundle(
            site="upper extremity",
            image_summary="A dark irregular lesion with a central area of necrosis.",
            early_ddx_candidates=["MEL", "BCC", "AKIEC"],
            support_margin=48.2,
            subtype_support_margin=15.9,
            workflow_cell_id="qwen__ham10000__dataset_best",
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "medgemma_ham10000_extremity_melanoma_topk_promotion" not in result["fusion_decision"]["reasons"]

def test_qwen_scin_promotes_face_acne_grouped_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "DERMATITIS_ECZEMA", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            skill_outputs={"distribution_analysis_skill": {"body_location": "face"}},
        ),
    )

    assert result["final_diagnosis"] == "ACNE_ROSACEA_FOLLICULAR"
    assert "qwen_scin_face_acne_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_face_acne_requires_target_workflow_cell() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "DERMATITIS_ECZEMA", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            skill_outputs={"distribution_analysis_skill": {"body_location": "face"}},
            workflow_cell_id="qwen__scin__alternate_cell",
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_face_acne_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_face_acne_blocks_scaling_scarred_dermatitis_anchor() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["ACNE_ROSACEA_FOLLICULAR", "DERMATITIS_ECZEMA"],
            skill_outputs={
                "distribution_analysis_skill": {"body_location": "face"},
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "raised papules with scarring",
                    "surface": ["scaling and peeling"],
                },
            },
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_face_acne_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_promotes_torso_pustular_infection_grouped_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "DERMATITIS_ECZEMA", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
            clinical_metadata={"region": "torso_front"},
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "clustered papules with pustules",
                    "surface": ["vesicular crust"],
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "INFECTION_VIRAL_FUNGAL"
    assert "qwen_scin_torso_pustular_infection_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_torso_infection_blocks_negated_pustules() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "DERMATITIS_ECZEMA", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
            clinical_metadata={"region": "torso_front"},
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "eczema-like papules",
                    "surface": ["no pustules or vesicles"],
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_torso_pustular_infection_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_promotes_lower_body_joint_pain_vascular_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "VASCULAR_PURPURIC", "differential_diagnoses": ["VASCULAR_PURPURIC"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
            clinical_metadata={"region": "leg", "textures_present": ["flat"]},
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "flat purpuric macules",
                    "associated_context": ["systemic joint_pain"],
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "VASCULAR_PURPURIC"
    assert "qwen_scin_lower_body_joint_pain_vascular_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_vascular_pattern_requires_lower_body_region() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "VASCULAR_PURPURIC", "differential_diagnoses": ["VASCULAR_PURPURIC"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
            clinical_metadata={"region": "head_or_neck", "textures_present": ["flat"]},
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "flat purpuric macules",
                    "associated_context": ["systemic joint_pain"],
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_lower_body_joint_pain_vascular_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_promotes_buttocks_leg_foot_vascular_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "VASCULAR_PURPURIC", "differential_diagnoses": ["VASCULAR_PURPURIC"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "URTICARIA_BITE_FOLLICULITIS"],
            clinical_metadata={
                "region": "buttocks",
                "body_sites": ["buttocks", "leg", "foot_top_or_side"],
                "textures_present": ["flat"],
                "symptoms_present": ["bothersome_appearance", "increasing_size", "burning"],
            },
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "flat purpuric macules",
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "VASCULAR_PURPURIC"
    assert "qwen_scin_lower_body_joint_pain_vascular_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_promotes_back_hand_malignant_grouped_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "DERMATITIS_ECZEMA", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            clinical_metadata={"region": "back_of_hand"},
            skill_outputs={"metadata_consistency_skill": {"consistency_score": "high"}},
        ),
    )

    assert result["final_diagnosis"] == "MALIGNANT_PREMALIGNANT"
    assert "qwen_scin_back_hand_malignant_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_back_hand_malignant_blocks_secondary_body_site_only() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "DERMATITIS_ECZEMA", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            clinical_metadata={"region": "arm", "body_sites": ["back_of_hand"]},
            skill_outputs={"metadata_consistency_skill": {"consistency_score": "high"}},
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_back_hand_malignant_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_promotes_headneck_medium_risk_bcc_grouped_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis", "Basal Cell Carcinoma"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            clinical_metadata={"region": "head_or_neck"},
            skill_outputs={
                "distribution_analysis_skill": {"body_location": "neck"},
                "malignancy_risk_assessment_skill": {"risk_level": "medium"},
            },
        ),
    )

    assert result["final_diagnosis"] == "MALIGNANT_PREMALIGNANT"
    assert "qwen_scin_headneck_medium_risk_bcc_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_headneck_bcc_malignant_requires_medium_risk() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis", "Basal Cell Carcinoma"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            clinical_metadata={"region": "head_or_neck"},
            skill_outputs={
                "distribution_analysis_skill": {"body_location": "neck"},
                "malignancy_risk_assessment_skill": {"risk_level": "low"},
            },
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_headneck_medium_risk_bcc_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_headneck_bcc_malignant_requires_headneck_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis", "Basal Cell Carcinoma"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "ACNE_ROSACEA_FOLLICULAR"],
            clinical_metadata={"region": "leg"},
            skill_outputs={
                "distribution_analysis_skill": {"body_location": "leg"},
                "malignancy_risk_assessment_skill": {"risk_level": "medium"},
            },
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_headneck_medium_risk_bcc_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_promotes_pigment_nevus_topk_grouped_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis", "Nevus"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "PIGMENT_KERATOSIS_NEVUS"],
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "pigmented papules with pustules",
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "PIGMENT_KERATOSIS_NEVUS"
    assert "qwen_scin_pigment_nevus_topk_grouped_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_scin_pigment_nevus_requires_topk_signal() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "PIGMENT_KERATOSIS_NEVUS"],
            skill_outputs={
                "lesion_description_structuring_skill": {
                    "primary_lesion_morphology": "pigmented papules with pustules",
                }
            },
        ),
    )

    assert result["final_diagnosis"] == "Contact Dermatitis"
    assert "qwen_scin_pigment_nevus_topk_grouped_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_scin_grouped_promotions_preserve_non_dermatitis_anchor() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={"final_diagnosis": "Nevus", "differential_diagnoses": ["Nevus"], "confidence": "Moderate"},
        agent_output={"final_diagnosis": "Contact Dermatitis", "differential_diagnoses": ["Contact Dermatitis", "Nevus"], "confidence": "Moderate"},
        evidence_bundle=_qwen_scin_evidence_bundle(
            early_ddx_candidates=["DERMATITIS_ECZEMA", "PIGMENT_KERATOSIS_NEVUS"],
            skill_outputs={
                "distribution_analysis_skill": {"body_location": "face"},
                "lesion_description_structuring_skill": {"primary_lesion_morphology": "pigmented papules with pustules"},
            },
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert not any(reason.startswith("qwen_scin_") for reason in result["fusion_decision"]["reasons"])


def test_qwen_ham10000_promotes_pigmented_plaque_mel_from_topk() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular pigmented plaque with mottled brown and blue areas.",
            support_margin=41.8,
            subtype_support_margin=29.2,
        ),
    )

    assert result["final_diagnosis"] == "Malignant Melanoma"
    assert "qwen_ham10000_pigmented_plaque_mel_topk_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_mel_promotion_requires_workflow_cell() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular pigmented plaque with mottled brown and blue areas.",
            support_margin=41.8,
            subtype_support_margin=29.2,
            workflow_cell_id="qwen__ham10000__alternate_cell",
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_ham10000_pigmented_plaque_mel_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_mel_promotion_requires_plaque_signal() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular pigmented papule with mottled brown and blue areas.",
            support_margin=41.8,
            subtype_support_margin=29.2,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_ham10000_pigmented_plaque_mel_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_promotes_reddish_hyperpigmented_akiec_from_topk() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Actinic Keratosis", "Seborrheic Keratosis"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Reddish hyperpigmented irregular keratotic plaque.",
            support_margin=62.1,
            subtype_support_margin=11.4,
        ),
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_ham10000_reddish_hyperpigmented_akiec_topk_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_promotes_reddish_brown_nevus_before_akiec() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus", "Actinic Keratosis"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular hyperpigmented brown and red macule with asymmetric color.",
            support_margin=65.7,
            subtype_support_margin=15.8,
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "qwen_ham10000_reddish_brown_nevus_topk_promotion" in result["fusion_decision"]["reasons"]
    assert "qwen_ham10000_reddish_hyperpigmented_akiec_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_nevus_promotion_requires_subtype_window() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular hyperpigmented brown and red macule with asymmetric color.",
            support_margin=65.7,
            subtype_support_margin=24.0,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_ham10000_reddish_brown_nevus_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_akiec_promotion_requires_akiec_topk() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Nevus", "Seborrheic Keratosis"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Reddish hyperpigmented irregular keratotic plaque.",
            support_margin=62.1,
            subtype_support_margin=11.4,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_ham10000_reddish_hyperpigmented_akiec_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_promotes_central_depression_bcc_from_akiec_topk() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis", "Basal Cell Carcinoma", "Seborrheic Keratosis"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Lesion has central depression with mottled pigmentation.",
            support_margin=50.2,
            subtype_support_margin=20.1,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_ham10000_central_depression_bcc_topk_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_promotes_support_window_central_depression_bcc() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis", "Basal Cell Carcinoma", "Nevus"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular hyperpigmented nodule with central depression and pink coloration.",
            support_margin=65.8,
            subtype_support_margin=7.6,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "qwen_ham10000_central_depression_bcc_topk_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_bcc_promotion_requires_central_depression() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis", "Basal Cell Carcinoma", "Seborrheic Keratosis"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Lesion has mottled pigmentation and a smooth surface.",
            support_margin=50.2,
            subtype_support_margin=20.1,
        ),
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_ham10000_central_depression_bcc_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_promotes_red_asymmetric_bkl_from_akiec_topk() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Seborrheic Keratosis", "Dermatofibroma", "Vascular Lesion"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular hyperpigmented red-brown patch with asymmetric keratotic surface.",
            support_margin=61.2,
            subtype_support_margin=28.8,
        ),
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "qwen_ham10000_red_asymmetric_bkl_topk_promotion" in result["fusion_decision"]["reasons"]


def test_qwen_ham10000_bkl_promotion_requires_red_asymmetry() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Seborrheic Keratosis", "Dermatofibroma", "Vascular Lesion"],
            "confidence": "Medium",
        },
        evidence_bundle=_qwen_ham10000_evidence_bundle(
            selected_evidence_text="Irregular hyperpigmented brown patch with keratotic surface.",
            support_margin=61.2,
            subtype_support_margin=28.8,
        ),
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "qwen_ham10000_red_asymmetric_bkl_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_promotes_trunk_bkl_structure() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="trunk",
            surface_texture="smooth",
            clustering_pattern="solitary",
            support_margin=38.6,
            subtype_support_margin=0.7,
        ),
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "dermatollama_ham10000_bkl_structure_top1_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_promotes_back_cobblestone_bkl_structure() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="back",
            surface_texture="cobblestone-like",
            border_clarity="well-defined",
            border_irregularity="regular",
            clustering_pattern="clustered",
            support_margin=38.2,
            subtype_support_margin=0.5,
        ),
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "dermatollama_ham10000_bkl_structure_top1_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_promotes_agent_bkl_low_margin_structure() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Seborrheic Keratosis",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Seborrheic Keratosis"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="chest",
            primary_lesion_morphology="pigmented macule with reticular globules",
            surface_texture="rough",
            support_margin=42.4,
            subtype_support_margin=4.6,
        ),
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "dermatollama_ham10000_agent_bkl_structured_top1_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_promotes_agent_bkl_upper_extremity_rough_cluster() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Seborrheic Keratosis",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Malignant Melanoma", "Seborrheic Keratosis"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="upper extremity",
            primary_lesion_morphology="rough reticular lesion with globules",
            surface_texture="rough",
            clustering_pattern="clustered",
            support_margin=61.9,
            subtype_support_margin=16.5,
        ),
    )

    assert result["final_diagnosis"] == "Seborrheic Keratosis"
    assert "dermatollama_ham10000_agent_bkl_structured_top1_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_agent_bkl_promotion_is_cell_bound() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Seborrheic Keratosis",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Seborrheic Keratosis"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="chest",
            workflow_cell_id="qwen__ham10000__dataset_best",
            primary_lesion_morphology="pigmented macule with reticular globules",
            surface_texture="rough",
            support_margin=42.4,
            subtype_support_margin=4.6,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "dermatollama_ham10000_agent_bkl_structured_top1_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_agent_bkl_promotion_blocks_back_bcc_risk_pattern() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Seborrheic Keratosis",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Seborrheic Keratosis"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="back",
            primary_lesion_morphology="raised plaque with reticular network and globules",
            surface_texture="rough",
            support_margin=61.7,
            subtype_support_margin=22.5,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "dermatollama_ham10000_agent_bkl_structured_top1_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_promotes_truncal_reticular_nevus() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="trunk",
            primary_lesion_morphology="pigmented macule with reticular pattern",
            color=["dark brown", "reticular pattern"],
            surface_texture="smooth",
            support_margin=60.6,
            subtype_support_margin=16.2,
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_ham10000_truncal_reticular_nevus_top1_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_promotes_truncal_reticular_nevus_with_unknown_uncertainty() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="trunk",
            primary_lesion_morphology="pigmented macule with reticular pattern",
            color=["brown", "reticular pattern"],
            surface_texture="smooth",
            support_margin=63.2,
            subtype_support_margin=16.4,
            uncertainty_level="unknown",
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_ham10000_truncal_reticular_nevus_top1_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_truncal_reticular_nevus_blocks_raised_counterexample() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="trunk",
            primary_lesion_morphology="pigmented macule with reticular pattern and raised area",
            color=["dark brown", "reticular pattern"],
            surface_texture="rough",
            support_margin=63.2,
            subtype_support_margin=15.4,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "dermatollama_ham10000_truncal_reticular_nevus_top1_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_truncal_reticular_nevus_is_cell_bound() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="trunk",
            workflow_cell_id="qwen__ham10000__dataset_best",
            primary_lesion_morphology="pigmented macule with reticular pattern",
            color=["dark brown", "reticular pattern"],
            support_margin=60.6,
            subtype_support_margin=14.2,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "dermatollama_ham10000_truncal_reticular_nevus_top1_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_blocks_back_nevus_override_anchor() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Seborrheic Keratosis", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="back",
            surface_texture="smooth",
            border_clarity="poor",
            border_irregularity="irregular",
            support_margin=39.9,
            subtype_support_margin=1.4,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "dermatollama_ham10000_baseline_anchor_guard" in result["fusion_decision"]["reasons"]
    assert "dermatollama_ham10000_guarded_override" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_blocks_nevus_override_when_akiec_topk_would_drop() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Actinic Keratosis"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": [
                "Nevus",
                "Dermatofibroma",
                "Seborrheic Keratosis",
                "Vascular lesion",
                "Basal Cell Carcinoma",
                "Actinic Keratosis",
            ],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="scalp",
            surface_texture="smooth",
            support_margin=39.0,
            subtype_support_margin=1.4,
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "Actinic Keratosis" in result["differential_diagnoses"]
    assert "dermatollama_ham10000_guarded_override" not in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_preserves_non_back_nevus_override() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="scalp",
            surface_texture="smooth",
            support_margin=39.0,
            subtype_support_margin=1.4,
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_ham10000_guarded_override" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_adds_low_uncertainty_raw_agent_differential() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="upper extremity",
            support_margin=62.0,
            subtype_support_margin=15.0,
            uncertainty_level="low",
        ),
    )

    assert result["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "Nevus" in result["differential_diagnoses"]
    assert "dermatollama_ham10000_raw_agent_differential_expansion" in result["fusion_decision"]["reasons"]


def test_dermatollama_ham10000_raw_agent_differential_is_cell_bound() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "confidence": "medium",
        },
        evidence_bundle=_dermatollama_ham10000_evidence_bundle(
            localization="upper extremity",
            workflow_cell_id="qwen__ham10000__dataset_best",
            support_margin=62.0,
            subtype_support_margin=15.0,
            uncertainty_level="low",
        ),
    )

    assert "Nevus" not in result["differential_diagnoses"]
    assert "dermatollama_ham10000_raw_agent_differential_expansion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_promotes_nevus_to_benign_nodule_group() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Dermatofibroma"],
            "confidence": "High",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Dermatofibroma", "Fibroma Molle", "Nevus"],
            image_summary="Firm nodular lesion resembling a fibroma.",
            selected_evidence_text="visual evidence mentions a nodular cyst-like fibroma.",
            support_margin=55.0,
            subtype_support_margin=12.0,
            contradiction_count=5,
        ),
    )

    assert result["final_diagnosis"] == "BENIGN_TUMOR_CYST"
    assert "dermatollama_sd198_nevus_benign_nodule_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_nevus_benign_promotion_requires_workflow_cell() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Dermatofibroma"],
            "confidence": "High",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Dermatofibroma", "Fibroma Molle", "Nevus"],
            image_summary="Firm nodular lesion resembling a fibroma.",
            selected_evidence_text="visual evidence mentions a nodular cyst-like fibroma.",
            support_margin=55.0,
            subtype_support_margin=12.0,
            contradiction_count=5,
            workflow_cell_id="dermatollama__ham10000__baseline_guard_v1",
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_sd198_nevus_benign_nodule_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_nevus_benign_promotion_blocks_acne_like_anchor() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Acne"],
            "confidence": "Medium",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Nevus Comedonicus", "Acne", "Milia"],
            image_summary="A small papular comedonal lesion.",
            selected_evidence_text="comedonal papule with acne-like features.",
            support_margin=55.0,
            subtype_support_margin=12.0,
            contradiction_count=5,
        ),
    )

    assert result["final_diagnosis"] == "Nevus"
    assert "dermatollama_sd198_nevus_benign_nodule_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_promotes_actinic_scaly_papulosquamous_group() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis", "Ichthyosis"],
            "confidence": "High",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Ichthyosis", "Xerosis", "Seborrheic keratosis"],
            image_summary="Skin with rough dry scaly texture.",
            selected_evidence_text="dry scaly keratotic plaques with ichthyosis signal.",
            support_margin=56.0,
            subtype_support_margin=12.0,
            contradiction_count=2,
        ),
    )

    assert result["final_diagnosis"] == "PAPULOSQUAMOUS_KERATOTIC"
    assert "dermatollama_sd198_actinic_scaly_papulosquamous_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_actinic_scaly_promotion_requires_marker() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis", "Psoriasis"],
            "confidence": "High",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Psoriasis", "Actinic Keratosis"],
            image_summary="A pink papule with a smooth surface.",
            selected_evidence_text="smooth papule without scale.",
            support_margin=56.0,
            subtype_support_margin=12.0,
            contradiction_count=2,
        ),
    )

    assert result["final_diagnosis"] == "Actinic Keratosis"
    assert "dermatollama_sd198_actinic_scaly_papulosquamous_topk_promotion" not in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_actinic_scaly_promotion_allows_agent_papulosquamous_canonical() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Actinic Keratosis"],
            "confidence": "High",
        },
        agent_output={
            "final_diagnosis": "Hyperkeratosis Palmaris Et Plantaris",
            "differential_diagnoses": ["Hyperkeratosis Palmaris Et Plantaris", "Xerosis", "Actinic Keratosis"],
            "confidence": "Medium",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Hyperkeratosis Palmaris Et Plantaris", "Xerosis", "Tinea pedis"],
            image_summary="A foot plaque with dry keratotic scale.",
            selected_evidence_text="hyperkeratosis and xerosis support a papulosquamous group.",
            support_margin=56.0,
            subtype_support_margin=12.0,
            contradiction_count=2,
        ),
    )

    assert result["final_diagnosis"] == "PAPULOSQUAMOUS_KERATOTIC"
    assert "dermatollama_sd198_actinic_scaly_papulosquamous_topk_promotion" in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_rescues_crowe_sign_to_pigmentary_group() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Crowe's Sign",
            "differential_diagnoses": ["Crowe's Sign"],
            "confidence": "Medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Crowe's Sign"],
            "confidence": "Medium",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Pigmented nevus", "Seborrheic keratosis", "Melanocytic nevus"],
            image_summary="Skin with hair follicles and a small brown lesion.",
            selected_evidence_text="brown pigmented nevus-like lesion with hair follicle signal.",
            support_margin=36.0,
            subtype_support_margin=2.1,
            contradiction_count=1,
        ),
    )

    assert result["final_diagnosis"] == "PIGMENTARY_NEVUS_KERATOSIS"
    assert "dermatollama_sd198_crowe_sign_pigmentary_rescue" in result["fusion_decision"]["reasons"]


def test_dermatollama_sd198_crowe_sign_rescue_requires_pigmentary_initial_candidate() -> None:
    result = apply_conservative_agent_fusion(
        baseline_output={
            "final_diagnosis": "Crowe's Sign",
            "differential_diagnoses": ["Crowe's Sign"],
            "confidence": "Medium",
        },
        agent_output={
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Crowe's Sign"],
            "confidence": "Medium",
        },
        evidence_bundle=_dermatollama_sd198_evidence_bundle(
            early_ddx_candidates=["Actinic keratosis", "Malignant skin cancer"],
            image_summary="A pale linear scar.",
            selected_evidence_text="linear scar without pigmentary nevus support.",
            support_margin=36.0,
            subtype_support_margin=2.1,
            contradiction_count=1,
        ),
    )

    assert result["final_diagnosis"] == "Crowe's Sign"
    assert "dermatollama_sd198_crowe_sign_pigmentary_rescue" not in result["fusion_decision"]["reasons"]
