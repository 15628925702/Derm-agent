"""
Model-specific workflow routing configuration.

This module provides model-aware workflow profile selection to optimize
DermAgent performance for different vision-language models.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from workflow_evolution.runtime import maybe_apply_active_workflow_evolution


SPECIALIST_SKILLS = (
    "mel_nev_specialist_skill",
    "ack_scc_specialist_skill",
    "benign_mimic_specialist_skill",
)


DATASET_ALIASES = {
    "pad": "pad20",
    "pad20": "pad20",
    "pad_ufes_20": "pad20",
    "pad-ufes-20": "pad20",
    "isic": "isic2019",
    "isic2019": "isic2019",
    "scin": "scin",
    "sd198": "sd198",
    "sd-198": "sd198",
    "xiangya": "xiangya_sft",
    "xiangya_sft": "xiangya_sft",
    "ham": "ham10000",
    "ham10000": "ham10000",
}


# Explicit 6x6 workflow cells.  The qwen row is the tuned/current-best line;
# the other model rows intentionally fall back to model + dataset routing until
# each cell is tuned from 6x6 results.
MODEL_DATASET_WORKFLOW_PROFILES = {
    "Qwen2.5-VL-7B-Instruct": {
        "pad20": {
            "workflow_cell_id": "qwen__pad20__dataset_best",
            "label_space_id": "derm_six",
            "inherit_dataset_workflow": True,
            "block_model_workflow_profile": True,
        },
        "isic2019": {
            "workflow_cell_id": "qwen__isic2019__dataset_best",
            "label_space_id": "isic2019_full",
            "workflow_profile": "qwen_isic2019_archive_guard_workflow",
            "workflow_capabilities": ["baseline_anchored_final", "melanocytic_guard_reasoning"],
            "inherit_dataset_workflow": True,
            "disable_legacy_final_path": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
        },
        "scin": {
            "workflow_cell_id": "qwen__scin__grouped_best",
            "label_space_id": "scin_grouped",
            "environment": {"DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped"},
            "inherit_dataset_workflow": True,
            "block_model_workflow_profile": True,
        },
        "sd198": {
            "workflow_cell_id": "qwen__sd198__grouped_best",
            "label_space_id": "sd198_grouped",
            "environment": {"DERMAGENT_SD198_LABEL_SPACE_ID": "sd198_grouped"},
            "inherit_dataset_workflow": True,
            "block_model_workflow_profile": True,
        },
        "xiangya_sft": {
            "workflow_cell_id": "qwen__xiangya_sft__grouped_best",
            "label_space_id": "xiangya_sft_grouped",
            "inherit_dataset_workflow": True,
            "block_model_workflow_profile": True,
        },
        "ham10000": {
            "workflow_cell_id": "qwen__ham10000__dataset_best",
            "label_space_id": "ham10000_full",
            "inherit_dataset_workflow": True,
            "block_model_workflow_profile": True,
        },
    },
    "DermatoLlama-full": {
        "isic2019": {
            "workflow_cell_id": "dermatollama__isic2019__archive_guard_v1",
            "label_space_id": "isic2019_full",
            "workflow_profile": "dermatollama_isic2019_archive_guard_workflow",
            "workflow_capabilities": ["baseline_anchored_final", "melanocytic_guard_reasoning"],
            "inherit_dataset_workflow": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "ham10000": {
            "workflow_cell_id": "dermatollama__ham10000__baseline_guard_v1",
            "label_space_id": "ham10000_full",
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final"],
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "pad20": {
            "workflow_cell_id": "dermatollama__pad20__baseline_guard_v1",
            "label_space_id": "derm_six",
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final"],
            "skip_specialist_skills": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "xiangya_sft": {
            "workflow_cell_id": "dermatollama__xiangya_sft__baseline_guard_v1",
            "label_space_id": "xiangya_sft_grouped",
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final"],
            "skip_specialist_skills": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "scin": {
            "workflow_cell_id": "dermatollama__scin__grouped_guard_v1",
            "label_space_id": "scin_grouped",
            "environment": {"DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped"},
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final"],
            "skip_specialist_skills": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "sd198": {
            "workflow_cell_id": "dermatollama__sd198__grouped_guard_v1",
            "label_space_id": "sd198_grouped",
            "environment": {"DERMAGENT_SD198_LABEL_SPACE_ID": "sd198_grouped"},
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final"],
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
    },
    "medgemma-4b-it": {
        "scin": {
            "workflow_cell_id": "medgemma__scin__grouped_core_v1",
            "label_space_id": "scin_grouped",
            "environment": {"DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped"},
            "workflow_profile": "medgemma_scin_grouped_core_workflow",
            "workflow_capabilities": ["graded_conservative_fusion"],
            "inherit_dataset_workflow": True,
            "force_conservative_fusion": True,
            "disable_legacy_final_path": True,
            "force_disable_skills": [
                "metadata_consistency_skill",
                "uncertainty_assessment_skill",
                "information_gap_detection_skill",
                "contradiction_check_skill",
                "escalation_recommendation_skill",
                "ack_scc_specialist_skill",
                "benign_mimic_specialist_skill",
                "mel_nev_specialist_skill",
                "exclusion_reasoning_skill",
            ],
        },
        "sd198": {
            "workflow_cell_id": "medgemma__sd198__grouped_coarse_v1",
            "label_space_id": "sd198_grouped",
            "environment": {"DERMAGENT_SD198_LABEL_SPACE_ID": "sd198_grouped"},
            "workflow_profile": "coarse_taxonomy_workflow",
            "workflow_profile_mode": "replace",
            "workflow_capabilities": [
                "coarse_taxonomy_reasoning",
                "grouped_label_reasoning",
                "focal_lesion_reasoning",
                "graded_conservative_fusion",
            ],
            "replace_workflow_capabilities": True,
            "force_conservative_fusion": True,
            "disable_legacy_final_path": True,
            "force_disable_skills": [
                "metadata_consistency_skill",
                "uncertainty_assessment_skill",
                "information_gap_detection_skill",
                "contradiction_check_skill",
                "escalation_recommendation_skill",
                "ack_scc_specialist_skill",
                "benign_mimic_specialist_skill",
                "mel_nev_specialist_skill",
                "exclusion_reasoning_skill",
            ],
        },
        "pad20": {
            "workflow_cell_id": "medgemma__pad20__clinical_core_v2",
            "label_space_id": "derm_six",
            "workflow_profile": "medgemma_pad20_clinical_core_workflow",
            "inherit_dataset_workflow": True,
            "force_disable_skills": [
                "metadata_consistency_skill",
                "uncertainty_assessment_skill",
                "information_gap_detection_skill",
                "contradiction_check_skill",
                "escalation_recommendation_skill",
                "ack_scc_specialist_skill",
                "benign_mimic_specialist_skill",
                "mel_nev_specialist_skill",
                "exclusion_reasoning_skill",
            ],
        },
        "isic2019": {
            "workflow_cell_id": "medgemma__isic2019__archive_guard_v1",
            "label_space_id": "isic2019_full",
            "workflow_profile": "medgemma_isic2019_archive_guard_workflow",
            "workflow_capabilities": ["baseline_anchored_final", "melanocytic_guard_reasoning"],
            "inherit_dataset_workflow": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
            "force_disable_skills": [
                "metadata_consistency_skill",
                "uncertainty_assessment_skill",
                "information_gap_detection_skill",
                "contradiction_check_skill",
                "escalation_recommendation_skill",
            ],
        },
        "ham10000": {
            "workflow_cell_id": "medgemma__ham10000__akiec_face_guard_v1",
            "label_space_id": "ham10000_full",
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final"],
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
    },
    "Hulu-Med-7B": {
        "isic2019": {
            "workflow_cell_id": "hulumed__isic2019__archive_guard_v1",
            "label_space_id": "isic2019_full",
            "workflow_profile": "hulumed_isic2019_archive_guard_workflow",
            "workflow_capabilities": ["baseline_anchored_final", "melanocytic_guard_reasoning"],
            "inherit_dataset_workflow": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "pad20": {
            "workflow_cell_id": "hulumed__pad20__clinical_guard_v1",
            "label_space_id": "derm_six",
            "workflow_profile": "hulumed_pad20_clinical_guard_workflow",
            "workflow_capabilities": ["baseline_anchored_final", "clinical_subtype_guard_reasoning"],
            "inherit_dataset_workflow": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "sd198": {
            "workflow_cell_id": "hulumed__sd198__grouped_coarse_guard_v1",
            "label_space_id": "sd198_grouped",
            "environment": {"DERMAGENT_SD198_LABEL_SPACE_ID": "sd198_grouped"},
            "workflow_profile": "coarse_taxonomy_workflow",
            "workflow_profile_mode": "replace",
            "workflow_capabilities": ["baseline_anchored_final", "coarse_taxonomy_reasoning", "grouped_label_reasoning"],
            "replace_workflow_capabilities": True,
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "ham10000": {
            "workflow_cell_id": "hulumed__ham10000__akiec_guard_v1",
            "label_space_id": "ham10000_full",
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final", "sparse_lesion_reasoning"],
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
        "scin": {
            "workflow_cell_id": "hulumed__scin__grouped_guard_v1",
            "label_space_id": "scin_grouped",
            "environment": {"DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped"},
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final", "grouped_label_reasoning"],
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
    },
    "Llama-3.2-11B-Vision-Instruct": {
        "scin": {
            "workflow_cell_id": "llama__scin__grouped_guard_v1",
            "label_space_id": "scin_grouped",
            "environment": {"DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped"},
            "inherit_dataset_workflow": True,
            "workflow_capabilities": ["baseline_anchored_final", "grouped_label_reasoning"],
            "force_conservative_fusion": True,
            "fallback_on_malformed_final": True,
            "disable_legacy_final_path": True,
        },
    },
}


# Model-specific workflow overrides
MODEL_WORKFLOW_PROFILES = {
    # SkinVL performs poorly with complex multi-step reasoning
    # Use simpler, more direct workflow
    "SkinVL-MM": {
        "workflow_profile": "direct_baseline_workflow",
        "workflow_capabilities": ["minimal_reasoning", "direct_prediction"],
        "skip_specialist_skills": True,
        "skip_experience_retrieval": True,
        "force_conservative_fusion": True,
        "fallback_on_malformed_final": True,
        "disable_legacy_final_path": True,
    },
    "SkinVL": {
        "alias_for": "SkinVL-MM",
    },

    # Qwen performs well with full DermAgent workflow
    "Qwen2.5-VL-7B-Instruct": {
        "workflow_profile": None,  # Use default inference
        "enable_all_capabilities": True,
    },

    # Llama shows mixed results - use conservative approach
    "Llama-3.2-11B-Vision-Instruct": {
        "workflow_profile": "conservative_archive_workflow",
        "workflow_capabilities": ["baseline_anchored_final"],
        "conservative_fusion_weight": 0.7,  # More conservative
        "force_conservative_fusion": True,
        "fallback_on_malformed_final": True,
        "disable_legacy_final_path": True,
    },

    # Hulu-Med shows good results with standard workflow
    "Hulu-Med-7B": {
        "workflow_profile": None,  # Use default inference
    },

    # MedGemma - not yet tested
    "medgemma-4b-it": {
        "workflow_profile": None,  # Use default inference
    },

    # DermatoLlama - not yet tested
    "DermatoLlama-full": {
        "workflow_profile": None,  # Use default inference
    },
}


def normalize_dataset_key(dataset_name: str | None) -> str:
    raw = str(dataset_name or "").strip().lower()
    if not raw:
        return ""
    return DATASET_ALIASES.get(raw, raw)


def _normalize_model_key(model_name: str) -> str:
    raw_model = str(model_name or "").strip()
    if not raw_model:
        return ""
    if raw_model in MODEL_WORKFLOW_PROFILES:
        return raw_model
    lowered = raw_model.lower()
    if "skinvl" in lowered:
        return "SkinVL-MM"
    for known_model in MODEL_WORKFLOW_PROFILES:
        if lowered == known_model.lower():
            return known_model
    return raw_model


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)


def _resolve_model_config(model_name: str) -> tuple[str, dict[str, Any]]:
    model_key = _normalize_model_key(model_name)
    model_config = dict(MODEL_WORKFLOW_PROFILES.get(model_key, {}))
    alias_for = str(model_config.get("alias_for", "")).strip()
    if alias_for:
        model_key = alias_for
        model_config = dict(MODEL_WORKFLOW_PROFILES.get(alias_for, {}))
    return model_key, model_config


def get_model_dataset_workflow_profile(model_name: str, dataset_name: str | None) -> dict[str, Any]:
    model_key, _ = _resolve_model_config(model_name)
    dataset_key = normalize_dataset_key(dataset_name)
    if not model_key or not dataset_key:
        return {}
    model_cells = MODEL_DATASET_WORKFLOW_PROFILES.get(model_key, {})
    cell_config = maybe_apply_active_workflow_evolution(
        model_key=model_key,
        dataset_key=dataset_key,
        static_cell_config=model_cells.get(dataset_key, {}),
    )
    if not cell_config:
        return {}
    cell = _copy_config(cell_config)
    cell["model_workflow_profile"] = model_key
    cell["model_name"] = str(model_name or "").strip()
    cell["dataset_name"] = dataset_key
    cell["workflow_routing_priority"] = "model_dataset"
    return cell


def dataset_environment_overrides_for_model_dataset(model_name: str, dataset_name: str | None) -> dict[str, str]:
    cell = get_model_dataset_workflow_profile(model_name, dataset_name)
    return {
        str(key): str(value)
        for key, value in dict(cell.get("environment", {}) or {}).items()
        if str(key).strip() and str(value).strip()
    }


def get_model_workflow_overrides(
    model_name: str,
    *,
    dataset_name: str | None = None,
    base_workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get workflow overrides for a specific model.

    Args:
        model_name: The model identifier (e.g., "SkinVL-MM")
        dataset_name: Optional dataset name for dataset-specific tuning
        base_workflow_context: Base workflow context to merge with overrides

    Returns:
        Dictionary of workflow overrides to pass to run_agent via execution_overrides
    """
    model_dataset_config = get_model_dataset_workflow_profile(model_name, dataset_name)
    if model_dataset_config:
        return _build_workflow_overrides(model_dataset_config, model_name=model_name)

    model_key, model_config = _resolve_model_config(model_name)
    if not model_config:
        # No specific config for this model, return empty overrides
        return {}

    model_config["workflow_routing_priority"] = "model"
    model_config["model_workflow_profile"] = model_key
    model_config["model_name"] = str(model_name or "").strip()

    overrides = _build_workflow_overrides(model_config, model_name=model_name)
    if set(overrides.keys()).issubset({"model_workflow_profile", "model_name", "workflow_routing_priority"}):
        return {}

    base_context = dict(base_workflow_context or {})
    base_workflow_profile = str(base_context.get("workflow_profile", "")).strip().lower()

    # Workflow-specific model overlays.  This keeps dataset routing as the base
    # layer and only adds model behavior for the affected workflow cell.
    if (
        model_key == "Qwen2.5-VL-7B-Instruct"
        and base_workflow_profile == "clinical_full_taxonomy_lesion_workflow"
    ):
        overrides["workflow_profile"] = "clinical_malignant_guard_workflow"
        overrides["workflow_capabilities"] = list(
            dict.fromkeys(list(overrides.get("workflow_capabilities", [])) + ["baseline_anchored_final"])
        )
        overrides["force_conservative_fusion"] = True
        overrides["disable_legacy_final_path"] = True

    if overrides:
        overrides["model_workflow_profile"] = model_key
        overrides["model_name"] = str(model_name or "").strip()
        overrides.setdefault("workflow_routing_priority", "model")

    return overrides


def _build_workflow_overrides(config: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    source = dict(config or {})
    overrides: dict[str, Any] = {}

    # Apply workflow profile override if specified
    if "workflow_profile" in source and source["workflow_profile"] is not None:
        overrides["workflow_profile"] = source["workflow_profile"]
    if "workflow_profile_mode" in source:
        overrides["workflow_profile_mode"] = source["workflow_profile_mode"]
    if source.get("replace_workflow_capabilities"):
        overrides["replace_workflow_capabilities"] = True
    if source.get("inherit_dataset_workflow"):
        overrides["inherit_dataset_workflow"] = True
    if source.get("block_model_workflow_profile"):
        overrides["block_model_workflow_profile"] = True

    # Apply workflow capabilities override
    if "workflow_capabilities" in source:
        overrides["workflow_capabilities"] = source["workflow_capabilities"]

    # Apply specialist skill control
    if source.get("skip_specialist_skills"):
        overrides["skip_specialist_skills"] = True
        overrides["disabled_specialist_skills"] = list(SPECIALIST_SKILLS)
    if "disabled_specialist_skills" in source:
        overrides["disabled_specialist_skills"] = list(source.get("disabled_specialist_skills") or [])

    # Apply experience retrieval control
    if source.get("skip_experience_retrieval"):
        overrides["skip_experience_retrieval"] = True
        overrides["enable_experience_retrieval"] = False

    # Apply conservative fusion weight
    for key in (
        "allowed_skills",
        "force_disable_skills",
        "force_enable_skills",
        "enable_experience_retrieval",
        "enable_skill_retrieval",
        "enable_image_read_audit",
        "conservative_fusion_weight",
        "workflow_cell_id",
        "workflow_routing_priority",
        "label_space_id",
        "dataset_name",
    ):
        if key in source:
            overrides[key] = deepcopy(source[key])

    if source.get("force_conservative_fusion"):
        overrides["force_conservative_fusion"] = True

    if source.get("fallback_on_malformed_final"):
        overrides["fallback_on_malformed_final"] = True

    if source.get("disable_legacy_final_path"):
        overrides["disable_legacy_final_path"] = True

    if overrides:
        overrides["model_workflow_profile"] = str(source.get("model_workflow_profile", "")).strip()
        overrides["model_name"] = str(model_name or "").strip()
    return overrides


def apply_model_workflow_to_case(
    case_input: Any,
    model_name: str,
    *,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Overlay model-specific routing onto an existing dataset workflow context.

    Dataset workflow fields remain the base layer.  Model routing only adds
    model-specific metadata and records the original dataset workflow profile,
    so the two routing axes can coexist as a 6x6 model x dataset matrix.
    """
    overrides = get_model_workflow_overrides(
        model_name,
        dataset_name=dataset_name or getattr(case_input, "dataset_name", None),
        base_workflow_context=getattr(case_input, "workflow_context", None),
    )
    if not overrides:
        return {}

    workflow_context = dict(getattr(case_input, "workflow_context", None) or {})
    dataset_key = str(dataset_name or getattr(case_input, "dataset_name", "") or "").strip()
    label_space_override = str(overrides.get("label_space_id", "") or "").strip()
    if label_space_override:
        case_input.label_space_id = label_space_override
        metadata = dict(getattr(case_input, "metadata", None) or {})
        metadata["label_space_id"] = label_space_override
        case_input.metadata = metadata
    label_space_id = str(getattr(case_input, "label_space_id", "") or "").strip()
    if dataset_key:
        workflow_context.setdefault("dataset_name", dataset_key)
    if label_space_id:
        workflow_context["label_space_id"] = label_space_id
    dataset_workflow_profile = str(workflow_context.get("workflow_profile", "")).strip()
    requested_workflow_profile = str(overrides.get("workflow_profile", "")).strip()
    workflow_profile_mode = str(overrides.get("workflow_profile_mode", "model_overlay")).strip().lower()
    if requested_workflow_profile and workflow_profile_mode in {"dataset", "dataset_override", "replace"}:
        workflow_context["workflow_profile"] = requested_workflow_profile
        dataset_workflow_profile = requested_workflow_profile
        model_workflow_profile = str(overrides.get("model_workflow_profile", "")).strip()
    elif overrides.get("block_model_workflow_profile"):
        model_workflow_profile = ""
    else:
        model_workflow_profile = requested_workflow_profile
    if dataset_workflow_profile:
        workflow_context.setdefault("dataset_workflow_profile", dataset_workflow_profile)
    if model_workflow_profile:
        workflow_context["model_workflow_profile"] = model_workflow_profile
    if overrides.get("workflow_cell_id"):
        workflow_context["workflow_cell_id"] = str(overrides.get("workflow_cell_id", "")).strip()
    if overrides.get("workflow_routing_priority"):
        workflow_context["workflow_routing_priority"] = str(overrides.get("workflow_routing_priority", "")).strip()
    if overrides.get("workflow_capabilities"):
        model_capabilities = [
            str(item).strip()
            for item in overrides.get("workflow_capabilities", [])
            if str(item).strip()
        ]
        if overrides.get("replace_workflow_capabilities"):
            workflow_context["workflow_capabilities"] = list(dict.fromkeys(model_capabilities))
        else:
            existing_capabilities = [
                str(item).strip()
                for item in workflow_context.get("workflow_capabilities", [])
                if str(item).strip()
            ]
            workflow_context["workflow_capabilities"] = list(
                dict.fromkeys(existing_capabilities + model_capabilities)
            )
    workflow_context["model_workflow_routing"] = {
        "model_name": str(model_name or "").strip(),
        "model_profile": str(overrides.get("model_workflow_profile", "")).strip(),
        "workflow_cell_id": str(overrides.get("workflow_cell_id", "")).strip(),
        "workflow_routing_priority": str(overrides.get("workflow_routing_priority", "")).strip(),
        "label_space_id": label_space_id,
        "dataset_workflow_profile": dataset_workflow_profile,
        "model_workflow_profile": model_workflow_profile,
        "skip_specialist_skills": bool(overrides.get("skip_specialist_skills", False)),
        "skip_experience_retrieval": bool(overrides.get("skip_experience_retrieval", False)),
        "force_conservative_fusion": bool(overrides.get("force_conservative_fusion", False)),
        "fallback_on_malformed_final": bool(overrides.get("fallback_on_malformed_final", False)),
        "disable_legacy_final_path": bool(overrides.get("disable_legacy_final_path", False)),
    }
    if overrides.get("force_conservative_fusion"):
        workflow_context["force_conservative_fusion"] = True
    if overrides.get("fallback_on_malformed_final"):
        workflow_context["fallback_on_malformed_final"] = True
    if overrides.get("disable_legacy_final_path"):
        workflow_context["disable_legacy_final_path"] = True
    case_input.workflow_context = workflow_context
    return overrides


def execution_overrides_for_run_agent(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Translate model routing metadata into run_agent's supported switches."""
    source = dict(overrides or {})
    execution_overrides: dict[str, Any] = {}
    if source.get("skip_experience_retrieval"):
        execution_overrides["enable_experience_retrieval"] = False
    if "enable_experience_retrieval" in source:
        execution_overrides["enable_experience_retrieval"] = bool(source["enable_experience_retrieval"])
    if "enable_skill_retrieval" in source:
        execution_overrides["enable_skill_retrieval"] = bool(source["enable_skill_retrieval"])
    if "enable_image_read_audit" in source:
        execution_overrides["enable_image_read_audit"] = bool(source["enable_image_read_audit"])
    if "enable_physician_evidence_summary" in source:
        execution_overrides["enable_physician_evidence_summary"] = bool(source["enable_physician_evidence_summary"])
    return execution_overrides


def merge_model_workflow_policy_overrides(
    policy_config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a policy copy with model-specific planner controls applied."""
    source = dict(overrides or {})
    planner_override_keys = {
        "allowed_skills",
        "force_disable_skills",
        "force_enable_skills",
        "disabled_specialist_skills",
    }
    if not source.get("skip_specialist_skills") and not planner_override_keys.intersection(source.keys()):
        return policy_config

    policy = deepcopy(policy_config)
    planner_policy = dict(policy.get("planner_policy", {}) or {})
    disabled = list(planner_policy.get("force_disable_skills", []) or [])
    if source.get("skip_specialist_skills"):
        disabled.extend(source.get("disabled_specialist_skills") or SPECIALIST_SKILLS)
    else:
        disabled.extend(source.get("disabled_specialist_skills") or [])
    disabled.extend(source.get("force_disable_skills") or [])
    planner_policy["force_disable_skills"] = list(
        dict.fromkeys(str(skill).strip() for skill in disabled if str(skill).strip())
    )
    force_select = list(planner_policy.get("force_select_skills", []) or [])
    force_select.extend(source.get("force_enable_skills") or [])
    planner_policy["force_select_skills"] = list(
        dict.fromkeys(str(skill).strip() for skill in force_select if str(skill).strip())
    )
    if source.get("allowed_skills"):
        planner_policy["allowed_skills"] = list(
            dict.fromkeys(str(skill).strip() for skill in source.get("allowed_skills", []) if str(skill).strip())
        )
    policy["planner_policy"] = planner_policy
    base_policy_id = str(policy.get("policy_id", "policy")).strip() or "policy"
    route_suffix = (
        str(source.get("workflow_cell_id", "")).strip()
        or str(source.get("model_workflow_profile", "")).strip()
        or "model_workflow"
    )
    if not base_policy_id.endswith(f"__{route_suffix}"):
        policy["policy_id"] = f"{base_policy_id}__{route_suffix}"
    return policy


def should_use_simplified_workflow(model_name: str) -> bool:
    """
    Check if a model should use simplified workflow (bypass DermAgent features).

    Args:
        model_name: The model identifier

    Returns:
        True if model should use simplified workflow
    """
    _, model_config = _resolve_model_config(model_name)
    return model_config.get("skip_specialist_skills", False) or \
           model_config.get("skip_experience_retrieval", False)


def get_model_description(model_name: str) -> str:
    """
    Get a human-readable description of model-specific workflow configuration.

    Args:
        model_name: The model identifier

    Returns:
        Description string
    """
    model_key, model_config = _resolve_model_config(model_name)

    if not model_config:
        return f"{model_name}: Using default DermAgent workflow"

    if model_config.get("skip_specialist_skills"):
        return f"{model_name}: Using simplified direct prediction workflow (bypassing DermAgent features)"

    if "conservative_fusion_weight" in model_config:
        weight = model_config["conservative_fusion_weight"]
        return f"{model_name}: Using standard workflow with conservative fusion (weight={weight})"

    return f"{model_name}: Using standard DermAgent workflow"
