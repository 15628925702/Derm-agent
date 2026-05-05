"""
Model-specific workflow routing configuration.

This module provides model-aware workflow profile selection to optimize
DermAgent performance for different vision-language models.
"""

from __future__ import annotations

from typing import Any


# Model-specific workflow overrides
MODEL_WORKFLOW_PROFILES = {
    # SkinVL performs poorly with complex multi-step reasoning
    # Use simpler, more direct workflow
    "SkinVL-MM": {
        "workflow_profile": "direct_baseline_workflow",
        "workflow_capabilities": ["minimal_reasoning", "direct_prediction"],
        "skip_specialist_skills": True,
        "skip_experience_retrieval": True,
    },

    # Qwen performs well with full DermAgent workflow
    "Qwen2.5-VL-7B-Instruct": {
        "workflow_profile": None,  # Use default inference
        "enable_all_capabilities": True,
    },

    # Llama shows mixed results - use conservative approach
    "Llama-3.2-11B-Vision-Instruct": {
        "workflow_profile": None,  # Use default inference
        "conservative_fusion_weight": 0.7,  # More conservative
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
    model_key = str(model_name).strip()

    # Get model-specific config
    model_config = MODEL_WORKFLOW_PROFILES.get(model_key, {})

    if not model_config:
        # No specific config for this model, return empty overrides
        return {}

    overrides = {}

    # Apply workflow profile override if specified
    if "workflow_profile" in model_config and model_config["workflow_profile"] is not None:
        overrides["workflow_profile"] = model_config["workflow_profile"]

    # Apply workflow capabilities override
    if "workflow_capabilities" in model_config:
        overrides["workflow_capabilities"] = model_config["workflow_capabilities"]

    # Apply specialist skill control
    if model_config.get("skip_specialist_skills"):
        overrides["skip_specialist_skills"] = True

    # Apply experience retrieval control
    if model_config.get("skip_experience_retrieval"):
        overrides["skip_experience_retrieval"] = True

    # Apply conservative fusion weight
    if "conservative_fusion_weight" in model_config:
        overrides["conservative_fusion_weight"] = model_config["conservative_fusion_weight"]

    # Dataset-specific overrides (future extension point)
    if dataset_name:
        dataset_key = str(dataset_name).strip().lower()
        # Example: SkinVL might work better on certain datasets
        if model_key == "SkinVL-MM" and dataset_key in ("ham10000", "isic2019"):
            # Could add dataset-specific tuning here
            pass

    return overrides


def should_use_simplified_workflow(model_name: str) -> bool:
    """
    Check if a model should use simplified workflow (bypass DermAgent features).

    Args:
        model_name: The model identifier

    Returns:
        True if model should use simplified workflow
    """
    model_key = str(model_name).strip()
    model_config = MODEL_WORKFLOW_PROFILES.get(model_key, {})
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
    model_key = str(model_name).strip()
    model_config = MODEL_WORKFLOW_PROFILES.get(model_key, {})

    if not model_config:
        return f"{model_name}: Using default DermAgent workflow"

    if model_config.get("skip_specialist_skills"):
        return f"{model_name}: Using simplified direct prediction workflow (bypassing DermAgent features)"

    if "conservative_fusion_weight" in model_config:
        weight = model_config["conservative_fusion_weight"]
        return f"{model_name}: Using standard workflow with conservative fusion (weight={weight})"

    return f"{model_name}: Using standard DermAgent workflow"
