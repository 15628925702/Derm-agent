from __future__ import annotations

from typing import Any


def ensure_workflow_context(
    *,
    dataset_name: str | None,
    metadata: dict[str, Any] | None,
    label_space_id: str | None,
    workflow_context: dict[str, Any] | None,
) -> dict[str, Any]:
    context = dict(workflow_context or {})
    metadata = dict(metadata or {})
    dataset_key = str(dataset_name or "").strip().lower()
    label_space_key = str(label_space_id or metadata.get("label_space_id", "")).strip().lower()

    context.setdefault("label_granularity", _infer_label_granularity(label_space_key))
    context.setdefault("presentation_mode", _infer_presentation_mode(metadata=metadata))
    context.setdefault("workflow_profile", _infer_workflow_profile(metadata=metadata, label_space_id=label_space_key))
    context.setdefault("available_tests", _infer_available_tests(metadata=metadata, presentation_mode=context.get("presentation_mode")))
    context.setdefault("workflow_preference", _infer_workflow_preference(context))
    context.setdefault("hospital_type", "specialist_clinic")
    context.setdefault("time_budget", "standard")
    if "metadata_completeness" not in context:
        context["metadata_completeness"] = _infer_metadata_completeness(metadata)
    context.setdefault("workflow_capabilities", _infer_workflow_capabilities(context))
    return context


def workflow_profile_is(context: dict[str, Any] | None, profile_name: str) -> bool:
    if not context:
        return False
    return str(context.get("workflow_profile", "")).strip().lower() == str(profile_name).strip().lower()


def has_workflow_capability(context: dict[str, Any] | None, capability_name: str) -> bool:
    if not context:
        return False
    capabilities = {
        str(item).strip().lower()
        for item in (context.get("workflow_capabilities") or [])
        if str(item).strip()
    }
    return str(capability_name).strip().lower() in capabilities


def is_family_routing_case(*, workflow_context: dict[str, Any] | None, label_space_id: str | None = None) -> bool:
    return has_workflow_capability(workflow_context, "family_routing")


def is_sparse_lesion_case(*, workflow_context: dict[str, Any] | None) -> bool:
    return has_workflow_capability(workflow_context, "sparse_lesion_reasoning")


def is_coarse_taxonomy_case(*, workflow_context: dict[str, Any] | None) -> bool:
    return has_workflow_capability(workflow_context, "coarse_taxonomy_reasoning")


def uses_legacy_agent_final_path(workflow_context: dict[str, Any] | None) -> bool:
    if workflow_context and bool(workflow_context.get("disable_legacy_final_path", False)):
        return False
    return has_workflow_capability(workflow_context, "legacy_agent_final_reasoning")


def is_full_taxonomy_case(*, workflow_context: dict[str, Any] | None, label_space_id: str | None = None) -> bool:
    label_space_key = str(label_space_id or "").strip().lower()
    if "full" in label_space_key:
        return True
    return not has_workflow_capability(workflow_context, "grouped_label_reasoning")


def get_workflow_specialist_skills(workflow_context: dict[str, Any] | None) -> set[str]:
    if has_workflow_capability(workflow_context, "family_routing"):
        return set()
    if has_workflow_capability(workflow_context, "sparse_lesion_reasoning"):
        return {"mel_nev_specialist_skill", "benign_mimic_specialist_skill", "ack_scc_specialist_skill"}
    return {"mel_nev_specialist_skill", "ack_scc_specialist_skill", "benign_mimic_specialist_skill"}


def _infer_workflow_profile(*, metadata: dict[str, Any], label_space_id: str) -> str:
    presentation_mode = _infer_presentation_mode(metadata=metadata)
    granularity = _infer_label_granularity(label_space_id)
    case_source = str(metadata.get("case_source", "")).strip().lower()
    if case_source == "xiangya_sft":
        return "eczematous_family_routing_workflow"
    if presentation_mode == "diffuse_rash" and granularity == "grouped":
        return "family_routing_workflow"
    if granularity == "grouped":
        return "coarse_taxonomy_workflow"
    if _has_sparse_lesion_signature(metadata=metadata, label_space_id=label_space_id):
        return "sparse_lesion_workflow"
    if _has_clinical_lesion_metadata(metadata):
        return "clinical_full_taxonomy_lesion_workflow"
    if _has_image_archive_metadata(metadata):
        return "image_archive_full_taxonomy_lesion_workflow"
    if presentation_mode == "focal_lesion":
        return "full_taxonomy_lesion_workflow"
    return "default_workflow"


def _infer_label_granularity(label_space_id: str) -> str:
    if "grouped" in label_space_id:
        return "grouped"
    if label_space_id:
        return "fine"
    return "unknown"


def _infer_presentation_mode(*, metadata: dict[str, Any]) -> str:
    body_sites = metadata.get("body_sites")
    textures = metadata.get("textures_present")
    related_category = str(metadata.get("related_category", "")).strip().upper()

    if related_category == "RASH":
        return "diffuse_rash"
    if isinstance(body_sites, list) and len(body_sites) > 1:
        return "diffuse_rash"
    if isinstance(textures, list) and "flat" in {str(item).strip().lower() for item in textures} and related_category == "RASH":
        return "diffuse_rash"
    return "focal_lesion"


def _infer_available_tests(*, metadata: dict[str, Any], presentation_mode: Any = "") -> list[str]:
    tests = ["clinical_photo_only"]
    if str(presentation_mode or "").strip().lower() == "focal_lesion":
        tests.append("lesion_photo")
    if metadata.get("has_histopathology"):
        tests.append("histopathology_reference_hidden")
    return tests


def _infer_workflow_preference(context: dict[str, Any]) -> str:
    presentation_mode = str(context.get("presentation_mode", "")).strip().lower()
    if presentation_mode == "diffuse_rash":
        return "metadata_first"
    return "morphology_first"


def _infer_workflow_capabilities(context: dict[str, Any]) -> list[str]:
    profile = str(context.get("workflow_profile", "")).strip().lower()
    presentation_mode = str(context.get("presentation_mode", "")).strip().lower()
    label_granularity = str(context.get("label_granularity", "")).strip().lower()

    capabilities: list[str] = []
    if profile == "eczematous_family_routing_workflow":
        capabilities.extend(
            [
                "family_routing",
                "grouped_label_reasoning",
                "rash_reasoning",
                "eczematous_family_reasoning",
            ]
        )
    if profile == "family_routing_workflow":
        capabilities.extend(["family_routing", "grouped_label_reasoning"])
    if profile == "coarse_taxonomy_workflow":
        capabilities.extend(["coarse_taxonomy_reasoning", "grouped_label_reasoning"])
    if profile == "sparse_lesion_workflow":
        capabilities.extend(["sparse_lesion_reasoning", "focal_lesion_reasoning"])
    if profile == "clinical_full_taxonomy_lesion_workflow":
        capabilities.extend(["clinical_metadata_reasoning", "full_taxonomy_reasoning", "focal_lesion_reasoning", "legacy_agent_final_reasoning"])
    if profile == "image_archive_full_taxonomy_lesion_workflow":
        capabilities.extend(["image_archive_reasoning", "full_taxonomy_reasoning", "focal_lesion_reasoning", "legacy_agent_final_reasoning"])
    if profile == "full_taxonomy_lesion_workflow":
        capabilities.extend(["full_taxonomy_reasoning", "focal_lesion_reasoning"])
    if presentation_mode == "diffuse_rash":
        capabilities.append("rash_reasoning")
    if presentation_mode == "focal_lesion":
        capabilities.append("focal_lesion_reasoning")
    if label_granularity == "grouped":
        capabilities.append("grouped_label_reasoning")
    return list(dict.fromkeys(capabilities))


def _infer_metadata_completeness(metadata: dict[str, Any]) -> str:
    visible_values = 0
    for value in metadata.values():
        if value in (None, "", [], {}, ()):
            continue
        visible_values += 1
    if visible_values <= 2:
        return "minimal"
    if visible_values <= 8:
        return "partial"
    return "full"


def _has_sparse_lesion_signature(*, metadata: dict[str, Any], label_space_id: str) -> bool:
    label_space_key = str(label_space_id or metadata.get("label_space_id", "")).strip().lower()
    if label_space_key == "ham10000_full":
        return True
    return any(key in metadata for key in ("diagnosis_confidence", "has_histopathology", "localization"))


def _has_clinical_lesion_metadata(metadata: dict[str, Any]) -> bool:
    clinical_fields = {"grew", "changed", "bleed", "itch", "hurt", "diameter_1", "diameter_2", "elevation"}
    return bool(clinical_fields.intersection(metadata.keys()))


def _has_image_archive_metadata(metadata: dict[str, Any]) -> bool:
    archive_fields = {"anatom_site_general", "age_approx", "image"}
    return bool(archive_fields.intersection(metadata.keys()))
