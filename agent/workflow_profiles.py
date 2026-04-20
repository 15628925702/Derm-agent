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

    context.setdefault("workflow_profile", _infer_workflow_profile(dataset_key=dataset_key, metadata=metadata, label_space_id=label_space_key))
    context.setdefault("workflow_capabilities", _infer_workflow_capabilities(context))
    context.setdefault("label_granularity", _infer_label_granularity(label_space_key))
    context.setdefault("presentation_mode", _infer_presentation_mode(dataset_key=dataset_key, metadata=metadata))
    context.setdefault("available_tests", _infer_available_tests(dataset_key=dataset_key, metadata=metadata))
    context.setdefault("workflow_preference", _infer_workflow_preference(context))
    context.setdefault("hospital_type", "specialist_clinic")
    context.setdefault("time_budget", "standard")
    if "metadata_completeness" not in context:
        context["metadata_completeness"] = _infer_metadata_completeness(metadata)
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
    if has_workflow_capability(workflow_context, "family_routing"):
        return True
    return "grouped" in str(label_space_id or "").strip().lower()


def is_sparse_lesion_case(*, workflow_context: dict[str, Any] | None) -> bool:
    return has_workflow_capability(workflow_context, "sparse_lesion_reasoning")


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


def _infer_workflow_profile(*, dataset_key: str, metadata: dict[str, Any], label_space_id: str) -> str:
    presentation_mode = _infer_presentation_mode(dataset_key=dataset_key, metadata=metadata)
    granularity = _infer_label_granularity(label_space_id)
    if presentation_mode == "diffuse_rash" and granularity == "grouped":
        return "family_routing_workflow"
    if presentation_mode == "focal_lesion":
        return "sparse_lesion_workflow"
    return "default_workflow"


def _infer_label_granularity(label_space_id: str) -> str:
    if "grouped" in label_space_id:
        return "grouped"
    if label_space_id:
        return "fine"
    return "unknown"


def _infer_presentation_mode(*, dataset_key: str, metadata: dict[str, Any]) -> str:
    body_sites = metadata.get("body_sites")
    textures = metadata.get("textures_present")
    related_category = str(metadata.get("related_category", "")).strip().upper()

    if related_category == "RASH":
        return "diffuse_rash"
    if isinstance(body_sites, list) and len(body_sites) > 1:
        return "diffuse_rash"
    if isinstance(textures, list) and "flat" in {str(item).strip().lower() for item in textures} and related_category == "RASH":
        return "diffuse_rash"
    if dataset_key == "ham10000":
        return "focal_lesion"
    return "focal_lesion"


def _infer_available_tests(*, dataset_key: str, metadata: dict[str, Any]) -> list[str]:
    tests = ["clinical_photo_only"]
    if dataset_key in {"pad_ufes_20", "isic2019", "ham10000"}:
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
    if profile == "family_routing_workflow":
        capabilities.extend(["family_routing", "grouped_label_reasoning"])
    if profile == "sparse_lesion_workflow":
        capabilities.extend(["sparse_lesion_reasoning", "focal_lesion_reasoning"])
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
