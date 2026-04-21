from __future__ import annotations

from copy import deepcopy
from typing import Any

KERATINOCYTE_FAMILY = {
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Seborrheic Keratosis",
}

KERATINOCYTE_MALIGNANT_LABELS = {
    "actinic keratosis",
    "ack",
    "akiec",
    "basal cell carcinoma",
    "bcc",
    "squamous cell carcinoma",
    "scc",
    "bowen disease",
}


def apply_conservative_agent_fusion(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    decision = decide_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )
    if str(decision.get("fusion_mode", "")).strip().lower() == "off":
        chosen = dict(agent_output)
        chosen["fusion_decision"] = decision
        return chosen
    if decision.get("consensus_override_label"):
        chosen = dict(agent_output)
        chosen["final_diagnosis"] = str(decision["consensus_override_label"]).strip()
    else:
        chosen = dict(agent_output if decision["use_agent_output"] else baseline_output)
    if decision.get("use_agent_output") and decision.get("merge_baseline_differentials"):
        chosen["differential_diagnoses"] = _merge_differentials(
            primary=list(chosen.get("differential_diagnoses", []) or []),
            baseline=list(baseline_output.get("differential_diagnoses", []) or []),
            final_label=str(chosen.get("final_diagnosis", "")).strip(),
        )
    rationale = str(chosen.get("rationale", "")).strip()
    note = _build_fusion_note(decision)
    if note:
        rationale = f"{rationale} {note}".strip() if rationale else note
    follow_up = list(chosen.get("follow_up_considerations", []) or [])
    caution_line = _build_caution_line(decision)
    if caution_line and caution_line not in follow_up:
        follow_up.append(caution_line)
    chosen["rationale"] = rationale
    chosen["follow_up_considerations"] = follow_up
    chosen["fusion_decision"] = decision
    return chosen


def decide_conservative_agent_fusion(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(evidence_bundle.get("evidence_decision_policy", {}) or {})
    risk_layer = dict(policy.get("risk_layer", {}) or {})
    diagnosis_layer = dict(policy.get("diagnosis_override_layer", {}) or {})
    evidence_policy = dict(evidence_bundle.get("evidence_calibration_debug", {}).get("policy", {}) or {})

    baseline_label = str(baseline_output.get("final_diagnosis", "")).strip()
    agent_label = str(agent_output.get("final_diagnosis", "")).strip()
    baseline_confidence = str(baseline_output.get("confidence", "")).strip().lower()
    agent_confidence = str(agent_output.get("confidence", "")).strip().lower()
    fusion_mode = str(evidence_policy.get("conservative_fusion_mode", "soft")).strip().lower() or "soft"

    selected_evidence_present = bool(diagnosis_layer.get("selected_evidence_present", False))
    workflow_context = dict(diagnosis_layer.get("workflow_context", {}) or {})
    override_allowed = bool(diagnosis_layer.get("override_allowed", False))
    malignancy_override_allowed = bool(diagnosis_layer.get("malignancy_override_allowed", False))
    subtype_override_allowed = bool(diagnosis_layer.get("subtype_override_allowed", False))
    family_override_allowed = bool(diagnosis_layer.get("family_override_allowed", False))
    support_margin = _safe_float(diagnosis_layer.get("support_margin"))
    subtype_support_margin = _safe_float(diagnosis_layer.get("subtype_support_margin"))
    uncertainty_level = str(diagnosis_layer.get("uncertainty_level", "")).strip().lower()
    override_mode = str(diagnosis_layer.get("override_mode", "")).strip().lower()
    baseline_preview = dict(risk_layer.get("baseline_preview", {})) if isinstance(risk_layer.get("baseline_preview", {}), dict) else {}
    baseline_differentials = [str(item).strip() for item in baseline_output.get("differential_diagnoses", []) if str(item).strip()]
    initial_ddx = [str(item).strip() for item in baseline_preview.get("early_ddx_candidates", []) if str(item).strip()]
    consensus_candidates = _consensus_candidates(
        baseline_label=baseline_label,
        baseline_differentials=baseline_differentials,
        initial_ddx=initial_ddx,
        agent_label=agent_label,
    )

    use_agent_output = True
    reasons: list[str] = []
    merge_baseline_differentials = False
    consensus_override_label = ""

    if fusion_mode == "off":
        use_agent_output = True
        merge_baseline_differentials = False
        reasons.append("fusion_mode_off_passthrough")
    elif fusion_mode == "soft":
        if not agent_label or not baseline_label:
            use_agent_output = False
            reasons.append("missing_final_diagnosis")
        elif agent_label == baseline_label:
            use_agent_output = True
            reasons.append("agent_matches_baseline")
            merge_baseline_differentials = True
        elif (
            selected_evidence_present
            and support_margin >= 12.0
            and agent_label in consensus_candidates
            and agent_confidence in {"medium", "high"}
        ):
            use_agent_output = True
            merge_baseline_differentials = True
            reasons.append("soft_mode_high_support_consensus_override")
        elif (
            not selected_evidence_present
            and _allow_clinical_bcc_consensus_override(
                workflow_context=workflow_context,
                baseline_label=baseline_label,
                agent_label=agent_label,
                consensus_candidates=consensus_candidates,
                agent_confidence=agent_confidence,
            )
        ):
            consensus_override_label = "Basal Cell Carcinoma"
            use_agent_output = True
            merge_baseline_differentials = True
            reasons.append("clinical_bcc_consensus_override_without_selected_evidence")
        elif (
            not selected_evidence_present
            and _allow_image_archive_consensus_override(
                workflow_context=workflow_context,
                baseline_label=baseline_label,
                agent_label=agent_label,
                consensus_candidates=consensus_candidates,
                initial_ddx=initial_ddx,
                agent_confidence=agent_confidence,
            )
        ):
            use_agent_output = True
            merge_baseline_differentials = True
            reasons.append("image_archive_consensus_override_without_selected_evidence")
        elif not selected_evidence_present:
            use_agent_output = False
            reasons.append("no_selected_evidence")
        elif _allow_sparse_lesion_safe_override(
            workflow_context=workflow_context,
            baseline_label=baseline_label,
            agent_label=agent_label,
            initial_ddx=initial_ddx,
            selected_evidence_present=selected_evidence_present,
            subtype_support_margin=subtype_support_margin,
            uncertainty_level=uncertainty_level,
            agent_confidence=agent_confidence,
        ):
            use_agent_output = True
            merge_baseline_differentials = True
            reasons.append("sparse_lesion_safe_override")
        elif family_override_allowed and override_mode == "family_override":
            use_agent_output = True
            merge_baseline_differentials = True
            reasons.append("family_override_allowed")
        elif _allow_keratinocyte_subtype_override(
            baseline_label=baseline_label,
            agent_label=agent_label,
            selected_evidence_present=selected_evidence_present,
            subtype_support_margin=subtype_support_margin,
            uncertainty_level=uncertainty_level,
            agent_confidence=agent_confidence,
            evidence_policy=evidence_policy,
        ):
            use_agent_output = True
            merge_baseline_differentials = True
            reasons.append("keratinocyte_subtype_override_allowed")
        elif not malignancy_override_allowed:
            use_agent_output = False
            reasons.append("malignancy_override_not_allowed")
        elif override_mode == "risk_only" and not subtype_override_allowed:
            use_agent_output = False
            reasons.append("risk_only_mode_without_subtype_override")
        elif uncertainty_level in {"high", "unknown"} and support_margin < 7.0:
            use_agent_output = False
            reasons.append("uncertainty_too_high_for_label_change")
        elif subtype_support_margin < 3.0 and baseline_confidence in {"medium", "high"}:
            use_agent_output = False
            reasons.append("subtype_support_margin_too_low")
        elif agent_confidence == "low" and baseline_confidence in {"medium", "high"}:
            use_agent_output = False
            reasons.append("agent_confidence_below_baseline")
        else:
            merge_baseline_differentials = True
    else:
        # hard
        if not agent_label or not baseline_label:
            use_agent_output = False
            reasons.append("missing_final_diagnosis")
        elif agent_label == baseline_label:
            if (
                selected_evidence_present
                and consensus_candidates
                and baseline_label not in consensus_candidates
                and _all_in_family(consensus_candidates, KERATINOCYTE_FAMILY)
                and support_margin >= 5.5
                and uncertainty_level not in {"high", "unknown"}
            ):
                consensus_override_label = consensus_candidates[0]
                use_agent_output = True
                merge_baseline_differentials = True
                reasons.append("hard_mode_keratinocyte_consensus_override")
            else:
                use_agent_output = True
                reasons.append("agent_matches_baseline")
        elif not selected_evidence_present:
            use_agent_output = False
            reasons.append("no_selected_evidence")
        elif not malignancy_override_allowed:
            use_agent_output = False
            reasons.append("malignancy_override_not_allowed")
        elif override_mode == "risk_only" and not subtype_override_allowed:
            use_agent_output = False
            reasons.append("risk_only_mode_without_subtype_override")
        elif uncertainty_level in {"high", "unknown"} and support_margin < 8.0:
            use_agent_output = False
            reasons.append("uncertainty_too_high_for_label_change")
        elif subtype_support_margin < 4.0 and baseline_confidence in {"medium", "high"}:
            use_agent_output = False
            reasons.append("subtype_support_margin_too_low")
        elif agent_confidence == "low" and baseline_confidence in {"medium", "high"}:
            use_agent_output = False
            reasons.append("agent_confidence_below_baseline")

    if use_agent_output:
        reasons.append("use_agent_output")
    else:
        reasons.append("fallback_to_baseline")

    return {
        "use_agent_output": bool(use_agent_output),
        "fusion_mode": fusion_mode,
        "merge_baseline_differentials": bool(merge_baseline_differentials),
        "consensus_override_label": consensus_override_label,
        "consensus_candidates": list(consensus_candidates),
        "baseline_label": baseline_label,
        "agent_label": agent_label,
        "baseline_confidence": baseline_confidence,
        "agent_confidence": agent_confidence,
        "selected_evidence_present": selected_evidence_present,
        "override_allowed": override_allowed,
        "malignancy_override_allowed": malignancy_override_allowed,
        "subtype_override_allowed": subtype_override_allowed,
        "family_override_allowed": family_override_allowed,
        "override_mode": override_mode,
        "support_margin": support_margin,
        "subtype_support_margin": subtype_support_margin,
        "uncertainty_level": uncertainty_level,
        "workflow_profile": str(workflow_context.get("workflow_profile", "")).strip(),
        "baseline_preview": deepcopy(baseline_preview),
        "reasons": reasons,
    }


def _build_fusion_note(decision: dict[str, Any]) -> str:
    mode = "agent conservative fusion"
    if decision.get("use_agent_output"):
        return f"{mode}: kept the agent diagnosis."
    return f"{mode}: evidence was not strong enough to override the baseline diagnosis, so the baseline label was preserved."


def _build_caution_line(decision: dict[str, Any]) -> str:
    reasons = [str(item).strip() for item in decision.get("reasons", []) if str(item).strip()]
    if not reasons:
        return ""
    return "Fusion note: " + "; ".join(reasons[:4])


def _merge_differentials(*, primary: list[Any], baseline: list[Any], final_label: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    ordered = [final_label] + [str(item).strip() for item in primary] + [str(item).strip() for item in baseline]
    for item in ordered:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged[:5]


def _all_in_family(values: list[str], family: set[str]) -> bool:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if not normalized:
        return False
    return all(item in family for item in normalized)


def _is_keratinocyte_malignant_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    if not normalized:
        return False
    return normalized in KERATINOCYTE_MALIGNANT_LABELS


def _allow_keratinocyte_subtype_override(
    *,
    baseline_label: str,
    agent_label: str,
    selected_evidence_present: bool,
    subtype_support_margin: float,
    uncertainty_level: str,
    agent_confidence: str,
    evidence_policy: dict[str, Any],
) -> bool:
    if not bool(evidence_policy.get("allow_keratinocyte_subtype_override", False)):
        return False
    if not selected_evidence_present:
        return False
    if uncertainty_level in {"high", "unknown"}:
        return False
    if agent_confidence not in {"moderate", "medium", "high"}:
        return False
    if subtype_support_margin < 3.0:
        return False
    if baseline_label == agent_label:
        return False
    return _is_keratinocyte_malignant_label(baseline_label) and _is_keratinocyte_malignant_label(agent_label)


def _allow_clinical_bcc_consensus_override(
    *,
    workflow_context: dict[str, Any],
    baseline_label: str,
    agent_label: str,
    consensus_candidates: list[str],
    agent_confidence: str,
) -> bool:
    capabilities = {
        str(item).strip().lower()
        for item in (workflow_context.get("workflow_capabilities") or [])
        if str(item).strip()
    }
    if "clinical_metadata_reasoning" not in capabilities:
        return False
    if agent_confidence not in {"moderate", "medium", "high"}:
        return False
    if str(agent_label).strip().lower() != "basal cell carcinoma":
        return False
    if str(baseline_label).strip().lower() == "basal cell carcinoma":
        return False
    return any(str(item).strip().lower() == "basal cell carcinoma" for item in consensus_candidates + [agent_label])


def _allow_image_archive_consensus_override(
    *,
    workflow_context: dict[str, Any],
    baseline_label: str,
    agent_label: str,
    consensus_candidates: list[str],
    initial_ddx: list[str],
    agent_confidence: str,
) -> bool:
    capabilities = {
        str(item).strip().lower()
        for item in (workflow_context.get("workflow_capabilities") or [])
        if str(item).strip()
    }
    if "image_archive_reasoning" not in capabilities:
        return False
    if agent_confidence not in {"moderate", "medium", "high"}:
        return False
    normalized_agent = str(agent_label).strip().lower()
    normalized_baseline = str(baseline_label).strip().lower()
    normalized_ddx = " | ".join(str(item).strip().lower() for item in initial_ddx if str(item).strip())
    normalized_consensus = {str(item).strip().lower() for item in consensus_candidates}
    if not initial_ddx:
        return False

    if normalized_agent == "basal cell carcinoma":
        if normalized_agent not in normalized_consensus:
            return False
        return "basal cell carcinoma" in normalized_ddx and normalized_baseline in {
            "actinic keratosis",
            "seborrheic keratosis",
            "nevus",
        }

    if normalized_agent == "malignant melanoma":
        melanoma_subtype_signal = any(
            term in normalized_ddx
            for term in (
                "superficial spreading melanoma",
                "lentigo maligna",
                "acral melanoma",
                "nodular melanoma",
            )
        )
        if not melanoma_subtype_signal:
            return False
        # Keep this path narrow: only permit upgrade from nevus-like baselines.
        # We explicitly do not flip BCC/SCC-style baselines to melanoma without
        # selected evidence, because that caused recent regressions.
        return normalized_baseline in {
            "nevus",
            "atypical nevus",
            "seborrheic keratosis",
        }

    return False


def _allow_sparse_lesion_safe_override(
    *,
    workflow_context: dict[str, Any],
    baseline_label: str,
    agent_label: str,
    initial_ddx: list[str],
    selected_evidence_present: bool,
    subtype_support_margin: float,
    uncertainty_level: str,
    agent_confidence: str,
) -> bool:
    capabilities = {
        str(item).strip().lower()
        for item in (workflow_context.get("workflow_capabilities") or [])
        if str(item).strip()
    }
    if "sparse_lesion_reasoning" not in capabilities:
        return False
    if not selected_evidence_present:
        return False
    if uncertainty_level in {"high", "unknown"}:
        return False
    if agent_confidence not in {"moderate", "medium", "high"}:
        return False
    if subtype_support_margin < 3.0:
        return False

    baseline_norm = str(baseline_label).strip().lower()
    agent_norm = str(agent_label).strip().lower()
    ddx_text = " | ".join(str(item).strip().lower() for item in initial_ddx if str(item).strip())

    if baseline_norm != "basal cell carcinoma":
        return False

    if agent_norm == "actinic keratosis":
        return "actinic keratosis" in ddx_text or "actinic" in ddx_text or "ack" in ddx_text

    if agent_norm == "nevus":
        return "nevus" in ddx_text or "atypical nevus" in ddx_text or "melanocytic nevus" in ddx_text

    return False


def _consensus_candidates(
    *,
    baseline_label: str,
    baseline_differentials: list[str],
    initial_ddx: list[str],
    agent_label: str,
) -> list[str]:
    normalized_baseline = [str(item).strip() for item in baseline_differentials if str(item).strip()]
    normalized_ddx = [str(item).strip() for item in initial_ddx if str(item).strip()]
    baseline_set = {item.lower(): item for item in normalized_baseline}
    ddx_set = {item.lower(): item for item in normalized_ddx}

    shared_keys = [key for key in baseline_set.keys() if key in ddx_set]
    if not shared_keys:
        return []

    ordered: list[str] = []
    preferred = [agent_label] + normalized_ddx + normalized_baseline + [baseline_label]
    for item in preferred:
        lowered = str(item).strip().lower()
        if lowered in shared_keys:
            canonical = baseline_set.get(lowered) or ddx_set.get(lowered) or str(item).strip()
            if canonical and canonical not in ordered:
                ordered.append(canonical)
    return ordered


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0
