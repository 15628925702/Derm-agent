from __future__ import annotations

from typing import Any

from utils.label_canonicalizer import canonicalize_prediction


THREE_CLASS_LABEL_SPACE = ["MEL", "BCC", "NEV"]


def build_conservative_fusion_output(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
    label_space: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    decision = decide_external_conservative_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
        label_space=label_space,
    )
    chosen = dict(agent_output if decision["use_agent_output"] else baseline_output)
    rationale = str(chosen.get("rationale", "")).strip()
    note = build_fusion_note(decision)
    if note:
        rationale = f"{rationale} {note}".strip() if rationale else note
    follow_up = list(chosen.get("follow_up_considerations", []) or [])
    caution_line = build_caution_line(decision)
    if caution_line and caution_line not in follow_up:
        follow_up.append(caution_line)
    chosen["rationale"] = rationale
    chosen["follow_up_considerations"] = follow_up
    chosen["fusion_decision"] = decision
    return chosen


def decide_external_conservative_fusion(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
    label_space: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    policy = dict(evidence_bundle.get("evidence_decision_policy", {}) or {})
    risk_layer = dict(policy.get("risk_layer", {}) or {})
    diagnosis_layer = dict(policy.get("diagnosis_override_layer", {}) or {})

    baseline_label = normalize_prediction_label(baseline_output, label_space=label_space)
    agent_label = normalize_prediction_label(agent_output, label_space=label_space)

    support_margin = safe_float(diagnosis_layer.get("support_margin"))
    subtype_support_margin = safe_float(diagnosis_layer.get("subtype_support_margin"))
    contradiction_count = safe_int(diagnosis_layer.get("contradiction_count"))
    consistent_retrieval_count = safe_int(diagnosis_layer.get("consistent_retrieval_count"))
    uncertainty_level = str(diagnosis_layer.get("uncertainty_level", "")).strip().lower()
    override_mode = str(diagnosis_layer.get("override_mode", "")).strip()
    malignancy_override_allowed = bool(diagnosis_layer.get("malignancy_override_allowed", False))
    subtype_override_allowed = bool(diagnosis_layer.get("subtype_override_allowed", False))
    specialist_support_present = bool(diagnosis_layer.get("specialist_support_present", False))
    opposing_quota_satisfied = bool(diagnosis_layer.get("opposing_quota_satisfied", False))
    subtype_support_quota_satisfied = bool(diagnosis_layer.get("subtype_support_quota_satisfied", False))
    risk_flag = str(risk_layer.get("risk_flag", "")).strip().lower()

    use_agent_output = True
    reasons: list[str] = []

    if not agent_label or not baseline_label:
        use_agent_output = False
        reasons.append("missing_canonical_label")
    elif agent_label == baseline_label:
        use_agent_output = True
        reasons.append("agent_matches_baseline")
    else:
        if not malignancy_override_allowed:
            use_agent_output = False
            reasons.append("malignancy_override_not_allowed")
        elif support_margin < 7.5:
            use_agent_output = False
            reasons.append("support_margin_below_external_threshold")
        elif consistent_retrieval_count < 1:
            use_agent_output = False
            reasons.append("insufficient_consistent_retrieval")
        elif contradiction_count > 1:
            use_agent_output = False
            reasons.append("too_many_contradictions")
        elif uncertainty_level in {"high", "unknown"}:
            use_agent_output = False
            reasons.append("uncertainty_too_high")
        elif not opposing_quota_satisfied:
            use_agent_output = False
            reasons.append("opposing_quota_not_satisfied")
        elif label_space:
            # In aligned 3-class external evaluations, only allow an override
            # when subtype evidence is clearly strong enough.
            if not subtype_override_allowed:
                use_agent_output = False
                reasons.append("subtype_override_not_allowed")
            elif subtype_support_margin < 5.0:
                use_agent_output = False
                reasons.append("subtype_support_margin_below_external_threshold")
            elif not specialist_support_present:
                use_agent_output = False
                reasons.append("missing_specialist_support")
            elif not subtype_support_quota_satisfied:
                use_agent_output = False
                reasons.append("subtype_support_quota_not_satisfied")
        else:
            # For binary external eval, keep the override conservative when the
            # agent is only weakly escalating away from a benign-looking baseline.
            if baseline_label == "benign" and agent_label == "malignant":
                if support_margin < 9.0:
                    use_agent_output = False
                    reasons.append("binary_escalation_margin_too_low")
                elif risk_flag not in {"malignancy_risk_high"}:
                    use_agent_output = False
                    reasons.append("binary_escalation_risk_not_high")

    if use_agent_output:
        reasons.append("use_agent_output")
    else:
        reasons.append("fallback_to_baseline")

    return {
        "use_agent_output": bool(use_agent_output),
        "baseline_label": baseline_label,
        "agent_label": agent_label,
        "override_mode": override_mode,
        "malignancy_override_allowed": malignancy_override_allowed,
        "subtype_override_allowed": subtype_override_allowed,
        "support_margin": support_margin,
        "subtype_support_margin": subtype_support_margin,
        "consistent_retrieval_count": consistent_retrieval_count,
        "contradiction_count": contradiction_count,
        "uncertainty_level": uncertainty_level,
        "risk_flag": risk_flag,
        "reasons": reasons,
    }


def normalize_prediction_label(output: dict[str, Any], *, label_space: list[str] | tuple[str, ...] | None) -> str:
    final_text = str(output.get("final_diagnosis", "")).strip()
    if label_space:
        return str(canonicalize_prediction(final_text, label_space).get("canonical_label", "")).strip()
    lowered = final_text.lower()
    malignant_terms = (
        "melanoma",
        "mel",
        "basal cell",
        "bcc",
        "actinic keratos",
        "ack",
        "squamous cell",
        "scc",
        "malignant lesion",
    )
    benign_terms = (
        "nevus",
        "naevus",
        "mole",
        "seborrheic keratos",
        "bkl",
        "dermatofibroma",
        "df",
        "vascular",
        "vasc",
        "angioma",
        "papilloma",
        "benign",
    )
    if any(token in lowered for token in malignant_terms):
        return "malignant"
    if any(token in lowered for token in benign_terms):
        return "benign"
    return "unknown"


def build_fusion_note(decision: dict[str, Any]) -> str:
    mode = "agent-guided external fusion"
    if decision.get("use_agent_output"):
        return f"{mode}: kept the agent override."
    return f"{mode}: evidence was not strong enough to override the direct baseline, so the baseline diagnosis was preserved."


def build_caution_line(decision: dict[str, Any]) -> str:
    reasons = [str(item).strip() for item in decision.get("reasons", []) if str(item).strip()]
    if not reasons:
        return ""
    return "Fusion note: " + "; ".join(reasons[:4])


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0
