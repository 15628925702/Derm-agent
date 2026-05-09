from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_paths import repo_root

FUSION_EXPERIENCE_ACCUMULATION_ENV = "DERMAGENT_ENABLE_FUSION_EXPERIENCE_ACCUMULATION"
FUSION_EXPERIENCE_PROPOSAL_PATH_ENV = "DERMAGENT_FUSION_EXPERIENCE_PROPOSAL_PATH"


def is_fusion_experience_accumulation_enabled() -> bool:
    """Return True only when fusion experience proposal capture is explicitly enabled."""
    return os.environ.get(FUSION_EXPERIENCE_ACCUMULATION_ENV, "").strip() == "1"


def maybe_record_fusion_experience_observation(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    """Append a pending fusion-experience observation when the opt-in switch is on.

    This hook is intentionally inert by default. It does not approve, import, or
    execute generated experience. Reviewers can inspect the JSONL observations
    and turn them into a tested code patch later.
    """
    if not is_fusion_experience_accumulation_enabled():
        return

    proposal_path = _proposal_path()
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal = _build_pending_observation(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
        decision=decision,
    )
    with proposal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(proposal, ensure_ascii=False, sort_keys=True) + "\n")


def _proposal_path() -> Path:
    configured = os.environ.get(FUSION_EXPERIENCE_PROPOSAL_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root() / "memory" / "fusion_experience" / "proposals" / "pending_fusion_experience.jsonl"


def _build_pending_observation(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    diagnosis_layer = dict(
        (evidence_bundle.get("evidence_decision_policy", {}) or {}).get("diagnosis_override_layer", {}) or {}
    )
    workflow_context = dict(diagnosis_layer.get("workflow_context", {}) or {})
    selected_evidence = list(evidence_bundle.get("selected_evidence", []) or [])
    return {
        "schema_version": "fusion_experience_observation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "pending_human_review",
        "runtime_effect": "none",
        "accumulation_switch": FUSION_EXPERIENCE_ACCUMULATION_ENV,
        "workflow_cell_id": str(workflow_context.get("workflow_cell_id", "")).strip(),
        "dataset_name": str(workflow_context.get("dataset_name", "")).strip(),
        "label_space_id": str(workflow_context.get("label_space_id", "")).strip(),
        "baseline_final_diagnosis": str(baseline_output.get("final_diagnosis", "")).strip(),
        "agent_final_diagnosis": str(agent_output.get("final_diagnosis", "")).strip(),
        "fusion_decision": {
            "use_agent_output": bool(decision.get("use_agent_output", False)),
            "consensus_override_label": str(decision.get("consensus_override_label", "")).strip(),
            "reasons": list(decision.get("reasons", []) or []),
        },
        "support_snapshot": {
            "selected_evidence_present": bool(diagnosis_layer.get("selected_evidence_present", False)),
            "support_margin": diagnosis_layer.get("support_margin"),
            "subtype_support_margin": diagnosis_layer.get("subtype_support_margin"),
            "uncertainty_level": str(diagnosis_layer.get("uncertainty_level", "")).strip(),
            "contradiction_count": diagnosis_layer.get("contradiction_count"),
        },
        "selected_evidence_summaries": [
            {
                "source_name": str(item.get("source_name", "")).strip(),
                "summary": str(item.get("summary", "")).strip()[:800],
            }
            for item in selected_evidence
            if isinstance(item, dict)
        ],
        "human_review_note": (
            "This is only an observation/proposal seed. It must not affect runtime "
            "until a reviewer turns it into a tested, workflow_cell_id-bound rule."
        ),
        "suggested_patch": "",
    }
