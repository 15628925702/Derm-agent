from __future__ import annotations

from typing import Any

from memory.fusion_experience.workflow_fusion_decision import (
    apply_conservative_agent_fusion as _apply_conservative_agent_fusion,
    decide_conservative_agent_fusion as _decide_conservative_agent_fusion,
)


def apply_conservative_agent_fusion(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    return _apply_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )


def decide_conservative_agent_fusion(
    *,
    baseline_output: dict[str, Any],
    agent_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    return _decide_conservative_agent_fusion(
        baseline_output=baseline_output,
        agent_output=agent_output,
        evidence_bundle=evidence_bundle,
    )
