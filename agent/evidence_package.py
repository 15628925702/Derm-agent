from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.aggregator import build_evidence_bundle
from agent.state import CaseState


@dataclass
class EvidencePackage:
    perception: dict[str, Any] = field(default_factory=dict)
    retrieved_experience: list[dict[str, Any]] = field(default_factory=list)
    skill_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    initial_perception_summary: dict[str, Any] = field(default_factory=dict)
    retrieved_raw_cases_summary: list[dict[str, Any]] = field(default_factory=list)
    retrieved_tactical_experiences_summary: list[dict[str, Any]] = field(default_factory=list)
    retrieved_abstract_experiences_summary: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_summary: dict[str, Any] = field(default_factory=dict)
    contradiction_summary: dict[str, Any] = field(default_factory=dict)
    information_gap_summary: dict[str, Any] = field(default_factory=dict)
    escalation_summary: dict[str, Any] = field(default_factory=dict)
    planner_rationale: dict[str, Any] = field(default_factory=dict)
    confusion_cluster_summary: dict[str, Any] = field(default_factory=dict)
    selected_evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_decision_policy: dict[str, Any] = field(default_factory=dict)
    serialized_evidence_text: str = ""
    evidence_calibration_debug: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: CaseState) -> "EvidencePackage":
        return cls(**build_evidence_bundle(state))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
