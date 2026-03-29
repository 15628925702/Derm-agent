from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SkillStats:
    call_count: int = 0
    selected_count: int = 0
    success_count: int = 0
    partially_helpful_count: int = 0
    helpful_count: int = 0
    harmful_count: int = 0
    failure_count: int = 0
    reusable_count: int = 0
    uncertainty_reduction_count: int = 0
    contradiction_detection_count: int = 0
    malignant_flag_support_count: int = 0
    evidence_strength_sum: float = 0.0
    evidence_strength_observation_count: int = 0
    average_evidence_strength: float = 0.0
    helpful_rate: float = 0.0
    failure_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillTrigger:
    condition: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillSchemaField:
    name: str
    field_type: str
    description: str
    allowed_values: list[str] = field(default_factory=list)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillExecutionContext:
    perception: dict[str, Any]
    metadata: dict[str, Any]
    selected_raw_cases: list[dict[str, Any]] = field(default_factory=list)
    selected_tactical_experiences: list[dict[str, Any]] = field(default_factory=list)
    selected_abstract_experiences: list[dict[str, Any]] = field(default_factory=list)
    related_abstract_experiences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompositeWorkflowBlueprint:
    seed_id: str
    future_skill_id: str
    trigger_pattern: dict[str, Any]
    component_skills: list[str]
    required_min_success_count: int = 2
    workflow_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_seed(cls, seed_payload: dict[str, Any]) -> "CompositeWorkflowBlueprint":
        promotion_interface = seed_payload.get("promotion_interface", {})
        return cls(
            seed_id=str(seed_payload.get("seed_id", "")),
            future_skill_id=str(promotion_interface.get("future_skill_id", "")),
            trigger_pattern=dict(seed_payload.get("trigger_pattern", {})),
            component_skills=[str(item) for item in seed_payload.get("skill_sequence", []) if str(item).strip()],
            required_min_success_count=int(promotion_interface.get("required_min_success_count", 2)),
            workflow_contract={
                "builder": promotion_interface.get("builder", "promote_composite_skill_seed"),
                "output_contract_source": promotion_interface.get("output_contract_source", "component_skill_union"),
                "trigger_pattern_source": promotion_interface.get(
                    "trigger_pattern_source", "composite_skill_seed.trigger_pattern"
                ),
            },
        )


@dataclass
class SkillStep:
    step_id: str
    title: str
    instruction: str
    evidence_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillObject:
    skill_id: str
    name: str
    description: str
    skill_type: str
    triggers: list[SkillTrigger]
    input_schema: list[SkillSchemaField]
    workflow_text: str
    steps: list[SkillStep]
    watch_outs: list[str]
    output_schema: list[SkillSchemaField]
    version: str
    source: str
    stats: SkillStats = field(default_factory=SkillStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "input_schema": [schema_field.to_dict() for schema_field in self.input_schema],
            "workflow_text": self.workflow_text,
            "steps": [step.to_dict() for step in self.steps],
            "watch_outs": list(self.watch_outs),
            "output_schema": [schema_field.to_dict() for schema_field in self.output_schema],
            "version": self.version,
            "source": self.source,
            "stats": self.stats.to_dict(),
        }
