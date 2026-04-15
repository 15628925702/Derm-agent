from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.label_space import is_malignant_label as _shared_is_malignant_label


DEFAULT_EXPERIENCE_ROOT = Path("/root/DermAgent/state/experience")
LEGACY_EXPERIENCE_JSON_PATH = Path("/root/DermAgent/state/experience_bank.json")
LEGACY_EXPERIENCE_JSONL_PATH = Path("/root/DermAgent/state/experience_bank.jsonl")


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def dedupe_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def is_malignant_label(
    label: str | None,
    *,
    label_space_id: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool | None:
    return _shared_is_malignant_label(
        label,
        label_space_id=label_space_id,
        dataset_name=dataset_name,
        metadata=metadata,
    )


@dataclass
class ExperienceRecord:
    case_id: str
    experience_type: str
    perception_summary: str
    skills_used: list[str]
    key_evidence: dict[str, Any]
    error_type: str
    confusion_pair: str | None
    learning_points: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawCaseMemory:
    case_id: str
    image_paths: list[str]
    metadata: dict[str, Any]
    dataset_name: str | None
    label_space_id: str | None
    true_label: str | None
    qwen_output: dict[str, Any]
    agent_output: dict[str, Any]
    final_decision: dict[str, Any]
    correctness: dict[str, Any]
    malignant_flag: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TacticalExperience:
    exp_id: str
    case_id: str
    condition: dict[str, Any]
    action: dict[str, Any]
    rationale: list[str]
    step_trace: list[str]
    outcome: dict[str, Any]
    reusable_scope: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompositeSkillSeed:
    seed_id: str
    trigger_pattern: dict[str, Any]
    skill_sequence: list[str]
    supporting_cases: list[str]
    success_count: int
    notes: list[str]
    promotion_interface: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AbstractExperience:
    abs_id: str
    type: str
    concept: str
    pattern_summary: dict[str, Any]
    supporting_cases: list[str]
    counter_cases: list[str]
    derived_rule: dict[str, Any]
    version: str
    seed_id: str | None = None
    composite_skill_seed: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
