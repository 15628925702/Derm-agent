from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LEAKY_METADATA_KEYS = {
    "diagnostic",
    "label",
    "reference_label",
    "true_label",
    "final_decision",
    "patient_id",
    "lesion_id",
    "img_id",
    "image_id",
    "biopsed",
}


@dataclass
class CaseInput:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    label: str | None = None
    reference_label: str | None = None
    dataset_name: str | None = None
    source_metadata_path: str | None = None

    def __post_init__(self) -> None:
        if self.label is None and self.reference_label is not None:
            self.label = self.reference_label
        if self.reference_label is None and self.label is not None:
            self.reference_label = self.label

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def clinical_metadata(self) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in self.metadata.items()
            if str(key).strip().lower() not in LEAKY_METADATA_KEYS
        }


@dataclass
class CaseState:
    case_input: CaseInput
    perception: dict[str, Any] = field(default_factory=dict)
    retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    skill_retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    retrieved_experience: list[dict[str, Any]] = field(default_factory=list)
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    planner_output: dict[str, Any] = field(default_factory=dict)
    skill_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    final_diagnosis: dict[str, Any] = field(default_factory=dict)
    reflection: dict[str, Any] = field(default_factory=dict)
    execution_record: dict[str, Any] = field(default_factory=dict)

    @property
    def image_path(self) -> Path:
        return Path(self.case_input.image_path)

    @property
    def clinical_metadata(self) -> dict[str, Any]:
        return self.case_input.clinical_metadata()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["case_input"]["image_exists"] = self.image_path.exists()
        payload["case_input"]["clinical_metadata"] = self.clinical_metadata
        return payload
