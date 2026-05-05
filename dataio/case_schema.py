from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FieldCandidate:
    field: str
    score: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseSourceConfig:
    data_root: Path
    metadata_path: Path
    image_root: Path
    image_field: str | None
    label_field: str | None
    case_id_fields: list[str] = field(default_factory=list)
    metadata_format: str = "csv"
    image_field_candidates: list[FieldCandidate] = field(default_factory=list)
    label_field_candidates: list[FieldCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["metadata_path"] = str(self.metadata_path)
        payload["image_root"] = str(self.image_root)
        payload["image_field_candidates"] = [candidate.to_dict() for candidate in self.image_field_candidates]
        payload["label_field_candidates"] = [candidate.to_dict() for candidate in self.label_field_candidates]
        return payload


@dataclass
class StandardizedCaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
