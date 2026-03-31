from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MALIGNANT_LABELS = {"mel", "bcc", "akiec"}
BENIGN_LABELS = {"nv", "bkl", "df", "vasc"}


def binary_label_for_ham10000(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in MALIGNANT_LABELS:
        return "malignant"
    if normalized in BENIGN_LABELS:
        return "benign"
    return "unknown"


@dataclass
class Ham10000CaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    original_label: str
    binary_label: str
    dataset_name: str = "HAM10000"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ham10000DatasetSummary:
    data_root: str
    metadata_csv: str
    image_dirs: list[str] = field(default_factory=list)
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
