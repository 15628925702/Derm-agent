from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALIGNED_LABEL_MAP = {
    "mel": "MEL",
    "bcc": "BCC",
    "nv": "NEV",
}


def aligned_label_for_ham10000(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    return ALIGNED_LABEL_MAP.get(normalized, "OTHER")


@dataclass
class Ham10000AlignedCaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    original_label: str
    aligned_label: str
    dataset_name: str = "HAM10000"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ham10000AlignedSubsetSummary:
    data_root: str
    metadata_csv: str
    image_dirs: list[str] = field(default_factory=list)
    total_ham10000_rows: int = 0
    aligned_subset_count: int = 0
    aligned_label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
