from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALIGNED_LABEL_MAP = {
    "MEL": "MEL",
    "BCC": "BCC",
    "NV": "NEV",
}


def aligned_label_for_isic2019(label: str | None) -> str:
    normalized = str(label or "").strip().upper()
    return ALIGNED_LABEL_MAP.get(normalized, "OTHER")


@dataclass
class Isic2019AlignedCaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    original_label: str
    aligned_label: str
    dataset_name: str = "ISIC2019"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Isic2019AlignedSubsetSummary:
    data_root: str
    ground_truth_csv: str
    metadata_csv: str
    image_dir: str
    total_ground_truth_rows: int = 0
    aligned_subset_count: int = 0
    aligned_label_counts: dict[str, int] = field(default_factory=dict)
    matched_image_count: int = 0
    missing_image_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
