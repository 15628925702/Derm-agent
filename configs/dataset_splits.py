from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SPLIT_ID = "pad_ufes_20_contiguous_v1"


@dataclass(frozen=True)
class FixedSplitDefinition:
    split_id: str
    dataset_name: str
    metadata_relpath: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    strategy: str
    notes: str


FIXED_SPLITS: dict[str, FixedSplitDefinition] = {
    DEFAULT_SPLIT_ID: FixedSplitDefinition(
        split_id=DEFAULT_SPLIT_ID,
        dataset_name="pad_ufes_20",
        metadata_relpath="pad_ufes_20/metadata.csv",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy="contiguous_by_metadata_index",
        notes=(
            "Contiguous ranges keep case-offset/limit evaluation compatible with the current "
            "frozen evaluation scripts while remaining deterministic and reproducible."
        ),
    ),
}


def available_split_ids() -> list[str]:
    return sorted(FIXED_SPLITS.keys())


def build_fixed_split_payload(
    split_id: str = DEFAULT_SPLIT_ID,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    definition = FIXED_SPLITS.get(split_id)
    if definition is None:
        raise KeyError(f"Unknown fixed split id `{split_id}`. Available: {available_split_ids()}")

    metadata_path = data_root / definition.metadata_relpath
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found for split `{split_id}`: {metadata_path}")

    rows = list(csv.DictReader(metadata_path.open("r", encoding="utf-8", newline="")))
    case_ids = [_build_case_id(row, row_index=index) for index, row in enumerate(rows)]
    total_cases = len(case_ids)
    if total_cases == 0:
        raise ValueError(f"Split `{split_id}` has no rows in metadata: {metadata_path}")

    train_count = int(total_cases * definition.train_ratio)
    remaining = total_cases - train_count
    val_count = remaining // 2
    test_count = remaining - val_count

    train_start = 0
    train_end = train_start + train_count - 1
    val_start = train_end + 1
    val_end = val_start + val_count - 1
    test_start = val_end + 1
    test_end = total_cases - 1

    payload = {
        "dataset_name": definition.dataset_name,
        "split_id": definition.split_id,
        "split_version": definition.split_id,
        "strategy": definition.strategy,
        "notes": definition.notes,
        "metadata_path": str(metadata_path),
        "total_cases": total_cases,
        "train_ratio": definition.train_ratio,
        "val_ratio": definition.val_ratio,
        "test_ratio": definition.test_ratio,
        "train_range": [train_start, train_end],
        "val_range": [val_start, val_end],
        "test_range": [test_start, test_end],
        "train": case_ids[train_start : train_end + 1],
        "val": case_ids[val_start : val_end + 1],
        "test": case_ids[test_start : test_end + 1],
    }
    return payload


def resolve_split_range(
    split_payload: dict[str, Any],
    split_name: str,
    *,
    count: int | None = None,
    offset_within_split: int = 0,
) -> tuple[int, int]:
    normalized = str(split_name).strip().lower()
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split name `{split_name}`.")
    range_key = f"{normalized}_range"
    raw_range = split_payload.get(range_key)
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ValueError(f"Split payload missing range `{range_key}`.")
    split_start, split_end = int(raw_range[0]), int(raw_range[1])
    available = split_end - split_start + 1
    if available <= 0:
        raise ValueError(f"Invalid range `{range_key}`: {raw_range}")
    local_offset = max(0, int(offset_within_split))
    if local_offset >= available:
        raise ValueError(
            f"offset_within_split={local_offset} is beyond available {normalized} cases={available}."
        )
    start_index = split_start + local_offset
    max_available = available - local_offset
    if count is None or int(count) <= 0:
        resolved_count = max_available
    else:
        resolved_count = min(int(count), max_available)
    return start_index, resolved_count


def write_fixed_split_json(
    output_path: Path,
    *,
    split_id: str = DEFAULT_SPLIT_ID,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    payload = build_fixed_split_payload(split_id, data_root=data_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _build_case_id(row: dict[str, Any], *, row_index: int) -> str:
    patient_id = str(row.get("patient_id", "")).strip()
    lesion_id = str(row.get("lesion_id", "")).strip()
    if patient_id and lesion_id:
        return f"{patient_id}_{lesion_id}"
    img_id = str(row.get("img_id", "")).strip()
    if img_id:
        return img_id
    return f"row_{row_index}"
