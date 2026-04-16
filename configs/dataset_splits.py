from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SPLIT_ID = "pad_ufes_20_contiguous_v1"
ISIC2019_SPLIT_ID = "isic2019_contiguous_v1"
HAM10000_SPLIT_ID = "ham10000_contiguous_v1"
HAM10000_BALANCED_SPLIT_ID = "ham10000_balanced_v1"


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
    ISIC2019_SPLIT_ID: FixedSplitDefinition(
        split_id=ISIC2019_SPLIT_ID,
        dataset_name="isic2019",
        metadata_relpath="isic2019/ISIC_2019_Training_Metadata.csv",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy="contiguous_by_metadata_index",
        notes=(
            "Contiguous split over ISIC2019 metadata rows for deterministic dataset-adaptation experiments."
        ),
    ),
    HAM10000_SPLIT_ID: FixedSplitDefinition(
        split_id=HAM10000_SPLIT_ID,
        dataset_name="ham10000",
        metadata_relpath="ham10000/HAM10000_metadata.csv",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy="contiguous_by_metadata_index",
        notes=(
            "Contiguous split over HAM10000 metadata rows for deterministic dataset-adaptation experiments."
        ),
    ),
    HAM10000_BALANCED_SPLIT_ID: FixedSplitDefinition(
        split_id=HAM10000_BALANCED_SPLIT_ID,
        dataset_name="ham10000",
        metadata_relpath="ham10000/HAM10000_metadata.csv",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy="stratified_by_dx_group",
        notes=(
            "Balanced split over HAM10000 original labels to avoid all-benign contiguous slices during adaptation smoke tests."
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
    if definition.strategy == "stratified_by_dx_group":
        return _build_stratified_dx_split_payload(definition=definition, metadata_path=metadata_path, rows=rows)
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


def _build_stratified_dx_split_payload(
    *,
    definition: FixedSplitDefinition,
    metadata_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_rows = [
        {
            "case_id": _build_case_id(row, row_index=index),
            "row_index": index,
            "dx": str(row.get("dx", "")).strip().lower(),
        }
        for index, row in enumerate(rows)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in case_rows:
        grouped.setdefault(item["dx"], []).append(item)

    rng = random.Random(42)
    train_cases: list[dict[str, Any]] = []
    val_cases: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []

    for dx, items in sorted(grouped.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_count = int(total * definition.train_ratio)
        remaining = total - train_count
        val_count = remaining // 2
        test_count = remaining - val_count
        train_cases.extend(shuffled[:train_count])
        val_cases.extend(shuffled[train_count : train_count + val_count])
        test_cases.extend(shuffled[train_count + val_count : train_count + val_count + test_count])

    train_cases = _interleave_cases_by_label(train_cases)
    val_cases = _interleave_cases_by_label(val_cases)
    test_cases = _interleave_cases_by_label(test_cases)

    ordered = train_cases + val_cases + test_cases
    train_ids = [item["case_id"] for item in train_cases]
    val_ids = [item["case_id"] for item in val_cases]
    test_ids = [item["case_id"] for item in test_cases]
    total_cases = len(ordered)
    train_end = len(train_ids) - 1
    val_start = len(train_ids)
    val_end = val_start + len(val_ids) - 1
    test_start = val_end + 1
    test_end = total_cases - 1

    return {
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
        "train_range": [0, train_end],
        "val_range": [val_start, val_end],
        "test_range": [test_start, test_end],
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "train_case_indices": [item["row_index"] for item in train_cases],
        "val_case_indices": [item["row_index"] for item in val_cases],
        "test_case_indices": [item["row_index"] for item in test_cases],
    }


def _interleave_cases_by_label(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("dx", "")).strip().lower(), []).append(item)

    ordered_labels = [label for label in sorted(grouped.keys()) if label]
    result: list[dict[str, Any]] = []
    while ordered_labels:
        next_labels: list[str] = []
        for label in ordered_labels:
            bucket = grouped.get(label, [])
            if not bucket:
                continue
            result.append(bucket.pop(0))
            if bucket:
                next_labels.append(label)
        ordered_labels = next_labels
    return result


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
    image = str(row.get("image", "")).strip()
    if image:
        return image
    image_id = str(row.get("image_id", "")).strip()
    if image_id:
        return image_id
    patient_id = str(row.get("patient_id", "")).strip()
    lesion_id = str(row.get("lesion_id", "")).strip()
    if patient_id and lesion_id:
        return f"{patient_id}_{lesion_id}"
    img_id = str(row.get("img_id", "")).strip()
    if img_id:
        return img_id
    return f"row_{row_index}"
