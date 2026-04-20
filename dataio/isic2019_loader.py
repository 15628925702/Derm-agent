from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.state import CaseInput
from agent.workflow_profiles import ensure_workflow_context


DEFAULT_ISIC2019_ROOT = Path("/root/DermAgent/data/isic2019")
ISIC2019_METADATA_LEAKY_KEYS = {
    "mel",
    "nv",
    "bcc",
    "ak",
    "bkl",
    "df",
    "vasc",
    "scc",
    "unk",
    "label",
    "reference_label",
    "true_label",
    "ground_truth",
    "canonical_label",
    "risk_label",
}

ISIC2019_PRIMARY_LABELS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")


@dataclass
class Isic2019CaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    original_label: str
    dataset_name: str = "ISIC2019"
    label_space_id: str = "isic2019_full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Isic2019DatasetSummary:
    data_root: str
    metadata_csv: str
    ground_truth_csv: str
    image_dir: str
    sample_count: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_isic2019_assets(data_root: str | Path = DEFAULT_ISIC2019_ROOT) -> Isic2019DatasetSummary:
    return _discover_isic2019_assets_cached(str(Path(data_root)))


@lru_cache(maxsize=16)
def _discover_isic2019_assets_cached(data_root: str) -> Isic2019DatasetSummary:
    root = Path(data_root)
    metadata_csv = root / "ISIC_2019_Training_Metadata.csv"
    ground_truth_csv = root / "ISIC_2019_Training_GroundTruth.csv"
    image_dir = root / "images" / "ISIC_2019_Training_Input"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"ISIC2019 metadata CSV not found: {metadata_csv}")
    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"ISIC2019 ground-truth CSV not found: {ground_truth_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"ISIC2019 image directory not found: {image_dir}")
    rows = load_isic2019_rows(data_root)
    counts = Counter(str(row.get("original_label", "")).strip() for row in rows if str(row.get("original_label", "")).strip())
    return Isic2019DatasetSummary(
        data_root=str(root),
        metadata_csv=str(metadata_csv),
        ground_truth_csv=str(ground_truth_csv),
        image_dir=str(image_dir),
        sample_count=len(rows),
        label_counts=dict(counts),
    )


def load_isic2019_rows(data_root: str | Path = DEFAULT_ISIC2019_ROOT) -> list[dict[str, Any]]:
    return list(_load_isic2019_rows_cached(str(Path(data_root))))


@lru_cache(maxsize=16)
def _load_isic2019_rows_cached(data_root: str) -> tuple[dict[str, Any], ...]:
    root = Path(data_root)
    metadata_csv = root / "ISIC_2019_Training_Metadata.csv"
    ground_truth_csv = root / "ISIC_2019_Training_GroundTruth.csv"

    metadata_rows: dict[str, dict[str, str]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("image", "")).strip()
            if image_id:
                metadata_rows[image_id] = dict(row)

    merged_rows: list[dict[str, Any]] = []
    with ground_truth_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("image", "")).strip()
            if not image_id:
                continue
            merged = dict(metadata_rows.get(image_id, {}))
            merged.update(dict(row))
            merged["original_label"] = _extract_original_label(row)
            merged_rows.append(merged)
    return tuple(merged_rows)


def standardize_isic2019_row(row: dict[str, Any], *, data_root: str | Path = DEFAULT_ISIC2019_ROOT) -> Isic2019CaseRecord:
    root = Path(data_root)
    image_id = str(row.get("image", "")).strip()
    if not image_id:
        raise ValueError("ISIC2019 row missing image id")
    image_path = root / "images" / "ISIC_2019_Training_Input" / f"{image_id}.jpg"
    metadata = {
        str(key): value
        for key, value in dict(row).items()
        if str(key).strip().lower() not in {item.lower() for item in ISIC2019_METADATA_LEAKY_KEYS}
    }
    metadata["label_space_id"] = "isic2019_full"
    return Isic2019CaseRecord(
        case_id=image_id,
        image_path=str(image_path),
        metadata=metadata,
        original_label=str(row.get("original_label", "")).strip(),
    )


def load_isic2019_record_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
) -> Isic2019CaseRecord:
    rows = load_isic2019_rows(data_root)
    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case-index {case_index} out of range for {len(rows)} rows")
    return standardize_isic2019_row(rows[case_index], data_root=data_root)


def load_isic2019_records(
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Isic2019CaseRecord]:
    rows = load_isic2019_rows(data_root)
    ordered_rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered_rows)
    if sample_size is not None:
        rng = random.Random(seed)
        take = min(max(0, sample_size), len(ordered_rows))
        ordered_rows = rng.sample(ordered_rows, take)
    if offset:
        ordered_rows = ordered_rows[offset:]
    if limit is not None:
        ordered_rows = ordered_rows[:limit]
    return [standardize_isic2019_row(row, data_root=data_root) for row in ordered_rows]


def load_isic2019_case_inputs(
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[CaseInput]:
    records = load_isic2019_records(
        data_root=data_root,
        limit=limit,
        offset=offset,
        sample_size=sample_size,
        seed=seed,
        shuffle=shuffle,
    )
    metadata_csv = Path(data_root) / "ISIC_2019_Training_Metadata.csv"
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.original_label,
            reference_label=record.original_label,
            dataset_name=record.dataset_name,
            label_space_id=record.label_space_id,
            source_metadata_path=str(metadata_csv),
            workflow_context=ensure_workflow_context(
                dataset_name=record.dataset_name,
                metadata=record.metadata,
                label_space_id=record.label_space_id,
                workflow_context=None,
            ),
        )
        for record in records
    ]


def load_isic2019_case_input_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
) -> CaseInput:
    record = load_isic2019_record_by_index(case_index, data_root=data_root)
    metadata_csv = Path(data_root) / "ISIC_2019_Training_Metadata.csv"
    return CaseInput(
        case_id=record.case_id,
        image_path=record.image_path,
        metadata=record.metadata,
        label=record.original_label,
        reference_label=record.original_label,
        dataset_name=record.dataset_name,
        label_space_id=record.label_space_id,
        source_metadata_path=str(metadata_csv),
        workflow_context=ensure_workflow_context(
            dataset_name=record.dataset_name,
            metadata=record.metadata,
            label_space_id=record.label_space_id,
            workflow_context=None,
        ),
    )


def _extract_original_label(row: dict[str, Any]) -> str:
    active_labels = [label for label in ISIC2019_PRIMARY_LABELS if str(row.get(label, "")).strip() == "1.0"]
    if not active_labels:
        return "UNK"
    return active_labels[0]
