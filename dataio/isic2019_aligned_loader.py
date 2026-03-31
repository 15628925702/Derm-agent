from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from agent.state import CaseInput
from dataio.isic2019_aligned_schema import (
    ALIGNED_LABEL_MAP,
    Isic2019AlignedCaseRecord,
    Isic2019AlignedSubsetSummary,
    aligned_label_for_isic2019,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_ISIC2019_ROOT = Path("/root/DermAgent/data/isic2019")
GROUND_TRUTH_COLUMNS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
ISIC2019_METADATA_LEAKY_KEYS = {
    "image",
    "image_id",
    "case_id",
    "label",
    "reference_label",
    "ground_truth",
    "true_label",
    "aligned_label",
    "original_label",
    "mel",
    "nv",
    "bcc",
    "ak",
    "bkl",
    "df",
    "vasc",
    "scc",
    "unk",
    "lesion_id",
}


def discover_isic2019_aligned_assets(data_root: str | Path = DEFAULT_ISIC2019_ROOT) -> Isic2019AlignedSubsetSummary:
    root = Path(data_root)
    ground_truth_csv = _discover_ground_truth_csv(root)
    metadata_csv = _discover_metadata_csv(root)
    image_dir = _discover_image_dir(root)
    ground_truth_rows = _read_csv_rows(ground_truth_csv)

    aligned_rows = [row for row in ground_truth_rows if _extract_original_label(row)]
    counts: dict[str, int] = {}
    matched_image_count = 0
    missing_image_count = 0
    for row in aligned_rows:
        original_label = _extract_original_label(row)
        aligned_label = aligned_label_for_isic2019(original_label)
        counts[aligned_label] = counts.get(aligned_label, 0) + 1
        image_id = str(row.get("image", "")).strip()
        try:
            _resolve_image_path(image_id=image_id, image_dir=image_dir)
            matched_image_count += 1
        except FileNotFoundError:
            missing_image_count += 1

    return Isic2019AlignedSubsetSummary(
        data_root=str(root),
        ground_truth_csv=str(ground_truth_csv),
        metadata_csv=str(metadata_csv),
        image_dir=str(image_dir),
        total_ground_truth_rows=len(ground_truth_rows),
        aligned_subset_count=len(aligned_rows),
        aligned_label_counts=dict(sorted(counts.items(), key=lambda item: item[0])),
        matched_image_count=matched_image_count,
        missing_image_count=missing_image_count,
    )


def load_isic2019_aligned_records(
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Isic2019AlignedCaseRecord]:
    assets = discover_isic2019_aligned_assets(data_root)
    ground_truth_rows = _read_csv_rows(Path(assets.ground_truth_csv))
    metadata_index = _metadata_rows_by_image_id(Path(assets.metadata_csv))
    selected_rows = [row for row in ground_truth_rows if _extract_original_label(row)]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected_rows)
    if sample_size is not None:
        rng = random.Random(seed)
        take = min(max(0, sample_size), len(selected_rows))
        selected_rows = rng.sample(selected_rows, take)
    if offset:
        selected_rows = selected_rows[offset:]
    if limit is not None:
        selected_rows = selected_rows[:limit]

    return [
        standardize_isic2019_aligned_row(
            row,
            image_dir=Path(assets.image_dir),
            metadata_row=metadata_index.get(str(row.get("image", "")).strip(), {}),
        )
        for row in selected_rows
    ]


def load_isic2019_aligned_records_by_case_ids(
    case_ids: list[str],
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
) -> list[Isic2019AlignedCaseRecord]:
    records = {
        record.case_id: record
        for record in load_isic2019_aligned_records(
            data_root=data_root,
            limit=None,
            offset=0,
            sample_size=None,
            seed=0,
            shuffle=False,
        )
    }
    missing = [case_id for case_id in case_ids if case_id not in records]
    if missing:
        raise KeyError(f"Missing ISIC2019 aligned case ids: {missing[:10]}")
    return [records[case_id] for case_id in case_ids]


def load_isic2019_aligned_case_inputs(
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[CaseInput]:
    records = load_isic2019_aligned_records(
        data_root=data_root,
        limit=limit,
        offset=offset,
        sample_size=sample_size,
        seed=seed,
        shuffle=shuffle,
    )
    assets = discover_isic2019_aligned_assets(data_root)
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.aligned_label,
            reference_label=record.aligned_label,
            dataset_name=record.dataset_name,
            source_metadata_path=assets.metadata_csv,
        )
        for record in records
    ]


def load_isic2019_aligned_case_inputs_by_case_ids(
    case_ids: list[str],
    *,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
) -> list[CaseInput]:
    records = load_isic2019_aligned_records_by_case_ids(case_ids, data_root=data_root)
    assets = discover_isic2019_aligned_assets(data_root)
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.aligned_label,
            reference_label=record.aligned_label,
            dataset_name=record.dataset_name,
            source_metadata_path=assets.metadata_csv,
        )
        for record in records
    ]


def standardize_isic2019_aligned_row(
    row: dict[str, Any],
    *,
    image_dir: Path | None = None,
    metadata_row: dict[str, Any] | None = None,
    data_root: str | Path = DEFAULT_ISIC2019_ROOT,
) -> Isic2019AlignedCaseRecord:
    assets = discover_isic2019_aligned_assets(data_root) if image_dir is None else None
    resolved_image_dir = image_dir or Path(assets.image_dir)
    image_id = str(row.get("image", "")).strip()
    image_path = _resolve_image_path(image_id=image_id, image_dir=resolved_image_dir)
    original_label = _extract_original_label(row)
    aligned_label = aligned_label_for_isic2019(original_label)
    metadata = _sanitize_isic2019_metadata(metadata_row or {})
    return Isic2019AlignedCaseRecord(
        case_id=image_id,
        image_path=str(image_path),
        metadata=metadata,
        original_label=str(original_label or "").strip().upper(),
        aligned_label=aligned_label,
    )


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _discover_ground_truth_csv(root: Path) -> Path:
    exact = root / "ISIC_2019_Training_GroundTruth.csv"
    if exact.exists():
        return exact
    candidates = sorted(root.glob("*GroundTruth*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"ISIC2019 ground-truth CSV not found under {root}")


def _discover_metadata_csv(root: Path) -> Path:
    exact = root / "ISIC_2019_Training_Metadata.csv"
    if exact.exists():
        return exact
    candidates = sorted(root.glob("*Metadata*.csv")) + sorted(root.glob("*metadata*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"ISIC2019 metadata CSV not found under {root}")


def _discover_image_dir(root: Path) -> Path:
    preferred = [
        root / "images" / "ISIC_2019_Training_Input",
        root / "ISIC_2019_Training_Input",
        root / "images",
    ]
    for candidate in preferred:
        if candidate.exists() and candidate.is_dir() and any(child.is_file() for child in candidate.iterdir()):
            return candidate
    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir() and "input" in candidate.name.lower() and any(child.is_file() for child in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"ISIC2019 image directory not found under {root}")


def _metadata_rows_by_image_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv_rows(path)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image", "")).strip()
        if image_id:
            index[image_id] = dict(row)
    return index


def _extract_original_label(row: dict[str, Any]) -> str | None:
    positive_columns = [
        column
        for column in GROUND_TRUTH_COLUMNS
        if float(row.get(column, 0) or 0.0) > 0.5
    ]
    if len(positive_columns) != 1:
        return None
    label = positive_columns[0]
    return label if label in ALIGNED_LABEL_MAP else None


def _resolve_image_path(*, image_id: str, image_dir: Path) -> Path:
    if not image_id:
        raise ValueError("ISIC2019 row missing image id")
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for ISIC2019 image id `{image_id}` not found under {image_dir}")


def _sanitize_isic2019_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(row).items()
        if str(key).strip().lower() not in ISIC2019_METADATA_LEAKY_KEYS and str(value).strip()
    }
