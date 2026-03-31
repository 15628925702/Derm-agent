from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from agent.state import CaseInput
from dataio.ham10000_schema import Ham10000CaseRecord, Ham10000DatasetSummary, binary_label_for_ham10000


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_HAM10000_ROOT = Path("/root/DermAgent/data/ham10000")
HAM10000_METADATA_LEAKY_KEYS = {
    "dx",
    "dx_type",
    "binary_label",
    "aligned_label",
    "label",
    "reference_label",
    "true_label",
    "ground_truth",
}


def discover_ham10000_assets(data_root: str | Path = DEFAULT_HAM10000_ROOT) -> Ham10000DatasetSummary:
    root = Path(data_root)
    metadata_csv = root / "HAM10000_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found: {metadata_csv}")
    image_dirs = [
        path
        for path in (
            root / "HAM10000_images_part_1",
            root / "HAM10000_images_part_2",
        )
        if path.exists() and path.is_dir()
    ]
    if not image_dirs:
        raise FileNotFoundError(f"HAM10000 image directories not found under {root}")
    rows = _read_csv_rows(metadata_csv)
    return Ham10000DatasetSummary(
        data_root=str(root),
        metadata_csv=str(metadata_csv),
        image_dirs=[str(path) for path in image_dirs],
        sample_count=len(rows),
    )


def load_ham10000_rows(data_root: str | Path = DEFAULT_HAM10000_ROOT) -> list[dict[str, Any]]:
    assets = discover_ham10000_assets(data_root)
    return _read_csv_rows(Path(assets.metadata_csv))


def standardize_ham10000_row(row: dict[str, Any], *, data_root: str | Path = DEFAULT_HAM10000_ROOT) -> Ham10000CaseRecord:
    assets = discover_ham10000_assets(data_root)
    image_path = _resolve_image_path(image_id=str(row.get("image_id", "")).strip(), image_dirs=[Path(path) for path in assets.image_dirs])
    original_label = str(row.get("dx", "")).strip().lower()
    metadata = _sanitize_ham10000_metadata(row)
    return Ham10000CaseRecord(
        case_id=str(row.get("image_id", "")).strip(),
        image_path=str(image_path),
        metadata=metadata,
        original_label=original_label,
        binary_label=binary_label_for_ham10000(original_label),
    )


def load_ham10000_records(
    *,
    data_root: str | Path = DEFAULT_HAM10000_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Ham10000CaseRecord]:
    rows = load_ham10000_rows(data_root)
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
    return [standardize_ham10000_row(row, data_root=data_root) for row in ordered_rows]


def load_ham10000_case_inputs(
    *,
    data_root: str | Path = DEFAULT_HAM10000_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[CaseInput]:
    records = load_ham10000_records(
        data_root=data_root,
        limit=limit,
        offset=offset,
        sample_size=sample_size,
        seed=seed,
        shuffle=shuffle,
    )
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.binary_label,
            reference_label=record.binary_label,
            dataset_name=record.dataset_name,
            source_metadata_path=str(Path(data_root) / "HAM10000_metadata.csv"),
        )
        for record in records
    ]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_image_path(*, image_id: str, image_dirs: list[Path]) -> Path:
    if not image_id:
        raise ValueError("HAM10000 row missing image_id")
    for image_dir in image_dirs:
        for extension in IMAGE_EXTENSIONS:
            candidate = image_dir / f"{image_id}{extension}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Image for HAM10000 image_id `{image_id}` not found under {image_dirs}")


def _sanitize_ham10000_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(row).items()
        if str(key).strip().lower() not in HAM10000_METADATA_LEAKY_KEYS
    }
