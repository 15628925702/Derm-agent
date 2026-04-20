from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.sd198_label_catalog import SD198_RAW_TO_CANONICAL
from agent.state import CaseInput
from dataio.case_schema import CaseSourceConfig


DEFAULT_SD198_ROOT = Path("/root/DermAgent/data/sd198/sd-198")
SD198_LABEL_SPACE_ID = "sd198_full"
SD198_METADATA_LEAKY_KEYS = {
    "class_id",
    "class_name",
    "original_label",
    "label",
    "reference_label",
    "true_label",
    "ground_truth",
    "target",
}


@dataclass
class Sd198CaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    original_label: str
    dataset_name: str = "sd198"
    label_space_id: str = SD198_LABEL_SPACE_ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Sd198DatasetSummary:
    data_root: str
    classes_txt: str
    images_txt: str
    image_class_labels_txt: str
    image_dir: str
    sample_count: int = 0
    class_count: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_sd198_assets(data_root: str | Path = DEFAULT_SD198_ROOT) -> Sd198DatasetSummary:
    return _discover_sd198_assets_cached(str(Path(data_root)))


@lru_cache(maxsize=16)
def _discover_sd198_assets_cached(data_root: str) -> Sd198DatasetSummary:
    root = Path(data_root)
    classes_txt = root / "classes.txt"
    images_txt = root / "images.txt"
    image_class_labels_txt = root / "image_class_labels.txt"
    image_dir = root / "images"
    if not classes_txt.exists():
        raise FileNotFoundError(f"SD-198 classes file not found: {classes_txt}")
    if not images_txt.exists():
        raise FileNotFoundError(f"SD-198 images file not found: {images_txt}")
    if not image_class_labels_txt.exists():
        raise FileNotFoundError(f"SD-198 image_class_labels file not found: {image_class_labels_txt}")
    if not image_dir.exists():
        raise FileNotFoundError(f"SD-198 image directory not found: {image_dir}")
    rows = load_sd198_rows(data_root)
    label_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("original_label", "")).strip()
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
    return Sd198DatasetSummary(
        data_root=str(root),
        classes_txt=str(classes_txt),
        images_txt=str(images_txt),
        image_class_labels_txt=str(image_class_labels_txt),
        image_dir=str(image_dir),
        sample_count=len(rows),
        class_count=len(_load_sd198_class_map(str(root))),
        label_counts=label_counts,
    )


def load_sd198_rows(data_root: str | Path = DEFAULT_SD198_ROOT) -> list[dict[str, Any]]:
    return list(_load_sd198_rows_cached(str(Path(data_root))))


@lru_cache(maxsize=16)
def _load_sd198_rows_cached(data_root: str) -> tuple[dict[str, Any], ...]:
    root = Path(data_root)
    class_map = _load_sd198_class_map(str(root))
    image_map = _load_sd198_image_map(str(root))
    label_map = _load_sd198_image_label_map(str(root))

    merged_rows: list[dict[str, Any]] = []
    for image_id in sorted(image_map.keys()):
        relative_path = image_map[image_id]
        class_id = label_map.get(image_id)
        if class_id is None:
            raise ValueError(f"SD-198 image_id `{image_id}` missing class label mapping")
        raw_label = class_map.get(class_id)
        if raw_label is None:
            raise ValueError(f"SD-198 class_id `{class_id}` missing class definition")
        merged_rows.append(
            {
                "image_id": image_id,
                "relative_image_path": relative_path,
                "class_id": class_id,
                "raw_label": raw_label,
                "original_label": SD198_RAW_TO_CANONICAL.get(raw_label, raw_label),
            }
        )
    return tuple(merged_rows)


def load_sd198_record_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_SD198_ROOT,
) -> Sd198CaseRecord:
    rows = load_sd198_rows(data_root)
    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case-index {case_index} out of range for {len(rows)} rows")
    return standardize_sd198_row(rows[case_index], data_root=data_root)


def load_sd198_records(
    *,
    data_root: str | Path = DEFAULT_SD198_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Sd198CaseRecord]:
    rows = load_sd198_rows(data_root)
    ordered_rows = [dict(row) for row in rows]
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
    return [standardize_sd198_row(row, data_root=data_root) for row in ordered_rows]


def load_sd198_case_inputs(
    *,
    data_root: str | Path = DEFAULT_SD198_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[CaseInput]:
    records = load_sd198_records(
        data_root=data_root,
        limit=limit,
        offset=offset,
        sample_size=sample_size,
        seed=seed,
        shuffle=shuffle,
    )
    source_path = Path(data_root) / "images.txt"
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.original_label,
            reference_label=record.original_label,
            dataset_name=record.dataset_name,
            label_space_id=record.label_space_id,
            source_metadata_path=str(source_path),
        )
        for record in records
    ]


def load_sd198_case_input_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_SD198_ROOT,
) -> CaseInput:
    record = load_sd198_record_by_index(case_index, data_root=data_root)
    source_path = Path(data_root) / "images.txt"
    return CaseInput(
        case_id=record.case_id,
        image_path=record.image_path,
        metadata=record.metadata,
        label=record.original_label,
        reference_label=record.original_label,
        dataset_name=record.dataset_name,
        label_space_id=record.label_space_id,
        source_metadata_path=str(source_path),
    )


def discover_sd198_case_source(data_root: str | Path = DEFAULT_SD198_ROOT) -> CaseSourceConfig:
    root = Path(data_root)
    return CaseSourceConfig(
        data_root=root,
        metadata_path=root / "images.txt",
        image_root=root / "images",
        image_field="relative_image_path",
        label_field="original_label",
        case_id_fields=["image_id"],
        metadata_format="indexed_text",
    )


def standardize_sd198_row(
    row: dict[str, Any],
    *,
    data_root: str | Path = DEFAULT_SD198_ROOT,
) -> Sd198CaseRecord:
    root = Path(data_root)
    image_id = int(row.get("image_id", 0))
    relative_image_path = str(row.get("relative_image_path", "")).strip()
    if not relative_image_path:
        raise ValueError(f"SD-198 row missing relative_image_path: {row}")
    image_path = root / "images" / relative_image_path
    if not image_path.exists():
        raise FileNotFoundError(f"SD-198 image not found: {image_path}")

    metadata = _sanitize_sd198_metadata(row)
    metadata["label_space_id"] = SD198_LABEL_SPACE_ID
    metadata["case_source"] = "sd198"
    return Sd198CaseRecord(
        case_id=f"sd198_{image_id:06d}",
        image_path=str(image_path),
        metadata=metadata,
        original_label=str(row.get("original_label", "")).strip(),
    )


@lru_cache(maxsize=16)
def _load_sd198_class_map(data_root: str) -> dict[int, str]:
    class_map: dict[int, str] = {}
    with (Path(data_root) / "classes.txt").open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            class_id_text, raw_label = line.split(" ", 1)
            class_map[int(class_id_text)] = raw_label.strip()
    return class_map


@lru_cache(maxsize=16)
def _load_sd198_image_map(data_root: str) -> dict[int, str]:
    image_map: dict[int, str] = {}
    with (Path(data_root) / "images.txt").open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_id_text, relative_path = line.split(" ", 1)
            image_map[int(image_id_text)] = relative_path.strip()
    return image_map


@lru_cache(maxsize=16)
def _load_sd198_image_label_map(data_root: str) -> dict[int, int]:
    label_map: dict[int, int] = {}
    with (Path(data_root) / "image_class_labels.txt").open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_id_text, class_id_text = line.split(" ", 1)
            label_map[int(image_id_text)] = int(class_id_text.strip())
    return label_map


def _sanitize_sd198_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        str(key): value
        for key, value in dict(row).items()
        if str(key).strip().lower() not in SD198_METADATA_LEAKY_KEYS
    }
    raw_label = str(row.get("raw_label", "")).strip()
    if raw_label:
        metadata["raw_label_name"] = raw_label
        metadata["class_name"] = raw_label.replace("_", " ")
        metadata["class_family"] = raw_label.split("(", 1)[0].replace("_", " ").strip()
    relative_path = str(row.get("relative_image_path", "")).strip()
    if relative_path:
        metadata["image_relpath"] = relative_path
        metadata["image_basename"] = Path(relative_path).name
        metadata["class_dir"] = Path(relative_path).parent.name
    return metadata
