from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from agent.state import CaseInput
from dataio.ham10000_aligned_schema import (
    ALIGNED_LABEL_MAP,
    Ham10000AlignedCaseRecord,
    Ham10000AlignedSubsetSummary,
    aligned_label_for_ham10000,
)
from dataio.ham10000_loader import (
    DEFAULT_HAM10000_ROOT,
    _sanitize_ham10000_metadata,
    discover_ham10000_assets,
    load_ham10000_rows,
    standardize_ham10000_row,
)


ALIGNED_ORIGINAL_LABELS = tuple(sorted(ALIGNED_LABEL_MAP.keys()))


def discover_ham10000_aligned_assets(data_root: str | Path = DEFAULT_HAM10000_ROOT) -> Ham10000AlignedSubsetSummary:
    assets = discover_ham10000_assets(data_root)
    rows = load_ham10000_rows(data_root)
    aligned_rows = [row for row in rows if str(row.get("dx", "")).strip().lower() in ALIGNED_ORIGINAL_LABELS]
    counts: dict[str, int] = {}
    for row in aligned_rows:
        label = aligned_label_for_ham10000(row.get("dx"))
        counts[label] = counts.get(label, 0) + 1
    return Ham10000AlignedSubsetSummary(
        data_root=assets.data_root,
        metadata_csv=assets.metadata_csv,
        image_dirs=list(assets.image_dirs),
        total_ham10000_rows=assets.sample_count,
        aligned_subset_count=len(aligned_rows),
        aligned_label_counts=dict(sorted(counts.items(), key=lambda item: item[0])),
    )


def load_ham10000_aligned_rows(data_root: str | Path = DEFAULT_HAM10000_ROOT) -> list[dict[str, Any]]:
    rows = load_ham10000_rows(data_root)
    return [row for row in rows if str(row.get("dx", "")).strip().lower() in ALIGNED_ORIGINAL_LABELS]


def standardize_ham10000_aligned_row(row: dict[str, Any], *, data_root: str | Path = DEFAULT_HAM10000_ROOT) -> Ham10000AlignedCaseRecord:
    standardized = standardize_ham10000_row(row, data_root=data_root)
    original_label = str(standardized.original_label).strip().lower()
    aligned_label = aligned_label_for_ham10000(original_label)
    metadata = _sanitize_ham10000_metadata(standardized.metadata)
    return Ham10000AlignedCaseRecord(
        case_id=standardized.case_id,
        image_path=standardized.image_path,
        metadata=metadata,
        original_label=original_label,
        aligned_label=aligned_label,
    )


def load_ham10000_aligned_records(
    *,
    data_root: str | Path = DEFAULT_HAM10000_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[Ham10000AlignedCaseRecord]:
    rows = list(load_ham10000_aligned_rows(data_root))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if sample_size is not None:
        rng = random.Random(seed)
        take = min(max(0, sample_size), len(rows))
        rows = rng.sample(rows, take)
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return [standardize_ham10000_aligned_row(row, data_root=data_root) for row in rows]


def load_ham10000_aligned_records_by_case_ids(
    case_ids: list[str],
    *,
    data_root: str | Path = DEFAULT_HAM10000_ROOT,
) -> list[Ham10000AlignedCaseRecord]:
    records = {
        record.case_id: record
        for record in load_ham10000_aligned_records(
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
        raise KeyError(f"Missing HAM10000 aligned case ids: {missing[:10]}")
    return [records[case_id] for case_id in case_ids]


def load_ham10000_aligned_case_inputs(
    *,
    data_root: str | Path = DEFAULT_HAM10000_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[CaseInput]:
    records = load_ham10000_aligned_records(
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
            label=record.aligned_label,
            reference_label=record.aligned_label,
            dataset_name=record.dataset_name,
            source_metadata_path=str(Path(data_root) / "HAM10000_metadata.csv"),
        )
        for record in records
    ]


def load_ham10000_aligned_case_inputs_by_case_ids(
    case_ids: list[str],
    *,
    data_root: str | Path = DEFAULT_HAM10000_ROOT,
) -> list[CaseInput]:
    records = load_ham10000_aligned_records_by_case_ids(case_ids, data_root=data_root)
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.aligned_label,
            reference_label=record.aligned_label,
            dataset_name=record.dataset_name,
            source_metadata_path=str(Path(data_root) / "HAM10000_metadata.csv"),
        )
        for record in records
    ]
