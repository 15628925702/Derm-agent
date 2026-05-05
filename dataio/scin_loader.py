from __future__ import annotations

import ast
import csv
import os
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.state import CaseInput
from agent.workflow_profiles import ensure_workflow_context
from dataio.case_schema import CaseSourceConfig


DEFAULT_SCIN_ROOT = Path("/root/DermAgent/data/scin/official_mirror")
SCIN_METADATA_LEAKY_KEYS = {
    "dermatologist_skin_condition_on_label_name",
    "dermatologist_skin_condition_confidence",
    "weighted_skin_condition_label",
    "dermatologist_fitzpatrick_skin_type_label_1",
    "dermatologist_fitzpatrick_skin_type_label_2",
    "dermatologist_fitzpatrick_skin_type_label_3",
    "monk_skin_tone_label_india",
    "monk_skin_tone_label_us",
    "label",
    "reference_label",
    "true_label",
    "ground_truth",
    "target",
}


@dataclass
class ScinCaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    original_label: str
    image_paths: list[str] = field(default_factory=list)
    dataset_name: str = "scin"
    label_space_id: str = "scin_full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScinDatasetSummary:
    data_root: str
    cases_csv: str
    labels_csv: str
    image_dir: str
    sample_count: int = 0
    labeled_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_scin_assets(data_root: str | Path = DEFAULT_SCIN_ROOT) -> ScinDatasetSummary:
    return _discover_scin_assets_cached(str(_resolve_scin_root(data_root)))


@lru_cache(maxsize=16)
def _discover_scin_assets_cached(data_root: str) -> ScinDatasetSummary:
    root = Path(data_root)
    cases_csv = root / "scin_cases.csv"
    labels_csv = root / "scin_labels.csv"
    image_dir = root / "images"
    if not cases_csv.exists():
        raise FileNotFoundError(f"SCIN cases CSV not found: {cases_csv}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"SCIN labels CSV not found: {labels_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"SCIN image directory not found: {image_dir}")
    rows = load_scin_rows(data_root)
    labeled_count = sum(1 for row in rows if str(row.get("original_label", "")).strip())
    return ScinDatasetSummary(
        data_root=str(root),
        cases_csv=str(cases_csv),
        labels_csv=str(labels_csv),
        image_dir=str(image_dir),
        sample_count=len(rows),
        labeled_count=labeled_count,
    )


def load_scin_rows(data_root: str | Path = DEFAULT_SCIN_ROOT) -> list[dict[str, Any]]:
    return list(_load_scin_rows_cached(str(_resolve_scin_root(data_root))))


@lru_cache(maxsize=16)
def _load_scin_rows_cached(data_root: str) -> tuple[dict[str, Any], ...]:
    root = Path(data_root)
    cases_csv = root / "scin_cases.csv"
    labels_csv = root / "scin_labels.csv"

    cases_by_id: dict[str, dict[str, str]] = {}
    with cases_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                cases_by_id[case_id] = dict(row)

    labels_by_id: dict[str, dict[str, str]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                labels_by_id[case_id] = dict(row)

    merged_rows: list[dict[str, Any]] = []
    for case_id, case_row in cases_by_id.items():
        merged = dict(case_row)
        merged.update(labels_by_id.get(case_id, {}))
        merged["original_label"] = _extract_original_label(merged)
        merged_rows.append(merged)
    return tuple(merged_rows)


def standardize_scin_row(row: dict[str, Any], *, data_root: str | Path = DEFAULT_SCIN_ROOT) -> ScinCaseRecord:
    root = _resolve_scin_root(data_root)
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("SCIN row missing case_id")

    image_paths = _resolve_image_paths(row=row, data_root=root)
    if not image_paths:
        raise FileNotFoundError(f"SCIN case `{case_id}` does not have a resolvable image path")

    metadata = _sanitize_scin_metadata(row=row, image_paths=image_paths)
    original_label = str(row.get("original_label", "")).strip()
    effective_label_space_id = str(os.getenv("DERMAGENT_SCIN_LABEL_SPACE_ID", "scin_full")).strip() or "scin_full"
    metadata["label_space_id"] = effective_label_space_id
    return ScinCaseRecord(
        case_id=case_id,
        image_path=image_paths[0],
        image_paths=image_paths,
        metadata=metadata,
        original_label=original_label,
        label_space_id=effective_label_space_id,
    )


def load_scin_record_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_SCIN_ROOT,
) -> ScinCaseRecord:
    rows = load_scin_rows(data_root)
    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case-index {case_index} out of range for {len(rows)} rows")
    return standardize_scin_row(rows[case_index], data_root=data_root)


def load_scin_records(
    *,
    data_root: str | Path = DEFAULT_SCIN_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
    labeled_only: bool = True,
) -> list[ScinCaseRecord]:
    rows = load_scin_rows(data_root)
    ordered_rows = [dict(row) for row in rows]
    if labeled_only:
        ordered_rows = [row for row in ordered_rows if str(row.get("original_label", "")).strip()]
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
    return [standardize_scin_row(row, data_root=data_root) for row in ordered_rows]


def load_scin_case_inputs(
    *,
    data_root: str | Path = DEFAULT_SCIN_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
    labeled_only: bool = True,
) -> list[CaseInput]:
    records = load_scin_records(
        data_root=data_root,
        limit=limit,
        offset=offset,
        sample_size=sample_size,
        seed=seed,
        shuffle=shuffle,
        labeled_only=labeled_only,
    )
    cases_csv = _resolve_scin_root(data_root) / "scin_cases.csv"
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.original_label,
            reference_label=record.original_label,
            dataset_name=record.dataset_name,
            label_space_id=record.label_space_id,
            source_metadata_path=str(cases_csv),
            workflow_context=ensure_workflow_context(
                dataset_name=record.dataset_name,
                metadata=record.metadata,
                label_space_id=record.label_space_id,
                workflow_context=None,
            ),
        )
        for record in records
    ]


def load_scin_case_input_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_SCIN_ROOT,
) -> CaseInput:
    record = load_scin_record_by_index(case_index, data_root=data_root)
    cases_csv = _resolve_scin_root(data_root) / "scin_cases.csv"
    return CaseInput(
        case_id=record.case_id,
        image_path=record.image_path,
        metadata=record.metadata,
        label=record.original_label,
        reference_label=record.original_label,
        dataset_name=record.dataset_name,
        label_space_id=record.label_space_id,
        source_metadata_path=str(cases_csv),
        workflow_context=ensure_workflow_context(
            dataset_name=record.dataset_name,
            metadata=record.metadata,
            label_space_id=record.label_space_id,
            workflow_context=None,
        ),
    )


def discover_scin_case_source(data_root: str | Path = DEFAULT_SCIN_ROOT) -> CaseSourceConfig:
    root = _resolve_scin_root(data_root)
    discover_scin_assets(root)
    return CaseSourceConfig(
        data_root=root,
        metadata_path=root / "scin_cases.csv",
        image_root=root / "images",
        image_field="image_1_path",
        label_field="original_label",
        case_id_fields=["case_id"],
        metadata_format="scin_csv",
    )


def _resolve_scin_root(data_root: str | Path = DEFAULT_SCIN_ROOT) -> Path:
    root = Path(data_root)
    if (root / "scin_cases.csv").exists() and (root / "scin_labels.csv").exists():
        return root
    nested = root / "official_mirror"
    if (nested / "scin_cases.csv").exists() and (nested / "scin_labels.csv").exists():
        return nested
    return root


def _extract_original_label(row: dict[str, Any]) -> str:
    weighted = str(row.get("weighted_skin_condition_label", "")).strip()
    if not weighted:
        return ""
    try:
        parsed = ast.literal_eval(weighted)
    except (SyntaxError, ValueError):
        return ""
    if not isinstance(parsed, dict) or not parsed:
        return ""

    best_label = ""
    best_score = float("-inf")
    for label, score in parsed.items():
        label_text = str(label).strip()
        if not label_text:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0
        if numeric_score > best_score or (numeric_score == best_score and label_text < best_label):
            best_label = label_text
            best_score = numeric_score
    return best_label


def _resolve_image_paths(*, row: dict[str, Any], data_root: Path) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for field in ("image_1_path", "image_2_path", "image_3_path"):
        raw_path = str(row.get(field, "")).strip()
        if not raw_path:
            continue
        candidate = data_root / raw_path
        if not candidate.exists():
            candidate = data_root / raw_path.replace("dataset/images/", "images/")
        if not candidate.exists():
            candidate = data_root / "images" / Path(raw_path).name
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            resolved.append(text)
    return resolved


def _sanitize_scin_metadata(*, row: dict[str, Any], image_paths: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metadata["label_space_id"] = "scin_full"
    metadata["case_source"] = "scin"
    metadata["image_count"] = len(image_paths)
    metadata["image_paths"] = list(image_paths)

    for key in (
        "age_group",
        "sex_at_birth",
        "fitzpatrick_skin_type",
        "combined_race",
        "related_category",
        "condition_duration",
    ):
        value = str(row.get(key, "")).strip()
        if value:
            metadata[key] = value

    body_parts = [
        key.replace("body_parts_", "")
        for key, value in row.items()
        if str(key).startswith("body_parts_") and str(value).strip().upper() == "YES"
    ]
    if body_parts:
        metadata["body_sites"] = body_parts
        if "region" not in metadata:
            metadata["region"] = body_parts[0]

    textures = [
        key.replace("textures_", "")
        for key, value in row.items()
        if str(key).startswith("textures_") and str(value).strip().upper() == "YES"
    ]
    if textures:
        metadata["textures_present"] = textures

    symptoms = [
        key.replace("condition_symptoms_", "")
        for key, value in row.items()
        if str(key).startswith("condition_symptoms_") and str(value).strip().upper() == "YES"
    ]
    if symptoms:
        metadata["symptoms_present"] = symptoms

    other_symptoms = [
        key.replace("other_symptoms_", "")
        for key, value in row.items()
        if str(key).startswith("other_symptoms_") and str(value).strip().upper() == "YES"
    ]
    if other_symptoms:
        metadata["other_symptoms_present"] = other_symptoms

    shot_types = [
        str(row.get(field, "")).strip()
        for field in ("image_1_shot_type", "image_2_shot_type", "image_3_shot_type")
        if str(row.get(field, "")).strip()
    ]
    if shot_types:
        metadata["shot_types"] = shot_types

    age_group = str(row.get("age_group", "")).strip()
    if age_group:
        metadata["age_group"] = age_group
    sex_at_birth = str(row.get("sex_at_birth", "")).strip()
    if sex_at_birth:
        metadata["sex_at_birth"] = sex_at_birth
    related_category = str(row.get("related_category", "")).strip()
    if related_category:
        metadata["related_category"] = related_category
    condition_duration = str(row.get("condition_duration", "")).strip()
    if condition_duration:
        metadata["condition_duration"] = condition_duration

    return metadata
