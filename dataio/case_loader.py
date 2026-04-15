from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from agent.state import CaseInput
from dataio.case_schema import CaseSourceConfig, FieldCandidate, StandardizedCaseRecord
from dataio.isic2019_loader import DEFAULT_ISIC2019_ROOT, load_isic2019_case_inputs


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def discover_case_source(data_root: str | Path) -> CaseSourceConfig:
    root = Path(data_root)
    metadata_candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"
    )
    if not metadata_candidates:
        raise FileNotFoundError(f"No CSV metadata file found under {root}")

    metadata_path = metadata_candidates[0]
    rows = _read_csv_rows(metadata_path)
    fieldnames = list(rows[0].keys()) if rows else []

    image_root = _discover_image_root(root)
    image_field_candidates = _infer_image_field_candidates(fieldnames, rows, image_root)
    label_field_candidates = _infer_label_field_candidates(fieldnames, rows)

    image_field = image_field_candidates[0].field if image_field_candidates else None
    label_field = label_field_candidates[0].field if label_field_candidates else None
    case_id_fields = _infer_case_id_fields(fieldnames)

    return CaseSourceConfig(
        data_root=root,
        metadata_path=metadata_path,
        image_root=image_root,
        image_field=image_field,
        label_field=label_field,
        case_id_fields=case_id_fields,
        metadata_format="csv",
        image_field_candidates=image_field_candidates,
        label_field_candidates=label_field_candidates,
    )


def load_case_by_index(case_index: int, data_root: str | Path) -> CaseInput:
    root = Path(data_root)
    if root.resolve() == DEFAULT_ISIC2019_ROOT.resolve():
        cases = load_isic2019_case_inputs(data_root=root)
        if case_index < 0 or case_index >= len(cases):
            raise IndexError(f"case-index {case_index} out of range for {len(cases)} rows")
        return cases[case_index]
    config = discover_case_source(data_root)
    rows = _read_csv_rows(config.metadata_path)
    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case-index {case_index} out of range for {len(rows)} rows")

    row = rows[case_index]
    standardized = standardize_row(row=row, config=config)
    return CaseInput(
        case_id=standardized.case_id,
        image_path=standardized.image_path,
        metadata=standardized.metadata,
        label=standardized.label,
        dataset_name=_infer_dataset_name(config),
        source_metadata_path=str(config.metadata_path),
    )


def sample_cases(count: int, data_root: str | Path, seed: int = 0) -> list[CaseInput]:
    root = Path(data_root)
    if root.resolve() == DEFAULT_ISIC2019_ROOT.resolve():
        cases = load_isic2019_case_inputs(data_root=root)
        if not cases:
            return []
        rng = random.Random(seed)
        sample_size = min(count, len(cases))
        indices = rng.sample(range(len(cases)), sample_size)
        return [cases[index] for index in indices]
    config = discover_case_source(data_root)
    rows = _read_csv_rows(config.metadata_path)
    if not rows:
        return []

    rng = random.Random(seed)
    sample_size = min(count, len(rows))
    indices = rng.sample(range(len(rows)), sample_size)
    cases: list[CaseInput] = []
    for index in indices:
        standardized = standardize_row(rows[index], config)
        cases.append(
            CaseInput(
                case_id=standardized.case_id,
                image_path=standardized.image_path,
                metadata=standardized.metadata,
                label=standardized.label,
                dataset_name=_infer_dataset_name(config),
                source_metadata_path=str(config.metadata_path),
            )
        )
    return cases


def standardize_row(row: dict[str, Any], config: CaseSourceConfig) -> StandardizedCaseRecord:
    case_id = _build_case_id(row, config.case_id_fields)
    image_name = str(row.get(config.image_field, "")).strip() if config.image_field else ""
    image_path = _resolve_image_path(config.image_root, image_name)
    label = str(row.get(config.label_field)).strip() if config.label_field and row.get(config.label_field) not in ("", None) else None
    return StandardizedCaseRecord(
        case_id=case_id,
        image_path=str(image_path),
        metadata=dict(row),
        label=label,
    )


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _discover_image_root(root: Path) -> Path:
    image_dirs = sorted(path for path in root.rglob("*") if path.is_dir() and "image" in path.name.lower())
    if image_dirs:
        return image_dirs[0]

    for candidate in sorted(path for path in root.rglob("*") if path.is_dir()):
        if any(file_path.suffix.lower() in IMAGE_EXTENSIONS for file_path in candidate.iterdir() if file_path.is_file()):
            return candidate
    raise FileNotFoundError(f"No image directory found under {root}")


def _infer_image_field_candidates(
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    image_root: Path,
) -> list[FieldCandidate]:
    image_names = {path.name for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    candidates: list[FieldCandidate] = []

    for field in fieldnames:
        score = 0
        lowered = field.lower()
        if "img" in lowered or "image" in lowered or "path" in lowered:
            score += 5

        non_empty_values = [str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()]
        matched_names = sum(1 for value in non_empty_values if value in image_names)
        extension_like = sum(1 for value in non_empty_values if Path(value).suffix.lower() in IMAGE_EXTENSIONS)
        score += matched_names * 3
        score += extension_like

        if score > 0:
            reason = f"name_hint={lowered}; matched_images={matched_names}; extension_like={extension_like}"
            candidates.append(FieldCandidate(field=field, score=score, reason=reason))

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _infer_label_field_candidates(fieldnames: list[str], rows: list[dict[str, Any]]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    sample_count = len(rows)

    for field in fieldnames:
        lowered = field.lower()
        non_empty_values = [str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()]
        unique_values = sorted(set(non_empty_values))
        unique_count = len(unique_values)
        score = 0

        if any(token in lowered for token in ("label", "class", "target", "diagn", "dx")):
            score += 10

        if 1 < unique_count <= min(50, max(2, sample_count // 5 if sample_count else 50)):
            score += 3

        if unique_count and all(len(value) <= 12 for value in unique_values[: min(unique_count, 10)]):
            score += 1

        if score > 0:
            reason = f"name_hint={lowered}; unique_count={unique_count}"
            candidates.append(FieldCandidate(field=field, score=score, reason=reason))

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _infer_case_id_fields(fieldnames: list[str]) -> list[str]:
    preferred = [field for field in ("patient_id", "lesion_id") if field in fieldnames]
    if preferred:
        return preferred
    fallback = [field for field in fieldnames if field.lower().endswith("_id")]
    return fallback[:2] if fallback else [fieldnames[0]] if fieldnames else ["row_id"]


def _build_case_id(row: dict[str, Any], case_id_fields: list[str]) -> str:
    parts = [str(row.get(field, "")).strip() for field in case_id_fields if str(row.get(field, "")).strip()]
    if parts:
        return "_".join(parts)
    for field, value in row.items():
        if str(value).strip():
            return f"case_{field}_{value}"
    return "case_unknown"


def _resolve_image_path(image_root: Path, image_name: str) -> Path:
    if not image_name:
        return image_root / "missing_image"
    direct_path = image_root / image_name
    if direct_path.exists():
        return direct_path
    matches = list(image_root.glob(f"**/{image_name}"))
    if matches:
        return matches[0]
    return direct_path


def _infer_dataset_name(config: CaseSourceConfig) -> str:
    metadata_parent = config.metadata_path.parent.name.strip()
    if metadata_parent:
        return metadata_parent
    image_parent = config.image_root.parent.name.strip()
    if image_parent:
        return image_parent
    return config.data_root.name.strip() or "unknown_dataset"
