from __future__ import annotations

import csv
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.state import CaseInput
from agent.workflow_profiles import ensure_workflow_context
from dataio.case_schema import CaseSourceConfig, FieldCandidate, StandardizedCaseRecord
from dataio.ham10000_loader import DEFAULT_HAM10000_ROOT, load_ham10000_case_input_by_index, load_ham10000_case_inputs
from dataio.isic2019_loader import DEFAULT_ISIC2019_ROOT, load_isic2019_case_input_by_index, load_isic2019_case_inputs
from dataio.scin_loader import DEFAULT_SCIN_ROOT, discover_scin_case_source, load_scin_case_input_by_index, load_scin_case_inputs
from dataio.sd198_loader import (
    DEFAULT_SD198_ROOT,
    discover_sd198_case_source,
    load_sd198_case_input_by_index,
    load_sd198_case_inputs,
)
from dataio.xiangya_sft_loader import (
    DEFAULT_XIANGYA_SFT_ROOT,
    discover_xiangya_sft_case_source,
    load_xiangya_sft_case_input_by_index,
    load_xiangya_sft_case_inputs,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetLoaderSpec:
    dataset_name: str
    load_by_index: Callable[..., CaseInput]
    load_all: Callable[..., list[CaseInput]]
    discover_source: Callable[..., CaseSourceConfig] | None = None
    canonical_roots: tuple[str, ...] = ()


# Registry: dataset_name -> DatasetLoaderSpec
# Populated via register_dataset_loader(); built-in datasets are pre-registered below.
_LOADER_REGISTRY: dict[str, DatasetLoaderSpec] = {}


def register_dataset_loader(
    dataset_name: str,
    *,
    load_by_index: Callable[[int], CaseInput],
    load_all: Callable[[], list[CaseInput]],
    discover_source: Callable[..., CaseSourceConfig] | None = None,
    data_root: str | Path | None = None,
    data_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> None:
    """Register a dataset loader so case_loader.py routes to it automatically.

    Args:
        dataset_name: Canonical dataset name (e.g. "my_dataset").
        load_by_index: fn(case_index, *, data_root=...) -> CaseInput
        load_all: fn(*, data_root=...) -> list[CaseInput]
        data_root: Optional canonical data root path for path-based routing.
        data_roots: Optional additional root aliases for the same dataset.
    """
    key = _normalize_dataset_key(dataset_name)
    canonical_roots: list[str] = []
    if data_root is not None:
        canonical_roots.append(_normalize_root_key(data_root))
    for root in data_roots or ():
        canonical_roots.append(_normalize_root_key(root))

    spec = DatasetLoaderSpec(
        dataset_name=key,
        load_by_index=load_by_index,
        load_all=load_all,
        discover_source=discover_source,
        canonical_roots=tuple(dict.fromkeys(canonical_roots)),
    )
    _LOADER_REGISTRY[key] = spec
    for root_key in spec.canonical_roots:
        _PATH_LOADER_REGISTRY[root_key] = key


# Path-based routing: resolved path string -> dataset_name key
_PATH_LOADER_REGISTRY: dict[str, str] = {}


def list_registered_dataset_loaders() -> list[DatasetLoaderSpec]:
    return [spec for _, spec in sorted(_LOADER_REGISTRY.items(), key=lambda item: item[0])]


def get_registered_dataset_loader(dataset_name: str) -> DatasetLoaderSpec | None:
    return _LOADER_REGISTRY.get(_normalize_dataset_key(dataset_name))


def resolve_registered_dataset_loader(root: str | Path) -> DatasetLoaderSpec | None:
    dataset_key = _PATH_LOADER_REGISTRY.get(_normalize_root_key(root))
    if not dataset_key:
        return None
    return _LOADER_REGISTRY.get(dataset_key)


def discover_case_source(data_root: str | Path) -> CaseSourceConfig:
    root = Path(data_root)
    spec = resolve_registered_dataset_loader(root)
    if spec is not None and spec.discover_source is not None:
        return _call_registered_discover_source(spec.discover_source, data_root=root)

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
    spec = resolve_registered_dataset_loader(root)
    if spec is not None:
        case = _call_registered_load_by_index(spec.load_by_index, case_index=case_index, data_root=root)
        case.workflow_context = ensure_workflow_context(
            dataset_name=case.dataset_name,
            metadata=case.metadata,
            label_space_id=case.label_space_id,
            workflow_context=case.workflow_context,
        )
        return case
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
        workflow_context=ensure_workflow_context(
            dataset_name=_infer_dataset_name(config),
            metadata=standardized.metadata,
            label_space_id=str(standardized.metadata.get("label_space_id", "")).strip() or None,
            workflow_context=None,
        ),
    )


def sample_cases(count: int, data_root: str | Path, seed: int = 0) -> list[CaseInput]:
    root = Path(data_root)
    spec = resolve_registered_dataset_loader(root)
    if spec is not None:
        cases = _call_registered_load_all(spec.load_all, data_root=root)
        if not cases:
            return []
        rng = random.Random(seed)
        sampled = [cases[i] for i in rng.sample(range(len(cases)), min(count, len(cases)))]
        for case in sampled:
            case.workflow_context = ensure_workflow_context(
                dataset_name=case.dataset_name,
                metadata=case.metadata,
                label_space_id=case.label_space_id,
                workflow_context=case.workflow_context,
            )
        return sampled
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
                workflow_context=ensure_workflow_context(
                    dataset_name=_infer_dataset_name(config),
                    metadata=standardized.metadata,
                    label_space_id=str(standardized.metadata.get("label_space_id", "")).strip() or None,
                    workflow_context=None,
                ),
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


def load_case_with_masking(
    case_index: int,
    data_root: str | Path,
    metadata_mask: list[str] | None = None,
) -> CaseInput:
    """
    加载病例，支持 metadata masking

    Args:
        case_index: 病例索引
        data_root: 数据根目录
        metadata_mask: 要保留的字段列表，例如 ["region", "age"]
                      如果为 None，保留所有字段
                      如果为 []，mask 所有字段

    Returns:
        CaseInput with masked metadata and workflow_context set
    """
    case = load_case_by_index(case_index, data_root)

    if metadata_mask is not None:
        # 只保留 metadata_mask 中的字段，其余置为 None
        masked_metadata = {}
        for field in case.metadata.keys():
            if field in metadata_mask:
                masked_metadata[field] = case.metadata[field]
            else:
                masked_metadata[field] = None
        case.metadata = masked_metadata

        # 同时在 workflow_context 中标记 metadata_completeness
        if not case.workflow_context:
            case.workflow_context = {}

        if len(metadata_mask) == 0:
            case.workflow_context["metadata_completeness"] = "minimal"
        elif len(metadata_mask) <= 2:
            case.workflow_context["metadata_completeness"] = "partial"
        else:
            case.workflow_context["metadata_completeness"] = "full"

    return case


def _normalize_dataset_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    if not key:
        raise ValueError("dataset_name must be a non-empty string")
    return key


def _normalize_root_key(root: str | Path) -> str:
    return str(Path(root).resolve())


def _call_registered_load_by_index(
    load_by_index: Callable[..., CaseInput],
    *,
    case_index: int,
    data_root: Path,
) -> CaseInput:
    signature = _safe_signature(load_by_index)
    if signature and _supports_keyword(signature, "data_root"):
        return load_by_index(case_index, data_root=data_root)
    if signature and _supports_positional_count(signature, minimum=2):
        return load_by_index(case_index, data_root)
    return load_by_index(case_index)


def _call_registered_load_all(
    load_all: Callable[..., list[CaseInput]],
    *,
    data_root: Path,
) -> list[CaseInput]:
    signature = _safe_signature(load_all)
    if signature and _supports_keyword(signature, "data_root"):
        return load_all(data_root=data_root)
    if signature and _supports_positional_count(signature, minimum=1):
        return load_all(data_root)
    return load_all()


def _call_registered_discover_source(
    discover_source: Callable[..., CaseSourceConfig],
    *,
    data_root: Path,
) -> CaseSourceConfig:
    signature = _safe_signature(discover_source)
    if signature and _supports_keyword(signature, "data_root"):
        return discover_source(data_root=data_root)
    if signature and _supports_positional_count(signature, minimum=1):
        return discover_source(data_root)
    return discover_source()


def _safe_signature(func: Callable[..., Any]) -> inspect.Signature | None:
    try:
        return inspect.signature(func)
    except (TypeError, ValueError):
        return None


def _supports_keyword(signature: inspect.Signature, parameter_name: str) -> bool:
    if parameter_name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _supports_positional_count(signature: inspect.Signature, *, minimum: int) -> bool:
    positional_count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional_count += 1
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return positional_count >= minimum


# Pre-register built-in datasets so new datasets can follow the same pattern.
register_dataset_loader(
    "ham10000",
    load_by_index=load_ham10000_case_input_by_index,
    load_all=load_ham10000_case_inputs,
    data_root=DEFAULT_HAM10000_ROOT,
)
register_dataset_loader(
    "isic2019",
    load_by_index=load_isic2019_case_input_by_index,
    load_all=load_isic2019_case_inputs,
    data_root=DEFAULT_ISIC2019_ROOT,
)
register_dataset_loader(
    "scin",
    load_by_index=load_scin_case_input_by_index,
    load_all=load_scin_case_inputs,
    discover_source=discover_scin_case_source,
    data_root=DEFAULT_SCIN_ROOT,
    data_roots=[DEFAULT_SCIN_ROOT.parent],
)
register_dataset_loader(
    "sd198",
    load_by_index=load_sd198_case_input_by_index,
    load_all=load_sd198_case_inputs,
    discover_source=discover_sd198_case_source,
    data_root=DEFAULT_SD198_ROOT,
    data_roots=[DEFAULT_SD198_ROOT.parent],
)
register_dataset_loader(
    "xiangya_sft",
    load_by_index=load_xiangya_sft_case_input_by_index,
    load_all=load_xiangya_sft_case_inputs,
    discover_source=discover_xiangya_sft_case_source,
    data_root=DEFAULT_XIANGYA_SFT_ROOT,
    data_roots=[DEFAULT_XIANGYA_SFT_ROOT.parent / "xiangya_sft6"],
)
