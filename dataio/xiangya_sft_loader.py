from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.label_space import canonicalize_label
from agent.state import CaseInput
from agent.workflow_profiles import ensure_workflow_context
from dataio.case_schema import CaseSourceConfig
from project_paths import data_root


DEFAULT_XIANGYA_SFT_ROOT = data_root() / "sft数据"
DEFAULT_XIANGYA_SFT_LABEL_SPACE_ID = "xiangya_sft_grouped"


@dataclass
class XiangyaSftCaseRecord:
    case_id: str
    image_path: str
    metadata: dict[str, Any]
    label: str
    raw_label: str
    dataset_name: str = "xiangya_sft"
    label_space_id: str = DEFAULT_XIANGYA_SFT_LABEL_SPACE_ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XiangyaSftDatasetSummary:
    data_root: str
    jsonl_path: str
    image_root: str
    sample_count: int
    multi_image_case_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_xiangya_sft_assets(data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT) -> XiangyaSftDatasetSummary:
    return _discover_xiangya_sft_assets_cached(str(Path(data_root)))


@lru_cache(maxsize=16)
def _discover_xiangya_sft_assets_cached(data_root: str) -> XiangyaSftDatasetSummary:
    root = Path(data_root)
    jsonl_path = root / "skin_xiangya.jsonl"
    image_root = root / "skin_merge"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Xiangya SFT jsonl not found: {jsonl_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Xiangya SFT image root not found: {image_root}")

    rows = load_xiangya_sft_rows(data_root)
    multi_image_case_count = sum(1 for row in rows if len(_extract_raw_image_paths(row)) > 1)
    return XiangyaSftDatasetSummary(
        data_root=str(root),
        jsonl_path=str(jsonl_path),
        image_root=str(image_root),
        sample_count=len(rows),
        multi_image_case_count=multi_image_case_count,
    )


def discover_xiangya_sft_case_source(data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT) -> CaseSourceConfig:
    root = Path(data_root)
    return CaseSourceConfig(
        data_root=root,
        metadata_path=root / "skin_xiangya.jsonl",
        image_root=root / "skin_merge",
        image_field="images",
        label_field="label",
        case_id_fields=["id"],
        metadata_format="jsonl_multimodal",
    )


def load_xiangya_sft_rows(data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT) -> list[dict[str, Any]]:
    return [dict(row) for row in _load_xiangya_sft_rows_cached(str(Path(data_root)))]


@lru_cache(maxsize=16)
def _load_xiangya_sft_rows_cached(data_root: str) -> tuple[dict[str, Any], ...]:
    root = Path(data_root)
    jsonl_path = root / "skin_xiangya.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Xiangya SFT jsonl not found: {jsonl_path}")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return tuple(rows)


def load_xiangya_sft_record_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT,
) -> XiangyaSftCaseRecord:
    rows = load_xiangya_sft_rows(data_root)
    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case-index {case_index} out of range for {len(rows)} rows")
    return standardize_xiangya_sft_row(rows[case_index], data_root=data_root)


def load_xiangya_sft_records(
    *,
    data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[XiangyaSftCaseRecord]:
    rows = load_xiangya_sft_rows(data_root)
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
    return [standardize_xiangya_sft_row(row, data_root=data_root) for row in ordered_rows]


def load_xiangya_sft_case_inputs(
    *,
    data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT,
    limit: int | None = None,
    offset: int = 0,
    sample_size: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[CaseInput]:
    records = load_xiangya_sft_records(
        data_root=data_root,
        limit=limit,
        offset=offset,
        sample_size=sample_size,
        seed=seed,
        shuffle=shuffle,
    )
    source_path = Path(data_root) / "skin_xiangya.jsonl"
    return [
        CaseInput(
            case_id=record.case_id,
            image_path=record.image_path,
            metadata=record.metadata,
            label=record.label,
            reference_label=record.label,
            dataset_name=record.dataset_name,
            label_space_id=record.label_space_id,
            source_metadata_path=str(source_path),
            workflow_context=ensure_workflow_context(
                dataset_name=record.dataset_name,
                metadata=record.metadata,
                label_space_id=record.label_space_id,
                workflow_context=None,
            ),
        )
        for record in records
    ]


def load_xiangya_sft_case_input_by_index(
    case_index: int,
    *,
    data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT,
) -> CaseInput:
    record = load_xiangya_sft_record_by_index(case_index, data_root=data_root)
    source_path = Path(data_root) / "skin_xiangya.jsonl"
    return CaseInput(
        case_id=record.case_id,
        image_path=record.image_path,
        metadata=record.metadata,
        label=record.label,
        reference_label=record.label,
        dataset_name=record.dataset_name,
        label_space_id=record.label_space_id,
        source_metadata_path=str(source_path),
        workflow_context=ensure_workflow_context(
            dataset_name=record.dataset_name,
            metadata=record.metadata,
            label_space_id=record.label_space_id,
            workflow_context=None,
        ),
    )


def standardize_xiangya_sft_row(
    row: dict[str, Any],
    *,
    data_root: str | Path = DEFAULT_XIANGYA_SFT_ROOT,
) -> XiangyaSftCaseRecord:
    root = Path(data_root)
    case_id = str(row.get("id", "")).strip()
    if not case_id:
        raise ValueError("Xiangya SFT row missing id")

    case_dir = _infer_case_dir(case_id)
    raw_image_paths = _extract_raw_image_paths(row)
    image_paths = _resolve_image_paths(
        raw_image_paths=raw_image_paths,
        data_root=root,
        case_dir=case_dir,
    )
    if not image_paths:
        raise FileNotFoundError(f"Xiangya SFT case `{case_id}` does not have a resolvable image path")

    assistant_message = _extract_message_content(row.get("messages", []), role="assistant")
    raw_label = _extract_raw_label(assistant_message=assistant_message)
    effective_label = _derive_grouped_label(
        raw_label=raw_label,
        assistant_message=assistant_message,
        raw_image_paths=raw_image_paths,
    )
    metadata = _build_safe_metadata(
        case_id=case_id,
        case_dir=case_dir,
        image_paths=image_paths,
        label=effective_label,
    )
    return XiangyaSftCaseRecord(
        case_id=case_id,
        image_path=image_paths[0],
        metadata=metadata,
        label=effective_label,
        raw_label=raw_label,
    )


def _extract_raw_image_paths(row: dict[str, Any]) -> list[str]:
    raw_paths = row.get("images", [])
    if not isinstance(raw_paths, list):
        return []
    return [str(item).strip() for item in raw_paths if str(item).strip()]


def _infer_case_dir(case_id: str) -> str:
    matched = re.search(r"case(\d+)", case_id, flags=re.IGNORECASE)
    if not matched:
        return case_id
    return matched.group(1)


def _resolve_image_paths(*, raw_image_paths: list[str], data_root: Path, case_dir: str) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_image_paths:
        path = _resolve_single_image_path(raw_path=raw_path, data_root=data_root, case_dir=case_dir)
        if not path.exists():
            raise FileNotFoundError(f"Xiangya SFT image not found: {path}")
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _resolve_single_image_path(*, raw_path: str, data_root: Path, case_dir: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    normalized = str(raw_path).replace("\\", "/").strip()
    if "skin_merge/" in normalized:
        relative = normalized.split("skin_merge/", 1)[1]
        remapped = data_root / "skin_merge" / relative
        if remapped.exists():
            return remapped

    fallback = data_root / "skin_merge" / case_dir / candidate.name
    if fallback.exists():
        return fallback

    matches = sorted((data_root / "skin_merge").rglob(candidate.name))
    if matches:
        return matches[0]
    return fallback


def _extract_message_content(messages: Any, *, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != role:
            continue
        return str(message.get("content", "")).strip()
    return ""


def _extract_raw_label(*, assistant_message: str) -> str:
    patterns = (
        r"诊断结果为([^。；\n]+)",
        r"最可能的诊断[为是:： ]+([^。；\n]+)",
        r"整体表现[^。；\n]*符合([^。；\n]+)",
        r"考虑为([^。；\n]+)",
        r"提示存在([^。；\n]+)的可能性",
    )
    for pattern in patterns:
        matched = re.search(pattern, assistant_message)
        if not matched:
            continue
        return _clean_diagnosis_snippet(matched.group(1))
    return ""


def _clean_diagnosis_snippet(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" 。；;，,：:")
    if "；" in cleaned:
        cleaned = cleaned.split("；", 1)[0].strip()
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0].strip()
    if "，" in cleaned:
        cleaned = cleaned.split("，", 1)[0].strip()
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    cleaned = re.sub(r"[)）]+$", "", cleaned).strip()
    return cleaned


def _derive_grouped_label(*, raw_label: str, assistant_message: str, raw_image_paths: list[str]) -> str:
    search_candidates = [raw_label, assistant_message]
    file_stems = " ".join(Path(item).stem for item in raw_image_paths if str(item).strip())
    if file_stems:
        search_candidates.append(file_stems)

    for candidate in search_candidates:
        if not candidate:
            continue
        canonical = canonicalize_label(
            candidate,
            dataset_name="xiangya_sft",
            label_space_id=DEFAULT_XIANGYA_SFT_LABEL_SPACE_ID,
        )
        if canonical:
            return canonical
    return "OTHER_INFLAMMATORY"


def _build_safe_metadata(*, case_id: str, case_dir: str, image_paths: list[str], label: str) -> dict[str, Any]:
    related_category = "APPENDAGE" if label == "HAIR_DISORDER" else "RASH"
    dominant_family = "eczematous_family" if label in {
        "CONTACT_DERMATITIS",
        "ATOPIC_DERMATITIS",
        "ECZEMA_DERMATITIS",
        "PERIORAL_DERMATITIS",
        "HERPETIC_ECZEMA",
    } else "other"
    return {
        "label_space_id": DEFAULT_XIANGYA_SFT_LABEL_SPACE_ID,
        "case_source": "xiangya_sft",
        "image_count": len(image_paths),
        "image_paths": list(image_paths),
        "related_category": related_category,
        "dominant_family": dominant_family,
        "task_type": "diagnosis_and_management",
        "language": "zh",
        "source_case_id": case_id,
        "source_case_dir": case_dir,
    }
