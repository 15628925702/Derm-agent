from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.contamination_guard import normalize_split_name
from agent.policy_config import CURRENT_STABLE_POLICY_PATH
from cognition.cognition_state import CognitionState
from configs.dataset_splits import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_ID, build_fixed_split_payload, resolve_split_range
from memory.experience_store import ExperienceStore


DEFAULT_SPLIT_STATE_ROOT = Path("/root/DermAgent/state/split_states")


def resolve_split_state_root() -> Path:
    raw = str(os.getenv("DERMAGENT_SPLIT_STATE_ROOT", "")).strip()
    return Path(raw) if raw else DEFAULT_SPLIT_STATE_ROOT


def resolve_policy_path() -> Path:
    return Path(str(CURRENT_STABLE_POLICY_PATH))


@dataclass(frozen=True)
class CaseSelection:
    split_id: str
    split_json_path: str
    data_split: str
    selection_mode: str
    case_offset: int
    offset_within_split: int
    limit: int
    case_indices: list[int]
    case_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedStatePaths:
    data_split: str
    split_state_root: str
    experience_root: str
    cognition_path: str
    policy_path: str
    strict_frozen_eval: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_split_payload(
    *,
    split_json: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    split_id: str = DEFAULT_SPLIT_ID,
) -> tuple[dict[str, Any], Path | None]:
    if split_json is not None:
        payload = json.loads(split_json.read_text(encoding="utf-8"))
        return payload, split_json

    # Auto-infer split_id from data_root if using default split_id
    from configs.dataset_splits import infer_split_id_from_data_root
    if split_id == DEFAULT_SPLIT_ID:
        inferred_split_id = infer_split_id_from_data_root(data_root)
        split_id = inferred_split_id

    return build_fixed_split_payload(split_id=split_id, data_root=data_root), None


def resolve_case_selection(
    *,
    data_split: str,
    split_payload: dict[str, Any],
    split_json_path: Path | None = None,
    limit: int,
    case_offset: int | None = None,
    offset_within_split: int = 0,
    strict: bool = True,
    case_offset_mode: str = "auto",
) -> CaseSelection:
    normalized_split = normalize_split_name(data_split, default="test")
    if normalized_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported data split `{data_split}`.")

    split_case_ids = list(split_payload.get(normalized_split, []) or [])
    if not split_case_ids:
        raise ValueError(f"Split payload has no case ids for split `{normalized_split}`.")

    explicit_indices = split_payload.get(f"{normalized_split}_case_indices")
    if isinstance(explicit_indices, list) and explicit_indices:
        normalized_offset_mode = str(case_offset_mode).strip().lower() or "auto"
        if case_offset not in (None, 0):
            if normalized_offset_mode not in {"auto", "split_relative"}:
                raise ValueError(
                    f"Explicit case-index split only supports split-relative offsets, got case_offset_mode={case_offset_mode}."
                )
        local_offset = max(0, int(case_offset or offset_within_split))
        if local_offset >= len(explicit_indices):
            raise ValueError(
                f"offset_within_split={local_offset} is beyond available {normalized_split} cases={len(explicit_indices)}."
            )
        resolved_count = len(explicit_indices) - local_offset if limit <= 0 else min(int(limit), len(explicit_indices) - local_offset)
        case_indices = [int(item) for item in explicit_indices[local_offset : local_offset + resolved_count]]
        selected_case_ids = split_case_ids[local_offset : local_offset + resolved_count]
        if strict and len(selected_case_ids) != len(case_indices):
            raise ValueError(
                f"Resolved count mismatch for explicit-index split `{normalized_split}`: "
                f"indices={len(case_indices)} case_ids={len(selected_case_ids)}"
            )
        split_id = str(split_payload.get("split_id", DEFAULT_SPLIT_ID)).strip() or DEFAULT_SPLIT_ID
        return CaseSelection(
            split_id=split_id,
            split_json_path=str(split_json_path) if split_json_path else "",
            data_split=normalized_split,
            selection_mode="explicit_case_index_list",
            case_offset=local_offset,
            offset_within_split=local_offset,
            limit=len(case_indices),
            case_indices=case_indices,
            case_ids=selected_case_ids,
        )

    raw_range = split_payload.get(f"{normalized_split}_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ValueError(f"Split payload missing `{normalized_split}_range`.")
    split_start = int(raw_range[0])
    split_end = int(raw_range[1])

    normalized_offset_mode = str(case_offset_mode).strip().lower() or "auto"

    if case_offset is None:
        absolute_offset, resolved_count = resolve_split_range(
            split_payload,
            normalized_split,
            count=limit,
            offset_within_split=offset_within_split,
        )
        local_offset = absolute_offset - split_start
        selection_mode = "split_offset"
    else:
        requested_offset = int(case_offset)
        treat_as_absolute = normalized_offset_mode == "absolute"
        if normalized_offset_mode == "auto":
            treat_as_absolute = split_start <= requested_offset <= split_end

        if treat_as_absolute:
            absolute_offset = requested_offset
            if strict and not (split_start <= absolute_offset <= split_end):
                raise ValueError(
                    f"case_offset={absolute_offset} is outside split `{normalized_split}` range "
                    f"[{split_start}, {split_end}]."
                )
            local_offset = max(0, absolute_offset - split_start)
            absolute_offset, resolved_count = resolve_split_range(
                split_payload,
                normalized_split,
                count=limit,
                offset_within_split=local_offset,
            )
            selection_mode = "absolute_case_offset"
        elif normalized_offset_mode in {"auto", "split_relative"}:
            absolute_offset, resolved_count = resolve_split_range(
                split_payload,
                normalized_split,
                count=limit,
                offset_within_split=max(0, requested_offset),
            )
            local_offset = absolute_offset - split_start
            selection_mode = "split_offset"
        else:
            raise ValueError(f"Unsupported case_offset_mode `{case_offset_mode}`.")

    case_indices = list(range(absolute_offset, absolute_offset + resolved_count))
    selected_case_ids = split_case_ids[local_offset : local_offset + resolved_count]
    if strict and len(selected_case_ids) != resolved_count:
        raise ValueError(
            f"Resolved count mismatch for split `{normalized_split}`: "
            f"expected {resolved_count}, got {len(selected_case_ids)} case ids."
        )

    split_id = str(split_payload.get("split_id", DEFAULT_SPLIT_ID)).strip() or DEFAULT_SPLIT_ID
    return CaseSelection(
        split_id=split_id,
        split_json_path=str(split_json_path) if split_json_path else "",
        data_split=normalized_split,
        selection_mode=selection_mode,
        case_offset=absolute_offset,
        offset_within_split=local_offset,
        limit=resolved_count,
        case_indices=case_indices,
        case_ids=selected_case_ids,
    )


def resolve_split_state_paths(
    *,
    data_split: str,
    strict_frozen_eval: bool,
    split_state_root: Path | None = None,
    policy_path: Path | None = None,
) -> ResolvedStatePaths:
    split_state_root = split_state_root or resolve_split_state_root()
    normalized_split = normalize_split_name(data_split, default="global")
    target_root = split_state_root / normalized_split
    experience_root = target_root / "experience"
    cognition_path = target_root / "cognition_state.json"
    resolved_policy_path = policy_path or resolve_policy_path()

    if strict_frozen_eval:
        missing: list[str] = []
        if not experience_root.exists():
            missing.append(str(experience_root))
        if not cognition_path.exists():
            missing.append(str(cognition_path))
        if not resolved_policy_path.exists():
            missing.append(str(resolved_policy_path))
        if missing:
            raise FileNotFoundError(
                "Strict frozen eval requires split-aware state paths to exist. Missing: "
                + ", ".join(missing)
            )

    return ResolvedStatePaths(
        data_split=normalized_split,
        split_state_root=str(target_root),
        experience_root=str(experience_root),
        cognition_path=str(cognition_path),
        policy_path=str(resolved_policy_path),
        strict_frozen_eval=bool(strict_frozen_eval),
    )


def ensure_split_state_paths(
    *,
    data_split: str,
    split_state_root: Path | None = None,
    policy_path: Path | None = None,
) -> ResolvedStatePaths:
    split_state_root = split_state_root or resolve_split_state_root()
    normalized_split = normalize_split_name(data_split, default="global")
    target_root = split_state_root / normalized_split
    experience_root = target_root / "experience"
    cognition_path = target_root / "cognition_state.json"
    ExperienceStore(experience_root, split_name=normalized_split)
    cognition_state = CognitionState.load(cognition_path)
    cognition_state.state_split = normalized_split
    cognition_state.save(cognition_path)
    return resolve_split_state_paths(
        data_split=normalized_split,
        strict_frozen_eval=True,
        split_state_root=split_state_root,
        policy_path=policy_path,
    )


def validate_selected_case_ids(*, actual_case_ids: list[str], expected_case_ids: list[str], context: str) -> None:
    if list(actual_case_ids) != list(expected_case_ids):
        raise ValueError(
            f"{context} case ids do not match the resolved split selection. "
            f"expected_count={len(expected_case_ids)} actual_count={len(actual_case_ids)}"
        )


def build_experiment_state_manifest(
    *,
    case_selection: CaseSelection,
    state_paths: ResolvedStatePaths,
    writeback_enabled: bool,
    strict_frozen_eval: bool,
) -> dict[str, Any]:
    return {
        "data_split": case_selection.data_split,
        "split_id": case_selection.split_id,
        "split_json_path": case_selection.split_json_path,
        "strict_frozen_eval": bool(strict_frozen_eval),
        "writeback_enabled": bool(writeback_enabled),
        "case_selection": case_selection.to_dict(),
        "state_paths": state_paths.to_dict(),
    }
