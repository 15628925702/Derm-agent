from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.experiment_state import (
    build_experiment_state_manifest,
    ensure_split_state_paths,
    resolve_case_selection,
    resolve_split_state_paths,
)


def test_resolve_case_selection_is_split_correct() -> None:
    split_payload = {
        "split_id": "toy_split_v1",
        "train_range": [0, 2],
        "val_range": [3, 4],
        "test_range": [5, 6],
        "train": ["train_0", "train_1", "train_2"],
        "val": ["val_0", "val_1"],
        "test": ["test_0", "test_1"],
    }

    selection = resolve_case_selection(
        data_split="val",
        split_payload=split_payload,
        limit=2,
        case_offset=3,
        strict=True,
    )

    assert selection.data_split == "val"
    assert selection.case_indices == [3, 4]
    assert selection.case_ids == ["val_0", "val_1"]
    assert selection.selection_mode == "absolute_case_offset"


def test_resolve_case_selection_auto_mode_uses_split_relative_offset_when_absolute_is_outside_split() -> None:
    split_payload = {
        "split_id": "toy_split_v1",
        "train_range": [0, 2],
        "val_range": [3, 4],
        "test_range": [5, 6],
        "train": ["train_0", "train_1", "train_2"],
        "val": ["val_0", "val_1"],
        "test": ["test_0", "test_1"],
    }

    selection = resolve_case_selection(
        data_split="test",
        split_payload=split_payload,
        limit=1,
        case_offset=0,
        strict=True,
    )

    assert selection.data_split == "test"
    assert selection.case_indices == [5]
    assert selection.case_ids == ["test_0"]
    assert selection.selection_mode == "split_offset"


def test_resolve_case_selection_supports_explicit_case_index_lists() -> None:
    split_payload = {
        "split_id": "toy_balanced_v1",
        "train": ["train_0", "train_1", "train_2"],
        "val": ["val_0", "val_1"],
        "test": ["test_0", "test_1"],
        "train_case_indices": [10, 20, 30],
        "val_case_indices": [40, 50],
        "test_case_indices": [60, 70],
    }

    selection = resolve_case_selection(
        data_split="test",
        split_payload=split_payload,
        limit=1,
        case_offset=0,
        strict=True,
    )

    assert selection.data_split == "test"
    assert selection.case_indices == [60]
    assert selection.case_ids == ["test_0"]
    assert selection.selection_mode == "explicit_case_index_list"


def test_resolve_split_state_paths_requires_split_specific_state_when_strict(tmp_path: Path) -> None:
    split_root = tmp_path / "split_states"
    test_root = split_root / "test"
    (test_root / "experience").mkdir(parents=True)
    (test_root / "cognition_state.json").write_text("{}", encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")

    resolved = resolve_split_state_paths(
        data_split="test",
        strict_frozen_eval=True,
        split_state_root=split_root,
        policy_path=policy_path,
    )

    assert resolved.data_split == "test"
    assert resolved.strict_frozen_eval is True
    assert resolved.experience_root.endswith("/split_states/test/experience")


def test_resolve_split_state_paths_fails_when_strict_state_missing(tmp_path: Path) -> None:
    split_root = tmp_path / "split_states"
    with pytest.raises(FileNotFoundError):
        resolve_split_state_paths(
            data_split="val",
            strict_frozen_eval=True,
            split_state_root=split_root,
            policy_path=tmp_path / "missing_policy.json",
        )


def test_build_experiment_state_manifest_preserves_split_and_writeback_flags(tmp_path: Path) -> None:
    split_payload = {
        "split_id": "toy_split_v1",
        "train_range": [0, 2],
        "val_range": [3, 4],
        "test_range": [5, 6],
        "train": ["train_0", "train_1", "train_2"],
        "val": ["val_0", "val_1"],
        "test": ["test_0", "test_1"],
    }
    selection = resolve_case_selection(data_split="test", split_payload=split_payload, limit=1, case_offset=5, strict=True)
    split_root = tmp_path / "split_states"
    test_root = split_root / "test"
    (test_root / "experience").mkdir(parents=True)
    (test_root / "cognition_state.json").write_text(json.dumps({"state_split": "test"}), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")
    state_paths = resolve_split_state_paths(
        data_split="test",
        strict_frozen_eval=True,
        split_state_root=split_root,
        policy_path=policy_path,
    )

    manifest = build_experiment_state_manifest(
        case_selection=selection,
        state_paths=state_paths,
        writeback_enabled=False,
        strict_frozen_eval=True,
    )

    assert manifest["data_split"] == "test"
    assert manifest["writeback_enabled"] is False
    assert manifest["strict_frozen_eval"] is True
    assert manifest["case_selection"]["case_ids"] == ["test_0"]


def test_ensure_split_state_paths_bootstraps_empty_split_state(tmp_path: Path) -> None:
    split_root = tmp_path / "split_states"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")

    resolved = ensure_split_state_paths(
        data_split="test",
        split_state_root=split_root,
        policy_path=policy_path,
    )

    assert Path(resolved.experience_root).exists()
    assert Path(resolved.cognition_path).exists()
    manifest = json.loads((Path(resolved.experience_root) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state_partition"]["split_name"] == "test"
