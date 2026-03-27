from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import build_fixed_split_payload, resolve_split_range
from configs.run_profiles import get_run_profile


def _write_metadata(path: Path, total_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "lesion_id", "img_id", "diagnostic"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(total_rows):
            writer.writerow(
                {
                    "patient_id": f"PAT_{index}",
                    "lesion_id": f"{1000 + index}",
                    "img_id": f"PAT_{index}_{1000 + index}.png",
                    "diagnostic": "NEV",
                }
            )


def test_build_fixed_split_payload_is_deterministic_and_contiguous(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    metadata_path = data_root / "pad_ufes_20" / "metadata.csv"
    _write_metadata(metadata_path, total_rows=10)

    payload = build_fixed_split_payload(data_root=data_root)

    assert payload["total_cases"] == 10
    assert payload["train_range"] == [0, 6]
    assert payload["val_range"] == [7, 7]
    assert payload["test_range"] == [8, 9]
    assert payload["train"][0] == "PAT_0_1000"
    assert payload["test"][-1] == "PAT_9_1009"


def test_resolve_split_range_caps_count_inside_split(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    metadata_path = data_root / "pad_ufes_20" / "metadata.csv"
    _write_metadata(metadata_path, total_rows=20)
    payload = build_fixed_split_payload(data_root=data_root)

    start_index, count = resolve_split_range(payload, "val", count=10)

    assert start_index == payload["val_range"][0]
    assert count == payload["val_range"][1] - payload["val_range"][0] + 1


def test_get_run_profile_returns_quick4h_defaults() -> None:
    profile = get_run_profile("quick4h_v1")

    assert profile.train_seed_cases == 24
    assert profile.stage_training_limit == 24
    assert profile.val_compare_cases == 8


def test_get_run_profile_returns_medium_signal_no_ablation_defaults() -> None:
    profile = get_run_profile("medium_signal_no_ablation_v1")

    assert profile.train_seed_cases == 96
    assert profile.stage_training_limit == 96
    assert profile.val_compare_cases == 32
    assert profile.test_eval_cases == 32
    assert profile.include_ablations is False
