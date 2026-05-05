from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_6x6_runner_can_be_sourced_without_running_main() -> None:
    result = subprocess.run(
        [
            "bash",
            "-lc",
            "source scripts/run_6x6_matrix_parallel.sh; declare -F write_dataset_split_json >/dev/null",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Starting 6x6 matrix experiment" not in result.stdout
    assert "Starting 6x6 matrix experiment" not in result.stderr


def test_6x6_scin_split_filters_to_labeled_cases(tmp_path: Path) -> None:
    split_json = tmp_path / "scin_split.json"
    command = (
        "source scripts/run_6x6_matrix_parallel.sh; "
        f"write_dataset_split_json scin scin_contiguous_v1 {PROJECT_ROOT / 'data'} {split_json}"
    )
    subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT, check=True)

    payload = json.loads(split_json.read_text(encoding="utf-8"))

    assert payload["dataset_name"] == "scin"
    assert payload["strategy"].endswith("_labeled_only")
    assert payload["total_cases"] == 3061
    assert len(payload["train_case_indices"]) == 2157
    assert len(payload["val_case_indices"]) == 443
    assert len(payload["test_case_indices"]) == 461

    from dataio.scin_loader import load_scin_rows

    rows = load_scin_rows(PROJECT_ROOT / "data" / "scin")
    for split_name in ("train", "val", "test"):
        assert all(str(rows[idx].get("original_label", "")).strip() for idx in payload[f"{split_name}_case_indices"])


def test_6x6_sd198_split_and_grouped_loader_are_compatible(tmp_path: Path) -> None:
    split_json = tmp_path / "sd198_split.json"
    command = (
        "source scripts/run_6x6_matrix_parallel.sh; "
        f"write_dataset_split_json sd198 sd198_balanced_v1 {PROJECT_ROOT / 'data'} {split_json}"
    )
    subprocess.run(["bash", "-lc", command], cwd=PROJECT_ROOT, check=True)

    payload = json.loads(split_json.read_text(encoding="utf-8"))

    assert payload["dataset_name"] == "sd198"
    assert payload["strategy"] == "stratified_by_sd198_label"
    assert len(payload["train_case_indices"]) == len(payload["train"])
    assert len(payload["test_case_indices"]) == len(payload["test"])

    from dataio.case_loader import load_case_by_index

    old_value = os.environ.get("DERMAGENT_SD198_LABEL_SPACE_ID")
    os.environ["DERMAGENT_SD198_LABEL_SPACE_ID"] = "sd198_grouped"
    try:
        case = load_case_by_index(payload["train_case_indices"][0], PROJECT_ROOT / "data" / "sd198")
    finally:
        if old_value is None:
            os.environ.pop("DERMAGENT_SD198_LABEL_SPACE_ID", None)
        else:
            os.environ["DERMAGENT_SD198_LABEL_SPACE_ID"] = old_value

    assert case.dataset_name == "sd198"
    assert case.label_space_id == "sd198_grouped"
