from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.ham10000_loader import (
    load_ham10000_case_input_by_index,
    load_ham10000_case_inputs,
    load_ham10000_record_by_index,
    load_ham10000_records,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_ham10000_records_and_case_inputs(tmp_path: Path) -> None:
    data_root = tmp_path / "ham10000"
    _write_csv(
        data_root / "HAM10000_metadata.csv",
        ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"],
        [
            {"lesion_id": "HAM_1", "image_id": "ISIC_0001", "dx": "mel", "dx_type": "histo", "age": "80.0", "sex": "male", "localization": "scalp"},
            {"lesion_id": "HAM_2", "image_id": "ISIC_0002", "dx": "nv", "dx_type": "histo", "age": "61.0", "sex": "female", "localization": "back"},
        ],
    )
    part1 = data_root / "HAM10000_images_part_1"
    part2 = data_root / "HAM10000_images_part_2"
    part1.mkdir(parents=True, exist_ok=True)
    part2.mkdir(parents=True, exist_ok=True)
    (part1 / "ISIC_0001.jpg").write_bytes(b"img1")
    (part2 / "ISIC_0002.jpg").write_bytes(b"img2")

    records = load_ham10000_records(data_root=data_root)
    cases = load_ham10000_case_inputs(data_root=data_root)

    assert len(records) == 2
    assert records[0].binary_label == "malignant"
    assert records[1].binary_label == "benign"
    assert cases[0].label_space_id == "ham10000_full"
    assert cases[0].label == "mel"
    assert cases[1].label == "nv"

    one_record = load_ham10000_record_by_index(1, data_root=data_root)
    one_case = load_ham10000_case_input_by_index(1, data_root=data_root)

    assert one_record.case_id == "ISIC_0002"
    assert one_case.case_id == "ISIC_0002"
    assert one_case.label == "nv"
