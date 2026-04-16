from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import HAM10000_SPLIT_ID, build_fixed_split_payload


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_fixed_split_payload_for_ham10000_uses_image_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_csv(
        data_root / "ham10000" / "HAM10000_metadata.csv",
        ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"],
        [
            {"lesion_id": "HAM_1", "image_id": "ISIC_0001", "dx": "bkl", "dx_type": "histo", "age": "80.0", "sex": "male", "localization": "scalp"},
            {"lesion_id": "HAM_2", "image_id": "ISIC_0002", "dx": "mel", "dx_type": "histo", "age": "61.0", "sex": "female", "localization": "back"},
            {"lesion_id": "HAM_3", "image_id": "ISIC_0003", "dx": "nv", "dx_type": "histo", "age": "45.0", "sex": "female", "localization": "leg"},
            {"lesion_id": "HAM_4", "image_id": "ISIC_0004", "dx": "bcc", "dx_type": "histo", "age": "54.0", "sex": "male", "localization": "arm"},
            {"lesion_id": "HAM_5", "image_id": "ISIC_0005", "dx": "df", "dx_type": "histo", "age": "30.0", "sex": "female", "localization": "foot"},
            {"lesion_id": "HAM_6", "image_id": "ISIC_0006", "dx": "vasc", "dx_type": "histo", "age": "23.0", "sex": "female", "localization": "trunk"},
            {"lesion_id": "HAM_7", "image_id": "ISIC_0007", "dx": "akiec", "dx_type": "histo", "age": "71.0", "sex": "male", "localization": "face"},
            {"lesion_id": "HAM_8", "image_id": "ISIC_0008", "dx": "nv", "dx_type": "histo", "age": "37.0", "sex": "female", "localization": "leg"},
            {"lesion_id": "HAM_9", "image_id": "ISIC_0009", "dx": "bcc", "dx_type": "histo", "age": "68.0", "sex": "male", "localization": "back"},
            {"lesion_id": "HAM_10", "image_id": "ISIC_0010", "dx": "bkl", "dx_type": "histo", "age": "59.0", "sex": "male", "localization": "scalp"},
        ],
    )

    payload = build_fixed_split_payload(split_id=HAM10000_SPLIT_ID, data_root=data_root)

    assert payload["dataset_name"] == "ham10000"
    assert payload["train"][0] == "ISIC_0001"
    assert payload["test"][-1] == "ISIC_0010"
