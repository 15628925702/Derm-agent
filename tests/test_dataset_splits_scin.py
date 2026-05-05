from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import SCIN_SPLIT_ID, build_fixed_split_payload


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_fixed_split_payload_for_scin_uses_case_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_csv(
        data_root / "scin" / "official_mirror" / "scin_cases.csv",
        ["case_id", "image_1_path", "related_category"],
        [
            {"case_id": "case_001", "image_1_path": "dataset/images/case_001.png", "related_category": "RASH"},
            {"case_id": "case_002", "image_1_path": "dataset/images/case_002.png", "related_category": "RASH"},
            {"case_id": "case_003", "image_1_path": "dataset/images/case_003.png", "related_category": "LESION"},
            {"case_id": "case_004", "image_1_path": "dataset/images/case_004.png", "related_category": "LESION"},
            {"case_id": "case_005", "image_1_path": "dataset/images/case_005.png", "related_category": "RASH"},
            {"case_id": "case_006", "image_1_path": "dataset/images/case_006.png", "related_category": "LESION"},
            {"case_id": "case_007", "image_1_path": "dataset/images/case_007.png", "related_category": "RASH"},
            {"case_id": "case_008", "image_1_path": "dataset/images/case_008.png", "related_category": "LESION"},
            {"case_id": "case_009", "image_1_path": "dataset/images/case_009.png", "related_category": "RASH"},
            {"case_id": "case_010", "image_1_path": "dataset/images/case_010.png", "related_category": "LESION"},
        ],
    )

    payload = build_fixed_split_payload(split_id=SCIN_SPLIT_ID, data_root=data_root)

    assert payload["dataset_name"] == "scin"
    assert payload["train"][0] == "case_001"
    assert payload["test"][-1] == "case_010"
