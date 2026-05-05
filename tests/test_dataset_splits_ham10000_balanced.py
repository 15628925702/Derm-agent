from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import HAM10000_BALANCED_SPLIT_ID, build_fixed_split_payload


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_balanced_ham10000_split_preserves_class_mixture(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    rows = []
    for index, dx in enumerate(["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"] * 4, start=1):
        rows.append(
            {
                "lesion_id": f"HAM_{index}",
                "image_id": f"ISIC_{index:04d}",
                "dx": dx,
                "dx_type": "histo",
                "age": "60.0",
                "sex": "male",
                "localization": "back",
            }
        )
    _write_csv(
        data_root / "ham10000" / "HAM10000_metadata.csv",
        ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"],
        rows,
    )

    payload = build_fixed_split_payload(split_id=HAM10000_BALANCED_SPLIT_ID, data_root=data_root)

    assert payload["dataset_name"] == "ham10000"
    assert payload["strategy"] == "stratified_by_dx_group"
    assert len(payload["test"]) > 0
    assert len(payload["test_case_indices"]) == len(payload["test"])
    test_case_ids = set(payload["test"])
    test_dx = Counter(row["dx"] for row in rows if row["image_id"] in test_case_ids)
    assert len(test_dx) > 1
    first_test_dx = [row["dx"] for row in rows if row["image_id"] in set(payload["test"][:7])]
    assert len(set(first_test_dx)) >= 4
