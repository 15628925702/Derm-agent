from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import ISIC2019_SPLIT_ID, build_fixed_split_payload


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_fixed_split_payload_for_isic2019_uses_image_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_csv(
        data_root / "isic2019" / "ISIC_2019_Training_Metadata.csv",
        ["image", "age_approx", "anatom_site_general", "lesion_id", "sex"],
        [
            {"image": "ISIC_0001", "age_approx": "55", "anatom_site_general": "torso", "lesion_id": "", "sex": "female"},
            {"image": "ISIC_0002", "age_approx": "60", "anatom_site_general": "arm", "lesion_id": "", "sex": "male"},
            {"image": "ISIC_0003", "age_approx": "45", "anatom_site_general": "leg", "lesion_id": "", "sex": "female"},
            {"image": "ISIC_0004", "age_approx": "41", "anatom_site_general": "back", "lesion_id": "", "sex": "male"},
            {"image": "ISIC_0005", "age_approx": "36", "anatom_site_general": "scalp", "lesion_id": "", "sex": "female"},
            {"image": "ISIC_0006", "age_approx": "52", "anatom_site_general": "torso", "lesion_id": "", "sex": "female"},
            {"image": "ISIC_0007", "age_approx": "33", "anatom_site_general": "arm", "lesion_id": "", "sex": "male"},
            {"image": "ISIC_0008", "age_approx": "27", "anatom_site_general": "leg", "lesion_id": "", "sex": "female"},
            {"image": "ISIC_0009", "age_approx": "48", "anatom_site_general": "back", "lesion_id": "", "sex": "male"},
            {"image": "ISIC_0010", "age_approx": "59", "anatom_site_general": "torso", "lesion_id": "", "sex": "female"},
        ],
    )

    payload = build_fixed_split_payload(split_id=ISIC2019_SPLIT_ID, data_root=data_root)

    assert payload["dataset_name"] == "isic2019"
    assert payload["train"][0] == "ISIC_0001"
    assert payload["test"][-1] == "ISIC_0010"
