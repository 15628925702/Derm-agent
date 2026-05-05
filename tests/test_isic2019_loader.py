from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.isic2019_loader import load_isic2019_case_inputs, load_isic2019_records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_isic2019_records_and_case_inputs(tmp_path: Path) -> None:
    data_root = tmp_path / "isic2019"
    _write_csv(
        data_root / "ISIC_2019_Training_Metadata.csv",
        ["image", "age_approx", "anatom_site_general", "lesion_id", "sex"],
        [
            {"image": "ISIC_0001", "age_approx": "55", "anatom_site_general": "torso", "lesion_id": "", "sex": "female"},
            {"image": "ISIC_0002", "age_approx": "60", "anatom_site_general": "arm", "lesion_id": "", "sex": "male"},
        ],
    )
    _write_csv(
        data_root / "ISIC_2019_Training_GroundTruth.csv",
        ["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"],
        [
            {"image": "ISIC_0001", "MEL": "1.0", "NV": "0.0", "BCC": "0.0", "AK": "0.0", "BKL": "0.0", "DF": "0.0", "VASC": "0.0", "SCC": "0.0", "UNK": "0.0"},
            {"image": "ISIC_0002", "MEL": "0.0", "NV": "1.0", "BCC": "0.0", "AK": "0.0", "BKL": "0.0", "DF": "0.0", "VASC": "0.0", "SCC": "0.0", "UNK": "0.0"},
        ],
    )
    image_dir = data_root / "images" / "ISIC_2019_Training_Input"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "ISIC_0001.jpg").write_bytes(b"img1")
    (image_dir / "ISIC_0002.jpg").write_bytes(b"img2")

    records = load_isic2019_records(data_root=data_root)
    cases = load_isic2019_case_inputs(data_root=data_root)

    assert len(records) == 2
    assert records[0].original_label == "MEL"
    assert records[1].original_label == "NV"
    assert cases[0].label_space_id == "isic2019_full"
    assert cases[0].label == "MEL"
    assert cases[1].label == "NV"
