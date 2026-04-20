from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import SD198_SPLIT_ID, build_fixed_split_payload


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_fixed_split_payload_for_sd198_uses_sd198_case_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    dataset_root = data_root / "sd198" / "sd-198"
    _write_lines(
        dataset_root / "classes.txt",
        [
            "1 Basal_Cell_Carcinoma",
            "2 Acne_Vulgaris",
        ],
    )
    _write_lines(
        dataset_root / "images.txt",
        [
            "1 Basal_Cell_Carcinoma/case_001.jpg",
            "2 Acne_Vulgaris/case_002.jpg",
            "3 Basal_Cell_Carcinoma/case_003.jpg",
            "4 Acne_Vulgaris/case_004.jpg",
            "5 Basal_Cell_Carcinoma/case_005.jpg",
            "6 Acne_Vulgaris/case_006.jpg",
            "7 Basal_Cell_Carcinoma/case_007.jpg",
            "8 Acne_Vulgaris/case_008.jpg",
            "9 Basal_Cell_Carcinoma/case_009.jpg",
            "10 Acne_Vulgaris/case_010.jpg",
        ],
    )
    _write_lines(
        dataset_root / "image_class_labels.txt",
        [
            "1 1",
            "2 2",
            "3 1",
            "4 2",
            "5 1",
            "6 2",
            "7 1",
            "8 2",
            "9 1",
            "10 2",
        ],
    )
    (dataset_root / "images" / "Basal_Cell_Carcinoma").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "Acne_Vulgaris").mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        class_dir = "Basal_Cell_Carcinoma" if i % 2 == 1 else "Acne_Vulgaris"
        (dataset_root / "images" / class_dir / f"case_{i:03d}.jpg").write_bytes(b"img")

    payload = build_fixed_split_payload(split_id=SD198_SPLIT_ID, data_root=data_root)

    assert payload["dataset_name"] == "sd198"
    assert payload["train"][0] == "sd198_000001"
    assert payload["test"][-1] == "sd198_000010"
    assert payload["train_case_indices"][0] == 0
    assert payload["test_case_indices"][-1] == 9
