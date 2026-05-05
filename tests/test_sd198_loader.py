from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.label_space import canonicalize_label, is_malignant_label, label_space_snapshot
from dataio.case_loader import discover_case_source, get_registered_dataset_loader, load_case_by_index
from dataio.sd198_loader import (
    discover_sd198_assets,
    discover_sd198_case_source,
    load_sd198_case_input_by_index,
    load_sd198_case_inputs,
    load_sd198_records,
)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_sd198_records_and_case_inputs(tmp_path: Path) -> None:
    data_root = tmp_path / "sd-198"
    _write_lines(
        data_root / "classes.txt",
        [
            "1 Basal_Cell_Carcinoma",
            "2 Acne_Vulgaris",
            "3 Malignant_Melanoma",
        ],
    )
    _write_lines(
        data_root / "images.txt",
        [
            "1 Basal_Cell_Carcinoma/case_001.jpg",
            "2 Acne_Vulgaris/case_002.jpg",
            "3 Malignant_Melanoma/case_003.jpg",
        ],
    )
    _write_lines(
        data_root / "image_class_labels.txt",
        [
            "1 1",
            "2 2",
            "3 3",
        ],
    )
    (data_root / "images" / "Basal_Cell_Carcinoma").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "Acne_Vulgaris").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "Malignant_Melanoma").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "Basal_Cell_Carcinoma" / "case_001.jpg").write_bytes(b"img1")
    (data_root / "images" / "Acne_Vulgaris" / "case_002.jpg").write_bytes(b"img2")
    (data_root / "images" / "Malignant_Melanoma" / "case_003.jpg").write_bytes(b"img3")

    assets = discover_sd198_assets(data_root=data_root)
    records = load_sd198_records(data_root=data_root)
    cases = load_sd198_case_inputs(data_root=data_root)
    one_case = load_sd198_case_input_by_index(2, data_root=data_root)

    assert assets.sample_count == 3
    assert assets.class_count == 3
    assert records[0].case_id == "sd198_000001"
    assert records[0].original_label == "Basal Cell Carcinoma"
    assert records[0].metadata["raw_label_name"] == "Basal_Cell_Carcinoma"
    assert records[0].metadata["label_space_id"] == "sd198_full"
    assert cases[1].dataset_name == "sd198"
    assert cases[1].label_space_id == "sd198_full"
    assert one_case.label == "Malignant Melanoma"


def test_sd198_case_loader_registration_and_label_space(tmp_path: Path) -> None:
    data_root = tmp_path / "sd-198"
    _write_lines(data_root / "classes.txt", ["1 Basal_Cell_Carcinoma"])
    _write_lines(data_root / "images.txt", ["1 Basal_Cell_Carcinoma/case_001.jpg"])
    _write_lines(data_root / "image_class_labels.txt", ["1 1"])
    (data_root / "images" / "Basal_Cell_Carcinoma").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "Basal_Cell_Carcinoma" / "case_001.jpg").write_bytes(b"img1")

    from dataio.case_loader import register_dataset_loader

    register_dataset_loader(
        "sd198_test",
        load_by_index=load_sd198_case_input_by_index,
        load_all=load_sd198_case_inputs,
        discover_source=discover_sd198_case_source,
        data_root=data_root,
    )

    assert get_registered_dataset_loader("sd198") is not None
    assert get_registered_dataset_loader("sd198_test") is not None
    source = discover_case_source(data_root)
    assert source.metadata_path.name == "images.txt"
    assert source.image_root.name == "images"
    case = load_case_by_index(0, data_root)
    assert case.case_id == "sd198_000001"

    assert canonicalize_label("Basal_Cell_Carcinoma", dataset_name="sd198") == "BASAL CELL CARCINOMA"
    assert canonicalize_label("Basal Cell Carcinoma", dataset_name="sd198") == "BASAL CELL CARCINOMA"
    assert is_malignant_label("Basal_Cell_Carcinoma", dataset_name="sd198") is True
    assert label_space_snapshot(dataset_name="sd198")["label_space_id"] == "sd198_full"


def test_sd198_loader_can_switch_to_grouped_label_space_via_env(tmp_path: Path) -> None:
    data_root = tmp_path / "sd-198"
    _write_lines(
        data_root / "classes.txt",
        [
            "1 Basal_Cell_Carcinoma",
            "2 Acne_Vulgaris",
        ],
    )
    _write_lines(
        data_root / "images.txt",
        [
            "1 Basal_Cell_Carcinoma/case_001.jpg",
            "2 Acne_Vulgaris/case_002.jpg",
        ],
    )
    _write_lines(
        data_root / "image_class_labels.txt",
        [
            "1 1",
            "2 2",
        ],
    )
    (data_root / "images" / "Basal_Cell_Carcinoma").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "Acne_Vulgaris").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "Basal_Cell_Carcinoma" / "case_001.jpg").write_bytes(b"img1")
    (data_root / "images" / "Acne_Vulgaris" / "case_002.jpg").write_bytes(b"img2")

    previous = os.environ.get("DERMAGENT_SD198_LABEL_SPACE_ID")
    os.environ["DERMAGENT_SD198_LABEL_SPACE_ID"] = "sd198_grouped"
    try:
        case = load_sd198_case_input_by_index(0, data_root=data_root)
    finally:
        if previous is None:
            os.environ.pop("DERMAGENT_SD198_LABEL_SPACE_ID", None)
        else:
            os.environ["DERMAGENT_SD198_LABEL_SPACE_ID"] = previous

    assert case.label_space_id == "sd198_grouped"
    assert case.label == "MALIGNANT_SKIN_CANCER"
    assert case.metadata["label_space_id"] == "sd198_grouped"
