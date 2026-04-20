from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.label_space import canonicalize_label, is_malignant_label, label_space_snapshot
from dataio.case_loader import load_case_by_index, resolve_registered_dataset_loader
from dataio.scin_loader import load_scin_case_input_by_index, load_scin_case_inputs, load_scin_records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_scin_records_and_case_inputs(tmp_path: Path) -> None:
    data_root = tmp_path / "official_mirror"
    _write_csv(
        data_root / "scin_cases.csv",
        [
            "case_id",
            "age_group",
            "sex_at_birth",
            "body_parts_arm",
            "condition_symptoms_itching",
            "related_category",
            "condition_duration",
            "image_1_path",
            "image_2_path",
            "image_3_path",
        ],
        [
            {
                "case_id": "case_001",
                "age_group": "AGE_ADULT",
                "sex_at_birth": "FEMALE",
                "body_parts_arm": "YES",
                "condition_symptoms_itching": "YES",
                "related_category": "RASH",
                "condition_duration": "ONE_WEEK",
                "image_1_path": "dataset/images/case_001.png",
                "image_2_path": "dataset/images/case_001_b.png",
                "image_3_path": "",
            },
            {
                "case_id": "case_002",
                "age_group": "AGE_ADULT",
                "sex_at_birth": "MALE",
                "body_parts_arm": "",
                "condition_symptoms_itching": "",
                "related_category": "LESION",
                "condition_duration": "ONE_MONTH",
                "image_1_path": "dataset/images/case_002.png",
                "image_2_path": "",
                "image_3_path": "",
            },
        ],
    )
    _write_csv(
        data_root / "scin_labels.csv",
        [
            "case_id",
            "dermatologist_skin_condition_on_label_name",
            "weighted_skin_condition_label",
        ],
        [
            {
                "case_id": "case_001",
                "dermatologist_skin_condition_on_label_name": "['Eczema']",
                "weighted_skin_condition_label": "{'Eczema': 0.6, 'Allergic Contact Dermatitis': 0.4}",
            },
            {
                "case_id": "case_002",
                "dermatologist_skin_condition_on_label_name": "['Basal Cell Carcinoma']",
                "weighted_skin_condition_label": "{'Basal Cell Carcinoma': 1.0}",
            },
        ],
    )
    image_dir = data_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "case_001.png").write_bytes(b"img1")
    (image_dir / "case_001_b.png").write_bytes(b"img2")
    (image_dir / "case_002.png").write_bytes(b"img3")

    records = load_scin_records(data_root=data_root)
    cases = load_scin_case_inputs(data_root=data_root)
    one_case = load_scin_case_input_by_index(1, data_root=data_root)

    assert len(records) == 2
    assert records[0].original_label == "Eczema"
    assert records[0].metadata["image_count"] == 2
    assert "weighted_skin_condition_label" not in records[0].metadata
    assert records[0].metadata["region"] == "arm"
    assert cases[0].dataset_name == "scin"
    assert cases[0].label_space_id == "scin_full"
    assert one_case.label == "Basal Cell Carcinoma"


def test_scin_case_loader_registration_and_label_space(tmp_path: Path) -> None:
    data_root = tmp_path / "official_mirror"
    _write_csv(
        data_root / "scin_cases.csv",
        ["case_id", "image_1_path"],
        [{"case_id": "case_001", "image_1_path": "dataset/images/case_001.png"}],
    )
    _write_csv(
        data_root / "scin_labels.csv",
        ["case_id", "weighted_skin_condition_label"],
        [{"case_id": "case_001", "weighted_skin_condition_label": "{'Basal Cell Carcinoma': 1.0}"}],
    )
    image_dir = data_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "case_001.png").write_bytes(b"img1")

    from dataio.case_loader import register_dataset_loader
    from dataio.scin_loader import load_scin_case_input_by_index, load_scin_case_inputs

    register_dataset_loader(
        "scin_test",
        load_by_index=load_scin_case_input_by_index,
        load_all=load_scin_case_inputs,
        data_root=data_root,
    )

    spec = resolve_registered_dataset_loader(data_root)
    assert spec is not None
    case = load_case_by_index(0, data_root)
    assert case.case_id == "case_001"

    assert canonicalize_label("Basal Cell Carcinoma", dataset_name="scin") == "BASAL CELL CARCINOMA"
    assert is_malignant_label("Basal Cell Carcinoma", dataset_name="scin") is True
    assert label_space_snapshot(dataset_name="scin")["label_space_id"] == "scin_full"


def test_scin_full_label_space_prefers_more_specific_full_labels() -> None:
    assert canonicalize_label("Acute dermatitis, NOS", dataset_name="scin") == "ACUTE DERMATITIS, NOS"
    assert canonicalize_label("Melanocytic Nevus", dataset_name="scin") == "MELANOCYTIC NEVUS"
    assert canonicalize_label("SCC/SCCIS", dataset_name="scin") == "SCC/SCCIS"
    assert is_malignant_label("SCC/SCCIS", dataset_name="scin") is True


def test_scin_loader_can_switch_to_grouped_label_space_via_env(tmp_path: Path) -> None:
    data_root = tmp_path / "official_mirror"
    _write_csv(
        data_root / "scin_cases.csv",
        ["case_id", "image_1_path"],
        [{"case_id": "case_001", "image_1_path": "dataset/images/case_001.png"}],
    )
    _write_csv(
        data_root / "scin_labels.csv",
        ["case_id", "weighted_skin_condition_label"],
        [{"case_id": "case_001", "weighted_skin_condition_label": "{'Eczema': 1.0}"}],
    )
    image_dir = data_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "case_001.png").write_bytes(b"img1")

    previous = os.environ.get("DERMAGENT_SCIN_LABEL_SPACE_ID")
    os.environ["DERMAGENT_SCIN_LABEL_SPACE_ID"] = "scin_grouped"
    try:
        case = load_scin_case_input_by_index(0, data_root=data_root)
    finally:
        if previous is None:
            os.environ.pop("DERMAGENT_SCIN_LABEL_SPACE_ID", None)
        else:
            os.environ["DERMAGENT_SCIN_LABEL_SPACE_ID"] = previous

    assert case.label_space_id == "scin_grouped"
    assert case.metadata["label_space_id"] == "scin_grouped"
