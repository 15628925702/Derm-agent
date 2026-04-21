from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.case_loader import (
    get_registered_dataset_loader,
    list_registered_dataset_loaders,
    load_case_by_index,
    register_dataset_loader,
    resolve_registered_dataset_loader,
    sample_cases,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_builtin_loader_registry_exposes_known_datasets() -> None:
    specs = {spec.dataset_name: spec for spec in list_registered_dataset_loaders()}

    assert "ham10000" in specs
    assert "isic2019" in specs
    assert "sd198" in specs
    assert "xiangya_sft" in specs
    assert get_registered_dataset_loader("ham10000") is not None
    assert get_registered_dataset_loader("isic2019") is not None
    assert get_registered_dataset_loader("sd198") is not None
    assert get_registered_dataset_loader("xiangya_sft") is not None


def test_registered_loader_routes_by_root_and_passes_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "toy_dataset"
    image_dir = data_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "case_001.jpg").write_bytes(b"img")
    _write_csv(
        data_root / "toy.csv",
        ["image_name", "label", "age"],
        [{"image_name": "case_001.jpg", "label": "toy_label", "age": "42"}],
    )

    captured_roots: list[Path] = []

    def _load_by_index(case_index: int, *, data_root: str | Path) -> object:
        captured_roots.append(Path(data_root))
        return load_case_by_index_generic(case_index=case_index, data_root=data_root)

    def _load_all(*, data_root: str | Path) -> list[object]:
        captured_roots.append(Path(data_root))
        return [load_case_by_index_generic(case_index=0, data_root=data_root)]

    register_dataset_loader(
        "toy_dataset",
        load_by_index=_load_by_index,
        load_all=_load_all,
        data_root=data_root,
    )

    spec = resolve_registered_dataset_loader(data_root)
    assert spec is not None
    assert spec.dataset_name == "toy_dataset"

    case = load_case_by_index(0, data_root)
    sampled = sample_cases(1, data_root, seed=0)

    assert case.case_id == "toy_case_0"
    assert sampled[0].case_id == "toy_case_0"
    assert captured_roots
    assert all(path.resolve() == data_root.resolve() for path in captured_roots)


def load_case_by_index_generic(*, case_index: int, data_root: str | Path):
    root = Path(data_root)
    if case_index != 0:
        raise IndexError("toy dataset only has one case")
    return _toy_case(root)


def _toy_case(root: Path):
    from agent.state import CaseInput

    return CaseInput(
        case_id="toy_case_0",
        image_path=str(root / "images" / "case_001.jpg"),
        metadata={"age": "42"},
        label="toy_label",
        reference_label="toy_label",
        dataset_name="toy_dataset",
        label_space_id="toy_label_space",
        source_metadata_path=str(root / "toy.csv"),
    )
