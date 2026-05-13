from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.experiment_state import load_split_payload, resolve_case_selection
from agent.label_space import canonicalize_label
from agent.sd198_label_catalog import sd198_grouped_label_for_text
from dataio.case_loader import load_case_by_index
from dataio.ham10000_loader import load_ham10000_rows
from dataio.isic2019_loader import ISIC2019_PRIMARY_LABELS, load_isic2019_rows
from dataio.scin_loader import _extract_original_label as extract_scin_label
from dataio.scin_loader import load_scin_rows
from dataio.sd198_loader import SD198_RAW_TO_CANONICAL, load_sd198_rows


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "paper_data" / "final_dataset_splits_20260509"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RATIO = 0.30
DEFAULT_SEED = 20260509


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    case_index: int
    label: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    split_dataset_name: str
    data_root: Path
    metadata_path: Path
    label_space_id: str
    anchor_split_path: Path
    anchor_take: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final 30/70 dataset splits: 30% final_compare_test, "
            "70% final_experience_train. The current eval300 cases are pinned "
            "into final_compare_test before expanding by a label-aware blend."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--compare-ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Validate generated split files after writing.")
    return parser.parse_args()


def dataset_specs(data_root: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="isic2019",
            split_dataset_name="isic2019",
            data_root=data_root / "isic2019",
            metadata_path=data_root / "isic2019" / "ISIC_2019_Training_GroundTruth.csv",
            label_space_id="isic2019_full",
            anchor_split_path=PROJECT_ROOT
            / "paper_data/final_3x3_delta_pilot_20260507/splits/isic2019_final_3x3_delta_pilot_split.json",
        ),
        DatasetSpec(
            key="ham10000",
            split_dataset_name="ham10000",
            data_root=data_root / "ham10000",
            metadata_path=data_root / "ham10000" / "HAM10000_metadata.csv",
            label_space_id="ham10000_full",
            anchor_split_path=PROJECT_ROOT
            / "paper_data/final_6x5_continuous_20260507/splits/ham10000_final_6x5_continuous_split.json",
            anchor_take=300,
        ),
        DatasetSpec(
            key="sd198",
            split_dataset_name="sd198",
            data_root=data_root / "sd198" / "sd-198",
            metadata_path=data_root / "sd198" / "sd-198" / "images.txt",
            label_space_id="sd198_grouped",
            anchor_split_path=PROJECT_ROOT
            / "paper_data/final_6x5_continuous_20260507/splits/sd198_final_6x5_continuous_split.json",
            anchor_take=300,
        ),
        DatasetSpec(
            key="scin",
            split_dataset_name="scin",
            data_root=data_root / "scin",
            metadata_path=data_root / "scin" / "official_mirror" / "scin_cases.csv",
            label_space_id="scin_grouped",
            anchor_split_path=PROJECT_ROOT
            / "paper_data/final_3x3_delta_pilot_20260507/splits/scin_final_3x3_delta_pilot_split.json",
        ),
        DatasetSpec(
            key="pad20",
            split_dataset_name="pad_ufes_20",
            data_root=data_root / "pad_ufes_20",
            metadata_path=data_root / "pad_ufes_20" / "metadata.csv",
            label_space_id="derm_six",
            anchor_split_path=PROJECT_ROOT
            / "paper_data/final_3x3_delta_pilot_20260507/splits/pad20_final_3x3_delta_pilot_split.json",
        ),
    ]


def load_cases(spec: DatasetSpec) -> list[CaseRow]:
    if spec.key == "isic2019":
        cases: list[CaseRow] = []
        for index, row in enumerate(load_isic2019_rows(spec.data_root)):
            case_id = str(row.get("image", "")).strip()
            label = next((label for label in ISIC2019_PRIMARY_LABELS if str(row.get(label, "")).strip() == "1.0"), "UNK")
            cases.append(CaseRow(case_id=case_id, case_index=index, label=label))
        return cases

    if spec.key == "ham10000":
        return [
            CaseRow(
                case_id=str(row.get("image_id", "")).strip(),
                case_index=index,
                label=str(row.get("dx", "")).strip().lower() or "UNKNOWN",
            )
            for index, row in enumerate(load_ham10000_rows(spec.data_root))
        ]

    if spec.key == "sd198":
        cases = []
        for index, row in enumerate(load_sd198_rows(spec.data_root)):
            image_id = int(row.get("image_id", 0))
            base_label = SD198_RAW_TO_CANONICAL.get(
                str(row.get("raw_label", "")).strip(),
                str(row.get("original_label", "")).strip(),
            )
            cases.append(
                CaseRow(
                    case_id=f"sd198_{image_id:06d}",
                    case_index=index,
                    label=sd198_grouped_label_for_text(base_label) or "UNKNOWN",
                )
            )
        return cases

    if spec.key == "scin":
        cases = []
        for index, row in enumerate(load_scin_rows(spec.data_root)):
            raw_label = extract_scin_label(row)
            if not raw_label:
                continue
            cases.append(
                CaseRow(
                    case_id=str(row.get("case_id", "")).strip(),
                    case_index=index,
                    label=canonicalize_label(raw_label, label_space_id="scin_grouped") or raw_label,
                )
            )
        return cases

    if spec.key == "pad20":
        rows = list(csv.DictReader(spec.metadata_path.open("r", encoding="utf-8", newline="")))
        return [
            CaseRow(
                case_id=f"{str(row.get('patient_id', '')).strip()}_{str(row.get('lesion_id', '')).strip()}",
                case_index=index,
                label=str(row.get("diagnostic", "")).strip() or "UNKNOWN",
            )
            for index, row in enumerate(rows)
        ]

    raise KeyError(f"Unsupported dataset: {spec.key}")


def load_anchor_cases(spec: DatasetSpec) -> tuple[list[str], list[int]]:
    payload = json.loads(spec.anchor_split_path.read_text(encoding="utf-8"))
    test_ids = list(payload.get("test", []) or [])
    test_indices = list(payload.get("test_case_indices", []) or [])
    if not test_indices:
        raise ValueError(f"Anchor split has no explicit test_case_indices: {spec.anchor_split_path}")
    if spec.anchor_take is not None:
        test_ids = test_ids[: int(spec.anchor_take)]
        test_indices = test_indices[: int(spec.anchor_take)]
    if len(test_ids) != len(test_indices):
        raise ValueError(
            f"Anchor split id/index mismatch for {spec.anchor_split_path}: "
            f"ids={len(test_ids)} indices={len(test_indices)}"
        )
    return [str(item) for item in test_ids], [int(item) for item in test_indices]


def select_compare_cases(
    *,
    cases: list[CaseRow],
    anchor_case_ids: list[str],
    anchor_case_indices: list[int],
    compare_count: int,
    seed: int,
) -> tuple[list[CaseRow], list[CaseRow], dict[str, Any]]:
    by_index = {case.case_index: case for case in cases}
    missing_anchors = [case_index for case_index in anchor_case_indices if case_index not in by_index]
    if missing_anchors:
        raise ValueError(f"Anchor indices not found in dataset: {missing_anchors[:10]} (n={len(missing_anchors)})")

    for case_id, case_index in zip(anchor_case_ids, anchor_case_indices):
        actual = by_index[case_index].case_id
        if actual != case_id:
            raise ValueError(f"Anchor id/index mismatch at index {case_index}: split={case_id} actual={actual}")

    selected_indices = set(anchor_case_indices)
    selected_counts = Counter(by_index[case_index].label for case_index in anchor_case_indices)
    full_counts = Counter(case.label for case in cases)
    anchor_counts = Counter(by_index[case_index].label for case_index in anchor_case_indices)
    total = len(cases)
    anchor_total = max(1, len(anchor_case_ids))

    label_order = sorted(full_counts)
    natural_props = {label: full_counts[label] / total for label in label_order}
    anchor_props = {label: anchor_counts[label] / anchor_total for label in label_order}
    target_props = {label: 0.5 * natural_props[label] + 0.5 * anchor_props[label] for label in label_order}
    prop_sum = sum(target_props.values()) or 1.0
    target_props = {label: target_props[label] / prop_sum for label in label_order}

    # Keep the compare split risk-sensitive without emptying rare labels from the
    # experience split. Existing anchors may exceed the cap, and are always kept.
    per_label_caps = {
        label: max(anchor_counts[label], int(full_counts[label] * 0.50))
        for label in label_order
    }

    remaining_by_label: dict[str, list[CaseRow]] = defaultdict(list)
    for case in cases:
        if case.case_index not in selected_indices:
            remaining_by_label[case.label].append(case)

    rng = random.Random(seed)
    for label in label_order:
        rng.shuffle(remaining_by_label[label])

    while len(selected_indices) < compare_count:
        best_label = None
        best_deficit = float("-inf")
        for label in label_order:
            if not remaining_by_label[label]:
                continue
            if selected_counts[label] >= per_label_caps[label]:
                continue
            desired_so_far = target_props[label] * compare_count
            deficit = desired_so_far - selected_counts[label]
            if deficit > best_deficit:
                best_deficit = deficit
                best_label = label
        if best_label is None:
            # All caps reached; fill naturally from any remaining label.
            for label in label_order:
                if remaining_by_label[label]:
                    best_label = label
                    break
        if best_label is None:
            break
        case = remaining_by_label[best_label].pop()
        selected_indices.add(case.case_index)
        selected_counts[case.label] += 1

    if len(selected_indices) != compare_count:
        raise RuntimeError(f"Could only select {len(selected_indices)} compare cases; expected {compare_count}.")

    compare_cases = [case for case in cases if case.case_index in selected_indices]
    train_cases = [case for case in cases if case.case_index not in selected_indices]
    return compare_cases, train_cases, {
        "full_label_counts": dict(sorted(full_counts.items())),
        "anchor_label_counts": dict(sorted(anchor_counts.items())),
        "compare_label_counts": dict(sorted(Counter(case.label for case in compare_cases).items())),
        "experience_train_label_counts": dict(sorted(Counter(case.label for case in train_cases).items())),
        "per_label_compare_caps": dict(sorted(per_label_caps.items())),
    }


def build_split_payload(
    *,
    spec: DatasetSpec,
    cases: list[CaseRow],
    compare_cases: list[CaseRow],
    train_cases: list[CaseRow],
    anchor_case_ids: list[str],
    anchor_case_indices: list[int],
    compare_ratio: float,
    seed: int,
    label_summary: dict[str, Any],
) -> dict[str, Any]:
    compare_cases = order_cases_label_interleaved(compare_cases, seed=seed + 101)
    train_cases = order_cases_label_interleaved(train_cases, seed=seed + 202)
    compare_case_ids = [case.case_id for case in compare_cases]
    train_case_ids = [case.case_id for case in train_cases]
    compare_indices = [case.case_index for case in compare_cases]
    train_indices = [case.case_index for case in train_cases]
    compare_index_set = {case.case_index for case in compare_cases}
    return {
        "dataset_name": spec.split_dataset_name,
        "split_id": f"{spec.key}_final_30_70_anchor_preserving_v1",
        "split_version": "final_20260509",
        "strategy": "final_anchor_preserving_30_70_label_blend_v1",
        "seed": seed,
        "metadata_path": str(spec.metadata_path),
        "label_space_id": spec.label_space_id,
        "total_cases": len(cases),
        "source_available_cases": len(cases),
        "compare_ratio": compare_ratio,
        "experience_train_ratio": round(1.0 - compare_ratio, 6),
        "anchor_split_path": str(spec.anchor_split_path),
        "anchor_case_count": len(anchor_case_ids),
        "anchor_cases_retained_in_compare": sum(1 for case_index in anchor_case_indices if case_index in compare_index_set),
        "anchor_note": "Anchor retention is validated by explicit case indices; case_id values may repeat in PAD20.",
        "notes": (
            "Final personal-use split. `train` / `final_experience_train` is for experience-bank accumulation; "
            "`test` / `final_compare_test` is for held-out comparison. The current eval300 window is pinned "
            "into the compare split, then expanded with a 50/50 blend of current-anchor label mix and natural "
            "label distribution while leaving training support for rare labels."
        ),
        "train": train_case_ids,
        "val": [],
        "test": compare_case_ids,
        "final_experience_train": train_case_ids,
        "final_compare_test": compare_case_ids,
        "train_case_indices": train_indices,
        "val_case_indices": [],
        "test_case_indices": compare_indices,
        "final_experience_train_case_indices": train_indices,
        "final_compare_test_case_indices": compare_indices,
        "train_range": [0, len(train_case_ids) - 1] if train_case_ids else [0, -1],
        "val_range": [len(train_case_ids), len(train_case_ids) - 1],
        "test_range": [0, len(compare_case_ids) - 1],
        "label_summary": label_summary,
    }


def order_cases_label_interleaved(cases: list[CaseRow], *, seed: int) -> list[CaseRow]:
    """Deterministically interleave labels so prefix windows are useful smoke sets."""
    grouped: dict[str, list[CaseRow]] = defaultdict(list)
    for case in cases:
        grouped[case.label].append(case)

    rng = random.Random(seed)
    for label in grouped:
        rng.shuffle(grouped[label])

    total = len(cases)
    if total == 0:
        return []
    target_props = {label: len(items) / total for label, items in grouped.items()}
    used = Counter()
    ordered: list[CaseRow] = []
    labels = sorted(grouped)
    while len(ordered) < total:
        prefix_size = len(ordered) + 1
        best_label = None
        best_deficit = float("-inf")
        for label in labels:
            if not grouped[label]:
                continue
            deficit = target_props[label] * prefix_size - used[label]
            if deficit > best_deficit:
                best_deficit = deficit
                best_label = label
        if best_label is None:
            break
        case = grouped[best_label].pop()
        ordered.append(case)
        used[best_label] += 1
    return ordered


def write_markdown(summary: list[dict[str, Any]], output_root: Path, compare_ratio: float) -> None:
    lines = [
        "# Final 数据集划分清单",
        "",
        f"- 生成时间标识：`final_20260509`",
        f"- 统一比例：`final_compare_test={compare_ratio:.0%}`，`final_experience_train={1 - compare_ratio:.0%}`",
        "- 用途：个人项目验证与经验库积累，不按论文随机评估口径命名。",
        "- 规则：全量 case 只进入一个集合；当前 eval300 锚点全部保留在 `final_compare_test`。",
        "",
        "| Dataset | 全量 | final_compare_test | final_experience_train | anchor retained | split json |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        lines.append(
            "| {dataset} | {total} | {compare} | {train} | {anchor}/{anchor_total} | `{split}` |".format(
                dataset=item["dataset"],
                total=item["total_cases"],
                compare=item["compare_cases"],
                train=item["experience_train_cases"],
                anchor=item["anchor_cases_retained"],
                anchor_total=item["anchor_case_count"],
                split=item["split_json"],
            )
        )
    lines.extend(
        [
            "",
            "## 执行方式",
            "",
            "对比验证继续使用现有评估脚本，只是把 `--split-json` 换成对应 final split，`--data-split test` 指向 `final_compare_test`。",
            "",
            "示例：",
            "",
            "```bash",
            "DERMAGENT_DATA_ROOT=/data/gh/DermAgent/data \\",
            "/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python scripts/compare_agent_vs_qwen.py \\",
            "  --data-root /data/gh/DermAgent/data/isic2019 \\",
            "  --split-json /data/gh/DermAgent/paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json \\",
            "  --data-split test \\",
            "  --limit 300 --case-offset 0 \\",
            "  --output-dir paper_data/final_dataset_splits_20260509/smoke_runs/isic2019_example",
            "```",
            "",
            "经验库构建时使用同一个 split JSON 的 `train` / `final_experience_train`，不要从 `test` 写回经验。",
        ]
    )
    (output_root / "FINAL数据集划分清单.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test_split(split_path: Path, data_root: Path) -> dict[str, Any]:
    payload, resolved = load_split_payload(split_json=split_path, data_root=data_root)
    train_indices = list(payload.get("train_case_indices", []) or [])
    test_indices = list(payload.get("test_case_indices", []) or [])
    train_index_set = set(train_indices)
    test_index_set = set(test_indices)
    if len(train_indices) != len(train_index_set):
        raise ValueError(f"{split_path}: duplicate train case indices")
    if len(test_indices) != len(test_index_set):
        raise ValueError(f"{split_path}: duplicate test case indices")
    if train_index_set & test_index_set:
        raise ValueError(f"{split_path}: train/test overlap in case indices")
    if len(train_index_set | test_index_set) != int(payload.get("total_cases", 0)):
        raise ValueError(
            f"{split_path}: train+test index coverage does not equal total_cases="
            f"{payload.get('total_cases')}"
        )
    anchor_payload = json.loads(Path(str(payload["anchor_split_path"])).read_text(encoding="utf-8"))
    anchor_indices = list(anchor_payload.get("test_case_indices", []) or [])[: int(payload["anchor_case_count"])]
    missing_anchor_indices = [case_index for case_index in anchor_indices if int(case_index) not in test_index_set]
    if missing_anchor_indices:
        raise ValueError(f"{split_path}: anchor indices missing from test: {missing_anchor_indices[:10]}")

    test_selection = resolve_case_selection(
        data_split="test",
        split_payload=payload,
        split_json_path=resolved,
        limit=min(8, len(payload["test"])),
        case_offset=0,
        strict=True,
    )
    train_selection = resolve_case_selection(
        data_split="train",
        split_payload=payload,
        split_json_path=resolved,
        limit=min(8, len(payload["train"])),
        case_offset=0,
        strict=True,
    )
    for index, case_id in zip(test_selection.case_indices, test_selection.case_ids):
        actual = load_case_by_index(index, data_root=data_root)
        if actual.case_id != case_id:
            raise ValueError(f"{split_path}: test case mismatch {case_id} != {actual.case_id}")
    for index, case_id in zip(train_selection.case_indices, train_selection.case_ids):
        actual = load_case_by_index(index, data_root=data_root)
        if actual.case_id != case_id:
            raise ValueError(f"{split_path}: train case mismatch {case_id} != {actual.case_id}")
    return {
        "split_json": str(split_path),
        "test_probe_cases": len(test_selection.case_ids),
        "train_probe_cases": len(train_selection.case_ids),
        "train_test_index_overlap": 0,
        "covered_case_indices": len(train_index_set | test_index_set),
        "declared_total_cases": int(payload.get("total_cases", 0)),
        "anchor_indices_retained": len(anchor_indices),
    }


def main() -> int:
    args = parse_args()
    if not (0.0 < float(args.compare_ratio) < 1.0):
        raise ValueError("--compare-ratio must be between 0 and 1.")

    output_root = args.output_root
    splits_root = output_root / "splits"
    if not args.dry_run:
        splits_root.mkdir(parents=True, exist_ok=True)

    run_summary: list[dict[str, Any]] = []
    for spec in dataset_specs(args.data_root):
        cases = load_cases(spec)
        compare_count = round(len(cases) * float(args.compare_ratio))
        anchor_case_ids, anchor_case_indices = load_anchor_cases(spec)
        compare_cases, train_cases, label_summary = select_compare_cases(
            cases=cases,
            anchor_case_ids=anchor_case_ids,
            anchor_case_indices=anchor_case_indices,
            compare_count=compare_count,
            seed=int(args.seed),
        )
        payload = build_split_payload(
            spec=spec,
            cases=cases,
            compare_cases=compare_cases,
            train_cases=train_cases,
            anchor_case_ids=anchor_case_ids,
            anchor_case_indices=anchor_case_indices,
            compare_ratio=float(args.compare_ratio),
            seed=int(args.seed),
            label_summary=label_summary,
        )
        split_path = splits_root / f"{spec.key}_final_30_70_split.json"
        item = {
            "dataset": spec.key,
            "split_json": str(split_path.relative_to(PROJECT_ROOT)),
            "data_root": str(spec.data_root),
            "total_cases": len(cases),
            "compare_cases": len(compare_cases),
            "experience_train_cases": len(train_cases),
            "anchor_case_count": len(anchor_case_ids),
            "anchor_cases_retained": payload["anchor_cases_retained_in_compare"],
        }
        run_summary.append(item)
        if not args.dry_run:
            split_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.dry_run:
        (output_root / "manifest.json").write_text(
            json.dumps(
                {
                    "split_family": "final",
                    "split_version": "final_20260509",
                    "compare_ratio": float(args.compare_ratio),
                    "experience_train_ratio": round(1.0 - float(args.compare_ratio), 6),
                    "seed": int(args.seed),
                    "splits": run_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_markdown(run_summary, output_root, float(args.compare_ratio))

    if args.self_test and not args.dry_run:
        test_results = []
        for item in run_summary:
            split_path = PROJECT_ROOT / item["split_json"]
            test_results.append(self_test_split(split_path, Path(item["data_root"])))
        (output_root / "self_test_results.json").write_text(
            json.dumps(test_results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
