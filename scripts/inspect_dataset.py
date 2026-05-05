from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
METADATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".tsv", ".parquet", ".xlsx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dataset structure without assuming a fixed schema.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--max-tree-depth", type=int, default=3, help="Maximum depth for the printed directory tree.")
    parser.add_argument("--top-k-labels", type=int, default=20, help="How many label values to show per candidate field.")
    return parser.parse_args()


def build_directory_tree(root: Path, max_depth: int) -> list[str]:
    lines = [f"{root.name}/"]

    def walk(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(current.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
        for index, entry in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "│   "
                walk(entry, prefix + extension, depth + 1)

    walk(root, "", 1)
    return lines


def scan_dataset(root: Path) -> dict[str, Any]:
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    image_files = [path for path in all_files if path.suffix.lower() in IMAGE_EXTENSIONS]
    metadata_files = [path for path in all_files if path.suffix.lower() in METADATA_EXTENSIONS]

    return {
        "data_root": root,
        "all_files": all_files,
        "image_files": image_files,
        "metadata_files": metadata_files,
    }


def inspect_csv_metadata(path: Path, image_files: list[Path], top_k_labels: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = list(rows[0].keys()) if rows else []
    image_file_names = {image_path.name for image_path in image_files}

    image_path_candidates = []
    label_field_candidates = []

    for field in fieldnames:
        values = [row.get(field, "") for row in rows]
        non_empty_values = [value for value in values if value not in ("", None)]
        unique_non_empty = sorted({str(value) for value in non_empty_values})
        unique_count = len(unique_non_empty)

        if non_empty_values:
            matched_images = sum(1 for value in non_empty_values if str(value) in image_file_names)
            extension_like = sum(1 for value in non_empty_values if Path(str(value)).suffix.lower() in IMAGE_EXTENSIONS)
            if matched_images > 0 or extension_like > 0 or "img" in field.lower() or "image" in field.lower():
                image_path_candidates.append(
                    {
                        "field": field,
                        "non_empty_count": len(non_empty_values),
                        "matched_image_files": matched_images,
                        "extension_like_count": extension_like,
                    }
                )

        lowered = field.lower()
        looks_like_label_name = any(token in lowered for token in ("label", "class", "target", "diagn", "dx"))
        distribution = Counter(str(value) for value in non_empty_values)
        reasonable_label_cardinality = 1 < unique_count <= min(50, max(2, len(rows) // 5 if rows else 50))
        if looks_like_label_name or reasonable_label_cardinality:
            label_field_candidates.append(
                {
                    "field": field,
                    "unique_count": unique_count,
                    "top_values": distribution.most_common(top_k_labels),
                    "looks_like_label_name": looks_like_label_name,
                }
            )

    label_field_candidates.sort(
        key=lambda item: (
            1 if item["looks_like_label_name"] else 0,
            -item["unique_count"],
        ),
        reverse=True,
    )
    image_path_candidates.sort(key=lambda item: (item["matched_image_files"], item["extension_like_count"]), reverse=True)

    return {
        "path": str(path),
        "sample_count": len(rows),
        "columns": fieldnames,
        "label_field_candidates": label_field_candidates,
        "image_path_field_candidates": image_path_candidates,
    }


def format_report(scan_result: dict[str, Any], metadata_reports: list[dict[str, Any]], max_tree_depth: int) -> str:
    tree_lines = build_directory_tree(scan_result["data_root"], max_tree_depth)
    lines: list[str] = []

    lines.append("== Data Directory Tree ==")
    lines.extend(tree_lines)
    lines.append("")

    lines.append("== File Summary ==")
    lines.append(f"Data root: {scan_result['data_root']}")
    lines.append(f"Total files: {len(scan_result['all_files'])}")
    lines.append(f"Image files: {len(scan_result['image_files'])}")
    lines.append(f"Metadata files: {len(scan_result['metadata_files'])}")
    lines.append("Possible data entry files:")
    for path in scan_result["metadata_files"]:
        lines.append(f"- {path}")
    lines.append("Possible image roots:")
    for image_root in sorted({path.parent for path in scan_result["image_files"]}):
        lines.append(f"- {image_root}")
    lines.append("")

    if not metadata_reports:
        lines.append("== Metadata Inspection ==")
        lines.append("No inspectable metadata files found.")
        return "\n".join(lines)

    for report in metadata_reports:
        lines.append(f"== Metadata Inspection: {report['path']} ==")
        lines.append(f"Sample count: {report['sample_count']}")
        lines.append(f"Metadata columns ({len(report['columns'])}): {', '.join(report['columns'])}")
        lines.append("Label field candidates and distributions:")
        if report["label_field_candidates"]:
            for candidate in report["label_field_candidates"]:
                lines.append(
                    f"- {candidate['field']} | unique={candidate['unique_count']} | "
                    f"name_hint={candidate['looks_like_label_name']}"
                )
                top_values = ", ".join(f"{label}:{count}" for label, count in candidate["top_values"])
                lines.append(f"  top values: {top_values}")
        else:
            lines.append("- none detected")

        lines.append("Image path field candidates:")
        if report["image_path_field_candidates"]:
            for candidate in report["image_path_field_candidates"]:
                lines.append(
                    f"- {candidate['field']} | non_empty={candidate['non_empty_count']} | "
                    f"matched_images={candidate['matched_image_files']} | "
                    f"extension_like={candidate['extension_like_count']}"
                )
        else:
            lines.append("- none detected")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    scan_result = scan_dataset(args.data_root)

    metadata_reports: list[dict[str, Any]] = []
    for metadata_path in scan_result["metadata_files"]:
        if metadata_path.suffix.lower() == ".csv":
            metadata_reports.append(
                inspect_csv_metadata(metadata_path, scan_result["image_files"], args.top_k_labels)
            )

    report = format_report(scan_result, metadata_reports, args.max_tree_depth)
    print(report)
    print("== JSON Summary ==")
    print(
        json.dumps(
            {
                "data_root": str(scan_result["data_root"]),
                "total_files": len(scan_result["all_files"]),
                "image_files": len(scan_result["image_files"]),
                "metadata_files": [str(path) for path in scan_result["metadata_files"]],
                "metadata_reports": metadata_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
