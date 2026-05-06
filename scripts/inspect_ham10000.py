from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from project_paths import data_root, outputs_root

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_DATA_ROOT = data_root() / "ham10000"
DEFAULT_OUTPUT_DIR = outputs_root() / "external_eval" / "ham10000" / "inspection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HAM10000 dataset layout and metadata/image matching.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="HAM10000 dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for inspection artifacts.")
    parser.add_argument("--tree-depth", type=int, default=3, help="Max directory depth to summarize.")
    parser.add_argument("--missing-example-limit", type=int, default=20, help="How many missing image examples to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"HAM10000 root not found: {data_root}")

    tree_lines = build_directory_tree(data_root, max_depth=max(1, int(args.tree_depth)))
    csv_candidates = sorted(path for path in data_root.rglob("*.csv") if path.is_file())
    image_dirs = discover_image_dirs(data_root)
    duplicate_analysis = analyze_duplicate_image_dirs(image_dirs)
    preferred_image_dirs = choose_preferred_image_dirs(image_dirs, duplicate_analysis)
    chosen_csv = choose_metadata_csv(csv_candidates)

    rows = read_csv_rows(chosen_csv) if chosen_csv else []
    columns = list(rows[0].keys()) if rows else []

    image_index = build_image_index(preferred_image_dirs)
    label_field_candidates = infer_label_field_candidates(columns, rows)
    image_field_candidates = infer_image_field_candidates(columns, rows, image_index)
    chosen_label_field = label_field_candidates[0]["field"] if label_field_candidates else None
    chosen_image_field = image_field_candidates[0]["field"] if image_field_candidates else None

    label_distribution = compute_label_distribution(rows, chosen_label_field)
    image_match_summary = compute_image_match_summary(
        rows=rows,
        field=chosen_image_field,
        image_index=image_index,
        missing_example_limit=max(1, int(args.missing_example_limit)),
    )

    summary = {
        "data_root": str(data_root),
        "common_layout_detected": {
            "HAM10000_metadata_csv": (data_root / "HAM10000_metadata.csv").exists(),
            "HAM10000_images_part_1": (data_root / "HAM10000_images_part_1").exists(),
            "HAM10000_images_part_2": (data_root / "HAM10000_images_part_2").exists(),
            "ham10000_images_part_1": (data_root / "ham10000_images_part_1").exists(),
            "ham10000_images_part_2": (data_root / "ham10000_images_part_2").exists(),
        },
        "directory_tree_summary_path": str(output_dir / "directory_tree.txt"),
        "csv_candidates": [str(path) for path in csv_candidates],
        "chosen_metadata_csv": str(chosen_csv) if chosen_csv else "",
        "sample_count": len(rows),
        "columns": columns,
        "label_field_candidates": label_field_candidates,
        "chosen_label_field": chosen_label_field,
        "label_distribution": label_distribution,
        "image_directories": [serialize_image_dir(path) for path in image_dirs],
        "preferred_image_directories": [str(path) for path in preferred_image_dirs],
        "duplicate_directory_analysis": duplicate_analysis,
        "image_field_candidates": image_field_candidates,
        "chosen_image_field": chosen_image_field,
        "image_match_summary": image_match_summary,
    }

    (output_dir / "directory_tree.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    (output_dir / "inspection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "inspection_summary.md").write_text(render_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_directory_tree(root: Path, *, max_depth: int) -> list[str]:
    lines = [f"{root}"]
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * depth
        if path.is_dir():
            direct_files = sum(1 for child in path.iterdir() if child.is_file())
            direct_image_files = sum(
                1
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
            )
            lines.append(f"{indent}{path.name}/ [files={direct_files}, image_files={direct_image_files}]")
        else:
            lines.append(f"{indent}{path.name}")
    return lines


def discover_image_dirs(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        direct_images = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS]
        if direct_images:
            result.append(path)
    return result


def serialize_image_dir(path: Path) -> dict[str, Any]:
    files = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS]
    return {
        "path": str(path),
        "image_count": len(files),
        "sample_files": [child.name for child in sorted(files)[:5]],
    }


def choose_metadata_csv(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    exact = [path for path in candidates if path.name == "HAM10000_metadata.csv"]
    if exact:
        return exact[0]
    metadata_named = [path for path in candidates if "metadata" in path.name.lower()]
    if metadata_named:
        return sorted(metadata_named)[0]
    return candidates[0]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_label_field_candidates(columns: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for column in columns:
        lowered = column.lower()
        values = [str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()]
        unique_count = len(set(values))
        score = 0
        if any(token in lowered for token in ("dx", "diagn", "label", "class", "target")):
            score += 10
        if 1 < unique_count <= 32:
            score += 3
        if values and all(len(value) <= 16 for value in list(set(values))[:10]):
            score += 1
        if score > 0:
            candidates.append(
                {
                    "field": column,
                    "score": score,
                    "reason": f"name_hint={lowered}; unique_count={unique_count}",
                }
            )
    return sorted(candidates, key=lambda item: (item["score"], item["field"]), reverse=True)


def build_image_index(image_dirs: list[Path]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for directory in image_dirs:
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                index[child.name].append(str(child))
    return dict(index)


def infer_image_field_candidates(columns: list[str], rows: list[dict[str, Any]], image_index: dict[str, list[str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for column in columns:
        lowered = column.lower()
        values = [str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()]
        if not values:
            continue
        matched = 0
        extension_like = 0
        for value in values:
            if Path(value).suffix.lower() in IMAGE_EXTENSIONS:
                extension_like += 1
            if resolve_image_candidates(value, image_index):
                matched += 1
        score = 0
        if any(token in lowered for token in ("image", "img", "path", "file", "filename", "image_id", "img_id")):
            score += 8
        if matched:
            score += min(20, matched // max(1, len(values) // 10 or 1))
        if extension_like:
            score += 1
        if score > 0:
            candidates.append(
                {
                    "field": column,
                    "score": score,
                    "matched_rows": matched,
                    "non_empty_rows": len(values),
                    "reason": f"name_hint={lowered}; matched_rows={matched}; extension_like={extension_like}",
                }
            )
    return sorted(candidates, key=lambda item: (item["matched_rows"], item["score"], item["field"]), reverse=True)


def resolve_image_candidates(raw_value: str, image_index: dict[str, list[str]]) -> list[str]:
    value = str(raw_value).strip()
    if not value:
        return []
    basename = Path(value).name
    possibilities = [basename]
    stem = Path(basename).stem
    if stem and basename == stem:
        possibilities.extend(f"{stem}{ext}" for ext in IMAGE_EXTENSIONS)
    matches: list[str] = []
    for candidate in possibilities:
        matches.extend(image_index.get(candidate, []))
    return sorted(set(matches))


def compute_label_distribution(rows: list[dict[str, Any]], field: str | None) -> dict[str, int]:
    if not field:
        return {}
    counter = Counter(str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip())
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def compute_image_match_summary(
    *,
    rows: list[dict[str, Any]],
    field: str | None,
    image_index: dict[str, list[str]],
    missing_example_limit: int,
) -> dict[str, Any]:
    if not field:
        return {
            "matched_rows": 0,
            "missing_rows": len(rows),
            "missing_examples": [],
        }
    matched_rows = 0
    missing_examples: list[dict[str, Any]] = []
    for row in rows:
        raw_value = str(row.get(field, "")).strip()
        matches = resolve_image_candidates(raw_value, image_index)
        if matches:
            matched_rows += 1
        elif len(missing_examples) < missing_example_limit:
            missing_examples.append(
                {
                    "value": raw_value,
                    "row_preview": {key: row.get(key) for key in list(row.keys())[:6]},
                }
            )
    return {
        "matched_rows": matched_rows,
        "missing_rows": len(rows) - matched_rows,
        "missing_examples": missing_examples,
    }


def analyze_duplicate_image_dirs(image_dirs: list[Path]) -> list[dict[str, Any]]:
    dir_maps: dict[str, dict[str, int]] = {}
    for directory in image_dirs:
        file_map: dict[str, int] = {}
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                file_map[child.name] = child.stat().st_size
        dir_maps[str(directory)] = file_map

    analysis: list[dict[str, Any]] = []
    paths = [str(path) for path in image_dirs]
    for idx, left in enumerate(paths):
        for right in paths[idx + 1 :]:
            left_map = dir_maps[left]
            right_map = dir_maps[right]
            overlap = set(left_map).intersection(right_map)
            if not overlap:
                continue
            identical_overlap = sum(1 for name in overlap if left_map.get(name) == right_map.get(name))
            fully_identical = (
                len(left_map) == len(right_map)
                and set(left_map) == set(right_map)
                and identical_overlap == len(overlap)
            )
            analysis.append(
                {
                    "left_dir": left,
                    "right_dir": right,
                    "left_count": len(left_map),
                    "right_count": len(right_map),
                    "overlap_count": len(overlap),
                    "identical_overlap_count": identical_overlap,
                    "fully_identical": fully_identical,
                    "recommended_keep": recommend_dir_to_keep(Path(left), Path(right)),
                    "recommended_delete": recommend_dir_to_delete(Path(left), Path(right)),
                }
            )
    return analysis


def choose_preferred_image_dirs(image_dirs: list[Path], duplicate_analysis: list[dict[str, Any]]) -> list[Path]:
    duplicate_delete_set = {
        str(item["recommended_delete"])
        for item in duplicate_analysis
        if item.get("fully_identical")
    }
    preferred: list[Path] = []
    for path in image_dirs:
        if str(path) in duplicate_delete_set:
            continue
        preferred.append(path)
    return preferred


def recommend_dir_to_keep(left: Path, right: Path) -> str:
    preferred_tokens = ("HAM10000_images_part_1", "HAM10000_images_part_2")
    for token in preferred_tokens:
        if left.name == token:
            return str(left)
        if right.name == token:
            return str(right)
    return str(sorted([left, right])[0])


def recommend_dir_to_delete(left: Path, right: Path) -> str:
    keep = recommend_dir_to_keep(left, right)
    return str(right if str(left) == keep else left)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# HAM10000 Inspection", ""]
    lines.append(f"- data_root: `{summary['data_root']}`")
    lines.append(f"- chosen_metadata_csv: `{summary['chosen_metadata_csv']}`")
    lines.append(f"- sample_count: `{summary['sample_count']}`")
    lines.append(f"- chosen_label_field: `{summary['chosen_label_field']}`")
    lines.append(f"- chosen_image_field: `{summary['chosen_image_field']}`")
    image_match = summary["image_match_summary"]
    lines.append(f"- matched_rows: `{image_match['matched_rows']}`")
    lines.append(f"- missing_rows: `{image_match['missing_rows']}`")
    lines.append("")
    lines.append("## CSV Candidates")
    lines.append("")
    for item in summary["csv_candidates"]:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Image Directories")
    lines.append("")
    for item in summary["image_directories"]:
        lines.append(f"- `{item['path']}` image_count={item['image_count']}")
    lines.append("")
    lines.append("## Label Distribution")
    lines.append("")
    for label, count in summary["label_distribution"].items():
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append("## Duplicate Directory Analysis")
    lines.append("")
    for item in summary["duplicate_directory_analysis"]:
        lines.append(
            f"- `{item['left_dir']}` vs `{item['right_dir']}`: overlap={item['overlap_count']}, "
            f"identical_overlap={item['identical_overlap_count']}, fully_identical={item['fully_identical']}, "
            f"keep=`{item['recommended_keep']}`, delete=`{item['recommended_delete']}`"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
