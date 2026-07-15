from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def find_first(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        return path
    return None


def ensure_target_layout(source_root: Path, target_root: Path) -> tuple[Path, Path]:
    jsonl_path = find_first(source_root, "skin_xiangya.jsonl")
    if jsonl_path is None:
        raise FileNotFoundError(f"Could not find skin_xiangya.jsonl under {source_root}")

    image_root = find_first(source_root, "skin_merge")
    if image_root is None or not image_root.is_dir():
        raise FileNotFoundError(f"Could not find skin_merge directory under {source_root}")

    target_root.mkdir(parents=True, exist_ok=True)
    target_jsonl = target_root / "skin_xiangya.jsonl"
    target_image_root = target_root / "skin_merge"

    if jsonl_path.resolve() != target_jsonl.resolve():
        shutil.copy2(jsonl_path, target_jsonl)

    if image_root.resolve() != target_image_root.resolve():
        if target_image_root.exists():
            shutil.rmtree(target_image_root)
        shutil.copytree(image_root, target_image_root)

    return target_jsonl, target_image_root


def summarize_jsonl(jsonl_path: Path) -> dict[str, int]:
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                total += 1
    return {"rows": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a Xiangya SFT dataset tree into DermAgent's expected layout.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()

    target_jsonl, target_image_root = ensure_target_layout(args.source_root.resolve(), args.target_root.resolve())
    summary = summarize_jsonl(target_jsonl)

    print(
        json.dumps(
            {
                "target_root": str(args.target_root.resolve()),
                "jsonl_path": str(target_jsonl),
                "image_root": str(target_image_root),
                "rows": summary["rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
