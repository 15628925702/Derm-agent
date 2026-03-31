from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.ham10000_aligned_loader import (
    DEFAULT_HAM10000_ROOT,
    discover_ham10000_aligned_assets,
    load_ham10000_aligned_records,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "ham10000_aligned_subset" / "inspection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HAM10000 aligned subset (mel/bcc/nv).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HAM10000_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets = discover_ham10000_aligned_assets(args.data_root)
    records = load_ham10000_aligned_records(
        data_root=args.data_root,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    matched_count = sum(1 for record in records if Path(record.image_path).exists())
    counts: dict[str, int] = {}
    for record in records:
        counts[record.original_label] = counts.get(record.original_label, 0) + 1

    payload = {
        "assets": assets.to_dict(),
        "request": {
            "limit": args.limit,
            "seed": args.seed,
            "shuffle": bool(args.shuffle),
        },
        "selected_sample_count": len(records),
        "original_label_counts": dict(sorted(counts.items(), key=lambda item: item[0])),
        "image_match": {
            "matched": matched_count,
            "missing": len(records) - matched_count,
        },
    }
    out_path = args.output_dir / "inspection_summary.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(out_path), "selected_sample_count": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
