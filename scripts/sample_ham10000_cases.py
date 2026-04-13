from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.ham10000_loader import DEFAULT_HAM10000_ROOT, discover_ham10000_assets, load_ham10000_records


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "ham10000" / "adapter_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample standardized HAM10000 cases.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HAM10000_ROOT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets = discover_ham10000_assets(args.data_root)
    records = load_ham10000_records(
        data_root=args.data_root,
        limit=args.limit,
        offset=args.offset,
        sample_size=args.sample_size,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    payload = {
        "assets": assets.to_dict(),
        "request": {
            "limit": args.limit,
            "offset": args.offset,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "shuffle": bool(args.shuffle),
        },
        "records": [record.to_dict() for record in records],
    }
    output_path = args.output_dir / "sample_ham10000_cases.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "record_count": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
