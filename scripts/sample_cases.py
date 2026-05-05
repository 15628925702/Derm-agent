from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.case_loader import discover_case_source, sample_cases


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample standardized cases from the discovered dataset source.")
    parser.add_argument("--count", type=int, default=3, help="Number of cases to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = discover_case_source(args.data_root)
    cases = sample_cases(count=args.count, data_root=args.data_root, seed=args.seed)

    payload = {
        "source_config": config.to_dict(),
        "sample_count": len(cases),
        "cases": [case.to_dict() for case in cases],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
