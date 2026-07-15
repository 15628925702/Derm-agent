from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.auditability_scoring import extract_tar_subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a lightweight auditability subset from a large DermAgent tar.gz export.")
    parser.add_argument("--tar-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--system-segment", default="final-boostrap")
    parser.add_argument("--dataset-segment")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    manifest = extract_tar_subset(
        tar_path=args.tar_path.resolve(),
        output_root=args.output_root.resolve(),
        system_segment=str(args.system_segment).strip(),
        dataset_segment=str(args.dataset_segment).strip() if args.dataset_segment else None,
        limit=max(int(args.limit), 1),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
