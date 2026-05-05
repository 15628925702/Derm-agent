from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.paper_exports import DEFAULT_OUTPUTS_ROOT, DEFAULT_PAPER_EXPORT_ROOT, export_paper_fig_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper figure-ready DermAgent data as CSV and JSON.")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT, help="Root directory containing experiment outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PAPER_EXPORT_ROOT / "fig_data", help="Directory for exported paper figure data files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_paper_fig_data(outputs_root=args.outputs_root, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
