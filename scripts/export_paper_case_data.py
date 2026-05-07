from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.paper_exports import DEFAULT_CASE_LEVEL_EXPORT_ROOT, export_compare_case_data, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compare reports into paper-ready workflow-free case-level CSV/JSON/XLSX files."
    )
    parser.add_argument(
        "--compare-report",
        type=Path,
        action="append",
        default=[],
        help="Path to a compare_agent_vs_qwen_*.json report. Can be provided multiple times.",
    )
    parser.add_argument(
        "--compare-report-glob",
        type=str,
        action="append",
        default=[],
        help="Glob for compare_agent_vs_qwen_*.json reports. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CASE_LEVEL_EXPORT_ROOT / "manual_exports",
        help="Directory for exported case-level files.",
    )
    parser.add_argument("--export-stem", type=str, default="case_level_compare_export", help="Output file stem.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = list(args.compare_report)
    for pattern in args.compare_report_glob:
        report_paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
    report_paths = [path for path in sorted(set(report_paths)) if _looks_like_compare_case_report(path)]
    manifest = export_compare_case_data(
        compare_report_paths=report_paths,
        output_dir=args.output_dir,
        export_stem=args.export_stem,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _looks_like_compare_case_report(path: Path) -> bool:
    if not path.is_file() or not path.name.startswith("compare_agent_vs_qwen_") or path.suffix != ".json":
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("run_config"), dict) and isinstance(payload.get("cases"), list)


if __name__ == "__main__":
    raise SystemExit(main())
