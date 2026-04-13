from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.experience_consolidator import DEFAULT_HARD_CASES_PATH, DEFAULT_MIN_SUPPORTING_CASES, DEFAULT_OUTPUT_DIR
from memory.experience_consolidator import consolidate_experiences, save_consolidation_outputs
from memory.experience_store import ExperienceStore


DEFAULT_EXPERIENCE_ROOT = PROJECT_ROOT / "state" / "experience"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate raw and tactical experiences into stable abstract experiences.")
    parser.add_argument("--experience-root", type=Path, default=DEFAULT_EXPERIENCE_ROOT, help="Experience store root.")
    parser.add_argument("--hard-cases-path", type=Path, default=DEFAULT_HARD_CASES_PATH, help="Hard case JSONL path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for consolidation outputs.")
    parser.add_argument(
        "--min-supporting-cases",
        type=int,
        default=DEFAULT_MIN_SUPPORTING_CASES,
        help="Minimum distinct supporting cases required before a consolidated abstract experience is emitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and output consolidation candidates without writing them back into the experience store.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store = ExperienceStore(args.experience_root)
    result = consolidate_experiences(
        store=store,
        hard_cases_path=args.hard_cases_path,
        min_supporting_cases=args.min_supporting_cases,
        write_to_store=not args.dry_run,
    )
    output_paths = save_consolidation_outputs(result, output_dir=args.output_dir)

    print(
        json.dumps(
            {
                "consolidated_count": len(result.consolidated_records),
                "records_path": output_paths["records_path"],
                "summary_path": output_paths["summary_path"],
                "write_to_store": not args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
