from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.hard_case_miner import DEFAULT_MAX_PER_CLUSTER, DEFAULT_MIN_IMPORTANCE
from agent.hard_case_miner import load_execution_records, mine_hard_cases


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hard_case_mining"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard cases from DermAgent case execution records.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing case execution records.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for mined hard case outputs.")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name.")
    parser.add_argument("--label", type=str, default=None, help="Filter by ground-truth label.")
    parser.add_argument("--confusion-pair", type=str, default=None, help="Filter by confusion tag or pair.")
    parser.add_argument("--skill", type=str, default=None, help="Filter by involved skill.")
    parser.add_argument("--min-importance", type=float, default=DEFAULT_MIN_IMPORTANCE, help="Minimum importance score to keep.")
    parser.add_argument("--max-per-cluster", type=int, default=DEFAULT_MAX_PER_CLUSTER, help="Maximum number of hard cases to keep per cluster.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    execution_records = load_execution_records(args.records_root)
    hard_cases, summary = mine_hard_cases(
        execution_records,
        dataset_name=args.dataset,
        label=args.label,
        confusion_pair=args.confusion_pair,
        skill_name=args.skill,
        min_importance=args.min_importance,
        max_per_cluster=args.max_per_cluster,
    )

    hard_cases_path = args.output_dir / "hard_cases.jsonl"
    with hard_cases_path.open("w", encoding="utf-8") as handle:
        for record in hard_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "records_scanned": len(execution_records),
                "hard_case_count": len(hard_cases),
                "hard_cases_path": str(hard_cases_path),
                "summary_path": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
