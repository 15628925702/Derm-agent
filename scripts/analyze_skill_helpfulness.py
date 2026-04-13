from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.hard_case_miner import load_execution_records
from agent.skill_helpfulness_analyzer import DEFAULT_MIN_CALLS, DEFAULT_TOP_K
from agent.skill_helpfulness_analyzer import analyze_skill_helpfulness, save_skill_helpfulness_outputs


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "skill_helpfulness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze skill helpfulness from DermAgent case execution records.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing case execution records.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for analysis outputs.")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name.")
    parser.add_argument("--skill", type=str, default=None, help="Filter by a single skill name.")
    parser.add_argument("--label", type=str, default=None, help="Filter by ground-truth label.")
    parser.add_argument("--confusion-pair", type=str, default=None, help="Filter by confusion pair text.")
    parser.add_argument("--min-calls", type=int, default=DEFAULT_MIN_CALLS, help="Minimum analyzed calls required to keep a skill in the report.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many common scenarios and failure modes to keep per skill.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    execution_records = load_execution_records(args.records_root)
    reports, summary = analyze_skill_helpfulness(
        execution_records,
        dataset_name=args.dataset,
        skill_name=args.skill,
        label=args.label,
        confusion_pair=args.confusion_pair,
        min_calls=args.min_calls,
        top_k=args.top_k,
    )
    output_paths = save_skill_helpfulness_outputs(reports, summary, args.output_dir)

    print(
        json.dumps(
            {
                "records_scanned": len(execution_records),
                "skill_report_count": len(reports),
                "reports_path": output_paths["reports_path"],
                "summary_path": output_paths["summary_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
