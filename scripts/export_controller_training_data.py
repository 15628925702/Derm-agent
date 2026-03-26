from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller_training import DEFAULT_OUTPUT_DIR
from agent.controller_training import export_controller_training_data, save_controller_training_data
from agent.hard_case_miner import load_execution_records


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learnable-controller training examples from execution records.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing case execution records.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported controller training data.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name filter.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    execution_records = load_execution_records(args.records_root)
    examples, summary = export_controller_training_data(execution_records, dataset_name=args.dataset)
    output_paths = save_controller_training_data(examples, summary, output_dir=args.output_dir)

    print(
        json.dumps(
            {
                "records_scanned": len(execution_records),
                "example_count": len(examples),
                "dataset_filter": args.dataset,
                "examples_path": output_paths["examples_path"],
                "summary_path": output_paths["summary_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
