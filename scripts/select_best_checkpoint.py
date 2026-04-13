from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.checkpoint_selection import (
    DEFAULT_CHECKPOINT_EXPORT_ROOT,
    DEFAULT_SELECTION_OUTPUT_ROOT,
    DEFAULT_TRAIN_RUNS_ROOT,
    build_selection_report,
    discover_checkpoint_candidates,
    save_selection_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Programmatically select the best validation and stable paper checkpoints.")
    parser.add_argument("--train-runs-root", type=Path, default=DEFAULT_TRAIN_RUNS_ROOT, help="Root containing staged training runs.")
    parser.add_argument("--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINT_EXPORT_ROOT, help="Checkpoint export root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SELECTION_OUTPUT_ROOT, help="Directory for checkpoint selection reports.")
    parser.add_argument("--run-ids", type=str, default="", help="Optional comma-separated run ids to consider.")
    parser.add_argument("--no-export-best-validation", action="store_true", help="Do not export the selected best validation bundle.")
    parser.add_argument("--no-export-stable-paper", action="store_true", help="Do not export the selected stable paper bundle.")
    return parser.parse_args()


def parse_run_ids(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_ids = parse_run_ids(args.run_ids)
    candidates = discover_checkpoint_candidates(args.train_runs_root, run_ids=run_ids or None)
    if not candidates:
        raise FileNotFoundError(f"No train run manifests discovered under {args.train_runs_root}")

    report = build_selection_report(
        candidates,
        train_runs_root=args.train_runs_root,
        checkpoints_root=args.checkpoints_root,
    )
    saved = save_selection_report(
        report,
        output_dir=args.output_dir,
        checkpoint_export_root=args.checkpoints_root,
        export_best_validation=not bool(args.no_export_best_validation),
        export_stable_paper=not bool(args.no_export_stable_paper),
    )
    print(
        json.dumps(
            {
                "selection_id": report["selection_id"],
                "report_path": saved["report_path"],
                "candidate_count": len(report.get("ranked_candidates", [])),
                "best_validation_candidate_id": report.get("selected", {}).get("best_validation_checkpoint", {}).get("candidate_id", ""),
                "stable_paper_candidate_id": report.get("selected", {}).get("stable_paper_checkpoint", {}).get("candidate_id", ""),
                "exports": saved.get("exports", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
