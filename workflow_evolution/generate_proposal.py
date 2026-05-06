from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_evolution.proposal_generator import DEFAULT_OUTPUT_DIR, generate_workflow_evolution_proposal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a disabled-by-default model x dataset workflow evolution proposal from compare reports."
    )
    parser.add_argument("--reports", nargs="+", required=True, help="Compare report JSON paths or glob patterns.")
    parser.add_argument("--model", required=True, help="Model name, e.g. Hulu-Med-7B.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. isic2019.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doctor-experience", type=Path, default=None, help="Optional physician experience note file.")
    parser.add_argument("--workflow-cell-id", default="", help="Optional proposed workflow_cell_id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = _expand_reports(args.reports)
    if not report_paths:
        raise SystemExit("No compare reports matched --reports.")
    proposal = generate_workflow_evolution_proposal(
        report_paths=report_paths,
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        doctor_experience_path=args.doctor_experience,
        base_workflow_cell_id=args.workflow_cell_id,
    )
    print(
        json.dumps(
            {
                "proposal_id": proposal["proposal_id"],
                "review_status": proposal["review_status"],
                "default_enabled": proposal["default_enabled"],
                "proposal_path": str(args.output_dir / proposal["proposal_id"] / "workflow_evolution_proposal.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _expand_reports(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.exists() and str(path) not in seen:
                seen.add(str(path))
                paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
