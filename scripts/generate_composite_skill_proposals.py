from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.composite_skill_proposal_generator import (
    DEFAULT_BATCH_CRITIQUE_PATH,
    DEFAULT_MIN_SUPPORTING_CASES,
    DEFAULT_PROPOSALS_DIR,
    DEFAULT_REFINEMENT_CANDIDATES_PATH,
    generate_composite_skill_proposals,
    save_composite_skill_proposals,
)


DEFAULT_EXPERIENCE_ROOT = PROJECT_ROOT / "state" / "experience"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewed composite skill proposals from composite seeds and batch critique.")
    parser.add_argument("--batch-critique-path", type=Path, default=DEFAULT_BATCH_CRITIQUE_PATH, help="Batch critique JSON path.")
    parser.add_argument(
        "--refinement-candidates-path",
        type=Path,
        default=DEFAULT_REFINEMENT_CANDIDATES_PATH,
        help="Skill refinement candidate JSONL path.",
    )
    parser.add_argument("--experience-root", type=Path, default=DEFAULT_EXPERIENCE_ROOT, help="Experience store root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROPOSALS_DIR, help="Directory for reviewed composite skill proposals.")
    parser.add_argument(
        "--min-supporting-cases",
        type=int,
        default=DEFAULT_MIN_SUPPORTING_CASES,
        help="Minimum supporting cases required before a reviewed proposal is emitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proposals, summary = generate_composite_skill_proposals(
        batch_critique_path=args.batch_critique_path,
        refinement_candidates_path=args.refinement_candidates_path,
        experience_root=args.experience_root,
        min_supporting_cases=args.min_supporting_cases,
    )
    output_paths = save_composite_skill_proposals(proposals, summary, output_dir=args.output_dir)

    print(
        json.dumps(
            {
                "proposal_count": len(proposals),
                "proposal_ids": [proposal.get("proposal_id") for proposal in proposals],
                "proposals_path": output_paths["proposals_path"],
                "summary_path": output_paths["summary_path"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
