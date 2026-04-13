from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.skill_refinement_candidate_generator import DEFAULT_HARD_CASES_PATH, DEFAULT_OUTPUT_DIR
from agent.skill_refinement_candidate_generator import DEFAULT_SKILL_HELPFULNESS_DIR
from agent.skill_refinement_candidate_generator import generate_skill_refinement_candidates, save_skill_refinement_candidates


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate skill refinement candidates from DermAgent execution evidence.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing execution records.")
    parser.add_argument("--hard-cases-path", type=Path, default=DEFAULT_HARD_CASES_PATH, help="Path to mined hard cases JSONL.")
    parser.add_argument("--skill-helpfulness-dir", type=Path, default=DEFAULT_SKILL_HELPFULNESS_DIR, help="Directory containing skill helpfulness outputs.")
    parser.add_argument("--experience-root", type=Path, default=None, help="Optional override for the experience store root.")
    parser.add_argument("--cognition-path", type=Path, default=None, help="Optional override for cognition state path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for refinement candidate outputs.")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name.")
    parser.add_argument("--skill", type=str, default=None, help="Filter by target skill name.")
    parser.add_argument(
        "--update-type",
        type=str,
        default=None,
        choices=["workflow_update", "trigger_update", "watchout_update", "output_schema_update", "split_skill", "merge_skill"],
        help="Filter by proposed update type.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates, summary = generate_skill_refinement_candidates(
        records_root=args.records_root,
        hard_cases_path=args.hard_cases_path,
        skill_helpfulness_dir=args.skill_helpfulness_dir,
        experience_root=args.experience_root,
        cognition_path=args.cognition_path,
        dataset_name=args.dataset,
        skill_name=args.skill,
        proposed_update_type=args.update_type,
    )
    output_paths = save_skill_refinement_candidates(candidates, summary, args.output_dir)

    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "candidates_path": output_paths["candidates_path"],
                "summary_path": output_paths["summary_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
