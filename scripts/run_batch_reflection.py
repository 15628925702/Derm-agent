from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.batch_reflection import (
    DEFAULT_HARD_CASES_PATH,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINEMENT_CANDIDATES_PATH,
    run_batch_reflection,
    save_batch_reflection_outputs,
)


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch-level reflection and cross-rollout critique over execution records.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing execution records.")
    parser.add_argument("--hard-cases-path", type=Path, default=DEFAULT_HARD_CASES_PATH, help="Hard case JSONL path.")
    parser.add_argument(
        "--refinement-candidates-path",
        type=Path,
        default=DEFAULT_REFINEMENT_CANDIDATES_PATH,
        help="Skill refinement candidate JSONL path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for batch reflection outputs.")
    parser.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help="Minimum repeated support required before a cross-rollout pattern is emitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    critique = run_batch_reflection(
        records_root=args.records_root,
        hard_cases_path=args.hard_cases_path,
        refinement_candidates_path=args.refinement_candidates_path,
        min_support=args.min_support,
    )
    output_paths = save_batch_reflection_outputs(critique, output_dir=args.output_dir)

    print(
        json.dumps(
            {
                "batch_id": critique.batch_id,
                "source_summary": critique.source_summary,
                "success_cluster_count": len(critique.success_clusters),
                "failure_cluster_count": len(critique.failure_clusters),
                "rule_candidate_count": len(critique.rule_candidates),
                "composite_skill_seed_candidate_count": len(critique.composite_skill_seed_candidates),
                "critique_path": output_paths["critique_path"],
                "summary_path": output_paths["summary_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
