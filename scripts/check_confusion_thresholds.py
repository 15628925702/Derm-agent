from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.confusion_skill_proposal_generator import DEFAULT_PROPOSALS_DIR, DEFAULT_THRESHOLD, generate_confusion_triggered_proposals
from cognition.cognition_state import CognitionState

DEFAULT_COGNITION_PATH = PROJECT_ROOT / "state" / "cognition_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check confusion pair thresholds and generate specialist skill proposals for pairs exceeding the threshold."
    )
    parser.add_argument("--cognition-path", type=Path, default=DEFAULT_COGNITION_PATH)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Minimum confusion count to trigger a proposal.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROPOSALS_DIR)
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name for confusion cluster coverage check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cognition = CognitionState.load(args.cognition_path)

    proposals = generate_confusion_triggered_proposals(
        cognition=cognition,
        threshold=args.threshold,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
    )

    summary = {
        "threshold": args.threshold,
        "total_confusion_pairs": len(cognition.known_confusion_patterns),
        "proposals_generated": len(proposals),
        "proposals": [
            {
                "proposal_id": p["proposal_id"],
                "confusion_pair": p["confusion_pair"],
                "confusion_count": p["confusion_count"],
                "suggested_skill_id": p["proposal_artifacts"]["suggested_skill_id"],
                "output_dir": str(args.output_dir),
            }
            for p in proposals
        ],
    }

    if not proposals:
        summary["message"] = f"No confusion pairs exceed threshold={args.threshold} or all are already covered by existing specialist skills."

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
