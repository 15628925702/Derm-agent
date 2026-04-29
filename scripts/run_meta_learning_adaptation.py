from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_learning.few_shot_adapter import DEFAULT_ADAPTATION_DIR, DEFAULT_MIN_CASES, FewShotAdapter


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run few-shot meta-learning adaptation for a new dataset. "
            "After running, set DERMAGENT_META_LEARNING_ENABLED=1 to activate the bias."
        )
    )
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name (e.g. 'new_clinic_v1').")
    parser.add_argument(
        "--cases-path",
        type=Path,
        required=True,
        help="Path to JSONL file with few-shot cases. Each line: {reference_label, skill_assessments, ...}.",
    )
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES, help=f"Minimum cases required (default: {DEFAULT_MIN_CASES}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ADAPTATION_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = _load_jsonl(args.cases_path)

    if not cases:
        print(json.dumps({"error": f"No cases loaded from {args.cases_path}."}, indent=2))
        return 1

    adapter = FewShotAdapter(adaptation_dir=args.output_dir)
    adaptation = adapter.adapt(cases, dataset_name=args.dataset_name, min_cases=args.min_cases)

    if "warning" in adaptation and len(adaptation) == 1:
        print(json.dumps(adaptation, ensure_ascii=False, indent=2))
        return 1

    out_path = adapter.save_adaptation(args.dataset_name, adaptation)

    print(json.dumps({
        "dataset_name": args.dataset_name,
        "case_count": adaptation.get("case_count"),
        "skills_with_bias": len(adaptation.get("retrieval_bias", {})),
        "workflow_patches": len(adaptation.get("workflow_preferences_patch", {})),
        "adaptation_path": str(out_path),
        "next_step": "Set DERMAGENT_META_LEARNING_ENABLED=1 to activate this adaptation.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
