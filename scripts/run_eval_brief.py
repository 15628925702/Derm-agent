from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_protocol import DEFAULT_EVAL_OUTPUT_ROOT, EvaluationTargetSpec, run_evaluation_suite
from agent.policy_config import load_policy, load_stable_policy
from integrations.openai_client import DermOpenAIClient


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen paper-level brief evaluation protocol.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of cases to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for traceability; current case selection remains deterministic.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start from this case index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation artifacts.")
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional policy config JSON. Defaults to the current stable policy.")
    parser.add_argument("--data-split", type=str, default="test", choices=("val", "test"), help="Evaluation split label for contamination guard and manifests.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Override per-request timeout in seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Override automatic retries for transient local inference failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()
    target_specs = [
        EvaluationTargetSpec(
            target_id="direct_baseline",
            label="Direct Baseline",
            target_type="baseline",
            mode="baseline",
            description="Direct Qwen baseline with no agent evidence package.",
        ),
        EvaluationTargetSpec(
            target_id="full_dermagent",
            label="Full DermAgent",
            target_type="full_agent",
            mode="agent",
            description="Full DermAgent with frozen experience/cognition/policy state and writeback disabled.",
        ),
    ]
    result = run_evaluation_suite(
        output_root=args.output_dir,
        data_root=args.data_root,
        client=client,
        policy_config=policy,
        target_specs=target_specs,
        limit=args.limit,
        case_offset=args.case_offset,
        seed=args.seed,
        suite_label="eval_brief",
        data_split=args.data_split,
    )
    print(
        json.dumps(
            {
                "eval_id": result["eval_id"],
                "run_root": result["run_root"],
                "evaluation_manifest_path": result["evaluation_manifest_path"],
                "result_manifest_path": result["result_manifest_path"],
                "comparisons": result["result_manifest"].get("comparisons", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
