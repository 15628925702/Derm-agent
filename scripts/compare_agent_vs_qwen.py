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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct Qwen baseline against full DermAgent under frozen evaluation mode.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of cases to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for traceability; current case selection remains deterministic.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start from this case index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON reports.")
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional policy config JSON. Defaults to the current stable policy.")
    parser.add_argument("--policy-label", type=str, default="", help="Optional human-readable label for this policy run.")
    parser.add_argument("--data-split", type=str, default="test", choices=("val", "test"), help="Evaluation split label for contamination guard and manifests.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Override per-request timeout in seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Override automatic retries for transient local inference failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    suite = run_evaluation_suite(
        output_root=args.output_dir,
        data_root=args.data_root,
        client=client,
        policy_config=policy,
        target_specs=target_specs,
        limit=args.limit,
        case_offset=args.case_offset,
        seed=args.seed,
        suite_label="compare_agent_vs_qwen",
        data_split=args.data_split,
    )

    result_manifest = suite["result_manifest"]
    target_results = {item["target"]["target_id"]: item for item in result_manifest.get("target_results", [])}
    baseline_summary = target_results.get("direct_baseline", {}).get("summary", {})
    agent_summary = target_results.get("full_dermagent", {}).get("summary", {})
    full_agent_jsonl = Path(target_results.get("full_dermagent", {}).get("artifacts", {}).get("records_jsonl_path", ""))
    agent_cases = []
    if full_agent_jsonl.exists():
        with full_agent_jsonl.open("r", encoding="utf-8") as handle:
            agent_cases = [json.loads(line) for line in handle if line.strip()]

    report = {
        "run_config": {
            "data_root": str(args.data_root),
            "limit": args.limit,
            "seed": args.seed,
            "case_offset": args.case_offset,
            "base_url": client.base_url,
            "model": client.model,
            "policy_id": policy.get("policy_id"),
            "policy_source_path": policy.get("source_path"),
            "policy_label": args.policy_label,
            "data_split": args.data_split,
            "evaluation_protocol_version": result_manifest.get("protocol_version"),
        },
        "summary": {
            "baseline": baseline_summary,
            "agent": agent_summary,
            "agent_vs_baseline": result_manifest.get("comparisons", {}).get("full_dermagent", {}).get("vs_baseline", {}),
        },
        "artifacts": {
            "evaluation_manifest_path": suite["evaluation_manifest_path"],
            "result_manifest_path": suite["result_manifest_path"],
            "run_root": suite["run_root"],
        },
        "cases": agent_cases,
    }
    output_path = args.output_dir / f"compare_agent_vs_qwen_{suite['eval_id'].split('_')[-1]}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": report["summary"],
                "report_path": str(output_path),
                "evaluation_manifest_path": suite["evaluation_manifest_path"],
                "result_manifest_path": suite["result_manifest_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
