#!/usr/bin/env python3
"""
支持 Metadata Masking 的评估脚本
用于阶段2：Metadata Missingness 实验
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_protocol import run_evaluation_suite, EvaluationTargetSpec
from agent.policy_config import load_policy, load_stable_policy
from integrations.openai_client import DermOpenAIClient
from dataio.case_loader import load_case_with_masking

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare agent vs baseline with metadata masking support."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--policy-label", type=str, default="")
    parser.add_argument("--metadata-mask", type=str, default=None,
                       help='Comma-separated fields to keep, e.g. "region,age". Empty string for minimal.')
    parser.add_argument("--data-split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--non-strict-frozen-eval", action="store_true")
    parser.add_argument("--client-timeout", type=float, default=None)
    parser.add_argument("--client-max-retries", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 解析 metadata_mask
    metadata_mask = None
    if args.metadata_mask is not None:
        if args.metadata_mask == "":
            metadata_mask = []  # Minimal: mask all
        else:
            metadata_mask = [f.strip() for f in args.metadata_mask.split(",")]

    print(f"Metadata mask: {metadata_mask}")

    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()

    # 如果有 metadata_mask，需要在 policy 中传递
    if metadata_mask is not None:
        policy["metadata_mask"] = metadata_mask

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
            description="Full DermAgent with frozen experience/cognition/policy state.",
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
        seed=0,
        suite_label="compare_agent_vs_qwen",
        data_split=args.data_split,
        split_json=args.split_json,
        strict_frozen_eval=not args.non_strict_frozen_eval,
    )

    result_manifest = suite["result_manifest"]
    target_results = {item["target"]["target_id"]: item for item in result_manifest.get("target_results", [])}
    baseline_summary = target_results.get("direct_baseline", {}).get("summary", {})
    agent_summary = target_results.get("full_dermagent", {}).get("summary", {})

    report = {
        "run_config": {
            "data_root": str(args.data_root),
            "limit": args.limit,
            "case_offset": args.case_offset,
            "metadata_mask": metadata_mask,
            "policy_label": args.policy_label,
            "data_split": args.data_split,
        },
        "summary": {
            "baseline": baseline_summary,
            "agent": agent_summary,
        },
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
