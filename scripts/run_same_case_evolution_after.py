from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.model_workflow_router import (
    apply_model_workflow_to_case,
    execution_overrides_for_run_agent,
    get_model_workflow_overrides,
    merge_model_workflow_policy_overrides,
)
from agent.policy_config import load_stable_policy, snapshot_policy
from agent.run_agent import run_agent
from dataio.xiangya_sft_loader import load_xiangya_sft_case_inputs
from integrations.openai_client import DermOpenAIClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a post-evolution same-case DermAgent demo.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--baseline-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8013/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Hulu-Med-4B")
    parser.add_argument("--split-state-root", type=Path, required=True)
    parser.add_argument("--enable-workflow-evolution", action="store_true")
    parser.add_argument("--allowed-skill", action="append", default=[], help="Restrict planner to these skills.")
    parser.add_argument("--force-select-skill", action="append", default=[], help="Force planner to select these skills.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    return parser.parse_args()


def load_case_by_id(data_root: Path, case_id: str):
    for case_input in load_xiangya_sft_case_inputs(data_root=data_root):
        if str(case_input.case_id).strip() == case_id:
            return case_input
    raise ValueError(f"Case id not found: {case_id}")


def main() -> int:
    args = parse_args()
    os.environ["DERMAGENT_SPLIT_STATE_ROOT"] = str(args.split_state_root)
    if args.enable_workflow_evolution:
        os.environ["DERMAGENT_ENABLE_WORKFLOW_EVOLUTION"] = "1"
    else:
        os.environ.pop("DERMAGENT_ENABLE_WORKFLOW_EVOLUTION", None)

    case_input = load_case_by_id(args.data_root, args.case_id)
    routed_overrides = apply_model_workflow_to_case(case_input, args.model)

    policy = snapshot_policy(load_stable_policy())
    model_workflow_overrides = routed_overrides or get_model_workflow_overrides(
        args.model,
        dataset_name=case_input.dataset_name,
        base_workflow_context=case_input.workflow_context,
    )
    if model_workflow_overrides:
        policy = merge_model_workflow_policy_overrides(policy, model_workflow_overrides)
    execution_overrides = execution_overrides_for_run_agent(model_workflow_overrides)
    planner_policy = dict(policy.get("planner_policy", {}) or {})
    if args.allowed_skill:
        planner_policy["allowed_skills"] = list(dict.fromkeys(str(item).strip() for item in args.allowed_skill if str(item).strip()))
    if args.force_select_skill:
        planner_policy["force_select_skills"] = list(
            dict.fromkeys(str(item).strip() for item in args.force_select_skill if str(item).strip())
        )
    policy["planner_policy"] = planner_policy

    baseline_record = json.loads(args.baseline_record.read_text(encoding="utf-8"))
    baseline_override = dict(baseline_record.get("qwen_final", {}) or {})

    client = DermOpenAIClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state, evidence_package = run_agent(
        case_input,
        client=client,
        policy_config=policy,
        output_dir=args.output_dir,
        enable_writeback=False,
        run_mode="frozen_eval",
        data_split="test",
        strict_frozen_writeback_guard=False,
        execution_overrides=execution_overrides,
        baseline_diagnosis_override=baseline_override,
    )

    summary = {
        "case_id": state.case_input.case_id,
        "workflow_context": state.case_input.workflow_context,
        "selected_skills": state.planner_output.get("selected_skills", []),
        "skill_retrieval_candidates": state.skill_retrieval_bundle.get("candidate_skill_names", []),
        "final_diagnosis": state.final_diagnosis,
        "baseline_override": baseline_override,
        "contact_atopic_skill_output": state.skill_outputs.get("contact_atopic_specialist_skill", {}),
        "evidence_package_path": str(args.output_dir / state.case_input.case_id / "evidence_package.json"),
        "execution_record_path": str(args.output_dir / "records" / state.case_input.case_id / "case_execution_record.json"),
    }
    (args.output_dir / "same_case_evolution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
