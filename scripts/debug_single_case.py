from __future__ import annotations

import argparse
import json
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
from agent.policy_config import load_stable_policy
from agent.run_agent import run_agent
from dataio.case_loader import load_case_by_index
from integrations.openai_client import DermOpenAIClient


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
SPECIALIST_SKILLS = ("mel_nev_specialist_skill", "ack_scc_specialist_skill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a single DermAgent MVP case.")
    parser.add_argument("--case-index", type=int, default=0, help="Row index from metadata.csv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--enable-writeback", action="store_true", help="Enable online writeback (disabled by default to avoid contamination during debug).")
    parser.add_argument("--data-split", type=str, default="train", choices=("train", "val", "test"), help="Logical split label for this debug run.")
    parser.add_argument("--run-mode", type=str, default="debug", help="Runtime mode label attached to execution record state metadata.")
    parser.add_argument("--client-base-url", type=str, default=None, help="Optional OpenAI-compatible base URL override.")
    parser.add_argument("--client-api-key", type=str, default=None, help="Optional OpenAI-compatible API key override.")
    parser.add_argument("--client-model", type=str, default=None, help="Optional model-name override.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Optional client timeout in seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Optional client retry count.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for case artifacts and execution records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_input = load_case_by_index(args.case_index, args.data_root)
    client = DermOpenAIClient(
        base_url=args.client_base_url,
        api_key=args.client_api_key,
        model=args.client_model,
        timeout=args.client_timeout,
        max_retries=args.client_max_retries,
    )
    model_name = args.client_model or client.model
    overrides = get_model_workflow_overrides(
        model_name,
        dataset_name=case_input.dataset_name,
        base_workflow_context=case_input.workflow_context,
    )
    if overrides:
        apply_model_workflow_to_case(
            case_input,
            model_name,
            dataset_name=case_input.dataset_name,
        )
        print(
            json.dumps(
                {
                    "model_workflow_routing": {
                        "model": model_name,
                        "dataset": case_input.dataset_name,
                        "dataset_workflow_profile": (case_input.workflow_context or {}).get("dataset_workflow_profile", ""),
                        "model_workflow_profile": (case_input.workflow_context or {}).get("model_workflow_profile", ""),
                        "execution_overrides": execution_overrides_for_run_agent(overrides),
                        "skip_specialist_skills": bool(overrides.get("skip_specialist_skills", False)),
                        "skip_experience_retrieval": bool(overrides.get("skip_experience_retrieval", False)),
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    policy_config = None
    if overrides.get("skip_specialist_skills"):
        policy_config = merge_model_workflow_policy_overrides(
            load_stable_policy().to_dict(),
            overrides,
        )
    state, _ = run_agent(
        case_input,
        client=client,
        policy_config=policy_config,
        output_dir=args.output_dir,
        enable_writeback=bool(args.enable_writeback),
        run_mode=str(args.run_mode),
        data_split=str(args.data_split),
        execution_overrides={
            **overrides,
            **execution_overrides_for_run_agent(overrides),
        },
    )

    abstract_by_id = {
        str(record.get("source_id", "")).strip(): record
        for record in state.retrieval_bundle.get("abstract_results", [])
        if str(record.get("source_id", "")).strip()
    }
    specialist_experience_references = {}
    for skill_name in SPECIALIST_SKILLS:
        output = state.skill_outputs.get(skill_name, {})
        referenced_ids = [
            str(item).strip()
            for item in output.get("referenced_experiences", [])
            if str(item).strip() in abstract_by_id
        ]
        specialist_experience_references[skill_name] = {
            "referenced_confusion_patterns": output.get("referenced_confusion_patterns", []),
            "referenced_abstract_experience_ids": referenced_ids,
            "referenced_abstract_experiences": [abstract_by_id[item] for item in referenced_ids],
            "critical_supporting_evidence": output.get("critical_supporting_evidence", []),
            "counterexample_watchouts": output.get("counterexample_watchouts", []),
        }

    summary = dict(state.execution_record)
    summary["skill_retrieval_debug"] = {
        "candidate_skill_ids": state.skill_retrieval_bundle.get("candidate_skill_ids", []),
        "candidate_skill_names": state.skill_retrieval_bundle.get("candidate_skill_names", []),
        "skill_id_to_name": state.skill_retrieval_bundle.get("skill_id_to_name", {}),
        "match_reasons": state.skill_retrieval_bundle.get("match_reasons", {}),
        "trigger_hits": state.skill_retrieval_bundle.get("trigger_hits", {}),
        "retrieval_scores": state.skill_retrieval_bundle.get("retrieval_scores", {}),
    }
    summary["specialist_experience_references"] = specialist_experience_references
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
