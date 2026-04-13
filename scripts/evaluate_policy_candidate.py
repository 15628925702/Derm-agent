from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.contamination_guard import build_split_state_version, normalize_split_name
from agent.experiment_state import load_split_payload, resolve_case_selection, resolve_split_state_paths, validate_selected_case_ids
from agent.execution_record import enrich_execution_record_with_baseline, save_case_execution_record
from agent.policy_config import (
    CURRENT_STABLE_POLICY_PATH,
    POLICY_VERSIONS_DIR,
    load_policy,
    load_stable_policy,
    promote_candidate_to_stable,
    reject_or_rollback_candidate,
    save_evaluation_record,
    save_policy,
)
from agent.policy_evaluation import build_policy_summary, gate_policy_candidate
from agent.run_agent import run_agent
from cognition.cognition_state import CognitionState
from dataio.case_loader import load_case_by_index
from integrations.openai_client import DermOpenAIClient
from memory.experience_bank import ExperienceBank


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "policy_evaluation"
DEFAULT_SPLIT_STATE_ROOT = PROJECT_ROOT / "state" / "split_states"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservatively evaluate a candidate planner policy against the current stable policy.")
    parser.add_argument("--candidate-config", type=Path, required=True, help="Candidate policy JSON path.")
    parser.add_argument("--stable-config", type=Path, default=CURRENT_STABLE_POLICY_PATH, help="Stable policy JSON path.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of cases to evaluate.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start case index.")
    parser.add_argument("--data-split", type=str, default="val", choices=("train", "val", "test"), help="Logical data split label. Candidate evaluation should usually use `val`.")
    parser.add_argument("--split-json", type=Path, default=None, help="Optional fixed split JSON. If omitted, the built-in deterministic split is used.")
    parser.add_argument("--non-strict-frozen-eval", action="store_true", help="Allow snapshotting from resolved split state paths even if some files are missing.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation outputs.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Override local client timeout in seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Override local client retry count.")
    parser.add_argument("--auto-promote", action="store_true", help="Promote the candidate to stable if it passes the gate.")
    parser.add_argument("--auto-rollback", action="store_true", help="Mark the candidate as rolled back/rejected on gate failure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stable_policy = load_policy(args.stable_config)
    candidate_policy = load_policy(args.candidate_config)
    candidate_version_path = save_policy(candidate_policy, POLICY_VERSIONS_DIR / f"{candidate_policy.policy_id}.json", register=True)
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    split_payload, resolved_split_json = load_split_payload(split_json=args.split_json, data_root=args.data_root)
    case_data_root = resolve_case_data_root(data_root=args.data_root, split_payload=split_payload)
    case_selection = resolve_case_selection(
        data_split=args.data_split,
        split_payload=split_payload,
        split_json_path=resolved_split_json,
        limit=args.limit,
        case_offset=args.case_offset,
        strict=True,
    )
    state_paths = resolve_split_state_paths(
        data_split=case_selection.data_split,
        strict_frozen_eval=not args.non_strict_frozen_eval,
    )

    stable_records = evaluate_policy_run(
        stable_policy.to_dict(),
        policy_label="stable",
        client=client,
        data_root=case_data_root,
        case_selection=case_selection.to_dict(),
        state_paths=state_paths.to_dict(),
        output_dir=args.output_dir / "stable_run",
        data_split=case_selection.data_split,
    )
    candidate_records = evaluate_policy_run(
        candidate_policy.to_dict(),
        policy_label="candidate",
        client=client,
        data_root=case_data_root,
        case_selection=case_selection.to_dict(),
        state_paths=state_paths.to_dict(),
        output_dir=args.output_dir / "candidate_run",
        data_split=case_selection.data_split,
    )

    stable_summary = build_policy_summary(stable_records)
    candidate_summary = build_policy_summary(candidate_records)
    gate_decision = gate_policy_candidate(
        stable_summary=stable_summary,
        candidate_summary=candidate_summary,
        evaluation_gate=candidate_policy.evaluation_gate,
    )

    evaluated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evaluation_record = {
        "evaluated_at": evaluated_at,
        "stable_policy": stable_policy.to_dict(),
        "candidate_policy": candidate_policy.to_dict(),
        "stable_summary": stable_summary,
        "candidate_summary": candidate_summary,
        "gate_decision": gate_decision,
        "run_config": {
            "data_root": str(args.data_root),
            "case_data_root": str(case_data_root),
            "limit": args.limit,
            "case_offset": args.case_offset,
            "data_split": case_selection.data_split,
            "split_json": str(args.split_json) if args.split_json else "",
            "strict_frozen_eval": not args.non_strict_frozen_eval,
            "client_timeout": client.timeout,
            "client_max_retries": client.max_retries,
        },
        "artifacts": {
            "stable_output_dir": str(args.output_dir / "stable_run"),
            "candidate_output_dir": str(args.output_dir / "candidate_run"),
        },
    }

    evaluation_output_path = args.output_dir / f"policy_eval_{candidate_policy.policy_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    evaluation_output_path.write_text(json.dumps(evaluation_record, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation_record["evaluation_record_path"] = str(evaluation_output_path)
    policy_eval_state_path = save_evaluation_record(evaluation_record, policy_id=candidate_policy.policy_id)

    policy_action = {"action": "no_change", "path": "", "reason": ""}
    if gate_decision.get("passed") and args.auto_promote:
        promoted_path = promote_candidate_to_stable(candidate_policy, evaluation_record=evaluation_record)
        policy_action = {
            "action": "promoted_to_stable",
            "path": str(promoted_path),
            "reason": "Candidate passed the conservative gate.",
        }
    elif gate_decision.get("rollback_required") and args.auto_rollback:
        rejected_path = reject_or_rollback_candidate(
            candidate_policy,
            evaluation_record=evaluation_record,
            rollback_to_policy_id=stable_policy.policy_id,
            reason="Candidate failed the conservative evaluation gate.",
        )
        policy_action = {
            "action": "rolled_back",
            "path": str(rejected_path),
            "reason": "Candidate failed the conservative gate and was not enabled.",
        }

    print(
        json.dumps(
            {
                "candidate_policy_id": candidate_policy.policy_id,
                "stable_policy_id": stable_policy.policy_id,
                "candidate_version_path": str(candidate_version_path),
                "gate_decision": gate_decision,
                "evaluation_output_path": str(evaluation_output_path),
                "policy_eval_state_path": str(policy_eval_state_path),
                "policy_action": policy_action,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def resolve_case_data_root(*, data_root: Path, split_payload: dict[str, Any]) -> Path:
    metadata_path = Path(str(split_payload.get("metadata_path", "")).strip())
    if metadata_path.exists():
        return metadata_path.parent
    return data_root


def evaluate_policy_run(
    policy_config: dict[str, Any],
    *,
    policy_label: str,
    client: DermOpenAIClient,
    data_root: Path,
    case_selection: dict[str, Any],
    state_paths: dict[str, Any],
    output_dir: Path,
    data_split: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    actual_case_ids: list[str] = []
    case_indices = list(case_selection.get("case_indices", []))
    total_cases = len(case_indices)
    for index, case_index in enumerate(case_indices, start=1):
        percent = (index / total_cases * 100.0) if total_cases else 100.0
        print(
            f"[policy-eval {policy_label} {index}/{total_cases} ({percent:.1f}%)] case_index={case_index}",
            flush=True,
        )
        case_input = load_case_by_index(case_index=case_index, data_root=data_root)
        actual_case_ids.append(case_input.case_id)
        records.append(
            evaluate_case(
                case_input,
                client=client,
                output_dir=output_dir,
                policy_config=policy_config,
                policy_label=policy_label,
                data_split=data_split,
                state_paths=state_paths,
            )
        )
    validate_selected_case_ids(
        actual_case_ids=actual_case_ids,
        expected_case_ids=[str(item) for item in case_selection.get("case_ids", [])],
        context=f"policy evaluation `{policy_label}`",
    )
    return records


def evaluate_case(
    case_input,
    *,
    client: DermOpenAIClient,
    output_dir: Path,
    policy_config: dict[str, Any],
    policy_label: str,
    data_split: str,
    state_paths: dict[str, Any],
) -> dict[str, Any]:
    baseline_response = client.baseline_diagnosis(case_input)
    normalized_split = normalize_split_name(data_split, default="val")
    policy_snapshot = deepcopy(policy_config)
    policy_snapshot["state_split"] = normalized_split
    policy_snapshot["state_version"] = str(policy_snapshot.get("state_version", "")).strip() or build_split_state_version(
        component_id="policy_config",
        split_name=normalized_split,
        payload={
            "policy_id": policy_snapshot.get("policy_id"),
            "version": policy_snapshot.get("version"),
            "planner_policy": policy_snapshot.get("planner_policy", {}),
            "retrieval_policy": policy_snapshot.get("retrieval_policy", {}),
            "evidence_policy": policy_snapshot.get("evidence_policy", {}),
        },
    )

    split_root = Path(str(state_paths.get("split_state_root", DEFAULT_SPLIT_STATE_ROOT / normalized_split)))
    frozen_bank = ExperienceBank(root=Path(str(state_paths.get("experience_root", split_root / "experience"))))
    frozen_cognition = CognitionState.load(Path(str(state_paths.get("cognition_path", split_root / "cognition_state.json"))))
    frozen_cognition.state_split = normalized_split
    agent_state, _ = run_agent(
        case_input=case_input,
        client=client,
        experience_bank=frozen_bank,
        cognition=deepcopy(frozen_cognition),
        policy_config=policy_snapshot,
        output_dir=output_dir / "artifacts",
        enable_writeback=False,
        run_mode=f"{normalized_split}_frozen_inference",
        data_split=normalized_split,
    )
    enriched_record = enrich_execution_record_with_baseline(agent_state.execution_record, baseline_qwen=baseline_response)
    enriched_record["policy_run"] = {
        "policy_id": policy_config.get("policy_id"),
        "policy_label": policy_label,
        "data_split": normalized_split,
    }
    save_case_execution_record(enriched_record, output_dir / "records")
    return enriched_record


if __name__ == "__main__":
    raise SystemExit(main())
