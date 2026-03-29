from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.experiment_state import ensure_split_state_paths
from configs.dataset_splits import DEFAULT_SPLIT_ID, build_fixed_split_payload, resolve_split_range, write_fixed_split_json
from configs.run_profiles import available_profile_ids, get_run_profile


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "smoke_cycles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact end-to-end DermAgent smoke cycle with a fixed split and tracked profile."
    )
    parser.add_argument("--profile", type=str, default="quick4h_v1", choices=available_profile_ids())
    parser.add_argument("--split-id", type=str, default=DEFAULT_SPLIT_ID)
    parser.add_argument("--run-id", type=str, default="", help="Optional run id. Defaults to profile + UTC timestamp.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--client-timeout", type=float, default=180.0)
    parser.add_argument("--client-max-retries", type=int, default=3)
    parser.add_argument("--clean-outputs", action="store_true", help="Delete /root/DermAgent/outputs/* before starting.")
    parser.add_argument("--resume-seed-case-index", type=int, default=None, help="Resume the seed writeback loop from this case index.")
    parser.add_argument("--controller-checkpoint-in", type=Path, default=None, help="Optional controller checkpoint to warm-start learned components.")
    parser.add_argument("--retrieval-checkpoint-in", type=Path, default=None, help="Optional retrieval checkpoint to warm-start learned components.")
    parser.add_argument(
        "--abort-on-seed-failure",
        action="store_true",
        help="Abort immediately if a single seed case fails. Default is to record the failure and continue.",
    )
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-paper-exports", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(
    command: list[str],
    *,
    step_name: str,
    log_path: Path,
    dry_run: bool,
    cwd: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if dry_run:
        return {
            "step_name": step_name,
            "command": command,
            "returncode": 0,
            "dry_run": True,
        }

    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {step_name} =====\n")
        handle.write("COMMAND: " + " ".join(command) + "\n")
        if completed.stdout:
            handle.write("\n[stdout]\n" + completed.stdout + "\n")
        if completed.stderr:
            handle.write("\n[stderr]\n" + completed.stderr + "\n")
        handle.write(f"\n[returncode] {completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step `{step_name}` failed with returncode={completed.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"See log: {log_path}"
        )
    return {
        "step_name": step_name,
        "command": command,
        "returncode": completed.returncode,
        "dry_run": False,
    }


def clean_outputs_root(outputs_root: Path) -> None:
    outputs_root.mkdir(parents=True, exist_ok=True)
    for child in outputs_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _latest_path(root: Path, pattern: str) -> str:
    matches = sorted(root.glob(pattern))
    return str(matches[-1]) if matches else ""


def build_key_results(run_root: Path) -> dict[str, Any]:
    train_runs_root = run_root / "train_runs"
    comparison_root = run_root / "comparison"
    evaluation_root = run_root / "evaluation_protocol"
    ablations_root = run_root / "ablations"
    candidate_policy_path = _latest_path(
        train_runs_root,
        "*/stage3_policy_candidate_evaluation/candidate_policy.json",
    )

    return {
        "run_root": str(run_root),
        "smoke_cycle_manifest": str(run_root / "smoke_cycle_manifest.json"),
        "fixed_split_json": str(run_root / "fixed_split.json"),
        "train_seed_log": str(run_root / "train_seed_debug.log"),
        "smoke_cycle_log": str(run_root / "smoke_cycle.log"),
        "key_paths": {
            "train_run_manifest": _latest_path(train_runs_root, "*/train_run_manifest.json"),
            "candidate_policy_path": candidate_policy_path,
            "checkpoint_selection_report": _latest_path(run_root / "checkpoint_selection", "checkpoint_selection_*.json"),
            "compare_report": _latest_path(comparison_root, "compare_agent_vs_qwen_*.json"),
            "eval_result_manifest": _latest_path(evaluation_root, "*/result_manifest.json"),
            "ablation_summary": _latest_path(ablations_root, "*/ablation_matrix_summary.json"),
            "skill_helpfulness_summary": str(run_root / "skill_helpfulness" / "skill_helpfulness_summary.json"),
            "hard_case_summary": str(run_root / "hard_case_mining" / "summary.json"),
            "batch_reflection_summary": str(run_root / "batch_reflection" / "summary.json"),
            "contamination_audit_report": _latest_path(
                run_root / "analysis" / "contamination_audit",
                "contamination_audit_*.json",
            ),
            "paper_tables_manifest": str(run_root / "paper_exports" / "tables" / "paper_tables_manifest.json"),
            "paper_fig_data_manifest": str(run_root / "paper_exports" / "fig_data" / "paper_fig_data_manifest.json"),
        },
    }


def main() -> int:
    args = parse_args()
    profile = get_run_profile(args.profile)
    run_id = args.run_id.strip() or f"{profile.profile_id}_{utc_compact()}"

    run_root = args.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "smoke_cycle.log"
    split_json_path = run_root / "fixed_split.json"

    if args.clean_outputs and not args.dry_run:
        clean_outputs_root(PROJECT_ROOT / "outputs")
        run_root.mkdir(parents=True, exist_ok=True)

    split_payload = build_fixed_split_payload(args.split_id, data_root=args.data_root)
    if not args.dry_run:
        write_fixed_split_json(split_json_path, split_id=args.split_id, data_root=args.data_root)
        for split_name in ("train", "val", "test"):
            ensure_split_state_paths(data_split=split_name)

    train_offset, train_count = resolve_split_range(
        split_payload,
        "train",
        count=profile.train_seed_cases,
    )
    val_offset, val_compare_count = resolve_split_range(
        split_payload,
        "val",
        count=profile.val_compare_cases,
    )
    test_offset, test_eval_count = resolve_split_range(
        split_payload,
        "test",
        count=profile.test_eval_cases,
    )
    _, ablation_count = resolve_split_range(
        split_payload,
        "test",
        count=profile.ablation_cases,
    )

    summary: dict[str, Any] = {
        "schema_version": "smoke_training_cycle_v1",
        "run_id": run_id,
        "profile": profile.to_dict(),
        "split_id": args.split_id,
        "run_root": str(run_root),
        "split_json_path": str(split_json_path),
        "ranges": {
            "train": {"case_offset": train_offset, "limit": train_count},
            "val": {"case_offset": val_offset, "limit": val_compare_count},
            "test": {"case_offset": test_offset, "limit": test_eval_count},
            "ablation_test": {"case_offset": test_offset, "limit": ablation_count},
        },
        "seed_failure_count": 0,
        "seed_failures": [],
        "steps": [],
    }

    seed_log_path = run_root / "train_seed_debug.log"
    seed_start_index = train_offset if args.resume_seed_case_index is None else max(train_offset, int(args.resume_seed_case_index))
    for case_index in range(seed_start_index, train_offset + train_count):
        step_name = f"seed_train_case_{case_index}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "debug_single_case.py"),
            "--case-index",
            str(case_index),
            "--enable-writeback",
            "--data-split",
            "train",
            "--run-mode",
            run_id,
        ]
        try:
            summary["steps"].append(
                run_command(
                    command,
                    step_name=step_name,
                    log_path=seed_log_path,
                    dry_run=args.dry_run,
                )
            )
        except Exception as exc:
            failure_payload = {
                "step_name": step_name,
                "case_index": case_index,
                "command": command,
                "error": str(exc),
            }
            summary["seed_failure_count"] += 1
            summary["seed_failures"].append(failure_payload)
            if args.abort_on_seed_failure:
                raise
            if not args.dry_run:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n===== {step_name} FAILED BUT CONTINUING =====\n")
                    handle.write(json.dumps(failure_payload, ensure_ascii=False, indent=2) + "\n")
            continue

    command_plan = [
        (
            "mine_hard_cases",
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "mine_hard_cases.py"),
                "--records-root",
                str(PROJECT_ROOT / "outputs"),
                "--output-dir",
                str(run_root / "hard_case_mining"),
            ],
        ),
        (
            "analyze_skill_helpfulness",
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "analyze_skill_helpfulness.py"),
                "--records-root",
                str(PROJECT_ROOT / "outputs"),
                "--output-dir",
                str(run_root / "skill_helpfulness"),
            ],
        ),
    ]

    if profile.include_batch_reflection:
        command_plan.extend(
            [
                (
                    "run_batch_reflection",
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "run_batch_reflection.py"),
                        "--records-root",
                        str(PROJECT_ROOT / "outputs"),
                        "--hard-cases-path",
                        str(run_root / "hard_case_mining" / "hard_cases.jsonl"),
                        "--output-dir",
                        str(run_root / "batch_reflection"),
                    ],
                ),
                (
                    "consolidate_experiences",
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "consolidate_experiences.py"),
                        "--experience-root",
                        str(PROJECT_ROOT / "state" / "experience"),
                        "--hard-cases-path",
                        str(run_root / "hard_case_mining" / "hard_cases.jsonl"),
                        "--output-dir",
                        str(run_root / "consolidation"),
                    ],
                ),
                (
                    "generate_skill_refinement_candidates",
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "generate_skill_refinement_candidates.py"),
                        "--records-root",
                        str(PROJECT_ROOT / "outputs"),
                        "--hard-cases-path",
                        str(run_root / "hard_case_mining" / "hard_cases.jsonl"),
                        "--skill-helpfulness-dir",
                        str(run_root / "skill_helpfulness"),
                        "--experience-root",
                        str(PROJECT_ROOT / "state" / "experience"),
                        "--output-dir",
                        str(run_root / "skill_refinement_candidates"),
                    ],
                ),
                (
                    "generate_composite_skill_proposals",
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "generate_composite_skill_proposals.py"),
                        "--batch-critique-path",
                        str(run_root / "batch_reflection" / "batch_critique.json"),
                        "--refinement-candidates-path",
                        str(run_root / "skill_refinement_candidates" / "skill_refinement_candidates.jsonl"),
                        "--experience-root",
                        str(PROJECT_ROOT / "state" / "experience"),
                        "--output-dir",
                        str(run_root / "composite_skill_proposals"),
                    ],
                ),
            ]
        )

    command_plan.extend(
        [
            (
                "train_learned_components",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_learned_components.py"),
                    "--stages",
                    "0,1,2,3,4",
                    "--run-id",
                    run_id,
                    "--records-root",
                    str(PROJECT_ROOT / "outputs"),
                    "--output-dir",
                    str(run_root / "train_runs"),
                    "--checkpoint-out-dir",
                    str(run_root / "checkpoints"),
                    "--training-split",
                    "train",
                    "--split-json",
                    str(split_json_path),
                    "--epochs",
                    str(profile.training_epochs),
                    "--limit",
                    str(profile.stage_training_limit),
                    "--case-offset",
                    str(val_offset),
                    "--stage3-data-split",
                    "val",
                    "--client-timeout",
                    str(args.client_timeout),
                    "--client-max-retries",
                    str(args.client_max_retries),
                ],
            ),
            (
                "select_best_checkpoint",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "select_best_checkpoint.py"),
                    "--train-runs-root",
                    str(run_root / "train_runs"),
                    "--checkpoints-root",
                    str(run_root / "checkpoints"),
                    "--output-dir",
                    str(run_root / "checkpoint_selection"),
                    "--run-ids",
                    run_id,
                ],
            ),
            (
                "compare_agent_vs_qwen",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "compare_agent_vs_qwen.py"),
                    "--data-root",
                    str(args.data_root),
                    "--limit",
                    str(val_compare_count),
                    "--case-offset",
                    str(val_offset),
                    "--data-split",
                    "val",
                    "--split-json",
                    str(split_json_path),
                    "--output-dir",
                    str(run_root / "comparison"),
                    "--policy-config",
                    str(run_root / "train_runs" / run_id / "stage3_policy_candidate_evaluation" / "candidate_policy.json"),
                    "--policy-label",
                    f"{run_id}_candidate_policy",
                    "--client-timeout",
                    str(args.client_timeout),
                    "--client-max-retries",
                    str(args.client_max_retries),
                ],
            ),
            (
                "run_eval_brief",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_eval_brief.py"),
                    "--data-root",
                    str(args.data_root),
                    "--limit",
                    str(test_eval_count),
                    "--case-offset",
                    str(test_offset),
                    "--data-split",
                    "test",
                    "--split-json",
                    str(split_json_path),
                    "--output-dir",
                    str(run_root / "evaluation_protocol"),
                    "--policy-config",
                    str(run_root / "train_runs" / run_id / "stage3_policy_candidate_evaluation" / "candidate_policy.json"),
                    "--client-timeout",
                    str(args.client_timeout),
                    "--client-max-retries",
                    str(args.client_max_retries),
                ],
            ),
        ]
    )

    if profile.include_ablations and not args.skip_ablations:
        command_plan.append(
            (
                "run_ablations",
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_ablations.py"),
                    "--data-root",
                    str(args.data_root),
                    "--output-dir",
                    str(run_root / "ablations"),
                    "--mode",
                    "smoke",
                    "--limit",
                    str(ablation_count),
                    "--case-offset",
                    str(test_offset),
                    "--data-split",
                    "test",
                    "--split-json",
                    str(split_json_path),
                    "--client-timeout",
                    str(args.client_timeout),
                    "--client-max-retries",
                    str(args.client_max_retries),
                ],
            )
        )

    train_learned_components_command = next(
        (command for step_name, command in command_plan if step_name == "train_learned_components"),
        None,
    )
    if train_learned_components_command is not None:
        if args.controller_checkpoint_in:
            train_learned_components_command.extend(
                ["--controller-checkpoint-in", str(args.controller_checkpoint_in.expanduser().resolve())]
            )
        if args.retrieval_checkpoint_in:
            train_learned_components_command.extend(
                ["--retrieval-checkpoint-in", str(args.retrieval_checkpoint_in.expanduser().resolve())]
            )

    command_plan.append(
        (
            "audit_experiment_state",
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "audit_experiment_state.py"),
                "--records-root",
                str(run_root / "evaluation_protocol"),
                "--eval-root",
                str(run_root / "evaluation_protocol"),
                "--split-json",
                str(split_json_path),
                "--fail-on-issues",
            ],
        )
    )

    if profile.include_paper_exports and not args.skip_paper_exports:
        command_plan.extend(
            [
                (
                    "export_paper_tables",
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "export_paper_tables.py"),
                        "--outputs-root",
                        str(run_root),
                        "--output-dir",
                        str(run_root / "paper_exports" / "tables"),
                    ],
                ),
                (
                    "export_paper_fig_data",
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "export_paper_fig_data.py"),
                        "--outputs-root",
                        str(run_root),
                        "--output-dir",
                        str(run_root / "paper_exports" / "fig_data"),
                    ],
                ),
            ]
        )

    for step_name, command in command_plan:
        summary["steps"].append(
            run_command(
                command,
                step_name=step_name,
                log_path=log_path,
                dry_run=args.dry_run,
            )
        )

    summary_path = run_root / "smoke_cycle_manifest.json"
    summary["summary_path"] = str(summary_path)
    key_results = build_key_results(run_root)
    key_results_path = run_root / "key_results.json"
    summary["key_results_path"] = str(key_results_path)
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        key_results_path.write_text(json.dumps(key_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
