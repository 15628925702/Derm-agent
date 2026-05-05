from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for train_learned_components stage3/4 completion, then run the remaining smoke-run tail steps."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--val-limit", type=int, required=True)
    parser.add_argument("--val-offset", type=int, required=True)
    parser.add_argument("--test-limit", type=int, required=True)
    parser.add_argument("--test-offset", type=int, required=True)
    parser.add_argument("--client-timeout", type=float, default=180.0)
    parser.add_argument("--client-max-retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def read_status(train_manifest_path: Path) -> str:
    if not train_manifest_path.exists():
        return "missing"
    payload = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    return str(payload.get("status", "missing")).strip() or "missing"


def run_command(command: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    train_run_root = run_root / "train_runs" / args.run_id
    train_manifest_path = train_run_root / "train_run_manifest.json"
    candidate_policy_path = train_run_root / "stage3_policy_candidate_evaluation" / "candidate_policy.json"

    while True:
        status = read_status(train_manifest_path)
        print(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} waiting_for_stage34 status={status}",
            flush=True,
        )
        if status == "completed":
            break
        if status == "failed":
            print("upstream_stage34_failed", flush=True)
            return 1
        time.sleep(max(5, int(args.poll_seconds)))

    if not candidate_policy_path.exists():
        raise FileNotFoundError(f"Candidate policy missing: {candidate_policy_path}")

    run_command(
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
            args.run_id,
        ]
    )
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compare_agent_vs_qwen.py"),
            "--data-root",
            str(args.data_root),
            "--limit",
            str(args.val_limit),
            "--case-offset",
            str(args.val_offset),
            "--data-split",
            "val",
            "--split-json",
            str(args.split_json),
            "--output-dir",
            str(run_root / "comparison"),
            "--policy-config",
            str(candidate_policy_path),
            "--policy-label",
            f"{args.run_id}_candidate_policy",
            "--client-timeout",
            str(args.client_timeout),
            "--client-max-retries",
            str(args.client_max_retries),
        ],
        extra_env={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY")},
    )
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_eval_brief.py"),
            "--data-root",
            str(args.data_root),
            "--limit",
            str(args.test_limit),
            "--case-offset",
            str(args.test_offset),
            "--data-split",
            "test",
            "--split-json",
            str(args.split_json),
            "--output-dir",
            str(run_root / "evaluation_protocol"),
            "--policy-config",
            str(candidate_policy_path),
            "--client-timeout",
            str(args.client_timeout),
            "--client-max-retries",
            str(args.client_max_retries),
        ],
        extra_env={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY")},
    )
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_experiment_state.py"),
            "--records-root",
            str(run_root / "evaluation_protocol"),
            "--eval-root",
            str(run_root / "evaluation_protocol"),
            "--split-json",
            str(args.split_json),
            "--fail-on-issues",
        ]
    )
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_paper_tables.py"),
            "--outputs-root",
            str(run_root),
            "--output-dir",
            str(run_root / "paper_exports" / "tables"),
        ]
    )
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_paper_fig_data.py"),
            "--outputs-root",
            str(run_root),
            "--output-dir",
            str(run_root / "paper_exports" / "fig_data"),
        ]
    )
    print("tail_resume_completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
