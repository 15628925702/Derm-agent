from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import CURRENT_STABLE_POLICY_PATH, load_policy
from scripts.train_controller import DEFAULT_EXAMPLES_PATH, FALLBACK_EXAMPLES_PATH, resolve_examples_path


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "controller_sparsification"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a sparse learned controller candidate, then optionally run a small frozen policy compare.")
    parser.add_argument("--examples-path", type=Path, default=DEFAULT_EXAMPLES_PATH, help="Controller training examples JSONL.")
    parser.add_argument("--split-json", type=Path, default=None, help="Optional fixed split JSON used by controller training and frozen compare.")
    parser.add_argument("--base-policy", type=Path, default=CURRENT_STABLE_POLICY_PATH, help="Base stable policy JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for sparse-controller artifacts.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--frozen-eval-split", type=str, default="val", choices=("val", "test"), help="Frozen compare split.")
    parser.add_argument("--frozen-eval-limit", type=int, default=8, help="Small frozen compare case count.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start offset for frozen compare.")
    parser.add_argument("--skip-frozen-compare", action="store_true", help="Only train/evaluate the controller checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--client-timeout", type=float, default=None)
    parser.add_argument("--client-max-retries", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = resolve_examples_path(args.examples_path) if args.examples_path == DEFAULT_EXAMPLES_PATH else args.examples_path
    if not examples_path.exists() and args.examples_path == DEFAULT_EXAMPLES_PATH and FALLBACK_EXAMPLES_PATH.exists():
        examples_path = FALLBACK_EXAMPLES_PATH
    if not examples_path.exists():
        raise FileNotFoundError(f"Controller examples not found: {examples_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_dir / f"sparse_eval_{stamp}"
    checkpoint_dir = run_root / "checkpoints"
    eval_dir = run_root / "eval"
    compare_dir = run_root / "frozen_compare"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = "controller_sparse_candidate.pt"
    train_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_controller.py"),
        "--examples-path",
        str(examples_path),
        "--output-dir",
        str(checkpoint_dir),
        "--checkpoint-name",
        checkpoint_name,
        "--selection-profile",
        "conservative_sparse",
        "--seed",
        str(int(args.seed)),
        "--epochs",
        str(max(1, int(args.epochs))),
    ]
    if args.split_json:
        train_cmd.extend(["--split-json", str(args.split_json)])
    run_command(train_cmd, cwd=PROJECT_ROOT)

    checkpoint_path = checkpoint_dir / checkpoint_name
    metrics_path = checkpoint_dir / f"{checkpoint_path.stem}_metrics.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint missing: {checkpoint_path}")

    eval_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_controller.py"),
        "--checkpoint-path",
        str(checkpoint_path),
        "--examples-path",
        str(examples_path),
        "--split",
        "all",
        "--output-path",
        str(eval_dir / "controller_eval.json"),
        "--predictions-path",
        str(eval_dir / "controller_predictions.jsonl"),
    ]
    run_command(eval_cmd, cwd=PROJECT_ROOT)

    candidate_policy_path = run_root / "candidate_policy.json"
    candidate_policy = build_sparse_candidate_policy(
        base_policy_path=args.base_policy,
        checkpoint_path=checkpoint_path,
    )
    candidate_policy_path.write_text(json.dumps(candidate_policy, ensure_ascii=False, indent=2), encoding="utf-8")

    frozen_compare_path = ""
    if not args.skip_frozen_compare:
        compare_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_policy_candidate.py"),
            "--candidate-config",
            str(candidate_policy_path),
            "--stable-config",
            str(args.base_policy),
            "--data-root",
            str(args.data_root),
            "--limit",
            str(max(1, int(args.frozen_eval_limit))),
            "--case-offset",
            str(max(0, int(args.case_offset))),
            "--data-split",
            str(args.frozen_eval_split),
            "--output-dir",
            str(compare_dir),
        ]
        if args.split_json:
            compare_cmd.extend(["--split-json", str(args.split_json)])
        if args.client_timeout is not None:
            compare_cmd.extend(["--client-timeout", str(float(args.client_timeout))])
        if args.client_max_retries is not None:
            compare_cmd.extend(["--client-max-retries", str(int(args.client_max_retries))])
        run_command(compare_cmd, cwd=PROJECT_ROOT)
        frozen_compare_path = str(compare_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "examples_path": str(examples_path),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "controller_eval_path": str(eval_dir / "controller_eval.json"),
        "controller_predictions_path": str(eval_dir / "controller_predictions.jsonl"),
        "candidate_policy_path": str(candidate_policy_path),
        "frozen_compare_dir": frozen_compare_path,
        "train_command": train_cmd,
        "eval_command": eval_cmd,
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "manifest_path": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


def build_sparse_candidate_policy(*, base_policy_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    base_policy = load_policy(base_policy_path).to_dict()
    checkpoint = json.loads((checkpoint_path.parent / f"{checkpoint_path.stem}_metrics.json").read_text(encoding="utf-8"))
    selection_policy = {}
    payload_path = checkpoint_path
    torch_payload = None
    try:
        import torch  # local import to avoid unnecessary dependency at import time

        torch_payload = torch.load(payload_path, map_location="cpu")
        selection_policy = dict(torch_payload.get("selection_policy", {}))
    except Exception:
        selection_policy = {}
    previous_policy_id = str(base_policy.get("policy_id", "")).strip() or "stable_default_policy"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_policy["policy_id"] = f"sparse_controller_candidate_{stamp}"
    base_policy["status"] = "candidate"
    base_policy["parent_policy_id"] = previous_policy_id
    base_policy["description"] = "Sparse-hybrid learned controller candidate focused on reducing over-selection."
    planner_policy = dict(base_policy.get("planner_policy", {}))
    planner_policy.update(
        {
            "controller_family": "learned_supervised",
            "controller_checkpoint_path": str(checkpoint_path),
            "learned_controller_select_threshold": float(selection_policy.get("threshold", planner_policy.get("learned_controller_select_threshold", 0.52))),
            "learned_controller_top_k": int(selection_policy.get("target_top_k", 3) or 3),
            "learned_controller_force_top_k": 0,
            "learned_controller_mode": "sparse_hybrid",
            "learned_controller_sparse_rule_floor": 9,
            "learned_controller_protected_min_score": 7,
        }
    )
    base_policy["planner_policy"] = planner_policy
    base_policy["change_summary"] = {
        **dict(base_policy.get("change_summary", {})),
        "planner": [
            "Switch learned controller to sparse-hybrid mode.",
            "Use conservative sparse checkpoint with tighter threshold/top-k and harmful-aware training.",
        ],
    }
    base_policy["evidence_refs"] = {
        **dict(base_policy.get("evidence_refs", {})),
        "controller_metrics_report": checkpoint,
        "controller_checkpoint_path": str(checkpoint_path),
    }
    return base_policy


def run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")


if __name__ == "__main__":
    raise SystemExit(main())
