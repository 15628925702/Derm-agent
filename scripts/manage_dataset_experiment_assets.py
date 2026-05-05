from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.contamination_guard import normalize_split_name
from agent.experiment_state import ensure_split_state_paths
from agent.policy_config import ensure_policy_store, load_policy
from cognition.cognition_state import CognitionState
from memory.experience_store import ExperienceStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage isolated asset roots for a dataset-specific DermAgent adaptation experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize isolated policy/split-state/output/checkpoint roots for a new dataset experiment.",
    )
    init_parser.add_argument("--experiment-id", type=str, required=True, help="Short experiment identifier.")
    init_parser.add_argument("--base-policy-config", type=Path, default=None, help="Optional source policy JSON to seed the isolated policy root.")
    init_parser.add_argument("--assets-root", type=Path, default=PROJECT_ROOT / "state" / "dataset_adaptation")
    init_parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_adaptation")
    init_parser.add_argument("--checkpoints-root", type=Path, default=PROJECT_ROOT / "outputs" / "checkpoints" / "dataset_adaptation")

    promote_parser = subparsers.add_parser(
        "promote-state",
        help="Copy a bootstrap split state into one or more evaluation splits under the same isolated root.",
    )
    promote_parser.add_argument("--split-state-root", type=Path, required=True, help="Isolated split-state root.")
    promote_parser.add_argument("--source-split", type=str, default="train", help="Source split to copy from.")
    promote_parser.add_argument(
        "--target-splits",
        type=str,
        default="val,test",
        help="Comma-separated target splits to overwrite, e.g. `val,test`.",
    )

    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_seed_policy(*, base_policy_config: Path, policy_root: Path) -> dict[str, Any]:
    policy_root.mkdir(parents=True, exist_ok=True)
    source_policy = _sanitize_policy_for_heuristic_only(load_policy(base_policy_config).to_dict())
    target_policy_path = policy_root / "current_stable_policy.json"
    source_policy["source_path"] = str(target_policy_path)
    _write_json(target_policy_path, source_policy)
    manifest_path = policy_root / "manifest.json"
    manifest_payload = {
        "stable_policy_id": source_policy.get("policy_id"),
        "stable_policy_path": str(target_policy_path),
        "policies": [
            {
                "policy_id": source_policy.get("policy_id"),
                "version": source_policy.get("version"),
                "status": source_policy.get("status", "stable"),
                "source_path": str(target_policy_path),
            }
        ],
        "evaluations": [],
    }
    _write_json(manifest_path, manifest_payload)
    return source_policy


def _sanitize_policy_for_heuristic_only(policy: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(policy)
    planner_policy = dict(sanitized.get("planner_policy", {}) or {})
    retrieval_policy = dict(sanitized.get("retrieval_policy", {}) or {})
    evidence_policy = dict(sanitized.get("evidence_policy", {}) or {})

    planner_policy["controller_family"] = "heuristic"
    planner_policy["controller_checkpoint_path"] = ""
    planner_policy["learned_controller_top_k"] = 0
    planner_policy["learned_controller_force_top_k"] = 0

    retrieval_policy["enable_learned_retrieval_reranker"] = False
    retrieval_policy["retrieval_reranker_checkpoint_path"] = ""

    evidence_policy["calibrator_mode"] = "heuristic"
    evidence_policy["calibrator_checkpoint_path"] = ""

    sanitized["planner_policy"] = planner_policy
    sanitized["retrieval_policy"] = retrieval_policy
    sanitized["evidence_policy"] = evidence_policy
    return sanitized


def init_experiment_assets(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = str(args.experiment_id).strip()
    if not experiment_id:
        raise ValueError("experiment_id must be non-empty.")

    asset_root = args.assets_root / experiment_id
    policy_root = asset_root / "policy"
    split_state_root = asset_root / "split_states"
    output_root = args.outputs_root / experiment_id
    checkpoint_root = args.checkpoints_root / experiment_id

    policy_root.mkdir(parents=True, exist_ok=True)
    split_state_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if args.base_policy_config is not None:
        seeded_policy = _copy_seed_policy(base_policy_config=args.base_policy_config, policy_root=policy_root)
    else:
        import os

        previous_root = os.environ.get("DERMAGENT_POLICY_ROOT")
        os.environ["DERMAGENT_POLICY_ROOT"] = str(policy_root)
        try:
            ensure_policy_store()
            seeded_policy = _sanitize_policy_for_heuristic_only(
                load_policy(policy_root / "current_stable_policy.json").to_dict()
            )
            _write_json(policy_root / "current_stable_policy.json", seeded_policy)
        finally:
            if previous_root is None:
                os.environ.pop("DERMAGENT_POLICY_ROOT", None)
            else:
                os.environ["DERMAGENT_POLICY_ROOT"] = previous_root

    for split_name in ("train", "val", "test"):
        ensure_split_state_paths(data_split=split_name, split_state_root=split_state_root, policy_path=policy_root / "current_stable_policy.json")

    env_payload = {
        "DERMAGENT_POLICY_ROOT": str(policy_root),
        "DERMAGENT_SPLIT_STATE_ROOT": str(split_state_root),
        "DERMAGENT_DATASET_EXPERIMENT_OUTPUT_ROOT": str(output_root),
        "DERMAGENT_DATASET_EXPERIMENT_CHECKPOINT_ROOT": str(checkpoint_root),
    }
    env_path = asset_root / "experiment.env"
    env_path.write_text(
        "\n".join(f'export {key}="{value}"' for key, value in env_payload.items()) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment_id": experiment_id,
        "asset_root": str(asset_root),
        "policy_root": str(policy_root),
        "split_state_root": str(split_state_root),
        "output_root": str(output_root),
        "checkpoint_root": str(checkpoint_root),
        "env_path": str(env_path),
        "seed_policy_id": seeded_policy.get("policy_id", ""),
        "seed_policy_version": seeded_policy.get("version", ""),
    }
    _write_json(asset_root / "asset_manifest.json", manifest)
    return manifest


def promote_split_state(args: argparse.Namespace) -> dict[str, Any]:
    split_state_root = args.split_state_root
    source_split = normalize_split_name(args.source_split, default="train")
    target_splits = [
        normalize_split_name(item, default="")
        for item in str(args.target_splits).split(",")
        if normalize_split_name(item, default="")
    ]
    if not target_splits:
        raise ValueError("At least one target split is required.")

    source_root = split_state_root / source_split
    source_experience_root = source_root / "experience"
    source_cognition_path = source_root / "cognition_state.json"
    if not source_experience_root.exists():
        raise FileNotFoundError(f"Missing source experience root: {source_experience_root}")
    if not source_cognition_path.exists():
        raise FileNotFoundError(f"Missing source cognition path: {source_cognition_path}")

    promotion_records: list[dict[str, Any]] = []
    for target_split in target_splits:
        target_root = split_state_root / target_split
        target_experience_root = target_root / "experience"
        target_cognition_path = target_root / "cognition_state.json"

        if target_experience_root.exists():
            shutil.rmtree(target_experience_root)
        target_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_experience_root, target_experience_root)
        ExperienceStore(target_experience_root, split_name=target_split).refresh_metadata()

        shutil.copy2(source_cognition_path, target_cognition_path)
        cognition = CognitionState.load(target_cognition_path)
        cognition.state_split = target_split
        cognition.save(target_cognition_path)

        promotion_records.append(
            {
                "source_split": source_split,
                "target_split": target_split,
                "target_experience_root": str(target_experience_root),
                "target_cognition_path": str(target_cognition_path),
            }
        )

    payload = {
        "split_state_root": str(split_state_root),
        "source_split": source_split,
        "target_splits": target_splits,
        "promotions": promotion_records,
    }
    _write_json(split_state_root / "promotion_manifest.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.command == "init":
        payload = init_experiment_assets(args)
    elif args.command == "promote-state":
        payload = promote_split_state(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
