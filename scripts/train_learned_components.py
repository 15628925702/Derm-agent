from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller_training import export_controller_training_data, save_controller_training_data
from agent.hard_case_miner import load_execution_records
from agent.policy_config import CURRENT_STABLE_POLICY_PATH, load_policy


PIPELINE_SCHEMA_VERSION = "staged_training_pipeline_v1"
DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_TRAIN_RUNS_ROOT = PROJECT_ROOT / "outputs" / "train_runs"
DEFAULT_CHECKPOINT_EXPORT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
RUN_MANIFEST_NAME = "train_run_manifest.json"

STAGE_NAMES: dict[int, str] = {
    0: "bootstrap_collect_data",
    1: "supervised_controller_training",
    2: "retrieval_scorer_training",
    3: "policy_candidate_evaluation",
    4: "stable_checkpoint_export",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged training pipeline for DermAgent learnable components.")
    parser.add_argument("--stages", type=str, default="0,1,2,3,4", help="Comma-separated stage ids or `all`.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id. Default uses UTC timestamp.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing execution records.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root used in stage 3 policy evaluation.")
    parser.add_argument("--stable-policy-config", type=Path, default=CURRENT_STABLE_POLICY_PATH, help="Stable policy config path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_RUNS_ROOT, help="Training run root directory.")
    parser.add_argument("--checkpoint-out-dir", type=Path, default=DEFAULT_CHECKPOINT_EXPORT_ROOT, help="Global checkpoint export directory.")
    parser.add_argument("--dataset-filter", type=str, default="", help="Optional dataset name filter for records/examples.")
    parser.add_argument("--training-split", type=str, default="train", choices=("train", "val", "test"), help="Expected split label for training-time data collection in stage 0.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max case count for stage 0 and stage 3.")
    parser.add_argument("--epochs", type=int, default=30, help="Epoch count for stage 1 and stage 2.")
    parser.add_argument("--split-json", type=Path, default=None, help="Optional split json with case ids.")
    parser.add_argument("--seed", type=int, default=42, help="Reserved training seed for stage scripts.")
    parser.add_argument("--case-offset", type=int, default=0, help="Case offset for stage 3 policy candidate evaluation.")
    parser.add_argument("--stage3-data-split", type=str, default="val", choices=("train", "val", "test"), help="Logical split label for stage 3 policy candidate evaluation.")
    parser.add_argument("--controller-examples-path", type=Path, default=None, help="Optional controller examples JSONL path for stage 1.")
    parser.add_argument("--controller-checkpoint-in", type=Path, default=None, help="Optional pre-trained controller checkpoint input.")
    parser.add_argument("--retrieval-checkpoint-in", type=Path, default=None, help="Optional pre-trained retrieval scorer checkpoint input.")
    parser.add_argument("--evidence-calibrator-checkpoint-in", type=Path, default=None, help="Optional pre-trained evidence calibrator checkpoint input.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Optional local Qwen client timeout for stage 3.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Optional local Qwen client retries for stage 3.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare manifests/commands without running stage scripts.")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_stage_selection(raw: str) -> list[int]:
    text = str(raw).strip().lower()
    if text in {"", "all"}:
        return list(STAGE_NAMES.keys())
    parsed: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        stage_id = int(token)
        if stage_id not in STAGE_NAMES:
            raise ValueError(f"Unsupported stage id `{stage_id}`. Allowed: {sorted(STAGE_NAMES.keys())}")
        parsed.add(stage_id)
    if not parsed:
        raise ValueError("No valid stages selected.")
    return sorted(parsed)


def build_stage_dir(run_root: Path, stage_id: int) -> Path:
    return run_root / f"stage{stage_id}_{STAGE_NAMES[stage_id]}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_id_split(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    result: set[str] = set()
    for key in ("train", "val", "test", "train_case_ids", "val_case_ids", "test_case_ids"):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            case_id = str(item).strip()
            if case_id:
                result.add(case_id)
    return result


def apply_record_filters(
    records: list[dict[str, Any]],
    *,
    dataset_filter: str,
    case_id_whitelist: set[str],
    expected_split: str,
    limit: int,
) -> list[dict[str, Any]]:
    filtered = list(records)
    dataset_name = dataset_filter.strip()
    if dataset_name:
        filtered = [row for row in filtered if str(row.get("dataset_name", "")).strip() == dataset_name]
    if case_id_whitelist:
        filtered = [row for row in filtered if str(row.get("case_id", "")).strip() in case_id_whitelist]
    normalized_split = str(expected_split).strip().lower()
    if normalized_split:
        filtered = [
            row
            for row in filtered
            if str(row.get("state_versions", {}).get("data_split", "")).strip().lower() in {"", normalized_split}
        ]
    filtered = [
        row
        for row in filtered
        if "frozen" not in str(row.get("state_versions", {}).get("run_mode", "")).strip().lower()
    ]
    filtered.sort(key=lambda row: (str(row.get("timestamp", "")), str(row.get("case_id", ""))))
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def resolve_controller_checkpoint(stage_context: dict[str, Any], args: argparse.Namespace) -> str:
    stage_ckpt = str(stage_context.get("stage1_controller_checkpoint_path", "")).strip()
    if stage_ckpt:
        return stage_ckpt
    if args.controller_checkpoint_in:
        return str(args.controller_checkpoint_in.expanduser())
    return ""


def resolve_retrieval_checkpoint(stage_context: dict[str, Any], args: argparse.Namespace) -> str:
    stage_ckpt = str(stage_context.get("stage2_retrieval_checkpoint_path", "")).strip()
    if stage_ckpt:
        return stage_ckpt
    if args.retrieval_checkpoint_in:
        return str(args.retrieval_checkpoint_in.expanduser())
    return ""


def resolve_evidence_calibrator_checkpoint(stage_context: dict[str, Any], args: argparse.Namespace) -> str:
    stage_ckpt = str(stage_context.get("evidence_calibrator_checkpoint_path", "")).strip()
    if stage_ckpt:
        return stage_ckpt
    if args.evidence_calibrator_checkpoint_in:
        return str(args.evidence_calibrator_checkpoint_in.expanduser())
    return ""


def hydrate_stage_context_from_existing_run(
    run_root: Path,
    *,
    selected_stages: list[int],
    stage_context: dict[str, Any],
) -> None:
    for stage_id in sorted(STAGE_NAMES):
        if stage_id in selected_stages:
            continue
        manifest_path = build_stage_dir(run_root, stage_id) / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        outputs = dict(manifest.get("outputs", {}) or {})
        checkpoint = dict(manifest.get("checkpoint", {}) or {})
        if stage_id == 0:
            stage_context["stage0_records_root"] = str(outputs.get("records_snapshot_root", "")).strip()
            stage_context["stage0_controller_examples_path"] = str(outputs.get("controller_examples_path", "")).strip()
            stage_context["stage0_data_version"] = str(outputs.get("data_version", "")).strip()
        elif stage_id == 1:
            stage_context["stage1_controller_checkpoint_path"] = str(outputs.get("checkpoint_path", "")).strip()
            stage_context["stage1_controller_metrics_path"] = str(outputs.get("metrics_path", "")).strip()
            stage_context["stage1_controller_version"] = str(outputs.get("version", "")).strip() or str(
                checkpoint.get("version", "")
            ).strip()
        elif stage_id == 2:
            stage_context["stage2_retrieval_checkpoint_path"] = str(outputs.get("checkpoint_path", "")).strip()
            stage_context["stage2_retrieval_metrics_path"] = str(outputs.get("metrics_path", "")).strip()
            stage_context["stage2_retrieval_version"] = str(outputs.get("version", "")).strip() or str(
                checkpoint.get("version", "")
            ).strip()
        elif stage_id == 3:
            candidate_policy_path = str(outputs.get("candidate_policy_path", "")).strip()
            if not candidate_policy_path:
                candidate_policy_path = str(manifest.get("inputs", {}).get("candidate_policy_path", "")).strip()
            if not candidate_policy_path:
                fallback = build_stage_dir(run_root, 3) / "candidate_policy.json"
                if fallback.exists():
                    candidate_policy_path = str(fallback)
            stage_context["stage3_candidate_policy_path"] = candidate_policy_path
            checkpoint_version = str(checkpoint.get("version", "")).strip()
            if checkpoint_version and "@v" in checkpoint_version:
                policy_id, version = checkpoint_version.split("@v", 1)
                stage_context["stage3_candidate_policy_id"] = policy_id.strip()
                stage_context["stage3_candidate_policy_version"] = version.strip()
            elif candidate_policy_path and Path(candidate_policy_path).exists():
                candidate_policy = read_json(Path(candidate_policy_path))
                stage_context["stage3_candidate_policy_id"] = str(candidate_policy.get("policy_id", "")).strip()
                stage_context["stage3_candidate_policy_version"] = str(candidate_policy.get("version", "")).strip()


def build_initial_run_manifest(
    *,
    run_manifest_path: Path,
    run_id: str,
    args: argparse.Namespace,
    selected_stages: list[int],
) -> dict[str, Any]:
    if run_manifest_path.exists():
        existing = read_json(run_manifest_path)
        existing["selected_stages"] = selected_stages
        existing["stage_names"] = {str(stage_id): STAGE_NAMES[stage_id] for stage_id in selected_stages}
        existing.setdefault("paths", {})
        existing["paths"]["run_root"] = str(run_manifest_path.parent)
        existing["paths"]["run_manifest_path"] = str(run_manifest_path)
        existing["status"] = "running"
        existing.pop("error", None)
        existing.pop("failed_at", None)
        return existing

    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "running",
        "selected_stages": selected_stages,
        "stage_names": {str(stage_id): STAGE_NAMES[stage_id] for stage_id in selected_stages},
        "train_eval_separation": {
            "formal_eval_scripts_not_called": True,
            "policy_candidate_eval_output_under_train_run": True,
        },
        "config": {
            "records_root": str(args.records_root),
            "data_root": str(args.data_root),
            "stable_policy_config": str(args.stable_policy_config),
            "dataset_filter": args.dataset_filter.strip(),
            "training_split": str(args.training_split),
            "limit": int(args.limit),
            "epochs": int(args.epochs),
            "split_json": str(args.split_json) if args.split_json else "",
            "seed": int(args.seed),
            "case_offset": int(args.case_offset),
            "stage3_data_split": str(args.stage3_data_split),
            "controller_examples_path": str(args.controller_examples_path) if args.controller_examples_path else "",
            "controller_checkpoint_in": str(args.controller_checkpoint_in) if args.controller_checkpoint_in else "",
            "retrieval_checkpoint_in": str(args.retrieval_checkpoint_in) if args.retrieval_checkpoint_in else "",
            "evidence_calibrator_checkpoint_in": str(args.evidence_calibrator_checkpoint_in) if args.evidence_calibrator_checkpoint_in else "",
            "checkpoint_out_dir": str(args.checkpoint_out_dir),
            "dry_run": bool(args.dry_run),
        },
        "paths": {
            "run_root": str(run_manifest_path.parent),
            "run_manifest_path": str(run_manifest_path),
        },
        "stages": {},
    }


def run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    print(f"[pipeline] running command: {' '.join(command)}", flush=True)
    if dry_run:
        print("[pipeline] dry-run: command skipped", flush=True)
        return {"returncode": 0, "stdout": "", "stderr": "", "dry_run": True}
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    streamed_lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        streamed_lines.append(line)
    returncode = process.wait()
    stdout_text = "".join(streamed_lines)
    print(f"[pipeline] command finished: returncode={returncode}", flush=True)
    if returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cmd={' '.join(command)}\n"
            f"returncode={returncode}\n"
            f"stdout={stdout_text}\n"
            "stderr="
        )
    return {
        "returncode": returncode,
        "stdout": stdout_text,
        "stderr": "",
        "dry_run": False,
    }


def run_stage0_bootstrap(args: argparse.Namespace, *, run_id: str, run_root: Path, stage_context: dict[str, Any]) -> dict[str, Any]:
    stage_id = 0
    stage_dir = build_stage_dir(run_root, stage_id)
    started_at = utc_now()

    source_records = load_execution_records(args.records_root)
    split_case_ids = load_case_id_split(args.split_json)
    filtered_records = apply_record_filters(
        source_records,
        dataset_filter=args.dataset_filter,
        case_id_whitelist=split_case_ids,
        expected_split=str(args.training_split),
        limit=max(0, int(args.limit)),
    )

    records_snapshot_dir = stage_dir / "records_snapshot"
    records_snapshot_path = records_snapshot_dir / "case_execution_records.jsonl"
    sanitized_rows: list[dict[str, Any]] = []
    for row in filtered_records:
        copied = dict(row)
        copied.pop("_source_record_path", None)
        sanitized_rows.append(copied)
    write_jsonl(records_snapshot_path, sanitized_rows)

    examples, summary = export_controller_training_data(sanitized_rows, dataset_name=None)
    controller_training_dir = stage_dir / "controller_training_data"
    saved_paths = save_controller_training_data(examples, summary, output_dir=controller_training_dir)
    case_ids = [str(row.get("case_id", "")).strip() for row in sanitized_rows if str(row.get("case_id", "")).strip()]
    data_hash = hashlib.sha1("\n".join(sorted(case_ids)).encode("utf-8")).hexdigest()[:12] if case_ids else "empty"
    data_version = f"bootstrap_data_{run_id}_{data_hash}"

    stage_context["stage0_records_root"] = str(records_snapshot_dir)
    stage_context["stage0_controller_examples_path"] = str(saved_paths["examples_path"])
    stage_context["stage0_data_version"] = data_version
    stage_context["stage0_case_ids"] = case_ids

    finished_at = utc_now()
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "stage_name": STAGE_NAMES[stage_id],
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "records_root": str(args.records_root),
            "dataset_filter": args.dataset_filter.strip(),
            "split_json": str(args.split_json) if args.split_json else "",
            "training_split": str(args.training_split),
            "limit": max(0, int(args.limit)),
        },
        "outputs": {
            "records_snapshot_root": str(records_snapshot_dir),
            "records_snapshot_path": str(records_snapshot_path),
            "controller_examples_path": str(saved_paths["examples_path"]),
            "controller_summary_path": str(saved_paths["summary_path"]),
            "data_version": data_version,
        },
        "stats": {
            "records_scanned": len(source_records),
            "records_selected": len(filtered_records),
            "controller_example_count": len(examples),
            "dataset_counts": summary.get("dataset_counts", {}),
        },
        "checkpoint": {
            "component": "bootstrap_data_snapshot",
            "version": data_version,
        },
    }
    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def run_stage1_controller_training(args: argparse.Namespace, *, run_id: str, run_root: Path, stage_context: dict[str, Any]) -> dict[str, Any]:
    stage_id = 1
    stage_dir = build_stage_dir(run_root, stage_id)
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    checkpoint_name = f"controller_planner_scorer__candidate__{run_id}.pt"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_controller.py"),
        "--output-dir",
        str(checkpoint_dir),
        "--selection-profile",
        "conservative_sparse",
        "--epochs",
        str(max(1, int(args.epochs))),
        "--checkpoint-name",
        checkpoint_name,
        "--seed",
        str(int(args.seed)),
    ]
    examples_path = ""
    if args.controller_examples_path:
        examples_path = str(args.controller_examples_path.expanduser())
    elif stage_context.get("stage0_controller_examples_path"):
        examples_path = str(stage_context["stage0_controller_examples_path"])
    if examples_path:
        command.extend(["--examples-path", examples_path])
    dataset_filter = args.dataset_filter.strip()
    if dataset_filter:
        command.extend(["--dataset-filter", dataset_filter])
    if args.split_json:
        command.extend(["--split-json", str(args.split_json)])

    cmd_result = run_command(command, dry_run=bool(args.dry_run))
    checkpoint_path = checkpoint_dir / checkpoint_name
    metrics_path = checkpoint_dir / f"{checkpoint_path.stem}_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    stage_context["stage1_controller_checkpoint_path"] = str(checkpoint_path)
    stage_context["stage1_controller_metrics_path"] = str(metrics_path)
    stage_context["stage1_controller_version"] = f"controller_planner_scorer_v0.1.0+{run_id}"

    finished_at = utc_now()
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "stage_name": STAGE_NAMES[stage_id],
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "controller_examples_path": examples_path,
            "dataset_filter": dataset_filter,
            "split_json": str(args.split_json) if args.split_json else "",
            "epochs": max(1, int(args.epochs)),
            "seed": int(args.seed),
        },
        "outputs": {
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "version": stage_context["stage1_controller_version"],
            "stdout_tail": cmd_result.get("stdout", "")[-1200:],
        },
        "checkpoint": {
            "component": "controller_planner_scorer",
            "version": stage_context["stage1_controller_version"],
            "path": str(checkpoint_path),
        },
        "command": command,
        "dry_run": bool(args.dry_run),
        "metrics": metrics,
    }
    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def run_stage2_retrieval_training(args: argparse.Namespace, *, run_id: str, run_root: Path, stage_context: dict[str, Any]) -> dict[str, Any]:
    stage_id = 2
    stage_dir = build_stage_dir(run_root, stage_id)
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    checkpoint_name = f"retrieval_reranker__candidate__{run_id}.pt"
    records_root = str(stage_context.get("stage0_records_root") or args.records_root)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_retrieval_scorer.py"),
        "--records-root",
        records_root,
        "--output-dir",
        str(checkpoint_dir),
        "--epochs",
        str(max(1, int(args.epochs))),
        "--seed",
        str(int(args.seed)),
        "--checkpoint-name",
        checkpoint_name,
    ]
    dataset_filter = args.dataset_filter.strip()
    if dataset_filter:
        command.extend(["--dataset-filter", dataset_filter])
    if args.split_json:
        command.extend(["--split-json", str(args.split_json)])

    cmd_result = run_command(command, dry_run=bool(args.dry_run))
    checkpoint_path = checkpoint_dir / checkpoint_name
    metrics_path = checkpoint_dir / f"{checkpoint_path.stem}_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    stage_context["stage2_retrieval_checkpoint_path"] = str(checkpoint_path)
    stage_context["stage2_retrieval_metrics_path"] = str(metrics_path)
    stage_context["stage2_retrieval_version"] = f"retrieval_reranker_v0.1.0+{run_id}"

    finished_at = utc_now()
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "stage_name": STAGE_NAMES[stage_id],
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "records_root": records_root,
            "dataset_filter": dataset_filter,
            "split_json": str(args.split_json) if args.split_json else "",
            "epochs": max(1, int(args.epochs)),
            "seed": int(args.seed),
        },
        "outputs": {
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "version": stage_context["stage2_retrieval_version"],
            "stdout_tail": cmd_result.get("stdout", "")[-1200:],
        },
        "checkpoint": {
            "component": "retrieval_reranker",
            "version": stage_context["stage2_retrieval_version"],
            "path": str(checkpoint_path),
        },
        "command": command,
        "dry_run": bool(args.dry_run),
        "metrics": metrics,
    }
    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def build_stage3_candidate_policy(
    *,
    stable_policy: dict[str, Any],
    run_id: str,
    controller_checkpoint_path: str,
    retrieval_checkpoint_path: str,
    evidence_calibrator_checkpoint_path: str,
) -> dict[str, Any]:
    candidate = deepcopy(stable_policy)
    candidate["policy_id"] = f"candidate_stage4_step6_{run_id}"
    candidate["status"] = "candidate"
    candidate["parent_policy_id"] = str(stable_policy.get("policy_id", "")).strip()
    try:
        candidate["version"] = int(stable_policy.get("version", 1)) + 1
    except Exception:
        candidate["version"] = 1
    candidate["created_at"] = utc_now()
    candidate["description"] = "Auto-generated candidate policy from staged training pipeline."
    change_summary = dict(candidate.get("change_summary", {}))
    change_summary["pipeline"] = [f"staged_training_run:{run_id}"]
    candidate["change_summary"] = change_summary

    planner_policy = dict(candidate.get("planner_policy", {}))
    if controller_checkpoint_path:
        planner_policy["controller_family"] = "learned_supervised"
        planner_policy["controller_checkpoint_path"] = controller_checkpoint_path
    candidate["planner_policy"] = planner_policy

    retrieval_policy = dict(candidate.get("retrieval_policy", {}))
    if retrieval_checkpoint_path:
        retrieval_policy["enable_learned_retrieval_reranker"] = True
        retrieval_policy["retrieval_reranker_checkpoint_path"] = retrieval_checkpoint_path
    candidate["retrieval_policy"] = retrieval_policy

    evidence_policy = dict(candidate.get("evidence_policy", {}))
    if evidence_calibrator_checkpoint_path:
        evidence_policy["enable_evidence_calibrator"] = True
        evidence_policy["calibrator_mode"] = "hybrid"
        evidence_policy["calibrator_checkpoint_path"] = evidence_calibrator_checkpoint_path
    candidate["evidence_policy"] = evidence_policy
    return candidate


def run_stage3_policy_evaluation(args: argparse.Namespace, *, run_id: str, run_root: Path, stage_context: dict[str, Any]) -> dict[str, Any]:
    stage_id = 3
    stage_dir = build_stage_dir(run_root, stage_id)
    eval_output_dir = stage_dir / "evaluation"
    if eval_output_dir.exists() and not args.dry_run:
        shutil.rmtree(eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    stable_policy = load_policy(args.stable_policy_config).to_dict()
    controller_checkpoint_path = resolve_controller_checkpoint(stage_context, args)
    retrieval_checkpoint_path = resolve_retrieval_checkpoint(stage_context, args)
    evidence_calibrator_checkpoint_path = resolve_evidence_calibrator_checkpoint(stage_context, args)
    candidate_policy = build_stage3_candidate_policy(
        stable_policy=stable_policy,
        run_id=run_id,
        controller_checkpoint_path=controller_checkpoint_path,
        retrieval_checkpoint_path=retrieval_checkpoint_path,
        evidence_calibrator_checkpoint_path=evidence_calibrator_checkpoint_path,
    )
    candidate_policy_path = stage_dir / "candidate_policy.json"
    write_json(candidate_policy_path, candidate_policy)
    stage_context["stage3_candidate_policy_path"] = str(candidate_policy_path)
    stage_context["stage3_candidate_policy_id"] = str(candidate_policy.get("policy_id", ""))
    stage_context["stage3_candidate_policy_version"] = str(candidate_policy.get("version", ""))

    evaluation_limit = max(1, int(args.limit)) if int(args.limit) > 0 else 10
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_policy_candidate.py"),
        "--candidate-config",
        str(candidate_policy_path),
        "--stable-config",
        str(args.stable_policy_config),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(eval_output_dir),
        "--limit",
        str(evaluation_limit),
        "--case-offset",
        str(max(0, int(args.case_offset))),
        "--data-split",
        str(args.stage3_data_split),
    ]
    if args.split_json:
        command.extend(["--split-json", str(args.split_json)])
    if args.client_timeout is not None:
        command.extend(["--client-timeout", str(float(args.client_timeout))])
    if args.client_max_retries is not None:
        command.extend(["--client-max-retries", str(int(args.client_max_retries))])

    cmd_result = run_command(command, dry_run=bool(args.dry_run))
    eval_records = sorted(eval_output_dir.glob("policy_eval_*.json"))
    eval_record_path = str(eval_records[-1]) if eval_records else ""
    gate_decision: dict[str, Any] = {}
    if eval_record_path:
        payload = json.loads(Path(eval_record_path).read_text(encoding="utf-8"))
        gate_decision = dict(payload.get("gate_decision", {}))

    finished_at = utc_now()
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "stage_name": STAGE_NAMES[stage_id],
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "stable_policy_config": str(args.stable_policy_config),
            "candidate_policy_path": str(candidate_policy_path),
            "controller_checkpoint_path": controller_checkpoint_path,
            "retrieval_checkpoint_path": retrieval_checkpoint_path,
            "evidence_calibrator_checkpoint_path": evidence_calibrator_checkpoint_path,
            "data_root": str(args.data_root),
            "limit": evaluation_limit,
            "case_offset": max(0, int(args.case_offset)),
            "data_split": str(args.stage3_data_split),
        },
        "outputs": {
            "candidate_policy_path": str(candidate_policy_path),
            "evaluation_output_dir": str(eval_output_dir),
            "evaluation_record_path": eval_record_path,
            "gate_decision": gate_decision,
            "stdout_tail": cmd_result.get("stdout", "")[-1200:],
        },
        "checkpoint": {
            "component": "policy_candidate",
            "version": f"{stage_context['stage3_candidate_policy_id']}@v{stage_context['stage3_candidate_policy_version']}",
            "path": str(candidate_policy_path),
        },
        "command": command,
        "dry_run": bool(args.dry_run),
    }
    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def export_stable_checkpoints(
    *,
    run_id: str,
    stage4_dir: Path,
    checkpoint_out_dir: Path,
    controller_checkpoint_path: str,
    retrieval_checkpoint_path: str,
    evidence_calibrator_checkpoint_path: str,
    candidate_policy_path: str,
    dry_run: bool,
    export_tier: str = "run_candidate_export",
) -> dict[str, Any]:
    stage4_dir.mkdir(parents=True, exist_ok=True)
    export_local_dir = stage4_dir / "exported_checkpoints"
    export_global_dir = checkpoint_out_dir / run_id
    export_local_dir.mkdir(parents=True, exist_ok=True)
    export_global_dir.mkdir(parents=True, exist_ok=True)

    exported_components: list[dict[str, Any]] = []
    stamp = utc_compact()
    component_sources = [
        ("controller_planner_scorer", controller_checkpoint_path),
        ("retrieval_reranker", retrieval_checkpoint_path),
        ("evidence_calibrator", evidence_calibrator_checkpoint_path),
    ]
    for component_id, source in component_sources:
        source_path = Path(str(source).strip()) if str(source).strip() else None
        if source_path is None or not source_path.exists():
            continue
        filename = f"{component_id}__{export_tier}__{run_id}__{stamp}{source_path.suffix or '.pt'}"
        local_path = export_local_dir / filename
        global_path = export_global_dir / filename
        if not dry_run:
            shutil.copy2(source_path, local_path)
            shutil.copy2(source_path, global_path)
        version_id = f"{component_id}_v0.1.0+{run_id}"
        component_manifest = {
            "component_id": component_id,
            "version_id": version_id,
            "source_checkpoint_path": str(source_path),
            "export_local_path": str(local_path),
            "export_global_path": str(global_path),
            "exported_at": utc_now(),
        }
        write_json(stage4_dir / f"{component_id}_version.json", component_manifest)
        exported_components.append(component_manifest)

    policy_export_path = ""
    if candidate_policy_path:
        source_policy = Path(candidate_policy_path)
        if source_policy.exists():
            policy_name = f"policy_candidate__{export_tier}__{run_id}__{stamp}.json"
            policy_export_local = export_local_dir / policy_name
            policy_export_global = export_global_dir / policy_name
            if not dry_run:
                shutil.copy2(source_policy, policy_export_local)
                shutil.copy2(source_policy, policy_export_global)
            policy_export_path = str(policy_export_global)

    bundle_manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "exported_at": utc_now(),
        "export_tier": export_tier,
        "components": exported_components,
        "policy_candidate_export_path": policy_export_path,
        "local_export_dir": str(export_local_dir),
        "global_export_dir": str(export_global_dir),
    }
    write_json(stage4_dir / "stable_checkpoint_bundle_manifest.json", bundle_manifest)
    return bundle_manifest


def run_stage4_export(args: argparse.Namespace, *, run_id: str, run_root: Path, stage_context: dict[str, Any]) -> dict[str, Any]:
    stage_id = 4
    stage_dir = build_stage_dir(run_root, stage_id)
    started_at = utc_now()
    print("[pipeline stage4] resolving exported checkpoints", flush=True)

    controller_checkpoint_path = resolve_controller_checkpoint(stage_context, args)
    retrieval_checkpoint_path = resolve_retrieval_checkpoint(stage_context, args)
    evidence_calibrator_checkpoint_path = resolve_evidence_calibrator_checkpoint(stage_context, args)
    candidate_policy_path = str(stage_context.get("stage3_candidate_policy_path", "")).strip()
    if not candidate_policy_path:
        fallback_candidate = build_stage_dir(run_root, 3) / "candidate_policy.json"
        if fallback_candidate.exists():
            candidate_policy_path = str(fallback_candidate)

    print(
        "[pipeline stage4] exporting controller/retrieval/evidence candidate bundle",
        flush=True,
    )
    bundle_manifest = export_stable_checkpoints(
        run_id=run_id,
        stage4_dir=stage_dir,
        checkpoint_out_dir=args.checkpoint_out_dir,
        controller_checkpoint_path=controller_checkpoint_path,
        retrieval_checkpoint_path=retrieval_checkpoint_path,
        evidence_calibrator_checkpoint_path=evidence_calibrator_checkpoint_path,
        candidate_policy_path=candidate_policy_path,
        dry_run=bool(args.dry_run),
        export_tier="run_candidate_export",
    )
    if not bundle_manifest.get("components"):
        raise ValueError("Stage 4 export found no controller/retrieval checkpoints to export.")
    print(
        f"[pipeline stage4] exported {len(bundle_manifest.get('components', []))} component(s)",
        flush=True,
    )

    finished_at = utc_now()
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "stage_name": STAGE_NAMES[stage_id],
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "controller_checkpoint_path": controller_checkpoint_path,
            "retrieval_checkpoint_path": retrieval_checkpoint_path,
            "evidence_calibrator_checkpoint_path": evidence_calibrator_checkpoint_path,
            "candidate_policy_path": candidate_policy_path,
            "checkpoint_out_dir": str(args.checkpoint_out_dir),
        },
        "outputs": bundle_manifest,
        "selection_protocol_note": "This stage exports run-local candidate artifacts only. Use scripts/select_best_checkpoint.py to choose best validation and stable paper checkpoints across runs.",
        "checkpoint": {
            "component": "stable_checkpoint_bundle",
            "version": f"stable_checkpoint_bundle_v1+{run_id}",
            "path": str(stage_dir / "stable_checkpoint_bundle_manifest.json"),
        },
        "dry_run": bool(args.dry_run),
    }
    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def run_stage(stage_id: int, args: argparse.Namespace, *, run_id: str, run_root: Path, stage_context: dict[str, Any]) -> dict[str, Any]:
    if stage_id == 0:
        return run_stage0_bootstrap(args, run_id=run_id, run_root=run_root, stage_context=stage_context)
    if stage_id == 1:
        return run_stage1_controller_training(args, run_id=run_id, run_root=run_root, stage_context=stage_context)
    if stage_id == 2:
        return run_stage2_retrieval_training(args, run_id=run_id, run_root=run_root, stage_context=stage_context)
    if stage_id == 3:
        return run_stage3_policy_evaluation(args, run_id=run_id, run_root=run_root, stage_context=stage_context)
    if stage_id == 4:
        return run_stage4_export(args, run_id=run_id, run_root=run_root, stage_context=stage_context)
    raise ValueError(f"Unsupported stage id: {stage_id}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_stages = parse_stage_selection(args.stages)
    run_id = args.run_id.strip() or f"learned_components_{utc_compact()}"
    run_root = args.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    run_manifest_path = run_root / RUN_MANIFEST_NAME
    run_manifest = build_initial_run_manifest(
        run_manifest_path=run_manifest_path,
        run_id=run_id,
        args=args,
        selected_stages=selected_stages,
    )
    write_json(run_manifest_path, run_manifest)

    stage_context: dict[str, Any] = {}
    hydrate_stage_context_from_existing_run(run_root, selected_stages=selected_stages, stage_context=stage_context)
    try:
        total_stages = len(selected_stages)
        print(
            f"[pipeline] run_id={run_id} selected_stages={selected_stages}",
            flush=True,
        )
        for index, stage_id in enumerate(selected_stages, start=1):
            print(
                f"[pipeline {index}/{total_stages}] starting stage{stage_id}_{STAGE_NAMES[stage_id]}",
                flush=True,
            )
            result = run_stage(stage_id, args, run_id=run_id, run_root=run_root, stage_context=stage_context)
            run_manifest["stages"][str(stage_id)] = {
                "stage_name": STAGE_NAMES[stage_id],
                "manifest_path": result["manifest_path"],
                "status": result["manifest"].get("status", "completed"),
                "checkpoint": result["manifest"].get("checkpoint", {}),
            }
            write_json(run_manifest_path, run_manifest)
            print(
                f"[pipeline {index}/{total_stages}] completed stage{stage_id}_{STAGE_NAMES[stage_id]}",
                flush=True,
            )
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["failed_at"] = utc_now()
        run_manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_manifest_path, run_manifest)
        raise

    run_manifest["status"] = "completed"
    run_manifest["finished_at"] = utc_now()
    run_manifest["resolved_checkpoints"] = {
        "controller_checkpoint_path": resolve_controller_checkpoint(stage_context, args),
        "retrieval_checkpoint_path": resolve_retrieval_checkpoint(stage_context, args),
        "evidence_calibrator_checkpoint_path": resolve_evidence_calibrator_checkpoint(stage_context, args),
    }
    write_json(run_manifest_path, run_manifest)
    print(
        json.dumps(
            {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "run_id": run_id,
                "run_root": str(run_root),
                "run_manifest_path": str(run_manifest_path),
                "selected_stages": selected_stages,
                "resolved_checkpoints": run_manifest.get("resolved_checkpoints", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
