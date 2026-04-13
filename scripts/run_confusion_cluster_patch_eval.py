from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.confusion_clusters import detect_confusion_clusters
from agent.policy_config import load_policy, load_stable_policy
from agent.run_agent import run_agent
from dataio.case_loader import discover_case_source, standardize_row
from integrations.openai_client import DermOpenAIClient


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_HARD_CASES = (
    PROJECT_ROOT
    / "outputs"
    / "smoke_cycles"
    / "medium_signal_no_ablation_10h_v1"
    / "hard_case_mining"
    / "hard_cases.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "confusion_cluster_patch_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused post-patch eval on key confusion clusters.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hard-cases-jsonl", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional policy config JSON; defaults to stable policy.")
    parser.add_argument("--cases-per-cluster", type=int, default=1)
    parser.add_argument("--control-indices", type=int, nargs="*", default=[0], help="Non-target regression control indices.")
    parser.add_argument("--data-split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--run-mode", type=str, default="confusion_cluster_patch_eval")
    parser.add_argument("--client-timeout", type=float, default=None)
    parser.add_argument("--client-max-retries", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()
    image_index_map = _build_image_index_map(args.data_root)
    selected_specs = _select_cluster_cases(
        hard_cases_path=args.hard_cases_jsonl,
        image_index_map=image_index_map,
        cases_per_cluster=max(1, int(args.cases_per_cluster)),
    )

    control_specs = []
    for case_index in args.control_indices:
        control_specs.append(
            {
                "cluster_id": "control",
                "case_index": int(case_index),
                "prior_case": None,
                "selection_reason": "non_target_regression_control",
            }
        )

    all_specs = selected_specs + control_specs
    case_runs: list[dict[str, Any]] = []
    for spec in all_specs:
        case_input = _load_case_by_index(spec["case_index"], args.data_root)
        state, evidence_package = run_agent(
            case_input=case_input,
            client=client,
            policy_config=policy,
            enable_writeback=False,
            run_mode=args.run_mode,
            data_split=args.data_split,
        )
        execution_record = dict(state.execution_record)
        case_runs.append(
            {
                "cluster_id": spec["cluster_id"],
                "case_index": spec["case_index"],
                "selection_reason": spec["selection_reason"],
                "prior_case": spec.get("prior_case"),
                "current_result": {
                    "case_id": execution_record.get("case_id"),
                    "image_path": execution_record.get("input_summary", {}).get("image_path"),
                    "ground_truth": execution_record.get("ground_truth"),
                    "qwen_final": execution_record.get("qwen_final"),
                    "evaluation": execution_record.get("evaluation"),
                    "planner_decision": execution_record.get("planner_decision"),
                    "retrieval_bundle": execution_record.get("retrieval_bundle"),
                    "evidence_bundle": evidence_package.to_dict(),
                },
            }
        )

    summary = _summarize_results(case_runs)
    output = {
        "eval_type": "focused_confusion_cluster_patch_eval",
        "policy_id": policy.get("policy_id"),
        "data_split": args.data_split,
        "selected_cluster_cases": selected_specs,
        "control_specs": control_specs,
        "summary": summary,
        "cases": case_runs,
    }
    timestamp = _timestamp_token()
    output_path = args.output_dir / f"confusion_cluster_patch_eval_{timestamp}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


def _load_case_by_index(case_index: int, data_root: Path):
    from dataio.case_loader import load_case_by_index

    return load_case_by_index(case_index, data_root)


def _build_image_index_map(data_root: Path) -> dict[str, int]:
    config = discover_case_source(data_root)
    with config.metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, int] = {}
    for index, row in enumerate(rows):
        standardized = standardize_row(row=row, config=config)
        image_name = Path(standardized.image_path).name
        if image_name and image_name not in mapping:
            mapping[image_name] = index
    return mapping


def _select_cluster_cases(
    *,
    hard_cases_path: Path,
    image_index_map: dict[str, int],
    cases_per_cluster: int,
) -> list[dict[str, Any]]:
    cluster_order = ("ack_bcc_scc", "ack_sek", "mel_nev")
    picked: dict[str, list[dict[str, Any]]] = {cluster_id: [] for cluster_id in cluster_order}
    used_images: set[str] = set()
    rows = _read_jsonl(hard_cases_path)

    for exact_only in (True, False):
        for row in rows:
            if len([items for items in picked.values() if len(items) >= cases_per_cluster]) == len(cluster_order):
                break
            source_record_path = Path(str(row.get("source_record_path", "")).strip())
            if not source_record_path.exists():
                continue
            source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
            image_path = str(source_record.get("input_summary", {}).get("image_path", "")).strip()
            image_name = Path(image_path).name
            if not image_name or image_name in used_images or image_name not in image_index_map:
                continue
            matched_cluster = _resolve_target_cluster(
                row=row,
                source_record=source_record,
                cluster_order=cluster_order,
                picked=picked,
                cases_per_cluster=cases_per_cluster,
                exact_only=exact_only,
            )
            if matched_cluster is None:
                continue
            used_images.add(image_name)
            picked[matched_cluster].append(
                {
                    "cluster_id": matched_cluster,
                    "case_index": image_index_map[image_name],
                    "selection_reason": "top_hard_case_from_previous_medium_run",
                    "prior_case": {
                        "case_id": row.get("case_id"),
                        "image_path": image_path,
                        "confusion_tags": row.get("confusion_tags", []),
                        "previous_agent_result": row.get("agent_result", {}),
                        "ground_truth": row.get("ground_truth", {}),
                        "importance_score": row.get("importance_score"),
                    },
                }
            )
    selected: list[dict[str, Any]] = []
    for cluster_id in cluster_order:
        selected.extend(picked[cluster_id])
    return selected


def _resolve_target_cluster(
    *,
    row: dict[str, Any],
    source_record: dict[str, Any],
    cluster_order: tuple[str, ...],
    picked: dict[str, list[dict[str, Any]]],
    cases_per_cluster: int,
    exact_only: bool,
) -> str | None:
    tags = " ".join(str(item) for item in row.get("confusion_tags", [])).lower()
    exact_priority = {
        "ack_sek": (
            "seborrheic keratosis->ack",
            "seborrheic keratosis_vs_ack",
            "actinic keratosis->sek",
            "actinic keratosis_vs_sek",
            "ack->sek",
        ),
        "mel_nev": (
            "malignant melanoma->nev",
            "malignant melanoma_vs_nev",
            "melanoma->nev",
        ),
        "ack_bcc_scc": (
            "actinic keratosis->bcc",
            "actinic keratosis_vs_bcc",
            "squamous cell carcinoma->bcc",
            "squamous cell carcinoma_vs_bcc",
            "bcc->ack",
            "bcc->scc",
        ),
    }
    for cluster_id in cluster_order:
        if len(picked[cluster_id]) >= cases_per_cluster:
            continue
        if any(token in tags for token in exact_priority.get(cluster_id, ())):
            return cluster_id
    if exact_only:
        return None
    clusters = detect_confusion_clusters(
        ddx_candidates=[str(item) for item in row.get("evidence_snapshot", {}).get("initial_ddx", []) if str(item).strip()],
        known_confusion_text=tags,
        image_summary=str(source_record.get("qwen_initial", {}).get("image_summary", "")),
    )
    return next(
        (
            cluster_id
            for cluster_id in cluster_order
            if cluster_id in clusters and len(picked[cluster_id]) < cases_per_cluster
        ),
        None,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summarize_results(case_runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_cluster: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "correct": 0, "topk_hit": 0, "improved_vs_prior": 0, "worsened_vs_prior": 0})
    for row in case_runs:
        cluster_id = str(row.get("cluster_id", "unknown"))
        current_eval = row.get("current_result", {}).get("evaluation", {}) or {}
        prior_case = row.get("prior_case") if isinstance(row.get("prior_case"), dict) else {}
        prior_eval = prior_case.get("previous_agent_result", {}) or {}
        by_cluster[cluster_id]["count"] += 1
        if current_eval.get("correct") is True:
            by_cluster[cluster_id]["correct"] += 1
        if current_eval.get("topk_hit") is True:
            by_cluster[cluster_id]["topk_hit"] += 1
        if prior_eval:
            prior_correct = prior_eval.get("correct")
            current_correct = current_eval.get("correct")
            if prior_correct is False and current_correct is True:
                by_cluster[cluster_id]["improved_vs_prior"] += 1
            elif prior_correct is True and current_correct is False:
                by_cluster[cluster_id]["worsened_vs_prior"] += 1
    return {"by_cluster": dict(by_cluster), "total_cases": len(case_runs)}


def _timestamp_token() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
