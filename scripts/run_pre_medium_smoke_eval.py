from __future__ import annotations

import argparse
import json
import statistics
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_protocol import EvaluationTargetSpec, run_evaluation_suite
from agent.policy_config import load_policy, load_stable_policy
from agent.skill_helpfulness_analyzer import analyze_skill_helpfulness
from agent.supervised_controller import (
    ControllerMLP,
    ControllerSelectionPolicy,
    flatten_training_example_features,
    multilabel_metrics,
    select_skills_with_policy_details,
    vectorize_feature_maps,
)
from integrations.openai_client import DermOpenAIClient
from scripts.audit_experiment_state import (
    audit_evaluation_manifests,
    audit_execution_records,
    audit_script_writeback_explicitness,
    load_split_map,
)
from scripts.train_controller import (
    FALLBACK_EXAMPLES_PATH,
    DEFAULT_EXAMPLES_PATH,
    build_candidate_mask_matrix,
    build_harmful_mask_matrix,
    build_label_matrix,
    build_target_k_list,
    extract_labels,
    load_examples,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "pre_medium_smoke_eval"
DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation_protocol"
DEFAULT_FIXED_SPLIT = PROJECT_ROOT / "outputs" / "smoke_cycles" / "medium_signal_no_ablation_v1" / "fixed_split.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "state" / "policy" / "current_stable_policy.json"

REPORT_SCHEMA_VERSION = "pre_medium_smoke_eval_v1"
REQUIRED_EVIDENCE_FIELDS = (
    "initial_perception_summary",
    "retrieved_raw_cases_summary",
    "retrieved_tactical_experiences_summary",
    "retrieved_abstract_experiences_summary",
    "uncertainty_summary",
    "contradiction_summary",
    "planner_rationale",
    "serialized_evidence_text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pre-medium smoke evaluation checklist before medium-scale training.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT, help="Root directory containing execution records.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT, help="Root directory containing frozen evaluation outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for report artifacts.")
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_PATH, help="Stable policy config used for config recommendation.")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_FIXED_SPLIT, help="Fixed split definition JSON.")
    parser.add_argument("--compare-limit", type=int, default=6, help="Small compare sample size.")
    parser.add_argument("--compare-split", type=str, default="val", choices=("val", "test"), help="Frozen split for smoke compare.")
    parser.add_argument("--case-offset", type=int, default=0, help="Case offset for compare smoke.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved seed for traceability.")
    parser.add_argument("--controller-checkpoint", type=Path, default=None, help="Optional learned controller checkpoint override.")
    parser.add_argument("--controller-examples", type=Path, default=None, help="Optional controller examples JSONL override.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Override Qwen client timeout seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Override Qwen client retries.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_json = args.split_json if args.split_json and args.split_json.exists() else None
    split_map = load_split_map(split_json) if split_json else {}

    checklist: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    warnings: list[str] = []
    blockers: list[str] = []

    contamination_item = run_contamination_audit_check(
        records_root=args.records_root,
        eval_root=args.eval_root,
        split_map=split_map,
    )
    checklist.append(contamination_item)
    _collect_status_messages(contamination_item, warnings=warnings, blockers=blockers)

    compare_payload: dict[str, Any] | None = None
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    rule_compare_item = run_rule_controller_compare_check(
        output_dir=args.output_dir / "compare",
        data_root=args.data_root,
        split_json=split_json,
        compare_limit=args.compare_limit,
        compare_split=args.compare_split,
        case_offset=args.case_offset,
        seed=args.seed,
        client=client,
    )
    checklist.append(rule_compare_item["check"])
    _collect_status_messages(rule_compare_item["check"], warnings=warnings, blockers=blockers)
    if rule_compare_item.get("payload"):
        compare_payload = rule_compare_item["payload"]
        artifacts["compare"] = {
            "evaluation_manifest_path": compare_payload.get("evaluation_manifest_path"),
            "result_manifest_path": compare_payload.get("result_manifest_path"),
            "run_root": compare_payload.get("run_root"),
            "report_path": compare_payload.get("report_path"),
        }

    controller_item = run_learned_controller_check(
        output_dir=args.output_dir / "controller_eval",
        checkpoint_override=args.controller_checkpoint,
        examples_override=args.controller_examples,
    )
    checklist.append(controller_item)
    _collect_status_messages(controller_item, warnings=warnings, blockers=blockers)
    if controller_item.get("artifacts"):
        artifacts["controller_eval"] = controller_item["artifacts"]

    helpfulness_item = run_helpfulness_realism_check(
        output_dir=args.output_dir / "helpfulness",
        compare_payload=compare_payload,
    )
    checklist.append(helpfulness_item)
    _collect_status_messages(helpfulness_item, warnings=warnings, blockers=blockers)
    if helpfulness_item.get("artifacts"):
        artifacts["helpfulness"] = helpfulness_item["artifacts"]

    evidence_item = run_evidence_bundle_check(compare_payload=compare_payload)
    checklist.append(evidence_item)
    _collect_status_messages(evidence_item, warnings=warnings, blockers=blockers)

    compare_quality_item = run_compare_quality_check(compare_payload=compare_payload)
    checklist.append(compare_quality_item)
    _collect_status_messages(compare_quality_item, warnings=warnings, blockers=blockers)

    overall_status = "FAIL" if blockers else ("WARNING" if warnings else "PASS")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "overall_status": overall_status,
        "summary": {
            "pass_count": sum(1 for item in checklist if item.get("status") == "PASS"),
            "warning_count": sum(1 for item in checklist if item.get("status") == "WARNING"),
            "fail_count": sum(1 for item in checklist if item.get("status") == "FAIL"),
            "ready_for_medium_training": overall_status != "FAIL",
        },
        "environment": build_environment_snapshot(
            policy_path=args.policy_config,
            split_json=split_json,
            compare_limit=args.compare_limit,
            compare_split=args.compare_split,
            records_root=args.records_root,
            eval_root=args.eval_root,
            client=client,
        ),
        "checklist": checklist,
        "blockers": blockers,
        "warnings": warnings,
        "artifacts": artifacts,
        "suggested_medium_training_config": build_recommended_medium_config(
            policy_path=args.policy_config,
            split_json=split_json,
            overall_status=overall_status,
        ),
    }

    report_path = args.output_dir / f"pre_medium_smoke_report_{compact_timestamp()}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = args.output_dir / "latest_pre_medium_smoke_report.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_status": overall_status,
                "report_path": str(report_path),
                "summary": report["summary"],
                "blockers": blockers,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_contamination_audit_check(
    *,
    records_root: Path,
    eval_root: Path,
    split_map: dict[str, set[str]],
) -> dict[str, Any]:
    execution_records = load_execution_records(records_root)
    record_checks = audit_execution_records(execution_records, split_map=split_map)
    eval_checks = audit_evaluation_manifests(eval_root, split_map=split_map)
    script_checks = audit_script_writeback_explicitness(PROJECT_ROOT / "scripts")
    issues = record_checks["issues"] + eval_checks["issues"] + script_checks["issues"]
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    if error_count > 0:
        status = "FAIL"
        detail = "Split contamination or frozen-eval violations detected."
    elif warning_count > 0:
        status = "WARNING"
        detail = "No blocking contamination found, but warnings remain."
    else:
        status = "PASS"
        detail = "Contamination audit clean for records, eval manifests, and scripts."
    return build_check_item(
        item_id="contamination_audit",
        title="Contamination Audit",
        status=status,
        thresholds={
            "pass": "0 error issues; warnings allowed only for WARNING.",
            "fail": "Any split-mismatch, frozen writeback, or explicit contamination error.",
        },
        observed={
            "execution_record_count": len(execution_records),
            "evaluation_manifest_count": eval_checks.get("evaluation_manifest_count", 0),
            "script_file_count": script_checks.get("script_file_count", 0),
            "error_count": error_count,
            "warning_count": warning_count,
            "sample_issues": issues[:5],
        },
        detail=detail,
        blocking=status == "FAIL",
    )


def run_rule_controller_compare_check(
    *,
    output_dir: Path,
    data_root: Path,
    split_json: Path | None,
    compare_limit: int,
    compare_split: str,
    case_offset: int,
    seed: int,
    client: DermOpenAIClient,
) -> dict[str, Any]:
    try:
        base_policy = load_stable_policy().to_dict()
        rule_policy = deepcopy(base_policy)
        planner_policy = dict(rule_policy.get("planner_policy", {}))
        planner_policy.update(
            {
                "controller_family": "heuristic",
                "controller_checkpoint_path": "",
                "learned_controller_weight": 0.0,
                "learned_controller_top_k": 0,
                "learned_controller_force_top_k": 0,
            }
        )
        rule_policy["planner_policy"] = planner_policy
        target_specs = [
            EvaluationTargetSpec(
                target_id="direct_baseline",
                label="Direct Baseline",
                target_type="baseline",
                mode="baseline",
                description="Direct Qwen baseline with no agent evidence package.",
            ),
            EvaluationTargetSpec(
                target_id="rule_controller_agent",
                label="Rule Controller Agent",
                target_type="full_agent",
                mode="agent",
                description="Full DermAgent using the heuristic controller only.",
            ),
        ]
        suite = run_evaluation_suite(
            output_root=output_dir,
            data_root=data_root,
            client=client,
            policy_config=rule_policy,
            target_specs=target_specs,
            limit=max(1, int(compare_limit)),
            case_offset=max(0, int(case_offset)),
            seed=int(seed),
            suite_label="pre_medium_smoke_compare",
            data_split=compare_split,
            split_json=split_json,
            strict_frozen_eval=True,
        )
        result_manifest = dict(suite.get("result_manifest", {}))
        target_results = {item["target"]["target_id"]: item for item in result_manifest.get("target_results", [])}
        baseline_summary = target_results.get("direct_baseline", {}).get("summary", {})
        agent_summary = target_results.get("rule_controller_agent", {}).get("summary", {})
        comparisons = result_manifest.get("comparisons", {}).get("rule_controller_agent", {}).get("vs_baseline", {})
        agent_records_path = Path(target_results.get("rule_controller_agent", {}).get("artifacts", {}).get("records_jsonl_path", ""))
        agent_records = load_jsonl(agent_records_path)
        report_payload = {
            "run_root": suite.get("run_root"),
            "evaluation_manifest_path": suite.get("evaluation_manifest_path"),
            "result_manifest_path": suite.get("result_manifest_path"),
            "report_path": "",
            "baseline_summary": baseline_summary,
            "agent_summary": agent_summary,
            "comparison": comparisons,
            "agent_records_path": str(agent_records_path) if agent_records_path else "",
            "agent_records": agent_records,
            "policy_id": rule_policy.get("policy_id"),
            "controller_family": planner_policy.get("controller_family"),
        }
        report_path = output_dir / "rule_controller_compare_summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report_payload["report_path"] = str(report_path)

        num_cases = int(agent_summary.get("num_cases", 0) or 0)
        if num_cases <= 0:
            status = "FAIL"
            detail = "Rule-controller compare completed without any evaluated agent cases."
        else:
            status = "PASS"
            detail = "Rule controller completed a frozen compare run successfully."
        check = build_check_item(
            item_id="rule_controller_runtime",
            title="Rule Controller Smoke Compare",
            status=status,
            thresholds={
                "pass": "Frozen compare run completes with >=1 agent case under heuristic controller.",
                "fail": "Compare crashes or evaluates 0 agent cases.",
            },
            observed={
                "controller_family": planner_policy.get("controller_family"),
                "policy_id": rule_policy.get("policy_id"),
                "num_cases": num_cases,
                "baseline_top1": safe_rate(baseline_summary, "top1"),
                "agent_top1": safe_rate(agent_summary, "top1"),
                "agent_records_path": str(agent_records_path) if agent_records_path else "",
            },
            detail=detail,
            blocking=status == "FAIL",
        )
        return {"check": check, "payload": report_payload}
    except Exception as exc:
        check = build_check_item(
            item_id="rule_controller_runtime",
            title="Rule Controller Smoke Compare",
            status="FAIL",
            thresholds={
                "pass": "Frozen compare run completes with >=1 agent case under heuristic controller.",
                "fail": "Any runtime failure.",
            },
            observed={"error": repr(exc)},
            detail="Rule-controller smoke compare failed to run.",
            blocking=True,
        )
        return {"check": check, "payload": None}


def run_learned_controller_check(
    *,
    output_dir: Path,
    checkpoint_override: Path | None,
    examples_override: Path | None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_override if checkpoint_override else discover_best_controller_checkpoint()
    examples_path = examples_override if examples_override else discover_controller_examples_path()
    if checkpoint_path is None or not checkpoint_path.exists():
        return build_check_item(
            item_id="learned_controller_selection",
            title="Learned Controller Sparsity",
            status="WARNING",
            thresholds={
                "pass": "Sparse-helpfulness checkpoint exists and avg predicted labels stays well below near-all-select.",
                "warning": "No suitable sparse controller checkpoint found yet.",
            },
            observed={
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
                "examples_path": str(examples_path) if examples_path else "",
            },
            detail="No learned-controller checkpoint available for sparse-selection validation; medium run can proceed with rule controller only.",
            blocking=False,
        )
    if examples_path is None or not examples_path.exists():
        return build_check_item(
            item_id="learned_controller_selection",
            title="Learned Controller Sparsity",
            status="WARNING",
            thresholds={
                "pass": "Sparse-helpfulness checkpoint exists and can be evaluated on exported controller examples.",
                "warning": "Checkpoint exists, but matching examples were not found.",
            },
            observed={"checkpoint_path": str(checkpoint_path), "examples_path": str(examples_path) if examples_path else ""},
            detail="Learned-controller checkpoint found, but no controller examples were available for evaluation.",
            blocking=False,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    examples = load_examples(examples_path)
    if not examples:
        return build_check_item(
            item_id="learned_controller_selection",
            title="Learned Controller Sparsity",
            status="WARNING",
            thresholds={
                "pass": "Learned controller can be evaluated on non-empty exported examples.",
                "warning": "Example file exists but resolved to 0 usable examples.",
            },
            observed={"checkpoint_path": str(checkpoint_path), "examples_path": str(examples_path)},
            detail="Learned-controller examples file was empty after loading.",
            blocking=False,
        )

    feature_vocab = dict(checkpoint.get("feature_vocab", {}))
    label_list = [str(item).strip() for item in checkpoint.get("label_list", []) if str(item).strip()]
    if not feature_vocab or not label_list:
        return build_check_item(
            item_id="learned_controller_selection",
            title="Learned Controller Sparsity",
            status="FAIL",
            thresholds={
                "pass": "Checkpoint exposes feature vocab and label list.",
                "fail": "Checkpoint missing core metadata.",
            },
            observed={"checkpoint_path": str(checkpoint_path), "has_feature_vocab": bool(feature_vocab), "has_label_list": bool(label_list)},
            detail="Learned-controller checkpoint metadata incomplete.",
            blocking=True,
        )

    feature_maps = [flatten_training_example_features(example) for example in examples]
    X = vectorize_feature_maps(feature_maps, feature_vocab)
    label_source = str(checkpoint.get("label_source", "selected") or "selected")
    y = build_label_matrix([extract_labels(example, label_source) for example in examples], label_list)
    harmful_mask = build_harmful_mask_matrix(examples, label_list)
    candidate_mask = build_candidate_mask_matrix(examples, label_list)
    target_k = build_target_k_list(examples)

    split_indices = resolve_eval_indices(checkpoint=checkpoint, num_examples=len(examples))
    X_eval = X[split_indices]
    y_eval = y[split_indices]
    harmful_eval = harmful_mask[split_indices]
    candidate_eval = candidate_mask[split_indices]
    target_k_eval = [target_k[index] for index in split_indices]
    examples_eval = [examples[index] for index in split_indices]

    model = ControllerMLP(
        input_dim=int(checkpoint.get("input_dim", len(feature_vocab))),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        output_dim=int(checkpoint.get("output_dim", len(label_list))),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", {}))
    model.eval()
    selection_policy_payload = dict(checkpoint.get("selection_policy", {}))
    selection_policy = ControllerSelectionPolicy(
        threshold=float(selection_policy_payload.get("threshold", checkpoint.get("threshold", 0.5))),
        target_top_k=max(1, int(selection_policy_payload.get("target_top_k", checkpoint.get("top_k", 5)))),
        min_select=max(1, int(selection_policy_payload.get("min_select", 4) or 1)),
        max_select=max(1, int(selection_policy_payload.get("max_select", 8) or 1)),
        top_k_buffer=max(0, int(selection_policy_payload.get("top_k_buffer", 1) or 0)),
    )

    with torch.no_grad():
        logits = model(X_eval)
        probs = torch.sigmoid(logits)
        if probs.shape == candidate_eval.shape:
            probs = probs * candidate_eval
    metrics = multilabel_metrics(
        y_eval,
        probs,
        threshold=selection_policy.threshold,
        top_k=selection_policy.target_top_k,
        target_k=target_k_eval,
        y_harmful=harmful_eval,
        min_select=selection_policy.min_select,
        max_select=selection_policy.max_select,
        top_k_buffer=selection_policy.top_k_buffer,
    )

    avg_predicted = _as_float(metrics.get("avg_predicted_labels"))
    avg_true = _as_float(metrics.get("avg_true_labels"))
    harmful_rate = _as_float(metrics.get("harmful_skill_over_selection_rate"))
    label_count = len(label_list)
    if label_source == "selected" and avg_predicted >= max(10.0, label_count - 1.0):
        status = "FAIL"
        detail = "Learned controller is still effectively near-all-select under the old selected-skill objective."
    elif avg_predicted >= max(10.0, label_count * 0.7):
        status = "FAIL"
        detail = "Learned controller still predicts too many skills per case."
    elif harmful_rate > 0.5:
        status = "WARNING"
        detail = "Learned controller is sparse enough, but harmful over-selection remains elevated."
    else:
        status = "PASS"
        detail = "Learned controller no longer behaves like a near-all-select model."

    predictions_rows = build_controller_prediction_rows(
        examples=examples_eval,
        probabilities=probs,
        label_list=label_list,
        selection_policy=selection_policy,
        target_k_eval=target_k_eval,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "learned_controller_eval.json"
    predictions_path = output_dir / "learned_controller_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    controller_report = {
        "checkpoint_path": str(checkpoint_path),
        "examples_path": str(examples_path),
        "label_source": label_source,
        "num_examples_eval": len(examples_eval),
        "selection_policy": selection_policy.to_dict(),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(controller_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return build_check_item(
        item_id="learned_controller_selection",
        title="Learned Controller Sparsity",
        status=status,
        thresholds={
            "pass": "avg_predicted_labels <= 8 and harmful over-selection <= 0.5.",
            "warning": "Sparse enough to avoid near-all-select, but harmful over-selection is still high or checkpoint remains imperfect.",
            "fail": "avg_predicted_labels > 10 or nearly all skills are still selected.",
        },
        observed={
            "checkpoint_path": str(checkpoint_path),
            "examples_path": str(examples_path),
            "label_source": label_source,
            "num_labels": label_count,
            "num_examples_eval": len(examples_eval),
            "avg_predicted_labels": avg_predicted,
            "avg_true_labels": avg_true,
            "micro_f1": _as_float(metrics.get("micro_f1")),
            "topk_hit_rate": _as_float(metrics.get("topk_hit_rate")),
            "harmful_skill_over_selection_rate": harmful_rate,
            "selection_policy": selection_policy.to_dict(),
        },
        detail=detail,
        blocking=status == "FAIL",
        artifacts={
            "report_path": str(report_path),
            "predictions_path": str(predictions_path),
        },
    )


def run_helpfulness_realism_check(
    *,
    output_dir: Path,
    compare_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    records = list(compare_payload.get("agent_records", [])) if compare_payload else []
    if not records:
        return build_check_item(
            item_id="skill_helpfulness_realism",
            title="Skill Helpfulness Realism",
            status="FAIL",
            thresholds={
                "pass": "Fresh smoke agent records available for helpful/harmful analysis.",
                "fail": "No fresh agent execution records were generated.",
            },
            observed={"record_count": 0},
            detail="Cannot judge helpful/harmful realism because no fresh compare records were available.",
            blocking=True,
        )

    reports, summary = analyze_skill_helpfulness(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "skill_helpfulness_summary.json"
    reports_path = output_dir / "skill_helpfulness_reports.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with reports_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")

    total_call_count = int(summary.get("totals", {}).get("call_count", 0) or 0)
    harmful_count = int(summary.get("totals", {}).get("harmful_count", 0) or 0)
    partially_helpful = int(summary.get("totals", {}).get("partially_helpful_count", 0) or 0)
    helpful_count = int(summary.get("totals", {}).get("helpful_count", 0) or 0)
    filtered_record_count = int(summary.get("filtered_record_count", 0) or 0)

    if total_call_count <= 0:
        status = "FAIL"
        detail = "No skill calls were present in fresh smoke records."
    elif harmful_count > 0:
        status = "PASS"
        detail = "Fresh smoke records produced non-zero harmful assignments, so the judge is no longer trivially optimistic."
    elif filtered_record_count < 10:
        status = "WARNING"
        detail = "Fresh smoke sample is small; harmful_count is still 0, so realism remains only partially validated."
    else:
        status = "FAIL"
        detail = "Fresh smoke sample is large enough that harmful_count=0 is still not credible."

    return build_check_item(
        item_id="skill_helpfulness_realism",
        title="Skill Helpfulness Realism",
        status=status,
        thresholds={
            "pass": "Fresh smoke helpfulness stats show non-zero harmful_count or clearly mixed judgments.",
            "warning": "Small smoke sample still has harmful_count=0.",
            "fail": "Large smoke sample still reports harmful_count=0 or no skill calls.",
        },
        observed={
            "filtered_record_count": filtered_record_count,
            "skill_count": int(summary.get("skill_count", 0) or 0),
            "call_count": total_call_count,
            "helpful_count": helpful_count,
            "partially_helpful_count": partially_helpful,
            "harmful_count": harmful_count,
            "average_evidence_strength": _as_float(summary.get("totals", {}).get("average_evidence_strength")),
            "top_harmful_skills": summary.get("top_harmful_skills", [])[:5],
        },
        detail=detail,
        blocking=status == "FAIL",
        artifacts={
            "summary_path": str(summary_path),
            "reports_path": str(reports_path),
        },
    )


def run_evidence_bundle_check(*, compare_payload: dict[str, Any] | None) -> dict[str, Any]:
    records = list(compare_payload.get("agent_records", [])) if compare_payload else []
    if not records:
        return build_check_item(
            item_id="evidence_bundle_quality",
            title="Evidence Bundle Quality",
            status="FAIL",
            thresholds={
                "pass": "Fresh smoke compare generates explainable serialized evidence bundles.",
                "fail": "No fresh agent records available for evidence inspection.",
            },
            observed={"record_count": 0},
            detail="Cannot assess evidence explainability or bloat without fresh compare records.",
            blocking=True,
        )

    lengths: list[int] = []
    line_counts: list[int] = []
    required_field_hits = 0
    explainability_hits = 0
    for record in records:
        evidence_bundle = dict(record.get("evidence_bundle", {}) or {})
        serialized = str(evidence_bundle.get("serialized_evidence_text", "") or "")
        lines = [line.strip() for line in serialized.splitlines() if line.strip()]
        lengths.append(len(serialized))
        line_counts.append(len(lines))
        if all(field in evidence_bundle for field in REQUIRED_EVIDENCE_FIELDS):
            required_field_hits += 1
        if _serialized_evidence_has_key_signals(record=record, serialized_text=serialized):
            explainability_hits += 1

    record_count = len(records)
    avg_len = statistics.mean(lengths) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    avg_lines = statistics.mean(line_counts) if line_counts else 0.0
    required_coverage = required_field_hits / record_count if record_count else 0.0
    explainability_coverage = explainability_hits / record_count if record_count else 0.0

    if required_coverage < 0.75 or avg_len > 13000 or max_len > 16000:
        status = "FAIL"
        detail = "Evidence bundle is missing key fields or has become too large for reliable medium-run prompting."
    elif avg_len > 9500 or max_len > 13000 or explainability_coverage < 0.7:
        status = "WARNING"
        detail = "Evidence bundle remains interpretable, but it is trending verbose or uneven in key-signal coverage."
    else:
        status = "PASS"
        detail = "Evidence bundle stays structured, interpretable, and reasonably compact."

    return build_check_item(
        item_id="evidence_bundle_quality",
        title="Evidence Bundle Quality",
        status=status,
        thresholds={
            "pass": "required-field coverage >= 0.75, avg serialized length <= 9500 chars, max <= 13000 chars.",
            "warning": "Still explainable but getting verbose or inconsistent.",
            "fail": "Missing key fields or clearly inflated bundles.",
        },
        observed={
            "record_count": record_count,
            "avg_serialized_char_len": round(avg_len, 2),
            "max_serialized_char_len": max_len,
            "avg_line_count": round(avg_lines, 2),
            "required_field_coverage": round(required_coverage, 4),
            "explainability_coverage": round(explainability_coverage, 4),
        },
        detail=detail,
        blocking=status == "FAIL",
    )


def run_compare_quality_check(*, compare_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not compare_payload:
        return build_check_item(
            item_id="compare_directionality",
            title="Small-Sample Compare Directionality",
            status="FAIL",
            thresholds={
                "pass": "Frozen baseline vs agent compare is available for smoke gating.",
                "fail": "No compare payload available.",
            },
            observed={},
            detail="Cannot judge smoke compare directionality because compare run did not complete.",
            blocking=True,
        )

    baseline_summary = dict(compare_payload.get("baseline_summary", {}) or {})
    agent_summary = dict(compare_payload.get("agent_summary", {}) or {})
    comparison = dict(compare_payload.get("comparison", {}) or {})
    top1_delta = _as_float(comparison.get("top1_delta"))
    topk_delta = _as_float(comparison.get("topk_delta"))
    malignant_delta = _as_float(comparison.get("malignant_recall_delta"))
    error_delta = _as_float(comparison.get("error_rate_delta"))

    if malignant_delta < 0.0 or top1_delta < -0.15 or error_delta > 0.15:
        status = "FAIL"
        detail = "Small frozen compare shows a material regression relative to direct baseline."
    elif top1_delta < -0.05 or error_delta > 0.05:
        status = "WARNING"
        detail = "Small compare is slightly weaker than baseline, but not yet catastrophically so."
    else:
        status = "PASS"
        detail = "Small compare remains directionally healthy for medium training."

    return build_check_item(
        item_id="compare_directionality",
        title="Small-Sample Compare Directionality",
        status=status,
        thresholds={
            "pass": "No malignant-recall drop; top1 delta >= -0.05; error-rate delta <= 0.05.",
            "warning": "Slight degradation tolerated on tiny smoke sample.",
            "fail": "Malignant recall drops or top1/error degrades materially.",
        },
        observed={
            "baseline_top1": safe_rate(baseline_summary, "top1"),
            "agent_top1": safe_rate(agent_summary, "top1"),
            "baseline_topk": safe_rate(baseline_summary, "topk"),
            "agent_topk": safe_rate(agent_summary, "topk"),
            "baseline_malignant_recall": safe_rate(baseline_summary, "malignant_recall"),
            "agent_malignant_recall": safe_rate(agent_summary, "malignant_recall"),
            "baseline_error_rate": safe_rate(baseline_summary, "error_rate"),
            "agent_error_rate": safe_rate(agent_summary, "error_rate"),
            "top1_delta": top1_delta,
            "topk_delta": topk_delta,
            "malignant_recall_delta": malignant_delta,
            "error_rate_delta": error_delta,
        },
        detail=detail,
        blocking=status == "FAIL",
    )


def discover_best_controller_checkpoint() -> Path | None:
    candidates = sorted(Path(PROJECT_ROOT).rglob("*.pt"))
    scored: list[tuple[float, Path]] = []
    for path in candidates:
        if "controller" not in path.name:
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            continue
        score = 0.0
        label_source = str(checkpoint.get("label_source", "") or "")
        selection_policy = dict(checkpoint.get("selection_policy", {}) or {})
        if label_source == "sparse_helpfulness":
            score += 100.0
        elif label_source == "helpful":
            score += 60.0
        elif label_source == "selected_plus_helpful":
            score += 40.0
        elif label_source == "selected":
            score -= 30.0
        if selection_policy:
            score += 20.0
        if "controller_training_smoke" in str(path):
            score += 15.0
        score += path.stat().st_mtime / 1_000_000_000.0
        scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def discover_controller_examples_path() -> Path | None:
    default_examples = DEFAULT_EXAMPLES_PATH if DEFAULT_EXAMPLES_PATH.exists() else None
    preferred = [
        PROJECT_ROOT / "outputs" / "controller_training_data_repair_manual" / "controller_training_examples.jsonl",
        PROJECT_ROOT / "outputs" / "controller_training_data_repair_verify" / "controller_training_examples.jsonl",
        PROJECT_ROOT / "outputs" / "controller_training_step3_check_v3" / "controller_training_examples.jsonl",
        PROJECT_ROOT / "outputs" / "smoke_cycles" / "medium_signal_no_ablation_v1" / "train_runs" / "medium_signal_no_ablation_v1" / "stage0_bootstrap_collect_data" / "controller_training_data" / "controller_training_examples.jsonl",
        default_examples,
        FALLBACK_EXAMPLES_PATH,
    ]
    for path in preferred:
        if path and Path(path).exists():
            return Path(path)
    discovered = sorted(Path(PROJECT_ROOT / "outputs").rglob("controller_training_examples.jsonl"))
    return discovered[-1] if discovered else None


def resolve_eval_indices(*, checkpoint: dict[str, Any], num_examples: int) -> list[int]:
    split_indices = dict(checkpoint.get("split_indices", {}) or {})
    for split_name in ("test", "val", "train"):
        indices = [int(item) for item in split_indices.get(split_name, []) if isinstance(item, int)]
        indices = [item for item in indices if 0 <= item < num_examples]
        if indices:
            return indices
    return list(range(num_examples))


def build_controller_prediction_rows(
    *,
    examples: list[dict[str, Any]],
    probabilities: torch.Tensor,
    label_list: list[str],
    selection_policy: ControllerSelectionPolicy,
    target_k_eval: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if probabilities.shape[0] != len(examples):
        return rows
    for index, example in enumerate(examples):
        prob_vector = probabilities[index].tolist()
        score_map = {skill_name: float(prob) for skill_name, prob in zip(label_list, prob_vector)}
        ranked = [item[0] for item in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]
        local_policy = ControllerSelectionPolicy(
            threshold=selection_policy.threshold,
            target_top_k=max(1, target_k_eval[index] if index < len(target_k_eval) else selection_policy.target_top_k),
            min_select=selection_policy.min_select,
            max_select=selection_policy.max_select,
            top_k_buffer=selection_policy.top_k_buffer,
            relative_margin=selection_policy.relative_margin,
            floor_score=selection_policy.floor_score,
            preserve_top1=selection_policy.preserve_top1,
        )
        selection_details = select_skills_with_policy_details(ranked_skills=ranked, score_map=score_map, policy=local_policy)
        selected = list(selection_details.get("selected_skills", []))
        rows.append(
            {
                "case_id": str(example.get("case_id", "")),
                "selected_skills_pred": selected,
                "ranked_skills_pred": ranked,
                "predicted_skill_count": len(selected),
                "target_k": int(target_k_eval[index] if index < len(target_k_eval) else selection_policy.target_top_k),
                "primary_positive_skills_true": [
                    str(item).strip()
                    for item in dict(example.get("outcome", {})).get("primary_positive_skills", [])
                    if str(item).strip()
                ],
                "harmful_skills_true": [
                    str(item).strip()
                    for item in dict(example.get("outcome", {})).get("explicit_negative_skills", [])
                    if str(item).strip()
                ],
            }
        )
    return rows
def run_helpfulness_records(compare_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list(compare_payload.get("agent_records", [])) if compare_payload else []


def build_environment_snapshot(
    *,
    policy_path: Path,
    split_json: Path | None,
    compare_limit: int,
    compare_split: str,
    records_root: Path,
    eval_root: Path,
    client: DermOpenAIClient,
) -> dict[str, Any]:
    stable_policy = load_policy(policy_path).to_dict()
    return {
        "policy_id": stable_policy.get("policy_id"),
        "policy_path": str(policy_path),
        "planner_policy": stable_policy.get("planner_policy", {}),
        "retrieval_policy": stable_policy.get("retrieval_policy", {}),
        "evidence_policy": stable_policy.get("evidence_policy", {}),
        "split_json": str(split_json) if split_json else "",
        "compare_limit": int(compare_limit),
        "compare_split": compare_split,
        "records_root": str(records_root),
        "eval_root": str(eval_root),
        "qwen_base_url": client.base_url,
        "qwen_model": client.model,
    }


def build_recommended_medium_config(
    *,
    policy_path: Path,
    split_json: Path | None,
    overall_status: str,
) -> dict[str, Any] | None:
    if overall_status == "FAIL":
        return None
    stable_policy = load_policy(policy_path).to_dict()
    return {
        "readiness_level": overall_status,
        "stable_policy_path": str(policy_path),
        "policy_id": stable_policy.get("policy_id"),
        "fixed_split_json": str(split_json) if split_json else "",
        "training_updates": {
            "train_split": "train",
            "writeback_enabled": True,
            "strict_split_state": True,
        },
        "frozen_validation": {
            "eval_split": "val",
            "strict_frozen_eval": True,
            "writeback_enabled": False,
        },
        "controller_training": {
            "label_source": "sparse_helpfulness",
            "min_select": 4,
            "max_select": 8,
            "target_top_k": 5,
            "top_k_buffer": 1,
            "fallback_to_rule_controller": True,
        },
        "retrieval_and_evidence": {
            "enable_learned_retrieval_reranker": False,
            "evidence_calibrator_mode": "heuristic",
            "omit_low_value_evidence": True,
        },
        "pre_and_post_checks": [
            "run contamination audit",
            "run pre-medium smoke compare on val split",
            "keep test split frozen and untouched until final evaluation",
        ],
    }


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_execution_records(records_root: Path) -> list[dict[str, Any]]:
    records_path = records_root / "case_execution_records.jsonl"
    if records_path.exists():
        return load_jsonl(records_path)
    discovered = sorted(records_root.rglob("case_execution_records.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in discovered:
        rows.extend(load_jsonl(path))
    return rows


def build_check_item(
    *,
    item_id: str,
    title: str,
    status: str,
    thresholds: dict[str, Any],
    observed: dict[str, Any],
    detail: str,
    blocking: bool,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "title": title,
        "status": status,
        "blocking": bool(blocking),
        "detail": detail,
        "thresholds": thresholds,
        "observed": observed,
        "artifacts": artifacts or {},
    }


def _collect_status_messages(item: dict[str, Any], *, warnings: list[str], blockers: list[str]) -> None:
    message = f"{item.get('title')}: {item.get('detail')}"
    if item.get("status") == "FAIL":
        blockers.append(message)
    elif item.get("status") == "WARNING":
        warnings.append(message)


def _serialized_evidence_has_key_signals(*, record: dict[str, Any], serialized_text: str) -> bool:
    text = serialized_text.lower()
    evidence_bundle = dict(record.get("evidence_bundle", {}) or {})
    checks: list[bool] = []
    checks.append(bool(text))
    if evidence_bundle.get("risk_flags"):
        checks.append(any(token in text for token in ("risk", "malignan", "danger", "suspicious")))
    if dict(evidence_bundle.get("contradiction_summary", {})).get("contradictions"):
        checks.append(any(token in text for token in ("contradiction", "conflict", "inconsisten")))
    uncertainty_summary = dict(evidence_bundle.get("uncertainty_summary", {}) or {})
    if uncertainty_summary.get("uncertainty_level") not in {"", "low", None} or uncertainty_summary.get("missing_information"):
        checks.append(any(token in text for token in ("uncertainty", "missing", "gap", "insufficient")))
    skill_outputs = dict(evidence_bundle.get("skill_outputs", {}) or {})
    if skill_outputs:
        checks.append("skill" in text or "analysis" in text or "evidence" in text)
    return all(checks) if checks else False


def safe_rate(summary: dict[str, Any], key: str) -> float | None:
    block = dict(summary.get(key, {}) or {})
    value = block.get("rate")
    return _as_float(value) if value is not None else None


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
