from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.policy_evaluation import gate_policy_candidate
from project_paths import outputs_root


CHECKPOINT_SELECTION_SCHEMA_VERSION = "checkpoint_selection_v1"
DEFAULT_TRAIN_RUNS_ROOT = outputs_root() / "train_runs"
DEFAULT_SELECTION_OUTPUT_ROOT = outputs_root() / "checkpoint_selection"
DEFAULT_CHECKPOINT_EXPORT_ROOT = outputs_root() / "checkpoints"
DEFAULT_SELECTION_WEIGHTS = {
    "malignant_recall": 0.30,
    "top1": 0.22,
    "topk": 0.12,
    "error_rate": 0.16,
    "subset_robustness": 0.12,
    "component_support": 0.08,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def discover_train_run_manifests(train_runs_root: str | Path, *, run_ids: set[str] | None = None) -> list[Path]:
    root = Path(train_runs_root)
    selected = {item.strip() for item in (run_ids or set()) if item.strip()}
    manifests: list[Path] = []
    for path in root.rglob("train_run_manifest.json"):
        run_id = path.parent.name
        if selected and run_id not in selected:
            continue
        manifests.append(path)
    manifests.sort(key=lambda item: (item.stat().st_mtime, str(item)), reverse=True)
    return manifests


def discover_checkpoint_candidates(train_runs_root: str | Path, *, run_ids: set[str] | None = None) -> list[dict[str, Any]]:
    return [build_candidate_from_run_manifest(path) for path in discover_train_run_manifests(train_runs_root, run_ids=run_ids)]


def build_candidate_from_run_manifest(run_manifest_path: str | Path) -> dict[str, Any]:
    run_manifest = read_json(run_manifest_path)
    run_root = Path(run_manifest.get("paths", {}).get("run_root") or Path(run_manifest_path).parent)
    run_id = str(run_manifest.get("run_id") or run_root.name).strip() or run_root.name

    stage1_manifest = _load_stage_manifest(run_manifest, "1")
    stage2_manifest = _load_stage_manifest(run_manifest, "2")
    stage3_manifest = _load_stage_manifest(run_manifest, "3")
    stage4_manifest = _load_stage_manifest(run_manifest, "4")

    controller_metrics_path = _path_or_empty(stage1_manifest.get("outputs", {}).get("metrics_path"))
    retrieval_metrics_path = _path_or_empty(stage2_manifest.get("outputs", {}).get("metrics_path"))
    controller_metrics = _read_json_if_exists(controller_metrics_path)
    retrieval_metrics = _read_json_if_exists(retrieval_metrics_path)

    policy_eval_path = _resolve_policy_eval_path(stage3_manifest, run_root)
    policy_eval = _read_json_if_exists(policy_eval_path)
    candidate_policy_path = _resolve_candidate_policy_path(stage3_manifest, policy_eval_path, run_root)
    candidate_policy = _read_json_if_exists(candidate_policy_path)

    components = {
        "controller_planner_scorer": _build_component_record(
            component_id="controller_planner_scorer",
            stage_manifest=stage1_manifest,
            metrics_path=controller_metrics_path,
            metrics=controller_metrics,
            fallback_checkpoint=_path_or_empty(run_manifest.get("resolved_checkpoints", {}).get("controller_checkpoint_path")),
        ),
        "retrieval_reranker": _build_component_record(
            component_id="retrieval_reranker",
            stage_manifest=stage2_manifest,
            metrics_path=retrieval_metrics_path,
            metrics=retrieval_metrics,
            fallback_checkpoint=_path_or_empty(run_manifest.get("resolved_checkpoints", {}).get("retrieval_checkpoint_path")),
        ),
    }

    evaluation_gate = dict(
        policy_eval.get("candidate_policy", {}).get("evaluation_gate", {})
        or candidate_policy.get("evaluation_gate", {})
        or {}
    )
    stable_summary = dict(policy_eval.get("stable_summary", {}) or {})
    candidate_summary = dict(policy_eval.get("candidate_summary", {}) or {})
    gate_decision = dict(policy_eval.get("gate_decision", {}) or {})
    if policy_eval and candidate_summary and stable_summary and not gate_decision:
        gate_decision = gate_policy_candidate(
            stable_summary=stable_summary,
            candidate_summary=candidate_summary,
            evaluation_gate=evaluation_gate,
        )

    ranking_features = compute_candidate_ranking_features(
        candidate_summary=candidate_summary,
        gate_decision=gate_decision,
        evaluation_gate=evaluation_gate,
        controller_metrics=controller_metrics,
        retrieval_metrics=retrieval_metrics,
    )
    candidate_id = str(
        policy_eval.get("candidate_policy", {}).get("policy_id")
        or candidate_policy.get("policy_id")
        or run_id
    ).strip() or run_id

    source_paths = {
        "run_manifest_path": str(run_manifest_path),
        "run_root": str(run_root),
        "stage1_manifest_path": _manifest_path_or_empty(run_manifest, "1"),
        "stage2_manifest_path": _manifest_path_or_empty(run_manifest, "2"),
        "stage3_manifest_path": _manifest_path_or_empty(run_manifest, "3"),
        "stage4_manifest_path": _manifest_path_or_empty(run_manifest, "4"),
        "policy_eval_path": str(policy_eval_path) if policy_eval_path else "",
        "candidate_policy_path": str(candidate_policy_path) if candidate_policy_path else "",
    }
    if stage4_manifest:
        source_paths["stage4_bundle_manifest_path"] = _path_or_empty(stage4_manifest.get("checkpoint", {}).get("path"))

    return {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "created_at": str(run_manifest.get("created_at", "")).strip(),
        "finished_at": str(run_manifest.get("finished_at", "")).strip(),
        "run_status": str(run_manifest.get("status", "")).strip(),
        "policy_id": str(policy_eval.get("candidate_policy", {}).get("policy_id") or candidate_policy.get("policy_id", "")).strip(),
        "policy_path": str(candidate_policy_path) if candidate_policy_path else "",
        "source_paths": source_paths,
        "components": components,
        "policy_evaluation": {
            "evaluation_record_path": str(policy_eval_path) if policy_eval_path else "",
            "evaluated_at": str(policy_eval.get("evaluated_at", "")).strip(),
            "gate_decision": gate_decision,
            "stable_summary": stable_summary,
            "candidate_summary": candidate_summary,
            "evaluation_gate": evaluation_gate,
        },
        "ranking_features": ranking_features,
        "selection_tier": infer_selection_tier(ranking_features),
    }


def compute_candidate_ranking_features(
    *,
    candidate_summary: dict[str, Any],
    gate_decision: dict[str, Any],
    evaluation_gate: dict[str, Any],
    controller_metrics: dict[str, Any],
    retrieval_metrics: dict[str, Any],
) -> dict[str, Any]:
    min_cases = int(evaluation_gate.get("minimum_cases", 10) or 10)
    num_with_truth = int(candidate_summary.get("num_with_ground_truth", 0) or 0)
    top1_rate = _safe_metric_rate(candidate_summary.get("top1"), default=0.0)
    topk_rate = _safe_metric_rate(candidate_summary.get("topk"), default=0.0)
    malignant_rate = _safe_metric_rate(candidate_summary.get("malignant_recall"), default=0.5)
    error_rate = _safe_metric_rate(candidate_summary.get("error_rate"), default=1.0)
    subset_robustness = compute_subset_robustness(candidate_summary, gate_decision)
    component_support = compute_component_support_score(controller_metrics, retrieval_metrics)
    case_coverage = min(1.0, num_with_truth / float(max(min_cases, 1)))
    gate_passed = bool(gate_decision.get("passed"))
    has_policy_eval = bool(candidate_summary)

    weights = deepcopy(DEFAULT_SELECTION_WEIGHTS)
    composite = 100.0 * (
        weights["malignant_recall"] * malignant_rate
        + weights["top1"] * top1_rate
        + weights["topk"] * topk_rate
        + weights["error_rate"] * max(0.0, 1.0 - error_rate)
        + weights["subset_robustness"] * subset_robustness
        + weights["component_support"] * component_support
    )
    composite += 5.0 * case_coverage
    if gate_passed:
        composite += 3.0
    if has_policy_eval and num_with_truth < min_cases:
        composite -= 10.0
    if has_policy_eval and gate_decision.get("rollback_required"):
        composite -= 12.0

    return {
        "has_policy_eval": has_policy_eval,
        "minimum_cases_required": min_cases,
        "num_with_ground_truth": num_with_truth,
        "has_sufficient_validation": has_policy_eval and num_with_truth >= min_cases,
        "gate_passed": gate_passed,
        "rollback_required": bool(gate_decision.get("rollback_required")),
        "decision": str(gate_decision.get("decision", "")).strip(),
        "top1_rate": round(top1_rate, 6),
        "topk_rate": round(topk_rate, 6),
        "malignant_recall_rate": round(malignant_rate, 6),
        "error_rate": round(error_rate, 6),
        "subset_robustness_score": round(subset_robustness, 6),
        "component_support_score": round(component_support, 6),
        "case_coverage_score": round(case_coverage, 6),
        "composite_selection_score": round(composite, 6),
    }


def compute_subset_robustness(candidate_summary: dict[str, Any], gate_decision: dict[str, Any]) -> float:
    subsets = dict(candidate_summary.get("key_confusion_subsets", {}) or {})
    deltas = dict(gate_decision.get("summary_delta", {}).get("key_confusion_subset_deltas", {}) or {})
    subset_scores: list[float] = []
    for subset_name, payload in subsets.items():
        if int(payload.get("num_cases", 0) or 0) <= 0:
            continue
        top1 = _safe_metric_rate(payload.get("top1"), default=0.0)
        topk = _safe_metric_rate(payload.get("topk"), default=0.0)
        malignant = _safe_metric_rate(payload.get("malignant_recall"), default=0.5)
        error_rate = _safe_metric_rate(payload.get("error_rate"), default=1.0)
        base = (0.38 * top1) + (0.18 * topk) + (0.24 * malignant) + (0.20 * max(0.0, 1.0 - error_rate))
        delta_payload = dict(deltas.get(subset_name, {}) or {})
        penalty = 0.0
        for key in ("top1_delta", "topk_delta", "malignant_recall_delta"):
            delta = delta_payload.get(key)
            if delta is not None and float(delta) < 0.0:
                penalty += min(0.5, abs(float(delta)))
        error_delta = delta_payload.get("error_rate_delta")
        if error_delta is not None and float(error_delta) > 0.0:
            penalty += min(0.5, abs(float(error_delta)))
        subset_scores.append(max(0.0, min(1.0, base - (0.25 * penalty))))
    if subset_scores:
        return sum(subset_scores) / len(subset_scores)
    overall_top1 = _safe_metric_rate(candidate_summary.get("top1"), default=0.0)
    overall_topk = _safe_metric_rate(candidate_summary.get("topk"), default=0.0)
    overall_malignant = _safe_metric_rate(candidate_summary.get("malignant_recall"), default=0.5)
    overall_error = _safe_metric_rate(candidate_summary.get("error_rate"), default=1.0)
    return (0.40 * overall_top1) + (0.20 * overall_topk) + (0.20 * overall_malignant) + (0.20 * max(0.0, 1.0 - overall_error))


def compute_component_support_score(controller_metrics: dict[str, Any], retrieval_metrics: dict[str, Any]) -> float:
    scores: list[float] = []
    controller_val = dict(controller_metrics.get("metrics", {}).get("val", {}) or {})
    if controller_val:
        micro_f1 = float(controller_val.get("micro_f1", 0.0) or 0.0)
        topk_hit = float(controller_val.get("topk_hit_rate", 0.0) or 0.0)
        exact_match = float(controller_val.get("exact_match_ratio", 0.0) or 0.0)
        scores.append((0.55 * micro_f1) + (0.35 * topk_hit) + (0.10 * exact_match))

    retrieval_val = dict(retrieval_metrics.get("metrics", {}).get("val", {}) or {})
    if retrieval_val:
        pointwise = dict(retrieval_val.get("pointwise", {}) or {})
        grouped = dict(retrieval_val.get("grouped", {}) or {})
        binary_f1 = float(pointwise.get("binary_f1", 0.0) or 0.0)
        top1_hit = float(grouped.get("top1_hit_rate", 0.0) or 0.0)
        scores.append((0.45 * binary_f1) + (0.55 * top1_hit))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def infer_selection_tier(features: dict[str, Any]) -> str:
    if bool(features.get("gate_passed")) and bool(features.get("has_sufficient_validation")):
        return "stable_paper_eligible"
    if bool(features.get("has_policy_eval")):
        return "best_validation_eligible"
    return "exploratory_checkpoint"


def build_selection_report(
    candidates: list[dict[str, Any]],
    *,
    train_runs_root: str | Path,
    checkpoints_root: str | Path,
) -> dict[str, Any]:
    ranked_candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
    ranked_payload: list[dict[str, Any]] = []
    for index, candidate in enumerate(ranked_candidates, start=1):
        enriched = deepcopy(candidate)
        enriched["selection_rank"] = index
        enriched["strengths"] = summarize_candidate_strengths(candidate)
        enriched["risks"] = summarize_candidate_risks(candidate)
        ranked_payload.append(enriched)

    validation_candidates = [item for item in ranked_payload if item.get("ranking_features", {}).get("has_policy_eval")]
    stable_candidates = [
        item
        for item in ranked_payload
        if item.get("ranking_features", {}).get("gate_passed")
        and item.get("ranking_features", {}).get("has_sufficient_validation")
    ]
    best_validation = validation_candidates[0] if validation_candidates else None
    stable_paper = stable_candidates[0] if stable_candidates else None
    exploratory_candidates = [item for item in ranked_payload if item.get("selection_tier") == "exploratory_checkpoint"]

    return {
        "schema_version": CHECKPOINT_SELECTION_SCHEMA_VERSION,
        "selection_id": f"checkpoint_selection_{utc_compact()}",
        "generated_at": utc_now(),
        "inputs": {
            "train_runs_root": str(train_runs_root),
            "checkpoints_root": str(checkpoints_root),
            "candidate_count": len(candidates),
        },
        "selection_criteria": {
            "tier_definitions": {
                "best_validation_checkpoint": "Highest-ranked candidate with frozen validation evidence, even if not yet eligible for stable paper export.",
                "stable_paper_checkpoint": "Highest-ranked candidate that passes conservative policy gate and has at least minimum validation coverage.",
                "exploratory_checkpoint": "Candidate lacking frozen validation evidence or failing to reach stable promotion requirements.",
            },
            "ranking_priority": [
                "stable-paper eligibility first",
                "malignant recall",
                "top1 accuracy",
                "topk hit rate",
                "lower error rate",
                "subset robustness on key confusion subsets",
                "component validation support",
                "validation case coverage",
                "recency as final tie-breaker",
            ],
            "score_weights": deepcopy(DEFAULT_SELECTION_WEIGHTS),
        },
        "ranked_candidates": ranked_payload,
        "selected": {
            "best_validation_checkpoint": _selection_stub(best_validation, selection_kind="best_validation_checkpoint"),
            "stable_paper_checkpoint": _selection_stub(stable_paper, selection_kind="stable_paper_checkpoint"),
            "exploratory_checkpoints": [
                _selection_stub(item, selection_kind="exploratory_checkpoint") for item in exploratory_candidates
            ],
        },
    }


def export_selected_checkpoint_bundle(
    candidate: dict[str, Any] | None,
    *,
    export_root: str | Path,
    export_label: str,
    selection_id: str,
    selection_report_path: str | Path,
) -> dict[str, Any]:
    if not candidate:
        return {}

    target_dir = Path(export_root) / export_label / f"{selection_id}__{candidate.get('run_id', 'unknown_run')}"
    target_dir.mkdir(parents=True, exist_ok=True)
    exported_components: list[dict[str, Any]] = []
    for component_id, payload in dict(candidate.get("components", {})).items():
        checkpoint_path = Path(str(payload.get("checkpoint_path", "")).strip()) if str(payload.get("checkpoint_path", "")).strip() else None
        if checkpoint_path is None or not checkpoint_path.exists():
            continue
        target_path = target_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, target_path)
        metrics_path = Path(str(payload.get("metrics_path", "")).strip()) if str(payload.get("metrics_path", "")).strip() else None
        copied_metrics_path = ""
        if metrics_path and metrics_path.exists():
            metrics_target = target_dir / metrics_path.name
            shutil.copy2(metrics_path, metrics_target)
            copied_metrics_path = str(metrics_target)
        exported_components.append(
            {
                "component_id": component_id,
                "source_checkpoint_path": str(checkpoint_path),
                "export_checkpoint_path": str(target_path),
                "source_metrics_path": str(metrics_path) if metrics_path else "",
                "export_metrics_path": copied_metrics_path,
            }
        )

    policy_path = Path(str(candidate.get("policy_path", "")).strip()) if str(candidate.get("policy_path", "")).strip() else None
    exported_policy_path = ""
    if policy_path and policy_path.exists():
        policy_target = target_dir / policy_path.name
        shutil.copy2(policy_path, policy_target)
        exported_policy_path = str(policy_target)

    export_manifest = {
        "schema_version": CHECKPOINT_SELECTION_SCHEMA_VERSION,
        "selection_id": selection_id,
        "export_label": export_label,
        "exported_at": utc_now(),
        "candidate_id": candidate.get("candidate_id"),
        "run_id": candidate.get("run_id"),
        "selection_tier": candidate.get("selection_tier"),
        "selection_report_path": str(selection_report_path),
        "export_dir": str(target_dir),
        "components": exported_components,
        "policy_export_path": exported_policy_path,
    }
    manifest_path = target_dir / "selected_checkpoint_bundle_manifest.json"
    write_json(manifest_path, export_manifest)
    export_manifest["manifest_path"] = str(manifest_path)
    return export_manifest


def save_selection_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
    checkpoint_export_root: str | Path,
    export_best_validation: bool = True,
    export_stable_paper: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selection_id = str(report.get("selection_id", "")).strip() or f"checkpoint_selection_{utc_compact()}"
    report_path = output_root / f"{selection_id}.json"
    write_json(report_path, report)

    exports: dict[str, Any] = {
        "selection_report_path": str(report_path),
        "best_validation_export": {},
        "stable_paper_export": {},
    }
    if export_best_validation:
        exports["best_validation_export"] = export_selected_checkpoint_bundle(
            report.get("selected", {}).get("best_validation_checkpoint", {}).get("candidate"),
            export_root=checkpoint_export_root,
            export_label="best_validation",
            selection_id=selection_id,
            selection_report_path=report_path,
        )
    if export_stable_paper:
        exports["stable_paper_export"] = export_selected_checkpoint_bundle(
            report.get("selected", {}).get("stable_paper_checkpoint", {}).get("candidate"),
            export_root=checkpoint_export_root,
            export_label="stable_paper",
            selection_id=selection_id,
            selection_report_path=report_path,
        )

    report["exports"] = exports
    write_json(report_path, report)
    return {"report_path": str(report_path), "exports": exports}


def summarize_candidate_strengths(candidate: dict[str, Any]) -> list[str]:
    features = dict(candidate.get("ranking_features", {}) or {})
    messages: list[str] = []
    if features.get("has_policy_eval"):
        messages.append(f"frozen validation composite score={features.get('composite_selection_score')}")
        messages.append(f"malignant_recall={features.get('malignant_recall_rate')}")
        messages.append(f"top1={features.get('top1_rate')}, topk={features.get('topk_rate')}")
        messages.append(f"subset_robustness={features.get('subset_robustness_score')}")
    if features.get("component_support_score", 0.0) > 0.0:
        messages.append(f"component_support={features.get('component_support_score')}")
    if features.get("gate_passed"):
        messages.append("passed conservative policy gate")
    return messages


def summarize_candidate_risks(candidate: dict[str, Any]) -> list[str]:
    features = dict(candidate.get("ranking_features", {}) or {})
    gate_decision = dict(candidate.get("policy_evaluation", {}).get("gate_decision", {}) or {})
    risks: list[str] = []
    if not features.get("has_policy_eval"):
        risks.append("no frozen validation record")
    if features.get("has_policy_eval") and not features.get("has_sufficient_validation"):
        risks.append(
            f"insufficient validation coverage: {features.get('num_with_ground_truth')}/{features.get('minimum_cases_required')}"
        )
    if gate_decision.get("rollback_required"):
        risks.extend(str(reason) for reason in gate_decision.get("reasons", []) if str(reason).strip())
    if features.get("error_rate", 1.0) > 0.25:
        risks.append(f"high validation error_rate={features.get('error_rate')}")
    return risks


def _selection_stub(candidate: dict[str, Any] | None, *, selection_kind: str) -> dict[str, Any]:
    if not candidate:
        return {"selection_kind": selection_kind, "selected": False, "candidate": None}
    return {
        "selection_kind": selection_kind,
        "selected": True,
        "candidate_id": candidate.get("candidate_id"),
        "run_id": candidate.get("run_id"),
        "selection_rank": candidate.get("selection_rank", 0),
        "candidate": deepcopy(candidate),
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    features = dict(candidate.get("ranking_features", {}) or {})
    return (
        bool(features.get("gate_passed")) and bool(features.get("has_sufficient_validation")),
        bool(features.get("has_policy_eval")),
        float(features.get("malignant_recall_rate", 0.0) or 0.0),
        float(features.get("top1_rate", 0.0) or 0.0),
        float(features.get("topk_rate", 0.0) or 0.0),
        -float(features.get("error_rate", 1.0) or 1.0),
        float(features.get("subset_robustness_score", 0.0) or 0.0),
        float(features.get("component_support_score", 0.0) or 0.0),
        float(features.get("case_coverage_score", 0.0) or 0.0),
        float(features.get("composite_selection_score", 0.0) or 0.0),
        str(candidate.get("policy_evaluation", {}).get("evaluated_at", "")),
        str(candidate.get("finished_at", "")),
        str(candidate.get("run_id", "")),
    )


def _load_stage_manifest(run_manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
    path = _manifest_path_or_empty(run_manifest, stage_id)
    return _read_json_if_exists(path)


def _manifest_path_or_empty(run_manifest: dict[str, Any], stage_id: str) -> str:
    return str(run_manifest.get("stages", {}).get(stage_id, {}).get("manifest_path", "")).strip()


def _resolve_policy_eval_path(stage3_manifest: dict[str, Any], run_root: Path) -> Path | None:
    direct = _path_or_none(stage3_manifest.get("outputs", {}).get("evaluation_record_path"))
    if direct and direct.exists():
        return direct
    stage_dir = Path(_path_or_empty(stage3_manifest.get("inputs", {}).get("candidate_policy_path"))).parent
    if stage_dir.exists():
        candidates = sorted((stage_dir / "evaluation").glob("policy_eval_*.json"))
        if candidates:
            return candidates[-1]
    alt = list(run_root.rglob("policy_eval_*.json"))
    return sorted(alt)[-1] if alt else None


def _resolve_candidate_policy_path(stage3_manifest: dict[str, Any], policy_eval_path: Path | None, run_root: Path) -> Path | None:
    direct = _path_or_none(stage3_manifest.get("inputs", {}).get("candidate_policy_path"))
    if direct and direct.exists():
        return direct
    if policy_eval_path and policy_eval_path.exists():
        sibling = policy_eval_path.parent.parent / "candidate_policy.json"
        if sibling.exists():
            return sibling
    alt = list(run_root.rglob("candidate_policy.json"))
    return sorted(alt)[-1] if alt else None


def _build_component_record(
    *,
    component_id: str,
    stage_manifest: dict[str, Any],
    metrics_path: str,
    metrics: dict[str, Any],
    fallback_checkpoint: str,
) -> dict[str, Any]:
    checkpoint_path = _path_or_empty(stage_manifest.get("outputs", {}).get("checkpoint_path"))
    if not checkpoint_path:
        checkpoint_path = _path_or_empty(stage_manifest.get("checkpoint", {}).get("path"))
    if not checkpoint_path:
        checkpoint_path = fallback_checkpoint
    metrics_path_resolved = metrics_path
    if not metrics_path_resolved and checkpoint_path:
        inferred = _infer_metrics_path_from_checkpoint(checkpoint_path)
        if inferred:
            metrics_path_resolved = inferred
            metrics = _read_json_if_exists(metrics_path_resolved)
    return {
        "component_id": component_id,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path_resolved,
        "metrics": metrics,
    }


def _read_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    target = _path_or_none(path)
    if target is None or not target.exists():
        return {}
    return read_json(target)


def _path_or_none(value: str | Path | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _path_or_empty(value: str | Path | None) -> str:
    target = _path_or_none(value)
    return str(target) if target else ""


def _safe_metric_rate(metric_block: dict[str, Any] | None, *, default: float) -> float:
    payload = dict(metric_block or {})
    rate = payload.get("rate")
    if rate is None:
        total = payload.get("total")
        if total in {0, None, ""}:
            return default
        return default
    return float(rate)


def _infer_metrics_path_from_checkpoint(checkpoint_path: str | Path) -> str:
    target = _path_or_none(checkpoint_path)
    if target is None:
        return ""
    metrics_candidate = target.with_name(f"{target.stem}_metrics.json")
    if metrics_candidate.exists():
        return str(metrics_candidate)
    return ""
