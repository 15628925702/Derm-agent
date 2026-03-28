from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_protocol import EvaluationTargetSpec, run_evaluation_suite
from agent.policy_config import load_policy, load_stable_policy
from agent.policy_evaluation import compare_policy_summaries
from dataio.case_loader import discover_case_source
from integrations.openai_client import DermOpenAIClient
from skills.registry import build_default_registry


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablations"
DEFAULT_MATRIX_CONFIG_PATH = PROJECT_ROOT / "configs" / "ablation_matrix_v1.json"

STAGE1_BASIC_SKILLS = (
    "morphology_analysis_skill",
    "color_pattern_analysis_skill",
    "border_surface_analysis_skill",
    "malignancy_risk_assessment_skill",
    "uncertainty_assessment_skill",
    "distribution_analysis_skill",
    "temporal_evolution_skill",
    "metadata_consistency_skill",
    "differential_compare_skill",
    "contradiction_check_skill",
)
SPECIALIST_SKILLS = ("mel_nev_specialist_skill", "ack_scc_specialist_skill")
MATRIX_SCHEMA_VERSION = "ablation_matrix_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-grade frozen ablation matrix for DermAgent.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root for ablation runs.")
    parser.add_argument("--matrix-config", type=Path, default=DEFAULT_MATRIX_CONFIG_PATH, help="Ablation matrix config JSON path.")
    parser.add_argument("--mode", type=str, default="smoke", choices=("smoke", "full"), help="Smoke uses small default limit; full uses all cases unless --limit is set.")
    parser.add_argument("--limit", type=int, default=None, help="Number of cases. If omitted: smoke=10, full=all available.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start from this case index.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for traceability; selection remains deterministic.")
    parser.add_argument("--data-split", type=str, default="test", choices=("val", "test"), help="Evaluation split label for contamination guard and manifests.")
    parser.add_argument("--split-json", type=Path, default=None, help="Optional fixed split JSON. If omitted, the built-in deterministic split is used.")
    parser.add_argument("--non-strict-frozen-eval", action="store_true", help="Allow snapshotting from resolved split state paths even if some files are missing.")
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional policy JSON path; default is current stable policy.")
    parser.add_argument("--ablations", type=str, default="", help="Comma-separated target_ids to run. Baseline and full anchor are auto-included.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional ablations (retrieval scorer / evidence calibrator pairs).")
    parser.add_argument("--strict-checkpoints", action="store_true", help="Fail if required checkpoint for a selected target is missing.")
    parser.add_argument("--learned-controller-checkpoint", type=Path, default=None, help="Override learned controller checkpoint path.")
    parser.add_argument("--retrieval-scorer-checkpoint", type=Path, default=None, help="Override retrieval scorer checkpoint path.")
    parser.add_argument("--evidence-calibrator-checkpoint", type=Path, default=None, help="Override evidence calibrator checkpoint path.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Override request timeout seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Override retry count for transient inference failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_config = load_matrix_config(args.matrix_config)
    effective_limit = resolve_case_limit(
        mode=args.mode,
        explicit_limit=args.limit,
        data_root=args.data_root,
        matrix_config=matrix_config,
        case_offset=max(0, int(args.case_offset)),
    )
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()
    checkpoint_catalog = resolve_checkpoint_catalog(
        policy=policy,
        learned_controller_checkpoint=args.learned_controller_checkpoint,
        retrieval_scorer_checkpoint=args.retrieval_scorer_checkpoint,
        evidence_calibrator_checkpoint=args.evidence_calibrator_checkpoint,
    )
    selected_target_ids = parse_target_ids(args.ablations)
    if selected_target_ids:
        selected_target_ids.update({"direct_qwen_baseline", "full_without_cognition_update"})
    target_specs, resolved_targets, skipped_targets = build_matrix_target_specs(
        matrix_config=matrix_config,
        include_optional=bool(args.include_optional),
        selected_target_ids=selected_target_ids,
        policy=policy,
        checkpoint_catalog=checkpoint_catalog,
        strict_checkpoints=bool(args.strict_checkpoints),
    )
    if not target_specs:
        raise ValueError("No ablation targets selected after filtering and checkpoint validation.")

    result = run_evaluation_suite(
        output_root=args.output_dir,
        data_root=args.data_root,
        client=client,
        policy_config=policy,
        target_specs=target_specs,
        limit=effective_limit,
        case_offset=max(0, int(args.case_offset)),
        seed=int(args.seed),
        suite_label="ablations",
        data_split=args.data_split,
        split_json=args.split_json,
        strict_frozen_eval=not args.non_strict_frozen_eval,
    )
    table_rows = build_ablation_result_rows(
        result_manifest=dict(result.get("result_manifest", {})),
        resolved_targets=resolved_targets,
    )
    run_root = Path(str(result["run_root"]))
    resolved_config_path = run_root / "ablation_matrix_config_resolved.json"
    summary_json_path = run_root / "ablation_matrix_summary.json"
    summary_csv_path = run_root / "ablation_matrix_summary.csv"

    resolved_payload = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_config_path": str(args.matrix_config),
        "mode": args.mode,
        "limit": effective_limit,
        "case_offset": int(args.case_offset),
        "seed": int(args.seed),
        "data_split": args.data_split,
        "split_json": str(args.split_json) if args.split_json else "",
        "strict_frozen_eval": not args.non_strict_frozen_eval,
        "include_optional": bool(args.include_optional),
        "checkpoint_catalog": checkpoint_catalog,
        "selected_target_ids": [spec.target_id for spec in target_specs],
        "resolved_targets": resolved_targets,
        "skipped_targets": skipped_targets,
    }
    resolved_config_path.write_text(json.dumps(resolved_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = {
        "eval_id": result["eval_id"],
        "run_root": str(run_root),
        "evaluation_manifest_path": result["evaluation_manifest_path"],
        "result_manifest_path": result["result_manifest_path"],
        "table_columns": list(table_rows[0].keys()) if table_rows else [],
        "rows": table_rows,
        "skipped_targets": skipped_targets,
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(table_rows, summary_csv_path)

    print(
        json.dumps(
            {
                "eval_id": result["eval_id"],
                "run_root": str(run_root),
                "evaluation_manifest_path": result["evaluation_manifest_path"],
                "result_manifest_path": result["result_manifest_path"],
                "ablation_config_resolved_path": str(resolved_config_path),
                "ablation_summary_json_path": str(summary_json_path),
                "ablation_summary_csv_path": str(summary_csv_path),
                "target_ids": [spec.target_id for spec in target_specs],
                "skipped_targets": skipped_targets,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_matrix_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Ablation matrix config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version", "")).strip() != MATRIX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported ablation matrix schema: {payload.get('schema_version')}")
    return payload


def parse_target_ids(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def resolve_case_limit(
    *,
    mode: str,
    explicit_limit: int | None,
    data_root: Path,
    matrix_config: dict[str, Any],
    case_offset: int,
) -> int:
    if explicit_limit is not None:
        if explicit_limit <= 0:
            raise ValueError("--limit must be > 0 when provided.")
        return int(explicit_limit)
    run_modes = dict(matrix_config.get("run_modes", {}))
    default_limit = run_modes.get(mode, {}).get("default_limit")
    if isinstance(default_limit, int) and default_limit > 0:
        return int(default_limit)
    if str(default_limit).strip().lower() == "all":
        total = count_total_cases(data_root)
        if case_offset >= total:
            raise ValueError(f"case_offset={case_offset} is beyond dataset size={total}.")
        return max(1, total - case_offset)
    return 10


def count_total_cases(data_root: Path) -> int:
    source = discover_case_source(data_root)
    with source.metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def resolve_checkpoint_catalog(
    *,
    policy: dict[str, Any],
    learned_controller_checkpoint: Path | None,
    retrieval_scorer_checkpoint: Path | None,
    evidence_calibrator_checkpoint: Path | None,
) -> dict[str, str]:
    planner_policy = dict(policy.get("planner_policy", {}))
    retrieval_policy = dict(policy.get("retrieval_policy", {}))
    evidence_policy = dict(policy.get("evidence_policy", {}))

    learned_controller = resolve_checkpoint_path(
        override=learned_controller_checkpoint,
        preferred_paths=_preferred_paths_from_strings([str(planner_policy.get("controller_checkpoint_path", ""))]),
        glob_candidates=[
            PROJECT_ROOT / "outputs" / "checkpoints" / "stable_paper" / "**" / "controller*.pt",
            PROJECT_ROOT / "state" / "trainable_components" / "controller_planner_scorer" / "candidates" / "*.pt",
            PROJECT_ROOT / "outputs" / "**" / "controller*.pt",
        ],
    )
    retrieval_scorer = resolve_checkpoint_path(
        override=retrieval_scorer_checkpoint,
        preferred_paths=_preferred_paths_from_strings([str(retrieval_policy.get("retrieval_reranker_checkpoint_path", ""))]),
        glob_candidates=[
            PROJECT_ROOT / "outputs" / "checkpoints" / "stable_paper" / "**" / "*retrieval*reranker*.pt",
            PROJECT_ROOT / "state" / "trainable_components" / "retrieval_reranker" / "candidates" / "*.pt",
            PROJECT_ROOT / "outputs" / "**" / "*retrieval*reranker*.pt",
        ],
    )
    evidence_calibrator = resolve_checkpoint_path(
        override=evidence_calibrator_checkpoint,
        preferred_paths=_preferred_paths_from_strings([str(evidence_policy.get("calibrator_checkpoint_path", ""))]),
        glob_candidates=[
            PROJECT_ROOT / "outputs" / "checkpoints" / "stable_paper" / "**" / "*evidence*calibrator*.pt",
            PROJECT_ROOT / "state" / "trainable_components" / "evidence_calibrator" / "candidates" / "*.pt",
            PROJECT_ROOT / "outputs" / "**" / "*evidence*calibrator*.pt",
        ],
    )
    return {
        "learned_controller_checkpoint": learned_controller,
        "retrieval_scorer_checkpoint": retrieval_scorer,
        "evidence_calibrator_checkpoint": evidence_calibrator,
    }


def resolve_checkpoint_path(
    *,
    override: Path | None,
    preferred_paths: list[Path],
    glob_candidates: list[Path],
) -> str:
    if override:
        path = override.expanduser()
        return str(path) if path.exists() else ""
    for path in preferred_paths:
        if path.exists():
            return str(path)
    found: list[Path] = []
    for pattern in glob_candidates:
        found.extend(path for path in PROJECT_ROOT.glob(str(pattern.relative_to(PROJECT_ROOT))) if path.is_file())
    if not found:
        return ""
    found.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    return str(found[0])


def _preferred_paths_from_strings(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        paths.append(Path(text).expanduser())
    return paths


def build_matrix_target_specs(
    *,
    matrix_config: dict[str, Any],
    include_optional: bool,
    selected_target_ids: set[str],
    policy: dict[str, Any],
    checkpoint_catalog: dict[str, str],
    strict_checkpoints: bool,
) -> tuple[list[EvaluationTargetSpec], list[dict[str, Any]], list[dict[str, Any]]]:
    all_skills = build_default_registry().list_names()
    non_stage1_basic = sorted(skill for skill in all_skills if skill not in set(STAGE1_BASIC_SKILLS))

    target_specs: list[EvaluationTargetSpec] = []
    resolved_targets: list[dict[str, Any]] = []
    skipped_targets: list[dict[str, Any]] = []

    requested_ids = set(selected_target_ids)
    for target_payload in matrix_config.get("targets", []):
        target = deepcopy(target_payload)
        target_id = str(target.get("target_id", "")).strip()
        if not target_id:
            continue
        if not bool(target.get("enabled", True)):
            skipped_targets.append({"target_id": target_id, "reason": "disabled_in_config"})
            continue
        is_optional = bool(target.get("optional", False))
        if is_optional and not include_optional:
            skipped_targets.append({"target_id": target_id, "reason": "optional_not_enabled"})
            continue
        if requested_ids and target_id not in requested_ids:
            continue

        policy_overrides = deepcopy(target.get("policy_overrides", {}))
        execution_overrides = deepcopy(target.get("execution_overrides", {}))
        skill_profile = str(target.get("skill_profile", "")).strip()
        if skill_profile == "stage1_basic_only":
            planner_overrides = dict(policy_overrides.get("planner_policy", {}))
            force_disable = list(planner_overrides.get("force_disable_skills", []))
            force_disable.extend(non_stage1_basic)
            planner_overrides["force_disable_skills"] = sorted(dict.fromkeys(force_disable))
            policy_overrides["planner_policy"] = planner_overrides

        policy_overrides = replace_checkpoint_placeholders(policy_overrides, checkpoint_catalog)
        requirements = dict(target.get("requirements", {}))
        unmet = unmet_requirements(requirements=requirements, checkpoint_catalog=checkpoint_catalog)
        if unmet:
            payload = {"target_id": target_id, "reason": "missing_required_checkpoint", "missing": unmet}
            if strict_checkpoints:
                raise RuntimeError(f"Target `{target_id}` missing checkpoints: {', '.join(unmet)}")
            skipped_targets.append(payload)
            continue

        spec = EvaluationTargetSpec(
            target_id=target_id,
            label=str(target.get("label", target_id)),
            target_type=str(target.get("target_type", "ablation")),
            mode=str(target.get("mode", "agent")),
            description=str(target.get("description", "")),
            policy_overrides=policy_overrides,
            execution_overrides=execution_overrides,
            experience_variant=str(target.get("experience_variant", "full")),
            cognition_variant=str(target.get("cognition_variant", "frozen")),
            ablation_tags=[str(item) for item in target.get("ablation_tags", []) if str(item).strip()],
            notes=[str(item) for item in target.get("notes", []) if str(item).strip()],
        )
        target_specs.append(spec)
        resolved_targets.append(
            {
                "target_id": target_id,
                "label": spec.label,
                "target_type": spec.target_type,
                "mode": spec.mode,
                "description": spec.description,
                "experiment_meaning": str(target.get("experiment_meaning", "")),
                "optional": bool(is_optional),
                "experience_variant": spec.experience_variant,
                "cognition_variant": spec.cognition_variant,
                "execution_overrides": deepcopy(spec.execution_overrides),
                "policy_overrides": deepcopy(spec.policy_overrides),
                "requirements": requirements,
                "ablation_tags": list(spec.ablation_tags),
            }
        )

    # Ensure baseline + full anchor are present for comparable tables.
    required_anchor_ids = {"direct_qwen_baseline", "full_without_cognition_update"}
    missing_anchors = required_anchor_ids - {spec.target_id for spec in target_specs}
    if missing_anchors:
        raise ValueError(f"Missing required matrix anchor targets: {sorted(missing_anchors)}")
    return target_specs, resolved_targets, skipped_targets


def replace_checkpoint_placeholders(payload: dict[str, Any], checkpoint_catalog: dict[str, str]) -> dict[str, Any]:
    text_payload = json.dumps(payload, ensure_ascii=False)
    text_payload = text_payload.replace("__AUTO_LEARNED_CONTROLLER_CHECKPOINT__", checkpoint_catalog.get("learned_controller_checkpoint", ""))
    text_payload = text_payload.replace("__AUTO_RETRIEVAL_SCORER_CHECKPOINT__", checkpoint_catalog.get("retrieval_scorer_checkpoint", ""))
    text_payload = text_payload.replace("__AUTO_EVIDENCE_CALIBRATOR_CHECKPOINT__", checkpoint_catalog.get("evidence_calibrator_checkpoint", ""))
    return json.loads(text_payload)


def unmet_requirements(*, requirements: dict[str, Any], checkpoint_catalog: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if bool(requirements.get("needs_learned_controller")) and not checkpoint_catalog.get("learned_controller_checkpoint"):
        missing.append("learned_controller_checkpoint")
    if bool(requirements.get("needs_retrieval_reranker")) and not checkpoint_catalog.get("retrieval_scorer_checkpoint"):
        missing.append("retrieval_scorer_checkpoint")
    if bool(requirements.get("needs_evidence_calibrator_checkpoint")) and not checkpoint_catalog.get("evidence_calibrator_checkpoint"):
        missing.append("evidence_calibrator_checkpoint")
    return missing


def build_ablation_result_rows(
    *,
    result_manifest: dict[str, Any],
    resolved_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_rows = {str(item.get("target", {}).get("target_id", "")): item for item in result_manifest.get("target_results", [])}
    comparisons = dict(result_manifest.get("comparisons", {}))
    baseline_id = "direct_qwen_baseline"
    full_anchor_id = "full_without_cognition_update"
    baseline_summary = dict(target_rows.get(baseline_id, {}).get("summary", {}))
    full_summary = dict(target_rows.get(full_anchor_id, {}).get("summary", {}))
    resolved_by_id = {str(item.get("target_id", "")): item for item in resolved_targets}

    rows: list[dict[str, Any]] = []
    for target_id, result in target_rows.items():
        summary = dict(result.get("summary", {}))
        target_spec = dict(result.get("target", {}))
        resolved = dict(resolved_by_id.get(target_id, {}))
        vs_baseline = dict(dict(comparisons.get(target_id, {})).get("vs_baseline", {}))
        vs_full = compare_policy_summaries(full_summary, summary) if full_summary and target_id != full_anchor_id else {}
        row = {
            "target_id": target_id,
            "label": target_spec.get("label", ""),
            "mode": target_spec.get("mode", ""),
            "target_type": target_spec.get("target_type", ""),
            "experiment_meaning": resolved.get("experiment_meaning", ""),
            "controller_family": dict(resolved.get("policy_overrides", {}).get("planner_policy", {})).get("controller_family", ""),
            "experience_variant": target_spec.get("experience_variant", ""),
            "cognition_variant": target_spec.get("cognition_variant", ""),
            "top1": _metric_rate(summary, "top1"),
            "topk": _metric_rate(summary, "topk"),
            "malignant_recall": _metric_rate(summary, "malignant_recall"),
            "error_rate": _metric_rate(summary, "error_rate"),
            "top1_delta_vs_baseline": vs_baseline.get("top1_delta"),
            "topk_delta_vs_baseline": vs_baseline.get("topk_delta"),
            "malignant_recall_delta_vs_baseline": vs_baseline.get("malignant_recall_delta"),
            "error_rate_delta_vs_baseline": vs_baseline.get("error_rate_delta"),
            "top1_delta_vs_full_anchor": vs_full.get("top1_delta"),
            "topk_delta_vs_full_anchor": vs_full.get("topk_delta"),
            "malignant_recall_delta_vs_full_anchor": vs_full.get("malignant_recall_delta"),
            "error_rate_delta_vs_full_anchor": vs_full.get("error_rate_delta"),
        }
        rows.append(row)
    rows.sort(key=lambda item: str(item.get("target_id", "")))
    return rows


def _metric_rate(summary: dict[str, Any], key: str) -> float | None:
    value = dict(summary.get(key, {})).get("rate")
    if value is None:
        return None
    return round(float(value), 6)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
