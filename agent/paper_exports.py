from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from project_paths import outputs_root


DEFAULT_OUTPUTS_ROOT = outputs_root()
DEFAULT_PAPER_EXPORT_ROOT = DEFAULT_OUTPUTS_ROOT / "paper_exports"
TABLE_FILE_ORDER = (
    "main_comparison_table",
    "ablation_table",
    "malignant_recall_table",
    "confusion_pair_summary_table",
    "hard_case_category_table",
    "skill_helpfulness_summary_table",
    "learning_curve_stage_performance_table",
    "error_breakdown_table",
)
FIG_FILE_ORDER = (
    "fig_main_comparison_long",
    "fig_ablation_long",
    "fig_confusion_pair_long",
    "fig_skill_helpfulness_long",
    "fig_learning_curve_long",
    "fig_hard_case_long",
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return rows
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})
    return target


def export_paper_tables(
    *,
    outputs_root: str | Path = DEFAULT_OUTPUTS_ROOT,
    output_dir: str | Path = DEFAULT_PAPER_EXPORT_ROOT / "tables",
) -> dict[str, Any]:
    root = Path(outputs_root)
    table_payloads = collect_table_payloads(root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for file_stem in TABLE_FILE_ORDER:
        rows = list(table_payloads.get(file_stem, []))
        json_path = write_json(output_root / f"{file_stem}.json", rows)
        csv_path = write_csv(output_root / f"{file_stem}.csv", rows)
        manifest_rows.append(
            {
                "file_stem": file_stem,
                "row_count": len(rows),
                "json_path": str(json_path),
                "csv_path": str(csv_path),
            }
        )

    manifest = {
        "tables_dir": str(output_root),
        "outputs_root": str(root),
        "files": manifest_rows,
    }
    manifest_path = write_json(output_root / "paper_tables_manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def export_paper_fig_data(
    *,
    outputs_root: str | Path = DEFAULT_OUTPUTS_ROOT,
    output_dir: str | Path = DEFAULT_PAPER_EXPORT_ROOT / "fig_data",
) -> dict[str, Any]:
    root = Path(outputs_root)
    table_payloads = collect_table_payloads(root)
    fig_payloads = build_figure_payloads(table_payloads)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for file_stem in FIG_FILE_ORDER:
        rows = list(fig_payloads.get(file_stem, []))
        json_path = write_json(output_root / f"{file_stem}.json", rows)
        csv_path = write_csv(output_root / f"{file_stem}.csv", rows)
        manifest_rows.append(
            {
                "file_stem": file_stem,
                "row_count": len(rows),
                "json_path": str(json_path),
                "csv_path": str(csv_path),
            }
        )

    manifest = {
        "fig_data_dir": str(output_root),
        "outputs_root": str(root),
        "files": manifest_rows,
    }
    manifest_path = write_json(output_root / "paper_fig_data_manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def collect_table_payloads(outputs_root: Path) -> dict[str, list[dict[str, Any]]]:
    main_rows = collect_main_comparison_rows(outputs_root)
    ablation_rows = collect_ablation_rows(outputs_root)
    hard_case_rows = collect_hard_case_rows(outputs_root)
    helpfulness_rows = collect_skill_helpfulness_rows(outputs_root)
    learning_rows = collect_learning_curve_rows(outputs_root)
    error_rows = collect_error_breakdown_rows(outputs_root)

    payloads = {
        "main_comparison_table": main_rows,
        "ablation_table": ablation_rows,
        "malignant_recall_table": collect_malignant_recall_rows(main_rows, ablation_rows),
        "confusion_pair_summary_table": collect_confusion_rows(outputs_root),
        "hard_case_category_table": hard_case_rows,
        "skill_helpfulness_summary_table": helpfulness_rows,
        "learning_curve_stage_performance_table": learning_rows,
        "error_breakdown_table": error_rows,
    }
    return payloads


def build_figure_payloads(table_payloads: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "fig_main_comparison_long": _longify_metrics(
            table_payloads.get("main_comparison_table", []),
            id_fields=("eval_id", "suite_label", "target_id", "label"),
            metric_fields=("top1", "topk", "malignant_recall", "error_rate"),
            delta_fields=("top1_delta_vs_baseline", "topk_delta_vs_baseline", "malignant_recall_delta_vs_baseline", "error_rate_delta_vs_baseline"),
        ),
        "fig_ablation_long": _longify_metrics(
            table_payloads.get("ablation_table", []),
            id_fields=("eval_id", "target_id", "label", "controller_family", "experience_variant"),
            metric_fields=("top1", "topk", "malignant_recall", "error_rate"),
            delta_fields=("top1_delta_vs_baseline", "top1_delta_vs_full_anchor", "topk_delta_vs_baseline", "malignant_recall_delta_vs_baseline", "error_rate_delta_vs_baseline"),
        ),
        "fig_confusion_pair_long": _longify_metrics(
            table_payloads.get("confusion_pair_summary_table", []),
            id_fields=("source_type", "source_id", "target_id", "confusion_pair"),
            metric_fields=("top1", "topk", "malignant_recall", "error_rate"),
            delta_fields=("top1_delta_vs_baseline", "error_rate_delta_vs_baseline"),
        ),
        "fig_skill_helpfulness_long": _longify_metrics(
            table_payloads.get("skill_helpfulness_summary_table", []),
            id_fields=("skill_name",),
            metric_fields=(
                "call_count",
                "helpful_count",
                "partially_helpful_count",
                "harmful_count",
                "average_evidence_strength",
                "helpful_rate",
                "harmful_rate",
            ),
            delta_fields=(),
        ),
        "fig_learning_curve_long": _longify_metrics(
            table_payloads.get("learning_curve_stage_performance_table", []),
            id_fields=("run_id", "stage_id", "stage_name", "component", "metric_split"),
            metric_fields=("primary_metric_value", "secondary_metric_value"),
            delta_fields=(),
            metric_name_overrides={
                "primary_metric_value": "primary_metric",
                "secondary_metric_value": "secondary_metric",
            },
            metric_value_name_fields={
                "primary_metric_value": "primary_metric_name",
                "secondary_metric_value": "secondary_metric_name",
            },
        ),
        "fig_hard_case_long": _longify_metrics(
            table_payloads.get("hard_case_category_table", []),
            id_fields=("source_id", "failure_type", "cluster_key"),
            metric_fields=("count", "average_importance"),
            delta_fields=(),
        ),
    }


def collect_main_comparison_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(outputs_root.glob("evaluation_protocol*/eval_brief_*/result_manifest.json")):
        payload = read_json(manifest_path)
        eval_id = str(payload.get("eval_id", manifest_path.parent.name))
        suite_label = str(payload.get("suite_label", "eval_brief"))
        comparisons = dict(payload.get("comparisons", {}))
        for target_result in payload.get("target_results", []):
            target = dict(target_result.get("target", {}))
            summary = dict(target_result.get("summary", {}))
            deltas = dict(comparisons.get(str(target.get("target_id", "")), {}).get("vs_baseline", {}))
            rows.append(
                {
                    "source_type": "evaluation_protocol",
                    "source_id": eval_id,
                    "eval_id": eval_id,
                    "suite_label": suite_label,
                    "target_id": target.get("target_id"),
                    "label": target.get("label"),
                    "mode": target.get("mode"),
                    "target_type": target.get("target_type"),
                    "description": target.get("description"),
                    "experience_variant": target.get("experience_variant"),
                    "cognition_variant": target.get("cognition_variant"),
                    "frozen_evaluation_mode": payload.get("frozen_evaluation_mode"),
                    "num_cases": summary.get("num_cases"),
                    "num_with_ground_truth": summary.get("num_with_ground_truth"),
                    "malignant_hits": summary.get("malignant_recall", {}).get("hits"),
                    "malignant_total": summary.get("malignant_recall", {}).get("total"),
                    "top1": _rate(summary.get("top1")),
                    "topk": _rate(summary.get("topk")),
                    "malignant_recall": _rate(summary.get("malignant_recall")),
                    "error_rate": _rate(summary.get("error_rate")),
                    "top1_delta_vs_baseline": deltas.get("top1_delta"),
                    "topk_delta_vs_baseline": deltas.get("topk_delta"),
                    "malignant_recall_delta_vs_baseline": deltas.get("malignant_recall_delta"),
                    "error_rate_delta_vs_baseline": deltas.get("error_rate_delta"),
                    "records_jsonl_path": target_result.get("artifacts", {}).get("records_jsonl_path", ""),
                }
            )
    for compare_path in sorted(outputs_root.glob("comparison*/compare_agent_vs_qwen_*.json")):
        payload = read_json(compare_path)
        summary = dict(payload.get("summary", {}))
        source_id = compare_path.stem
        rows.extend(
            [
                {
                    "source_type": "compare_script",
                    "source_id": source_id,
                    "eval_id": source_id,
                    "suite_label": "compare_agent_vs_qwen",
                    "target_id": "direct_baseline",
                    "label": "Direct Baseline",
                    "mode": "baseline",
                    "target_type": "baseline",
                    "description": "Direct baseline from comparison script summary.",
                    "experience_variant": "full",
                    "cognition_variant": "live_or_unspecified",
                    "frozen_evaluation_mode": False,
                    "num_cases": summary.get("num_cases"),
                    "num_with_ground_truth": summary.get("num_with_ground_truth"),
                    "malignant_hits": summary.get("baseline_malignant_recall_hits"),
                    "malignant_total": summary.get("malignant_case_count"),
                    "top1": summary.get("baseline_accuracy"),
                    "topk": summary.get("baseline_topk_accuracy"),
                    "malignant_recall": summary.get("baseline_malignant_recall"),
                    "error_rate": _invert_rate(summary.get("baseline_accuracy")),
                    "top1_delta_vs_baseline": 0.0,
                    "topk_delta_vs_baseline": 0.0,
                    "malignant_recall_delta_vs_baseline": 0.0 if summary.get("baseline_malignant_recall") is not None else None,
                    "error_rate_delta_vs_baseline": 0.0,
                    "records_jsonl_path": "",
                },
                {
                    "source_type": "compare_script",
                    "source_id": source_id,
                    "eval_id": source_id,
                    "suite_label": "compare_agent_vs_qwen",
                    "target_id": "full_dermagent",
                    "label": "Full DermAgent",
                    "mode": "agent",
                    "target_type": "agent",
                    "description": "DermAgent from comparison script summary.",
                    "experience_variant": "full",
                    "cognition_variant": "live_or_unspecified",
                    "frozen_evaluation_mode": False,
                    "num_cases": summary.get("num_cases"),
                    "num_with_ground_truth": summary.get("num_with_ground_truth"),
                    "malignant_hits": summary.get("agent_malignant_recall_hits"),
                    "malignant_total": summary.get("malignant_case_count"),
                    "top1": summary.get("agent_accuracy"),
                    "topk": summary.get("agent_topk_accuracy"),
                    "malignant_recall": summary.get("agent_malignant_recall"),
                    "error_rate": _invert_rate(summary.get("agent_accuracy")),
                    "top1_delta_vs_baseline": _subtract(summary.get("agent_accuracy"), summary.get("baseline_accuracy")),
                    "topk_delta_vs_baseline": _subtract(summary.get("agent_topk_accuracy"), summary.get("baseline_topk_accuracy")),
                    "malignant_recall_delta_vs_baseline": _subtract(summary.get("agent_malignant_recall"), summary.get("baseline_malignant_recall")),
                    "error_rate_delta_vs_baseline": _subtract(_invert_rate(summary.get("agent_accuracy")), _invert_rate(summary.get("baseline_accuracy"))),
                    "records_jsonl_path": str(compare_path.parent / "records" / "case_execution_records.jsonl"),
                },
            ]
        )
    return _sort_rows(rows, keys=("source_type", "source_id", "target_id"))


def collect_ablation_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.glob("**/ablation_matrix_summary.json")):
        payload = read_json(path)
        eval_id = str(payload.get("eval_id", path.parent.name))
        for row in payload.get("rows", []):
            enriched = {
                "source_type": "ablation_matrix",
                "source_id": eval_id,
                "eval_id": eval_id,
                **dict(row),
            }
            rows.append(enriched)
    return _sort_rows(rows, keys=("source_id", "target_id"))


def collect_malignant_recall_rows(
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in list(main_rows) + list(ablation_rows):
        rows.append(
            {
                "source_type": row.get("source_type"),
                "source_id": row.get("source_id"),
                "eval_id": row.get("eval_id"),
                "target_id": row.get("target_id"),
                "label": row.get("label"),
                "malignant_hits": row.get("malignant_hits"),
                "malignant_total": row.get("malignant_total"),
                "malignant_recall": row.get("malignant_recall"),
                "top1": row.get("top1"),
                "error_rate": row.get("error_rate"),
            }
        )
    return _sort_rows(rows, keys=("source_type", "source_id", "target_id"))


def collect_confusion_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(outputs_root.glob("**/result_manifest.json")):
        payload = read_json(manifest_path)
        eval_id = str(payload.get("eval_id", manifest_path.parent.name))
        comparisons = dict(payload.get("comparisons", {}))
        for target_result in payload.get("target_results", []):
            target = dict(target_result.get("target", {}))
            summary = dict(target_result.get("summary", {}))
            target_id = str(target.get("target_id", ""))
            confusion_summary = dict(summary.get("key_confusion_subsets", {}))
            confusion_deltas = dict(comparisons.get(target_id, {}).get("vs_baseline", {}).get("key_confusion_subset_deltas", {}))
            for confusion_pair, block in confusion_summary.items():
                delta_block = dict(confusion_deltas.get(confusion_pair, {}))
                rows.append(
                    {
                        "source_type": "result_manifest",
                        "source_id": eval_id,
                        "target_id": target_id,
                        "label": target.get("label"),
                        "confusion_pair": confusion_pair,
                        "num_cases": block.get("num_cases"),
                        "num_with_ground_truth": block.get("num_with_ground_truth"),
                        "top1": _rate(block.get("top1")),
                        "topk": _rate(block.get("topk")),
                        "malignant_recall": _rate(block.get("malignant_recall")),
                        "error_rate": _rate(block.get("error_rate")),
                        "top1_delta_vs_baseline": delta_block.get("top1_delta"),
                        "error_rate_delta_vs_baseline": delta_block.get("error_rate_delta"),
                    }
                )
    return _sort_rows(rows, keys=("source_id", "target_id", "confusion_pair"))


def collect_hard_case_rows(outputs_root: Path) -> list[dict[str, Any]]:
    summary_path = outputs_root / "hard_case_mining" / "summary.json"
    records_path = outputs_root / "hard_case_mining" / "hard_cases.jsonl"
    if not summary_path.exists():
        return []
    summary = read_json(summary_path)
    records = read_jsonl(records_path) if records_path.exists() else []
    rows: list[dict[str, Any]] = []
    for cluster in summary.get("clusters", []):
        cluster_key = str(cluster.get("cluster_key", ""))
        cluster_records = [row for row in records if str(row.get("cluster_key", "")) == cluster_key]
        label_counter = Counter(str(row.get("ground_truth", {}).get("canonical_label", "")).strip() or "unknown" for row in cluster_records)
        confusion_counter = Counter(
            "|".join(row.get("confusion_tags", [])) if row.get("confusion_tags") else "none"
            for row in cluster_records
        )
        rows.append(
            {
                "source_id": "hard_case_mining",
                "failure_type": cluster.get("failure_type"),
                "cluster_key": cluster_key,
                "count": cluster.get("count"),
                "average_importance": cluster.get("average_importance"),
                "example_case_ids": cluster.get("example_case_ids", []),
                "dominant_label": label_counter.most_common(1)[0][0] if label_counter else "",
                "dominant_confusion_pair": confusion_counter.most_common(1)[0][0] if confusion_counter else "",
            }
        )
    return _sort_rows(rows, keys=("failure_type", "cluster_key"))


def collect_skill_helpfulness_rows(outputs_root: Path) -> list[dict[str, Any]]:
    reports_path = outputs_root / "skill_helpfulness" / "skill_helpfulness_reports.jsonl"
    if not reports_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(reports_path):
        rows.append(
            {
                "skill_name": row.get("skill_name"),
                "call_count": row.get("call_count"),
                "selected_count": row.get("selected_count"),
                "helpful_count": row.get("helpful_count"),
                "partially_helpful_count": row.get("partially_helpful_count"),
                "harmful_count": row.get("harmful_count"),
                "average_evidence_strength": row.get("average_evidence_strength"),
                "helpful_rate": row.get("helpful_rate"),
                "harmful_rate": row.get("harmful_rate"),
                "uncertainty_reduction_count": row.get("uncertainty_reduction_count"),
                "contradiction_detection_count": row.get("contradiction_detection_count"),
                "malignant_flag_support_count": row.get("malignant_flag_support_count"),
                "common_use_scenarios": [item.get("name") for item in row.get("common_applicable_scenarios", [])],
                "common_failure_modes": row.get("common_failure_modes", []),
            }
        )
    return _sort_rows(rows, keys=("skill_name",))


def collect_learning_curve_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_manifest_path in sorted(outputs_root.glob("train_runs/*/train_run_manifest.json")):
        run_manifest = read_json(run_manifest_path)
        run_id = str(run_manifest.get("run_id", run_manifest_path.parent.name))
        stages = dict(run_manifest.get("stages", {}))
        for stage_id, stage_entry in sorted(stages.items(), key=lambda item: int(item[0])):
            stage_manifest_path = Path(str(stage_entry.get("manifest_path", "")).strip())
            if not stage_manifest_path.exists():
                continue
            stage_manifest = read_json(stage_manifest_path)
            rows.extend(_rows_from_stage_manifest(run_id, int(stage_id), stage_manifest))
    return _sort_rows(rows, keys=("run_id", "stage_id", "metric_split", "component"))


def collect_error_breakdown_rows(outputs_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source_type, source_id, target_id, record in iter_execution_records(outputs_root):
        categories = infer_error_categories(record)
        case_id = str(record.get("case_id", "")).strip()
        for category in categories:
            key = (source_type, source_id, target_id, category)
            slot = grouped.setdefault(
                key,
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "target_id": target_id,
                    "error_category": category,
                    "count": 0,
                    "case_ids": [],
                    "total_cases": 0,
                },
            )
            slot["count"] += 1
            slot["case_ids"].append(case_id)
        total_key = (source_type, source_id, target_id, "__total__")
        total_slot = grouped.setdefault(
            total_key,
            {
                "source_type": source_type,
                "source_id": source_id,
                "target_id": target_id,
                "error_category": "__total__",
                "count": 0,
                "case_ids": [],
                "total_cases": 0,
            },
        )
        total_slot["count"] += 1

    totals = {
        (slot["source_type"], slot["source_id"], slot["target_id"]): slot["count"]
        for slot in grouped.values()
        if slot["error_category"] == "__total__"
    }
    rows: list[dict[str, Any]] = []
    for slot in grouped.values():
        if slot["error_category"] == "__total__":
            continue
        total_cases = totals.get((slot["source_type"], slot["source_id"], slot["target_id"]), 0)
        rows.append(
            {
                "source_type": slot["source_type"],
                "source_id": slot["source_id"],
                "target_id": slot["target_id"],
                "error_category": slot["error_category"],
                "count": slot["count"],
                "rate": round(slot["count"] / total_cases, 6) if total_cases else None,
                "representative_case_ids": slot["case_ids"][:5],
            }
        )
    return _sort_rows(rows, keys=("source_type", "source_id", "target_id", "error_category"))


def iter_execution_records(outputs_root: Path):
    seen: set[Path] = set()
    for path in sorted(outputs_root.glob("**/records/case_execution_records.jsonl")):
        if path in seen:
            continue
        seen.add(path)
        source_type, source_id, target_id = classify_record_source(path)
        for row in read_jsonl(path):
            yield source_type, source_id, target_id, row


def classify_record_source(path: Path) -> tuple[str, str, str]:
    text = str(path)
    if "/evaluation_protocol" in text or "/comparison_step12/" in text or "/ablations_" in text:
        target_id = path.parents[1].name if len(path.parents) > 1 else path.parent.name
        run_root = path.parents[3] if len(path.parents) > 3 else path.parent
        return "evaluation_records", run_root.name, target_id
    if "/comparison/" in text:
        return "comparison_records", "comparison", "full_dermagent"
    return "execution_records", path.parent.parent.name, path.parent.parent.name


def infer_error_categories(record: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    evaluation = dict(record.get("evaluation", {}))
    baseline_correct = evaluation.get("baseline_correct")
    agent_correct = evaluation.get("correct")
    if baseline_correct is True and agent_correct is False:
        categories.append("baseline_correct_agent_wrong")
    if baseline_correct is False and agent_correct is True:
        categories.append("baseline_wrong_agent_correct")
    if baseline_correct is False and agent_correct is False:
        categories.append("both_wrong")
    if evaluation.get("malignant_recall_hit") is False and record.get("ground_truth", {}).get("malignant_flag") is True:
        categories.append("malignant_miss")

    contradiction_count = _extract_contradiction_count(record)
    if contradiction_count >= 2:
        categories.append("contradiction_heavy")
    uncertainty_level = _extract_uncertainty_level(record)
    if uncertainty_level in {"high", "very_high"}:
        categories.append("high_uncertainty")
    confusion_pair = _extract_confusion_pair(record)
    if confusion_pair and confusion_pair != "none":
        categories.append("confusion_pair_present")
    if not categories:
        categories.append("no_major_error_signal")
    return categories


def _rows_from_stage_manifest(run_id: str, stage_id: int, stage_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    stage_name = str(stage_manifest.get("stage_name", ""))
    status = str(stage_manifest.get("status", ""))
    started_at = stage_manifest.get("started_at")
    finished_at = stage_manifest.get("finished_at")
    checkpoint = dict(stage_manifest.get("checkpoint", {}))
    component = str(checkpoint.get("component", ""))
    version = str(checkpoint.get("version", ""))
    outputs = dict(stage_manifest.get("outputs", {}))
    rows: list[dict[str, Any]] = []

    if stage_id == 0:
        rows.append(
            {
                "run_id": run_id,
                "stage_id": stage_id,
                "stage_name": stage_name,
                "component": component or "bootstrap_data_snapshot",
                "version": version,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "metric_split": "stage",
                "primary_metric_name": "controller_example_count",
                "primary_metric_value": stage_manifest.get("stats", {}).get("controller_example_count"),
                "secondary_metric_name": "records_selected",
                "secondary_metric_value": stage_manifest.get("stats", {}).get("records_selected"),
                "data_version": outputs.get("data_version", ""),
            }
        )
        return rows

    metrics = dict(stage_manifest.get("metrics", {}))
    metric_blocks = _normalize_metric_blocks(metrics)
    if stage_id == 1:
        for split_name, metric_row in metric_blocks:
            rows.append(
                {
                    "run_id": run_id,
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "component": component or "controller_planner_scorer",
                    "version": version,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "metric_split": split_name,
                    "primary_metric_name": "micro_f1",
                    "primary_metric_value": metric_row.get("micro_f1"),
                    "secondary_metric_name": "topk_hit_rate",
                    "secondary_metric_value": metric_row.get("topk_hit_rate"),
                    "data_version": "",
                }
            )
        return rows
    if stage_id == 2:
        for split_name, metric_row in metric_blocks:
            pointwise = dict(metric_row.get("pointwise", {}))
            grouped = dict(metric_row.get("grouped", {}))
            rows.append(
                {
                    "run_id": run_id,
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "component": component or "retrieval_reranker",
                    "version": version,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "metric_split": split_name,
                    "primary_metric_name": "grouped_top1_hit_rate",
                    "primary_metric_value": grouped.get("top1_hit_rate"),
                    "secondary_metric_name": "pointwise_binary_f1",
                    "secondary_metric_value": pointwise.get("binary_f1"),
                    "data_version": "",
                }
            )
        return rows
    if stage_id == 3:
        gate_decision = dict(outputs.get("gate_decision", {}))
        eval_path = Path(str(outputs.get("evaluation_record_path", "")).strip()) if str(outputs.get("evaluation_record_path", "")).strip() else None
        candidate_summary = {}
        if eval_path and eval_path.exists():
            candidate_summary = read_json(eval_path).get("candidate_summary", {})
        rows.append(
            {
                "run_id": run_id,
                "stage_id": stage_id,
                "stage_name": stage_name,
                "component": component or "policy_candidate",
                "version": version,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "metric_split": stage_manifest.get("inputs", {}).get("data_split", "val"),
                "primary_metric_name": "top1",
                "primary_metric_value": _rate(candidate_summary.get("top1")),
                "secondary_metric_name": "malignant_recall",
                "secondary_metric_value": _rate(candidate_summary.get("malignant_recall")),
                "data_version": gate_decision.get("decision", ""),
            }
        )
        return rows
    rows.append(
        {
            "run_id": run_id,
            "stage_id": stage_id,
            "stage_name": stage_name,
            "component": component or "stable_checkpoint_bundle",
            "version": version,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "metric_split": "stage",
            "primary_metric_name": "exported_component_count",
            "primary_metric_value": len(stage_manifest.get("outputs", {}).get("components", [])),
            "secondary_metric_name": "export_tier",
            "secondary_metric_value": stage_manifest.get("outputs", {}).get("export_tier"),
            "data_version": "",
        }
    )
    return rows


def _normalize_metric_blocks(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    blocks: list[tuple[str, dict[str, Any]]] = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            blocks.append((str(key), value))
    if blocks:
        return blocks
    if isinstance(metrics, dict) and metrics:
        return [("stage", metrics)]
    return []


def _longify_metrics(
    rows: list[dict[str, Any]],
    *,
    id_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
    delta_fields: tuple[str, ...],
    metric_name_overrides: dict[str, str] | None = None,
    metric_value_name_fields: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    metric_name_overrides = metric_name_overrides or {}
    metric_value_name_fields = metric_value_name_fields or {}
    for row in rows:
        prefix = {field: row.get(field) for field in id_fields}
        for metric_field in metric_fields:
            payload.append(
                {
                    **prefix,
                    "metric": row.get(metric_value_name_fields.get(metric_field, ""), metric_name_overrides.get(metric_field, metric_field)),
                    "value": row.get(metric_field),
                    "delta_vs_baseline": row.get(f"{metric_field}_delta_vs_baseline"),
                }
            )
        for delta_field in delta_fields:
            payload.append(
                {
                    **prefix,
                    "metric": metric_name_overrides.get(delta_field, delta_field),
                    "value": row.get(delta_field),
                    "delta_vs_baseline": None,
                }
            )
    return payload


def _extract_contradiction_count(record: dict[str, Any]) -> int:
    contradiction_summary = dict(record.get("evidence_bundle", {}).get("contradiction_summary", {}) or {})
    if contradiction_summary.get("contradiction_count") is not None:
        return int(contradiction_summary.get("contradiction_count") or 0)
    if contradiction_summary.get("contradictions") and isinstance(contradiction_summary.get("contradictions"), list):
        return len(contradiction_summary.get("contradictions"))
    return int(record.get("hard_case", {}).get("evidence_snapshot", {}).get("contradiction_count", 0) or 0)


def _extract_uncertainty_level(record: dict[str, Any]) -> str:
    uncertainty = dict(record.get("uncertainty", {}) or {})
    level = str(uncertainty.get("uncertainty_level", "")).strip().lower()
    if level:
        return level
    evidence_uncertainty = dict(record.get("evidence_bundle", {}).get("uncertainty_summary", {}) or {})
    return str(evidence_uncertainty.get("uncertainty_level", "")).strip().lower()


def _extract_confusion_pair(record: dict[str, Any]) -> str:
    reflection_outcome = dict(record.get("reflection_summary", {}).get("case_outcome", {}) or {})
    value = str(reflection_outcome.get("confusion_pair", "")).strip().lower()
    if value:
        return value
    retrieval_confusion = str(record.get("skill_retrieval", {}).get("query_summary", {}).get("confusion_pair", "")).strip().lower()
    return retrieval_confusion


def _rate(block: Any) -> float | None:
    if isinstance(block, dict):
        value = block.get("rate")
        return round(float(value), 6) if value is not None else None
    if block is None:
        return None
    return round(float(block), 6)


def _invert_rate(value: Any) -> float | None:
    if value is None:
        return None
    return round(1.0 - float(value), 6)


def _subtract(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _sort_rows(rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in keys))
