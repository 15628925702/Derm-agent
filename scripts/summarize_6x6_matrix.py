from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_NAME = "matrix_metrics.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 6x6 matrix compare reports into a TSV.")
    parser.add_argument("matrix_roots", nargs="+", type=Path, help="One or more outputs/6x6_matrix_* roots.")
    parser.add_argument("--output", type=Path, default=None, help=f"Optional TSV path. Defaults to <first_root>/{DEFAULT_OUTPUT_NAME}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [root for root in args.matrix_roots if root.exists()]
    if not roots:
        raise SystemExit("No existing matrix roots were provided.")
    output_path = args.output or roots[0] / DEFAULT_OUTPUT_NAME
    rows = collect_matrix_rows(roots)
    write_tsv(output_path, rows)
    print(json.dumps({"rows": len(rows), "output_path": str(output_path)}, ensure_ascii=False))
    return 0


def collect_matrix_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for root in roots:
        summary_by_cell = load_summary_status(root / "summary.tsv")
        for report_path in sorted(root.glob("*/*/compare/compare_agent_vs_qwen_*.json")):
            if report_path.is_dir():
                continue
            relative_parts = report_path.relative_to(root).parts
            if len(relative_parts) < 4:
                continue
            model, dataset = relative_parts[0], relative_parts[1]
            key = (str(root), model, dataset)
            if key in seen:
                continue
            seen.add(key)
            payload = load_json(report_path)
            if not isinstance(payload, dict):
                continue
            row = summarize_report(
                root=root,
                model=model,
                dataset=dataset,
                report_path=report_path,
                payload=payload,
                status_row=summary_by_cell.get((model, dataset), {}),
            )
            rows.append(row)

        for (model, dataset), status_row in sorted(summary_by_cell.items()):
            if any(row["matrix_root"] == str(root) and row["model"] == model and row["dataset"] == dataset for row in rows):
                continue
            rows.append(
                {
                    "matrix_root": str(root),
                    "model": model,
                    "dataset": dataset,
                    "status": status_row.get("status", ""),
                    "bootstrap_status": status_row.get("bootstrap_status", ""),
                    "compare_status": status_row.get("compare_status", ""),
                    "num_cases": "",
                    "baseline_top1": "",
                    "agent_top1": "",
                    "top1_delta": "",
                    "baseline_topk": "",
                    "agent_topk": "",
                    "topk_delta": "",
                    "baseline_malignant_recall": "",
                    "agent_malignant_recall": "",
                    "malignant_recall_delta": "",
                    "regression_cases": "",
                    "improvement_cases": "",
                    "malformed_or_empty_final_cases": "",
                    "empty_initial_perception_cases": "",
                    "report_path": "",
                    "log_file": status_row.get("log_file", ""),
                }
            )

    return sorted(rows, key=lambda item: (item["matrix_root"], item["model"], item["dataset"]))


def summarize_report(
    *,
    root: Path,
    model: str,
    dataset: str,
    report_path: Path,
    payload: dict[str, Any],
    status_row: dict[str, str],
) -> dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    baseline = dict(summary.get("baseline", {}) or {})
    agent = dict(summary.get("agent", {}) or {})
    cases = list(payload.get("cases", []) or [])
    baseline_top1 = metric_rate(baseline, "top1")
    agent_top1 = metric_rate(agent, "top1")
    baseline_topk = metric_rate(baseline, "topk")
    agent_topk = metric_rate(agent, "topk")
    baseline_malignant = metric_rate(baseline, "malignant_recall")
    agent_malignant = metric_rate(agent, "malignant_recall")

    return {
        "matrix_root": str(root),
        "model": model,
        "dataset": dataset,
        "status": status_row.get("status", "OK"),
        "bootstrap_status": status_row.get("bootstrap_status", ""),
        "compare_status": status_row.get("compare_status", ""),
        "num_cases": metric_total(baseline, "top1") or metric_total(agent, "top1") or len(cases),
        "baseline_top1": format_value(baseline_top1),
        "agent_top1": format_value(agent_top1),
        "top1_delta": format_value(delta(agent_top1, baseline_top1)),
        "baseline_topk": format_value(baseline_topk),
        "agent_topk": format_value(agent_topk),
        "topk_delta": format_value(delta(agent_topk, baseline_topk)),
        "baseline_malignant_recall": format_value(baseline_malignant),
        "agent_malignant_recall": format_value(agent_malignant),
        "malignant_recall_delta": format_value(delta(agent_malignant, baseline_malignant)),
        "regression_cases": count_delta_cases(cases, negative=True),
        "improvement_cases": count_delta_cases(cases, negative=False),
        "malformed_or_empty_final_cases": count_malformed_or_empty_final(cases),
        "empty_initial_perception_cases": count_empty_initial_perception(cases),
        "report_path": str(report_path),
        "log_file": status_row.get("log_file", ""),
    }


def load_summary_status(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            model = str(row.get("model", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            if model and dataset:
                rows[(model, dataset)] = {str(key): str(value or "") for key, value in row.items()}
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix_root",
        "model",
        "dataset",
        "status",
        "bootstrap_status",
        "compare_status",
        "num_cases",
        "baseline_top1",
        "agent_top1",
        "top1_delta",
        "baseline_topk",
        "agent_topk",
        "topk_delta",
        "baseline_malignant_recall",
        "agent_malignant_recall",
        "malignant_recall_delta",
        "regression_cases",
        "improvement_cases",
        "malformed_or_empty_final_cases",
        "empty_initial_perception_cases",
        "report_path",
        "log_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric_rate(summary: dict[str, Any], metric_name: str) -> float | None:
    metric = summary.get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("rate")
    return float(value) if isinstance(value, int | float) else None


def metric_total(summary: dict[str, Any], metric_name: str) -> int | None:
    metric = summary.get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("total")
    return int(value) if isinstance(value, int) else None


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def count_delta_cases(cases: list[Any], *, negative: bool) -> int:
    count = 0
    target = -1 if negative else 1
    for case in cases:
        if not isinstance(case, dict):
            continue
        delta_payload = dict(case.get("evaluation", {}).get("agent_vs_baseline_delta", {}) or {})
        if any(delta_payload.get(key) == target for key in ("correct_delta", "topk_hit_delta", "malignant_recall_delta")):
            count += 1
    return count


def count_malformed_or_empty_final(cases: list[Any]) -> int:
    count = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        qwen_final = dict(case.get("qwen_final", {}) or {})
        final_label = str(qwen_final.get("final_diagnosis") or case.get("final_diagnosis") or "").strip()
        if is_malformed_final_label(final_label):
            count += 1
    return count


def count_empty_initial_perception(cases: list[Any]) -> int:
    count = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        initial = dict(case.get("qwen_initial", {}) or {})
        if not str(initial.get("image_summary", "")).strip() and not list(initial.get("ddx_candidates", []) or []):
            count += 1
    return count


def is_malformed_final_label(label: str) -> bool:
    text = str(label or "").strip()
    if not text:
        return True
    lowered = text.lower()
    markers = (
        "source_id",
        "retrieval_score",
        "raw_case_memory",
        "experience_type",
        "decision_trace",
        "skill retrieval",
        "boosted by skill",
        "correctness:",
        "\\\\",
        "{",
        "}",
        "[",
        "]",
    )
    return any(marker in lowered for marker in markers) or len(text) > 120


if __name__ == "__main__":
    raise SystemExit(main())
