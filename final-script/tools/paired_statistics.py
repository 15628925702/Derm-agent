#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


METRICS = ("top1", "topk", "malignant_recall", "error_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired statistics for DermAgent comparison runs.")
    parser.add_argument("--run-root", action="append", default=[], help="Comparison run root containing result_manifest.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for paired-statistics outputs.")
    return parser.parse_args()


def latest_run_root(search_root: Path) -> Optional[Path]:
    if not search_root.exists():
        return None
    if (search_root / "result_manifest.json").exists():
        return search_root
    manifests = sorted(search_root.rglob("result_manifest.json"), key=lambda path: (path.stat().st_mtime, str(path)))
    if not manifests:
        return None
    return manifests[-1].parent


def discover_default_run_roots() -> List[Path]:
    env_roots = [
        os.environ.get("QWEN_FINAL_COMPARISON_OUTPUT_ROOT", ""),
        os.environ.get("MEDGEMMA_FINAL_COMPARISON_OUTPUT_ROOT", ""),
        os.environ.get("SKINVL_FINAL_COMPARISON_OUTPUT_ROOT", ""),
        os.environ.get("QWEN_FINAL_EXTERNAL_OUTPUT_ROOT", ""),
    ]
    run_roots: List[Path] = []
    for root in env_roots:
        if not root:
            continue
        resolved = latest_run_root(Path(root))
        if resolved is not None:
            run_roots.append(resolved)
    if run_roots:
        return run_roots

    repo_root = Path(os.environ.get("DERMAGENT_REPO_ROOT", "/root/DermAgent"))
    manifests = sorted(
        repo_root.glob("outputs/**/comparison/**/result_manifest.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    recent_roots: List[Path] = []
    seen = set()
    for manifest in reversed(manifests):
        root = manifest.parent
        if root in seen:
            continue
        seen.add(root)
        recent_roots.append(root)
        if len(recent_roots) >= 4:
            break
    return recent_roots


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_records_jsonl(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            if case_id:
                records[case_id] = record
    return records


def metric_value(record: dict, metric: str) -> Optional[bool]:
    evaluation = record.get("evaluation", {})
    if metric == "top1":
        value = evaluation.get("correct")
    elif metric == "topk":
        value = evaluation.get("topk_hit")
    elif metric == "malignant_recall":
        value = evaluation.get("malignant_recall_hit")
    elif metric == "error_rate":
        correct = evaluation.get("correct")
        value = None if correct is None else (not bool(correct))
    else:
        raise ValueError(metric)
    if value is None:
        return None
    return bool(value)


def mcnemar_exact_pvalue(baseline_only: int, agent_only: int) -> float:
    discordant = baseline_only + agent_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(0, min(baseline_only, agent_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def summarize_metric(pairs: Iterable[Tuple[bool, bool]], metric: str) -> dict:
    both_positive = 0
    baseline_only = 0
    agent_only = 0
    both_negative = 0
    total = 0
    for baseline_value, agent_value in pairs:
        total += 1
        if baseline_value and agent_value:
            both_positive += 1
        elif baseline_value and not agent_value:
            baseline_only += 1
        elif not baseline_value and agent_value:
            agent_only += 1
        else:
            both_negative += 1
    baseline_rate = (both_positive + baseline_only) / total if total else None
    agent_rate = (both_positive + agent_only) / total if total else None
    delta = None if baseline_rate is None or agent_rate is None else agent_rate - baseline_rate
    preferred_direction = "lower" if metric == "error_rate" else "higher"
    return {
        "metric": metric,
        "n": total,
        "both_positive": both_positive,
        "baseline_only": baseline_only,
        "agent_only": agent_only,
        "both_negative": both_negative,
        "baseline_rate": baseline_rate,
        "agent_rate": agent_rate,
        "delta": delta,
        "preferred_direction": preferred_direction,
        "mcnemar_exact_pvalue": mcnemar_exact_pvalue(baseline_only, agent_only),
    }


def run_label(manifest: dict) -> str:
    eval_id = manifest.get("eval_id", "unknown_eval")
    targets = manifest.get("target_results", [])
    if len(targets) >= 2:
        baseline_label = targets[0].get("target", {}).get("label", "baseline")
        agent_label = targets[1].get("target", {}).get("label", "agent")
        return f"{eval_id}::{baseline_label}_vs_{agent_label}"
    return eval_id


def collect_run_summary(run_root: Path) -> dict:
    manifest = load_json(run_root / "result_manifest.json")
    targets = manifest.get("target_results", [])
    if len(targets) < 2:
        raise ValueError(f"Expected at least two targets in {run_root}")

    baseline_artifacts = targets[0]["artifacts"]
    agent_artifacts = targets[1]["artifacts"]
    baseline_records = load_records_jsonl(Path(baseline_artifacts["records_jsonl_path"]))
    agent_records = load_records_jsonl(Path(agent_artifacts["records_jsonl_path"]))
    case_ids = sorted(set(baseline_records) & set(agent_records))

    metrics_summary = []
    for metric in METRICS:
        paired_values: List[Tuple[bool, bool]] = []
        for case_id in case_ids:
            baseline_value = metric_value(baseline_records[case_id], metric)
            agent_value = metric_value(agent_records[case_id], metric)
            if baseline_value is None or agent_value is None:
                continue
            paired_values.append((baseline_value, agent_value))
        metrics_summary.append(summarize_metric(paired_values, metric))

    return {
        "run_root": str(run_root),
        "run_label": run_label(manifest),
        "eval_id": manifest.get("eval_id", ""),
        "case_count": len(case_ids),
        "baseline_label": targets[0].get("target", {}).get("label", "baseline"),
        "agent_label": targets[1].get("target", {}).get("label", "agent"),
        "baseline_summary": targets[0].get("summary", {}),
        "agent_summary": targets[1].get("summary", {}),
        "metrics": metrics_summary,
    }


def write_outputs(output_dir: Path, summaries: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "run_count": len(summaries),
        "runs": summaries,
    }
    (output_dir / "paired_statistics_summary.json").write_text(json.dumps(summary_json, indent=2))

    with (output_dir / "paired_statistics_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_label",
                "eval_id",
                "metric",
                "n",
                "baseline_rate",
                "agent_rate",
                "delta",
                "both_positive",
                "baseline_only",
                "agent_only",
                "both_negative",
                "preferred_direction",
                "mcnemar_exact_pvalue",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            for metric in summary["metrics"]:
                writer.writerow(
                    {
                        "run_label": summary["run_label"],
                        "eval_id": summary["eval_id"],
                        **metric,
                    }
                )

    with (output_dir / "paired_statistics_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_label",
                "eval_id",
                "run_root",
                "case_count",
                "baseline_label",
                "agent_label",
                "baseline_top1_rate",
                "agent_top1_rate",
                "baseline_topk_rate",
                "agent_topk_rate",
                "baseline_malignant_recall_rate",
                "agent_malignant_recall_rate",
                "baseline_error_rate",
                "agent_error_rate",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            baseline = summary["baseline_summary"]
            agent = summary["agent_summary"]
            writer.writerow(
                {
                    "run_label": summary["run_label"],
                    "eval_id": summary["eval_id"],
                    "run_root": summary["run_root"],
                    "case_count": summary["case_count"],
                    "baseline_label": summary["baseline_label"],
                    "agent_label": summary["agent_label"],
                    "baseline_top1_rate": baseline.get("top1", {}).get("rate"),
                    "agent_top1_rate": agent.get("top1", {}).get("rate"),
                    "baseline_topk_rate": baseline.get("topk", {}).get("rate"),
                    "agent_topk_rate": agent.get("topk", {}).get("rate"),
                    "baseline_malignant_recall_rate": baseline.get("malignant_recall", {}).get("rate"),
                    "agent_malignant_recall_rate": agent.get("malignant_recall", {}).get("rate"),
                    "baseline_error_rate": baseline.get("error_rate", {}).get("rate"),
                    "agent_error_rate": agent.get("error_rate", {}).get("rate"),
                }
            )


def main() -> None:
    args = parse_args()
    run_roots = [Path(path) for path in args.run_root] if args.run_root else discover_default_run_roots()
    run_roots = [path for path in run_roots if path.exists() and (path / "result_manifest.json").exists()]
    if not run_roots:
        raise SystemExit("No comparison run roots found. Pass --run-root or generate final comparison outputs first.")
    summaries = []
    total_runs = len(run_roots)
    for index, run_root in enumerate(run_roots, start=1):
        percent = (index / total_runs * 100.0) if total_runs else 100.0
        print(f"[progress paired {index}/{total_runs} ({percent:.1f}%)] run_root={run_root}", flush=True)
        summaries.append(collect_run_summary(run_root))
    write_outputs(Path(args.output_dir), summaries)
    print(json.dumps({"output_dir": args.output_dir, "run_count": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
