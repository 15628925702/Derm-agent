from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze aligned-subset compare outputs and summarize dominant error modes.")
    parser.add_argument("--compare-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.compare_json.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    if not cases:
        raise ValueError(f"No cases found in compare payload: {args.compare_json}")

    analysis = build_analysis(payload=payload, top_k=max(1, int(args.top_k)))
    json_path = args.output_dir / "error_analysis.json"
    md_path = args.output_dir / "error_analysis.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(analysis), encoding="utf-8")

    print(
        json.dumps(
            {
                "compare_json": str(args.compare_json),
                "output_json": str(json_path),
                "output_md": str(md_path),
                "dominant_agent_errors": analysis["agent_error_buckets"][: min(5, len(analysis["agent_error_buckets"]))],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_analysis(*, payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    cases = list(payload.get("cases", []))
    summary = dict(payload.get("summary", {}))
    run_config = dict(payload.get("run_config", {}))

    baseline_error_buckets = summarize_error_buckets(cases=cases, prefix="baseline")
    agent_error_buckets = summarize_error_buckets(cases=cases, prefix="agent")
    outcome_shift_buckets = summarize_outcome_shifts(cases)

    representative_failures = {
        "agent_nev_to_mel": representative_cases(
            cases,
            truth="NEV",
            pred_key="agent_final_aligned_label",
            pred="MEL",
            limit=top_k,
        ),
        "agent_bcc_to_nev": representative_cases(
            cases,
            truth="BCC",
            pred_key="agent_final_aligned_label",
            pred="NEV",
            limit=top_k,
        ),
        "agent_bcc_to_mel": representative_cases(
            cases,
            truth="BCC",
            pred_key="agent_final_aligned_label",
            pred="MEL",
            limit=top_k,
        ),
        "agent_mel_to_nev": representative_cases(
            cases,
            truth="MEL",
            pred_key="agent_final_aligned_label",
            pred="NEV",
            limit=top_k,
        ),
    }

    improved_cases = [
        summarize_case(case)
        for case in cases
        if bool(case.get("agent_correct")) and not bool(case.get("baseline_correct"))
    ][:top_k]
    worsened_cases = [
        summarize_case(case)
        for case in cases
        if bool(case.get("baseline_correct")) and not bool(case.get("agent_correct"))
    ][:top_k]

    return {
        "run_config": run_config,
        "summary": summary,
        "dataset_summary": payload.get("dataset_summary", {}),
        "num_cases": len(cases),
        "baseline_error_buckets": baseline_error_buckets[:top_k],
        "agent_error_buckets": agent_error_buckets[:top_k],
        "outcome_shift_buckets": outcome_shift_buckets[:top_k],
        "representative_failures": representative_failures,
        "representative_improved_cases": improved_cases,
        "representative_worsened_cases": worsened_cases,
    }


def summarize_error_buckets(*, cases: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for case in cases:
        truth = str(case.get("ground_truth_aligned_label", "")).strip()
        pred = str(case.get(f"{prefix}_final_aligned_label", "")).strip()
        if truth and pred and truth != pred:
            counter[(truth, pred)] += 1
    buckets = [
        {
            "truth": truth,
            "pred": pred,
            "count": count,
        }
        for (truth, pred), count in counter.most_common()
    ]
    return buckets


def summarize_outcome_shifts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for case in cases:
        baseline_correct = bool(case.get("baseline_correct"))
        agent_correct = bool(case.get("agent_correct"))
        if baseline_correct and agent_correct:
            counter["stable_correct"] += 1
        elif (not baseline_correct) and (not agent_correct):
            counter["stable_wrong"] += 1
        elif baseline_correct and not agent_correct:
            counter["worsened"] += 1
        elif (not baseline_correct) and agent_correct:
            counter["improved"] += 1
    return [{"bucket": name, "count": count} for name, count in counter.most_common()]


def representative_cases(
    cases: list[dict[str, Any]],
    *,
    truth: str,
    pred_key: str,
    pred: str,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [
        summarize_case(case)
        for case in cases
        if str(case.get("ground_truth_aligned_label", "")).strip() == truth
        and str(case.get(pred_key, "")).strip() == pred
    ]
    return selected[:limit]


def summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "ground_truth": case.get("ground_truth_aligned_label"),
        "baseline_pred": case.get("baseline_final_aligned_label"),
        "agent_pred": case.get("agent_final_aligned_label"),
        "baseline_final_diagnosis": case.get("baseline_final_diagnosis"),
        "agent_final_diagnosis": case.get("agent_final_diagnosis"),
        "baseline_topk": list(case.get("baseline_topk_aligned_labels", []) or []),
        "agent_topk": list(case.get("agent_topk_aligned_labels", []) or []),
        "image_path": case.get("image_path"),
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = ["# Aligned-Subset Error Analysis", ""]
    run_config = dict(analysis.get("run_config", {}))
    summary = dict(analysis.get("summary", {}))

    lines.append("## Run")
    for key in ("dataset_name", "model", "policy_id", "label_constraint_enabled"):
        if key in run_config:
            lines.append(f"- {key}: `{run_config.get(key)}`")
    lines.append("")

    lines.append("## Metrics")
    for target in ("baseline", "agent"):
        target_summary = dict(summary.get(target, {}))
        lines.append(f"### {target}")
        lines.append(f"- top1: `{format_rate(target_summary.get('top1_accuracy', {}).get('rate'))}`")
        lines.append(f"- topk: `{format_rate(target_summary.get('topk_accuracy', {}).get('rate'))}`")
        lines.append(
            f"- per_class_recall: `{json.dumps(target_summary.get('per_class_recall', {}), ensure_ascii=False, sort_keys=True)}`"
        )
        lines.append("")

    lines.append("## Agent Error Buckets")
    for bucket in analysis.get("agent_error_buckets", []):
        lines.append(f"- `{bucket['truth']} -> {bucket['pred']}`: `{bucket['count']}`")
    lines.append("")

    lines.append("## Outcome Shifts")
    for bucket in analysis.get("outcome_shift_buckets", []):
        lines.append(f"- `{bucket['bucket']}`: `{bucket['count']}`")
    lines.append("")

    lines.append("## Representative Worsened Cases")
    worsened = analysis.get("representative_worsened_cases", [])
    if worsened:
        for case in worsened:
            lines.append(
                f"- `{case['case_id']}` truth=`{case['ground_truth']}` baseline=`{case['baseline_pred']}` agent=`{case['agent_pred']}`"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Representative Improved Cases")
    improved = analysis.get("representative_improved_cases", [])
    if improved:
        for case in improved:
            lines.append(
                f"- `{case['case_id']}` truth=`{case['ground_truth']}` baseline=`{case['baseline_pred']}` agent=`{case['agent_pred']}`"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Focus Buckets")
    for key, items in analysis.get("representative_failures", {}).items():
        lines.append(f"### {key}")
        if items:
            for case in items:
                lines.append(
                    f"- `{case['case_id']}` truth=`{case['ground_truth']}` baseline=`{case['baseline_pred']}` agent=`{case['agent_pred']}`"
                )
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def format_rate(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
