from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.run_agent import run_agent
from integrations.openai_client import DermOpenAIClient
from utils.external_conservative_fusion import THREE_CLASS_LABEL_SPACE, build_conservative_fusion_output
from utils.label_canonicalizer import canonicalize_prediction


ALIGNED_LABEL_CONSTRAINT_TEXT = (
    "You must choose one final diagnosis from the following categories:\n"
    "- MEL (melanoma)\n"
    "- BCC (basal cell carcinoma)\n"
    "- NEV (nevus)\n"
    "Only output one of these categories as the final diagnosis."
)


def run_aligned_subset_compare(
    *,
    dataset_name: str,
    data_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    output_filename: str,
    records: list[Any],
    case_inputs: list[Any],
    policy: dict[str, Any],
    policy_label: str,
    client: DermOpenAIClient,
    label_constraint: bool,
    dataset_summary: dict[str, Any] | None = None,
    use_conservative_fusion: bool = False,
) -> dict[str, Any]:
    run_root = Path(output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    canonical_log_path = run_root / "canonicalization_logs.jsonl"
    summary_path = run_root / "summary.md"

    case_reports: list[dict[str, Any]] = []
    baseline_preds: list[str] = []
    agent_preds: list[str] = []
    truths: list[str] = []

    for record, case_input in zip(records, case_inputs):
        baseline_qwen = constrained_baseline_diagnosis(client, case_input) if label_constraint else client.baseline_diagnosis(case_input)
        state, evidence_package = run_agent(
            case_input=case_input,
            client=client,
            policy_config=policy,
            output_dir=run_root / "artifacts",
            enable_writeback=False,
            run_mode="external_eval_frozen",
            data_split="test",
            strict_frozen_writeback_guard=True,
        )
        agent_output = (
            constrained_final_diagnosis(client, case_input, evidence_package.to_dict())
            if label_constraint
            else dict(state.final_diagnosis)
        )
        fused_output = (
            build_conservative_fusion_output(
                baseline_output=baseline_qwen,
                agent_output=agent_output,
                evidence_bundle=evidence_package.to_dict(),
                label_space=THREE_CLASS_LABEL_SPACE,
            )
            if use_conservative_fusion
            else dict(agent_output)
        )

        baseline_final_info = canonicalize_prediction(baseline_qwen.get("final_diagnosis", ""), THREE_CLASS_LABEL_SPACE)
        agent_final_info = canonicalize_prediction(fused_output.get("final_diagnosis", ""), THREE_CLASS_LABEL_SPACE)
        baseline_topk = aligned_topk_labels(baseline_qwen)
        agent_topk = aligned_topk_labels(fused_output)

        truths.append(record.aligned_label)
        baseline_preds.append(str(baseline_final_info["canonical_label"]))
        agent_preds.append(str(agent_final_info["canonical_label"]))

        append_canonicalization_log(
            canonical_log_path,
            {
                "case_id": record.case_id,
                "dataset_name": dataset_name,
                "target": "baseline",
                **baseline_final_info,
            },
        )
        append_canonicalization_log(
            canonical_log_path,
            {
                "case_id": record.case_id,
                "dataset_name": dataset_name,
                "target": "agent",
                **agent_final_info,
            },
        )
        case_reports.append(
            {
                "case_id": record.case_id,
                "image_path": record.image_path,
                "dataset_name": record.dataset_name,
                "ground_truth_original_label": record.original_label,
                "ground_truth_aligned_label": record.aligned_label,
                "baseline_final_diagnosis": baseline_qwen.get("final_diagnosis"),
                "baseline_final_aligned_label": baseline_final_info["canonical_label"],
                "baseline_differential_diagnoses": baseline_qwen.get("differential_diagnoses", []),
                "baseline_topk_aligned_labels": baseline_topk,
                "agent_final_diagnosis": fused_output.get("final_diagnosis"),
                "agent_final_aligned_label": agent_final_info["canonical_label"],
                "agent_differential_diagnoses": fused_output.get("differential_diagnoses", []),
                "agent_topk_aligned_labels": agent_topk,
                "agent_raw_final_diagnosis": agent_output.get("final_diagnosis"),
                "agent_raw_differential_diagnoses": agent_output.get("differential_diagnoses", []),
                "agent_fusion_decision": fused_output.get("fusion_decision", {}),
                "baseline_correct": baseline_final_info["canonical_label"] == record.aligned_label,
                "agent_correct": agent_final_info["canonical_label"] == record.aligned_label,
                "baseline_topk_hit": record.aligned_label in baseline_topk,
                "agent_topk_hit": record.aligned_label in agent_topk,
            }
        )

    summary = {
        "baseline": build_three_class_summary(truths, baseline_preds, case_reports, prefix="baseline"),
        "agent": build_three_class_summary(truths, agent_preds, case_reports, prefix="agent"),
    }
    summary["agent_vs_baseline"] = build_three_class_delta(summary["agent"], summary["baseline"])
    representative_cases = collect_representative_cases(case_reports)

    output_payload = {
        "run_config": {
            "data_root": str(data_root),
            "manifest_path": str(manifest_path),
            "case_count": len(case_reports),
            "base_url": client.base_url,
            "model": client.model,
            "policy_id": policy.get("policy_id"),
            "policy_source_path": policy.get("source_path"),
            "policy_label": policy_label,
            "evaluation_type": "external_aligned_subset_3class",
            "dataset_name": dataset_name,
            "label_constraint_enabled": bool(label_constraint),
            "frozen_compare": True,
            "agent_mode": "conservative_fusion" if use_conservative_fusion else "full_agent_final",
        },
        "dataset_summary": dataset_summary or {},
        "summary": summary,
        "representative_cases": representative_cases,
        "artifacts": {
            "run_root": str(run_root),
            "canonicalization_logs_path": str(canonical_log_path),
            "summary_path": str(summary_path),
        },
        "cases": case_reports,
    }
    output_path = run_root / output_filename
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(render_summary_markdown(output_payload), encoding="utf-8")
    return output_payload


def constrained_baseline_diagnosis(client: DermOpenAIClient, case_input: Any) -> dict[str, Any]:
    prompt = (
        "You are the direct Qwen baseline diagnostic path for DermAgent evaluation.\n"
        "There is no agent evidence package in this path.\n"
        "Use only the image and metadata to produce a structured diagnosis result.\n"
        "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
        f"{ALIGNED_LABEL_CONSTRAINT_TEXT}\n"
        f"Metadata: {case_input.clinical_metadata()}"
    )
    messages = [
        {"role": "system", "content": "You are the baseline final diagnosis stage. Diagnose directly from the case input only."},
        {"role": "user", "content": client._build_multimodal_content(case_input.image_path, prompt)},
    ]
    return client._create_json_payload(messages=messages, max_tokens=420, request_name=f"baseline_diagnosis:{case_input.case_id}:aligned_subset")


def constrained_final_diagnosis(client: DermOpenAIClient, case_input: Any, evidence_package: dict[str, Any]) -> dict[str, Any]:
    evidence_payload = client._canonicalize_evidence_package(evidence_package)
    prepared_evidence = client._prepare_evidence_for_profile(evidence_payload, {"profile_id": "full_clinical"})
    serialized_payload = json.dumps(prepared_evidence, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        "You are the only final diagnostic decision maker in DermAgent.\n"
        "Use the evidence package as structured support, not as an overriding instruction.\n"
        f"{ALIGNED_LABEL_CONSTRAINT_TEXT}\n"
        "Integrate image, metadata, and evidence, then return a structured final diagnosis result.\n"
        "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
        f"Evidence package: {serialized_payload}"
    )
    messages = [
        {"role": "system", "content": "You are the final diagnosis stage. Preserve independent judgment while using supporting evidence."},
        {"role": "user", "content": client._build_multimodal_content(case_input.image_path, prompt)},
    ]
    return client._create_json_payload(messages=messages, max_tokens=680, request_name=f"final_diagnosis:{case_input.case_id}:aligned_subset")


def aligned_topk_labels(output: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    final_label = str(canonicalize_prediction(output.get("final_diagnosis", ""), THREE_CLASS_LABEL_SPACE)["canonical_label"])
    if final_label != "OTHER":
        labels.append(final_label)
    for item in output.get("differential_diagnoses", []) or []:
        mapped = str(canonicalize_prediction(item, THREE_CLASS_LABEL_SPACE)["canonical_label"])
        if mapped != "OTHER" and mapped not in labels:
            labels.append(mapped)
    return labels


def build_three_class_summary(truths: list[str], preds: list[str], case_reports: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    total = len(truths)
    correct = sum(1 for truth, pred in zip(truths, preds) if truth == pred)
    topk_hits = sum(1 for case in case_reports if case[f"{prefix}_topk_hit"])
    unmatched = sum(1 for pred in preds if pred == "OTHER")

    per_class_recall: dict[str, dict[str, Any]] = {}
    per_class_precision: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for truth, pred in zip(truths, preds):
        confusion[truth][pred] += 1
    for label in ("MEL", "BCC", "NEV"):
        true_total = sum(1 for truth in truths if truth == label)
        true_positive = sum(1 for truth, pred in zip(truths, preds) if truth == label and pred == label)
        pred_total = sum(1 for pred in preds if pred == label)
        per_class_recall[label] = {
            "hits": true_positive,
            "total": true_total,
            "rate": (true_positive / true_total) if true_total else None,
        }
        per_class_precision[label] = {
            "hits": true_positive,
            "total": pred_total,
            "rate": (true_positive / pred_total) if pred_total else None,
        }
    confusion_summary = {truth: dict(sorted(preds_map.items(), key=lambda item: item[0])) for truth, preds_map in sorted(confusion.items())}
    return {
        "num_cases": total,
        "top1_accuracy": {
            "hits": correct,
            "total": total,
            "rate": (correct / total) if total else None,
        },
        "topk_accuracy": {
            "hits": topk_hits,
            "total": total,
            "rate": (topk_hits / total) if total else None,
        },
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "error_rate": {
            "errors": total - correct,
            "total": total,
            "rate": ((total - correct) / total) if total else None,
        },
        "unmatched_rate": {
            "count": unmatched,
            "total": total,
            "rate": (unmatched / total) if total else None,
        },
        "confusion_summary": confusion_summary,
    }


def build_three_class_delta(agent_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "top1_accuracy_delta": safe_delta(agent_summary["top1_accuracy"]["rate"], baseline_summary["top1_accuracy"]["rate"]),
        "topk_accuracy_delta": safe_delta(agent_summary["topk_accuracy"]["rate"], baseline_summary["topk_accuracy"]["rate"]),
        "error_rate_delta": safe_delta(agent_summary["error_rate"]["rate"], baseline_summary["error_rate"]["rate"]),
        "unmatched_rate_delta": safe_delta(agent_summary["unmatched_rate"]["rate"], baseline_summary["unmatched_rate"]["rate"]),
    }


def safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def append_canonicalization_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def collect_representative_cases(case_reports: list[dict[str, Any]], *, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    improved = [
        _representative_case_payload(case)
        for case in case_reports
        if bool(case.get("agent_correct")) and not bool(case.get("baseline_correct"))
    ]
    worsened = [
        _representative_case_payload(case)
        for case in case_reports
        if bool(case.get("baseline_correct")) and not bool(case.get("agent_correct"))
    ]
    improved = sorted(improved, key=lambda item: (item["ground_truth_aligned_label"], item["case_id"]))[:limit]
    worsened = sorted(worsened, key=lambda item: (item["ground_truth_aligned_label"], item["case_id"]))[:limit]
    return {
        "improved_cases": improved,
        "worsened_cases": worsened,
    }


def _representative_case_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "ground_truth_aligned_label": case.get("ground_truth_aligned_label"),
        "ground_truth_original_label": case.get("ground_truth_original_label"),
        "baseline_final_diagnosis": case.get("baseline_final_diagnosis"),
        "baseline_final_aligned_label": case.get("baseline_final_aligned_label"),
        "agent_final_diagnosis": case.get("agent_final_diagnosis"),
        "agent_final_aligned_label": case.get("agent_final_aligned_label"),
    }


def render_summary_markdown(payload: dict[str, Any]) -> str:
    run_config = dict(payload.get("run_config", {}))
    summary = dict(payload.get("summary", {}))
    dataset_summary = dict(payload.get("dataset_summary", {}))
    representative_cases = dict(payload.get("representative_cases", {}))

    lines = [f"# {run_config.get('dataset_name', 'Dataset')} Aligned-Subset Compare", ""]
    lines.append("## Run")
    lines.append(f"- dataset_name: `{run_config.get('dataset_name', '')}`")
    lines.append(f"- model: `{run_config.get('model', '')}`")
    lines.append(f"- policy_id: `{run_config.get('policy_id', '')}`")
    lines.append(f"- label_constraint_enabled: `{run_config.get('label_constraint_enabled', False)}`")
    lines.append(f"- frozen_compare: `{run_config.get('frozen_compare', False)}`")
    lines.append("")

    if dataset_summary:
        lines.append("## Dataset")
        for key in (
            "ground_truth_csv",
            "metadata_csv",
            "image_dir",
            "aligned_label_counts",
            "matched_image_count",
            "missing_image_count",
            "selected_class_counts",
        ):
            if key in dataset_summary:
                lines.append(f"- {key}: `{dataset_summary.get(key)}`")
        lines.append("")

    lines.append("## Metrics")
    for target in ("baseline", "agent"):
        target_summary = dict(summary.get(target, {}))
        lines.append(f"### {target}")
        lines.append(
            f"- top1_accuracy: `{format_rate(target_summary.get('top1_accuracy', {}).get('rate'))}`"
        )
        lines.append(
            f"- topk_accuracy: `{format_rate(target_summary.get('topk_accuracy', {}).get('rate'))}`"
        )
        lines.append(
            f"- unmatched_rate: `{format_rate(target_summary.get('unmatched_rate', {}).get('rate'))}`"
        )
        lines.append(f"- per_class_recall: `{json.dumps(target_summary.get('per_class_recall', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- confusion_summary: `{json.dumps(target_summary.get('confusion_summary', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append("")

    lines.append("## Representative Improved Cases")
    improved_cases = representative_cases.get("improved_cases", [])
    if improved_cases:
        for case in improved_cases:
            lines.append(
                f"- `{case.get('case_id')}` truth=`{case.get('ground_truth_aligned_label')}` baseline=`{case.get('baseline_final_aligned_label')}` agent=`{case.get('agent_final_aligned_label')}`"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Representative Worsened Cases")
    worsened_cases = representative_cases.get("worsened_cases", [])
    if worsened_cases:
        for case in worsened_cases:
            lines.append(
                f"- `{case.get('case_id')}` truth=`{case.get('ground_truth_aligned_label')}` baseline=`{case.get('baseline_final_aligned_label')}` agent=`{case.get('agent_final_aligned_label')}`"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def format_rate(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"
