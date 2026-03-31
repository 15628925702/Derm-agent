from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import load_policy, load_stable_policy
from agent.run_agent import run_agent
from dataio.ham10000_loader import DEFAULT_HAM10000_ROOT, load_ham10000_case_inputs, load_ham10000_records
from integrations.openai_client import DermOpenAIClient
from utils.external_conservative_fusion import build_conservative_fusion_output


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "ham10000" / "comparison"

MALIGNANT_TERMS = (
    "melanoma",
    "mel",
    "basal cell",
    "bcc",
    "actinic keratos",
    "ack",
    "squamous cell",
    "scc",
    "malignant lesion",
)
BENIGN_TERMS = (
    "nevus",
    "naevus",
    "mole",
    "seborrheic keratos",
    "bkl",
    "dermatofibroma",
    "df",
    "vascular",
    "vasc",
    "angioma",
    "papilloma",
    "benign",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External HAM10000 benign-vs-malignant compare between direct Qwen and current agent.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HAM10000_ROOT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--balanced-by-original-label",
        action="store_true",
        help="Sample a near-balanced subset across original HAM10000 7-class labels. When enabled, case_offset is ignored.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--policy-label", type=str, default="")
    parser.add_argument(
        "--agent-risk-only-fallback",
        action="store_true",
        help="Use the agent risk layer only and keep a baseline-style diagnosis for external HAM10000 evaluation.",
    )
    parser.add_argument(
        "--conservative-fusion",
        action="store_true",
        help="Preserve direct baseline when external agent evidence is not strong enough to justify an override.",
    )
    parser.add_argument("--client-timeout", type=float, default=None)
    parser.add_argument("--client-max-retries", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # External HAM10000 should default to the conservative path unless the caller
    # explicitly requests the stricter risk-only fallback.
    if not args.conservative_fusion and not args.agent_risk_only_fallback:
        args.conservative_fusion = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()

    if args.balanced_by_original_label:
        all_records = load_ham10000_records(
            data_root=args.data_root,
            limit=None,
            offset=0,
            seed=args.seed,
            shuffle=False,
        )
        all_case_inputs = load_ham10000_case_inputs(
            data_root=args.data_root,
            limit=None,
            offset=0,
            seed=args.seed,
            shuffle=False,
        )
        record_by_case_id = {record.case_id: record for record in all_records}
        input_by_case_id = {case_input.case_id: case_input for case_input in all_case_inputs}
        selected_case_ids = select_balanced_case_ids(all_records, limit=args.limit, seed=args.seed)
        records = [record_by_case_id[case_id] for case_id in selected_case_ids]
        case_inputs = [input_by_case_id[case_id] for case_id in selected_case_ids]
    else:
        records = load_ham10000_records(
            data_root=args.data_root,
            limit=args.limit,
            offset=args.case_offset,
            seed=args.seed,
            shuffle=args.shuffle,
        )
        case_inputs = load_ham10000_case_inputs(
            data_root=args.data_root,
            limit=args.limit,
            offset=args.case_offset,
            seed=args.seed,
            shuffle=args.shuffle,
        )

    run_root = args.output_dir / f"compare_external_ham10000_{compact_timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    case_reports: list[dict[str, Any]] = []
    baseline_binary_predictions: list[str] = []
    agent_binary_predictions: list[str] = []
    ground_truth_binary_labels: list[str] = []

    for record, case_input in zip(records, case_inputs):
        baseline_qwen = client.baseline_diagnosis(case_input)
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
            build_risk_only_fallback_output(
                baseline_output=baseline_qwen,
                evidence_bundle=evidence_package.to_dict(),
            )
            if args.agent_risk_only_fallback
            else dict(state.final_diagnosis)
        )
        if args.conservative_fusion:
            agent_output = build_conservative_fusion_output(
                baseline_output=baseline_qwen,
                agent_output=agent_output,
                evidence_bundle=evidence_package.to_dict(),
                label_space=None,
            )

        baseline_binary = binary_label_from_output(baseline_qwen)
        agent_binary = binary_label_from_output(agent_output)
        ground_truth_binary = record.binary_label

        baseline_binary_predictions.append(baseline_binary)
        agent_binary_predictions.append(agent_binary)
        ground_truth_binary_labels.append(ground_truth_binary)

        case_reports.append(
            {
                "case_id": record.case_id,
                "image_path": record.image_path,
                "dataset_name": record.dataset_name,
                "ground_truth_original_label": record.original_label,
                "ground_truth_binary_label": ground_truth_binary,
                "baseline_final_diagnosis": baseline_qwen.get("final_diagnosis"),
                "baseline_differential_diagnoses": baseline_qwen.get("differential_diagnoses", []),
                "baseline_binary_label": baseline_binary,
                "agent_final_diagnosis": agent_output.get("final_diagnosis"),
                "agent_differential_diagnoses": agent_output.get("differential_diagnoses", []),
                "agent_binary_label": agent_binary,
                "agent_mode": (
                    "risk_only_fallback"
                    if args.agent_risk_only_fallback
                    else ("conservative_fusion" if args.conservative_fusion else "full_agent_final")
                ),
                "agent_fusion_decision": agent_output.get("fusion_decision", {}),
                "baseline_binary_correct": baseline_binary == ground_truth_binary,
                "agent_binary_correct": agent_binary == ground_truth_binary,
                "baseline_false_negative": ground_truth_binary == "malignant" and baseline_binary != "malignant",
                "agent_false_negative": ground_truth_binary == "malignant" and agent_binary != "malignant",
                "baseline_false_positive": ground_truth_binary == "benign" and baseline_binary == "malignant",
                "agent_false_positive": ground_truth_binary == "benign" and agent_binary == "malignant",
            }
        )

    summary = {
        "baseline": build_binary_summary(ground_truth_binary_labels, baseline_binary_predictions),
        "agent": build_binary_summary(ground_truth_binary_labels, agent_binary_predictions),
    }
    summary["agent_vs_baseline"] = build_binary_delta(summary["agent"], summary["baseline"])

    output_payload = {
        "run_config": {
            "data_root": str(args.data_root),
            "limit": args.limit,
                "case_offset": args.case_offset,
                "seed": args.seed,
                "shuffle": bool(args.shuffle),
                "balanced_by_original_label": bool(args.balanced_by_original_label),
                "base_url": client.base_url,
                "model": client.model,
                "policy_id": policy.get("policy_id"),
                "policy_source_path": policy.get("source_path"),
                "policy_label": args.policy_label,
                "evaluation_type": "external_binary_eval",
                "dataset_name": "HAM10000",
                "agent_mode": (
                    "risk_only_fallback"
                    if args.agent_risk_only_fallback
                    else ("conservative_fusion" if args.conservative_fusion else "full_agent_final")
                ),
                "sample_original_label_distribution": summarize_original_labels(records),
            },
        "summary": summary,
        "artifacts": {
            "run_root": str(run_root),
        },
        "cases": case_reports,
    }

    output_path = run_root / "compare_external_ham10000.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": summary,
                "report_path": str(output_path),
                "run_root": str(run_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def compact_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def summarize_original_labels(records: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.original_label).strip().lower()
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (item[0])))


def select_balanced_case_ids(records: list[Any], *, limit: int, seed: int) -> list[str]:
    import random

    grouped: dict[str, list[str]] = {}
    for record in records:
        label = str(record.original_label).strip().lower()
        grouped.setdefault(label, []).append(record.case_id)
    rng = random.Random(seed)
    ordered_labels = sorted(grouped)
    for label in ordered_labels:
        rng.shuffle(grouped[label])

    selected: list[str] = []
    # First ensure every label is represented when possible.
    for label in ordered_labels:
        if grouped[label] and len(selected) < limit:
            selected.append(grouped[label].pop())

    # Then round-robin the remaining capacity to keep the label mix roughly even.
    while len(selected) < limit:
        added = False
        for label in ordered_labels:
            if grouped[label] and len(selected) < limit:
                selected.append(grouped[label].pop())
                added = True
        if not added:
            break
    return selected


def binary_label_from_output(output: dict[str, Any]) -> str:
    final_text = str(output.get("final_diagnosis", "")).strip().lower()
    for token in MALIGNANT_TERMS:
        if token in final_text:
            return "malignant"
    for token in BENIGN_TERMS:
        if token in final_text:
            return "benign"

    for item in output.get("differential_diagnoses", []) or []:
        text = str(item).strip().lower()
        for token in MALIGNANT_TERMS:
            if token in text:
                return "malignant"
        for token in BENIGN_TERMS:
            if token in text:
                return "benign"
    return "unknown"


def build_binary_summary(truths: list[str], preds: list[str]) -> dict[str, Any]:
    total = len(truths)
    correct = sum(1 for truth, pred in zip(truths, preds) if truth == pred)
    malignant_total = sum(1 for truth in truths if truth == "malignant")
    benign_total = sum(1 for truth in truths if truth == "benign")
    malignant_hits = sum(1 for truth, pred in zip(truths, preds) if truth == "malignant" and pred == "malignant")
    benign_hits = sum(1 for truth, pred in zip(truths, preds) if truth == "benign" and pred == "benign")
    false_negatives = sum(1 for truth, pred in zip(truths, preds) if truth == "malignant" and pred != "malignant")
    false_positives = sum(1 for truth, pred in zip(truths, preds) if truth == "benign" and pred == "malignant")
    return {
        "num_cases": total,
        "binary_accuracy": {
            "hits": correct,
            "total": total,
            "rate": (correct / total) if total else None,
        },
        "malignant_recall": {
            "hits": malignant_hits,
            "total": malignant_total,
            "rate": (malignant_hits / malignant_total) if malignant_total else None,
        },
        "benign_recall": {
            "hits": benign_hits,
            "total": benign_total,
            "rate": (benign_hits / benign_total) if benign_total else None,
        },
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "error_rate": {
            "errors": total - correct,
            "total": total,
            "rate": ((total - correct) / total) if total else None,
        },
    }


def build_binary_delta(agent_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "binary_accuracy_delta": safe_delta(agent_summary["binary_accuracy"]["rate"], baseline_summary["binary_accuracy"]["rate"]),
        "malignant_recall_delta": safe_delta(agent_summary["malignant_recall"]["rate"], baseline_summary["malignant_recall"]["rate"]),
        "benign_recall_delta": safe_delta(agent_summary["benign_recall"]["rate"], baseline_summary["benign_recall"]["rate"]),
        "false_negative_delta": int(agent_summary["false_negatives"]) - int(baseline_summary["false_negatives"]),
        "false_positive_delta": int(agent_summary["false_positives"]) - int(baseline_summary["false_positives"]),
        "error_rate_delta": safe_delta(agent_summary["error_rate"]["rate"], baseline_summary["error_rate"]["rate"]),
    }


def safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def build_risk_only_fallback_output(
    *,
    baseline_output: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(evidence_bundle.get("evidence_decision_policy", {}))
    risk_layer = dict(policy.get("risk_layer", {}))
    diagnosis_layer = dict(policy.get("diagnosis_override_layer", {}))

    rationale_parts = [str(baseline_output.get("rationale", "")).strip()]
    risk_flag = str(risk_layer.get("risk_flag", "")).strip()
    if risk_flag:
        rationale_parts.append(f"Risk flag: {risk_flag}.")
    caution_flags = [str(item).strip() for item in risk_layer.get("caution_flags", []) if str(item).strip()]
    if caution_flags:
        rationale_parts.append(f"Caution: {'; '.join(caution_flags)}.")
    selected_summaries = [
        str(item.get("summary", "")).strip()
        for item in risk_layer.get("selected_evidence", [])[:3]
        if isinstance(item, dict) and str(item.get("summary", "")).strip()
    ]
    if selected_summaries:
        rationale_parts.append(f"Agent risk evidence: {'; '.join(selected_summaries)}.")
    fallback_reasons = [
        str(item).strip()
        for item in diagnosis_layer.get("why_not_confident_enough_to_override", [])[:4]
        if str(item).strip()
    ]
    if fallback_reasons:
        rationale_parts.append(f"Why not confident enough to override: {'; '.join(fallback_reasons)}.")

    follow_up = list(baseline_output.get("follow_up_considerations", []) or [])
    suggestion = str(risk_layer.get("follow_up_suggestion", "")).strip()
    if suggestion and suggestion not in follow_up:
        follow_up.append(suggestion)
    if caution_flags:
        caution_line = "Agent caution: " + "; ".join(caution_flags)
        if caution_line not in follow_up:
            follow_up.append(caution_line)

    return {
        "final_diagnosis": baseline_output.get("final_diagnosis"),
        "differential_diagnoses": list(baseline_output.get("differential_diagnoses", []) or []),
        "rationale": " ".join(part for part in rationale_parts if part).strip(),
        "confidence": baseline_output.get("confidence"),
        "follow_up_considerations": follow_up,
    }


if __name__ == "__main__":
    raise SystemExit(main())
