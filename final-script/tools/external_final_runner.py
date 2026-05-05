#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/root/DermAgent")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.policy_config import load_policy
from agent.run_agent import run_agent
from agent.state import CaseInput
from dataio.ham10000_loader import load_ham10000_case_inputs, load_ham10000_records
from dataio.isic2019_loader import load_isic2019_case_inputs, load_isic2019_records
from integrations.openai_client import DermOpenAIClient
from utils.external_conservative_fusion import THREE_CLASS_LABEL_SPACE, build_conservative_fusion_output


HAM10000_LABEL_MAP = {"mel": "MEL", "bcc": "BCC", "nv": "NEV"}
ISIC2019_LABEL_MAP = {"MEL": "MEL", "BCC": "BCC", "NV": "NEV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable source-based external final runner.")
    parser.add_argument("--dataset", required=True, choices=["ham10000", "isic2019"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--client-base-url", required=True)
    parser.add_argument("--client-api-key", required=True)
    parser.add_argument("--client-model", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--policy-label", default="")
    parser.add_argument("--conservative-generalization-layer", action="store_true")
    return parser.parse_args()


def align_label(text: str) -> str:
    raw = (text or "").strip().lower()
    if "mel" in raw:
        return "MEL"
    if "bcc" in raw or "basal" in raw:
        return "BCC"
    if "nev" in raw or "mole" in raw:
        return "NEV"
    return "OTHER"


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_ham10000_cases(data_root: Path, case_ids: list[str]) -> tuple[list[CaseInput], dict[str, str]]:
    records = load_ham10000_records(data_root=data_root, limit=None, offset=0)
    inputs = load_ham10000_case_inputs(data_root=data_root, limit=None, offset=0)
    record_by_case_id = {record.case_id: record for record in records}
    input_by_case_id = {case.case_id: case for case in inputs}

    resolved_cases: list[CaseInput] = []
    aligned_truths: dict[str, str] = {}
    for case_id in case_ids:
        record = record_by_case_id.get(case_id)
        case_input = input_by_case_id.get(case_id)
        if record is None or case_input is None:
            raise KeyError(f"Missing HAM10000 case `{case_id}` in loader output.")
        resolved_cases.append(case_input)
        aligned_truths[case_id] = HAM10000_LABEL_MAP.get(str(record.original_label).strip().lower(), "OTHER")
    return resolved_cases, aligned_truths


def build_isic2019_cases(data_root: Path, case_ids: list[str]) -> tuple[list[CaseInput], dict[str, str]]:
    records = load_isic2019_records(data_root=data_root, limit=None, offset=0)
    inputs = load_isic2019_case_inputs(data_root=data_root, limit=None, offset=0)
    record_by_case_id = {record.case_id: record for record in records}
    input_by_case_id = {case.case_id: case for case in inputs}

    resolved_cases: list[CaseInput] = []
    aligned_truths: dict[str, str] = {}
    for case_id in case_ids:
        base_case_id = case_id.removesuffix("_downsampled")
        record = record_by_case_id.get(base_case_id)
        case_input = input_by_case_id.get(base_case_id)
        if record is None or case_input is None:
            raise KeyError(f"Missing ISIC2019 case `{case_id}` (base `{base_case_id}`) in loader output.")
        resolved_cases.append(replace(case_input, case_id=case_id))
        aligned_truths[case_id] = ISIC2019_LABEL_MAP.get(str(record.original_label).strip().upper(), "OTHER")
    return resolved_cases, aligned_truths


def summarize(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_rows)
    baseline_top1 = sum(int(row["baseline_correct"]) for row in case_rows)
    agent_top1 = sum(int(row["agent_correct"]) for row in case_rows)
    baseline_topk = sum(int(row["baseline_topk_hit"]) for row in case_rows)
    agent_topk = sum(int(row["agent_topk_hit"]) for row in case_rows)
    return {
        "baseline": {
            "num_cases": total,
            "top1_accuracy": {"hits": baseline_top1, "total": total, "rate": baseline_top1 / total if total else None},
            "topk_accuracy": {"hits": baseline_topk, "total": total, "rate": baseline_topk / total if total else None},
            "error_rate": {"errors": total - baseline_top1, "total": total, "rate": (total - baseline_top1) / total if total else None},
        },
        "agent": {
            "num_cases": total,
            "top1_accuracy": {"hits": agent_top1, "total": total, "rate": agent_top1 / total if total else None},
            "topk_accuracy": {"hits": agent_topk, "total": total, "rate": agent_topk / total if total else None},
            "error_rate": {"errors": total - agent_top1, "total": total, "rate": (total - agent_top1) / total if total else None},
        },
        "agent_vs_baseline": {
            "top1_accuracy_delta": (agent_top1 - baseline_top1) / total if total else None,
            "topk_accuracy_delta": (agent_topk - baseline_topk) / total if total else None,
            "error_rate_delta": ((total - agent_top1) - (total - baseline_top1)) / total if total else None,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest_path))
    case_ids = list(manifest.get("selected_case_ids", []))
    data_root = Path(args.data_root)
    if args.dataset == "ham10000":
        cases, aligned_truths = build_ham10000_cases(data_root, case_ids)
        dataset_name = "HAM10000"
    else:
        cases, aligned_truths = build_isic2019_cases(data_root, case_ids)
        dataset_name = "ISIC2019"

    client = DermOpenAIClient(
        base_url=args.client_base_url,
        api_key=args.client_api_key,
        model=args.client_model,
    )
    policy = load_policy(Path(args.policy_config)).to_dict()

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []

    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        percent = (index / total_cases * 100.0) if total_cases else 100.0
        print(
            f"[progress external {index}/{total_cases} ({percent:.1f}%)] dataset={dataset_name} case_id={case.case_id}",
            flush=True,
        )
        baseline = client.baseline_diagnosis(case)
        state, evidence_package = run_agent(
            case_input=case,
            client=client,
            policy_config=policy,
            output_dir=artifacts_dir,
            enable_writeback=False,
            run_mode="external_final_eval",
            data_split="test",
            strict_frozen_writeback_guard=True,
        )
        agent = dict(state.final_diagnosis)
        if args.conservative_generalization_layer:
            agent = build_conservative_fusion_output(
                baseline_output=baseline,
                agent_output=agent,
                evidence_bundle=evidence_package.to_dict(),
                label_space=THREE_CLASS_LABEL_SPACE,
            )
        baseline_topk = [align_label(item) for item in baseline.get("differential_diagnoses", [])]
        agent_topk = [align_label(item) for item in agent.get("differential_diagnoses", [])]
        case_rows.append(
            {
                "case_id": case.case_id,
                "image_path": case.image_path,
                "dataset_name": dataset_name,
                "ground_truth_original_label": case.label,
                "ground_truth_aligned_label": aligned_truths[case.case_id],
                "baseline_final_diagnosis": baseline.get("final_diagnosis", ""),
                "baseline_final_aligned_label": align_label(baseline.get("final_diagnosis", "")),
                "baseline_differential_diagnoses": baseline.get("differential_diagnoses", []),
                "baseline_topk_aligned_labels": baseline_topk,
                "agent_final_diagnosis": agent.get("final_diagnosis", ""),
                "agent_final_aligned_label": align_label(agent.get("final_diagnosis", "")),
                "agent_differential_diagnoses": agent.get("differential_diagnoses", []),
                "agent_topk_aligned_labels": agent_topk,
                "agent_fusion_decision": agent.get("fusion_decision", {}),
                "baseline_correct": align_label(baseline.get("final_diagnosis", "")) == aligned_truths[case.case_id],
                "agent_correct": align_label(agent.get("final_diagnosis", "")) == aligned_truths[case.case_id],
                "baseline_topk_hit": aligned_truths[case.case_id] in baseline_topk,
                "agent_topk_hit": aligned_truths[case.case_id] in agent_topk,
            }
        )

    result = {
        "run_config": {
            "data_root": str(data_root),
            "manifest_path": str(args.manifest_path),
            "case_count": len(case_rows),
            "base_url": args.client_base_url,
            "model": args.client_model,
            "policy_id": policy.get("policy_id", ""),
            "policy_source_path": str(args.policy_config),
            "policy_label": args.policy_label,
            "evaluation_type": "external_aligned_subset_3class",
            "dataset_name": dataset_name,
            "label_constraint_enabled": True,
            "frozen_compare": True,
            "agent_mode": "full_agent_final",
            "conservative_generalization_layer": bool(args.conservative_generalization_layer),
        },
        "summary": summarize(case_rows),
        "artifacts": {"run_root": str(output_dir)},
        "cases": case_rows,
    }

    output_path = output_dir / f"compare_external_{args.dataset}_aligned_subset.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "case_count": len(case_rows)}, indent=2))


if __name__ == "__main__":
    main()
