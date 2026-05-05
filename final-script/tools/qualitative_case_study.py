#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional


BUCKETS = (
    "agent_top1_win",
    "baseline_top1_win",
    "agent_malignant_win",
    "retrieval_helped",
    "both_wrong",
    "high_uncertainty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build qualitative case-study bundles from comparison runs.")
    parser.add_argument("--run-root", action="append", default=[], help="Comparison run root containing result_manifest.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for qualitative outputs.")
    parser.add_argument("--max-per-bucket", type=int, default=4, help="Maximum cases copied for each bucket.")
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


def find_case_artifact_dir(target_dir: Path, case_id: str) -> Optional[Path]:
    for candidate in (
        target_dir / "artifacts" / case_id,
        target_dir / "records" / case_id,
    ):
        if candidate.exists():
            return candidate
    matches = sorted(target_dir.glob(f"**/{case_id}"))
    return matches[0] if matches else None


def diagnosis(record: dict) -> str:
    return (
        record.get("qwen_final", {}).get("final_diagnosis")
        or record.get("baseline_qwen", {}).get("final_diagnosis")
        or ""
    )


def confidence_label(record: dict) -> str:
    return (
        record.get("qwen_final", {}).get("confidence")
        or record.get("qwen_initial", {}).get("uncertainty", {}).get("level")
        or ""
    )


def selected_skills_count(record: dict) -> int:
    return len(record.get("selected_skills", []) or [])


def has_evidence(record: dict) -> bool:
    evidence_text = record.get("evidence_bundle", {}).get("serialized_evidence_text", "")
    return bool(evidence_text.strip())


def metric(record: dict, key: str) -> bool:
    value = record.get("evaluation", {}).get(key)
    return bool(value)


def rank_key(item: dict) -> tuple:
    return (
        -item["agent_selected_skills_count"],
        -int(item["agent_top1"]),
        -int(item["agent_topk"]),
        item["case_id"],
    )


def collect_candidates(run_root: Path) -> List[dict]:
    manifest = load_json(run_root / "result_manifest.json")
    targets = manifest.get("target_results", [])
    if len(targets) < 2:
        return []

    baseline_target = targets[0]
    agent_target = targets[1]
    baseline_target_dir = Path(baseline_target["artifacts"]["target_dir"])
    agent_target_dir = Path(agent_target["artifacts"]["target_dir"])
    baseline_records = load_records_jsonl(Path(baseline_target["artifacts"]["records_jsonl_path"]))
    agent_records = load_records_jsonl(Path(agent_target["artifacts"]["records_jsonl_path"]))
    case_ids = sorted(set(baseline_records) & set(agent_records))

    candidates: List[dict] = []
    for case_id in case_ids:
        baseline_record = baseline_records[case_id]
        agent_record = agent_records[case_id]
        baseline_correct = metric(baseline_record, "correct")
        agent_correct = metric(agent_record, "correct")
        baseline_topk = metric(baseline_record, "topk_hit")
        agent_topk = metric(agent_record, "topk_hit")
        baseline_malignant = metric(baseline_record, "malignant_recall_hit")
        agent_malignant = metric(agent_record, "malignant_recall_hit")
        agent_skills = selected_skills_count(agent_record)
        uncertainty = confidence_label(agent_record).strip().lower()
        candidate = {
            "run_root": str(run_root),
            "eval_id": manifest.get("eval_id", ""),
            "case_id": case_id,
            "ground_truth": agent_record.get("ground_truth", {}).get("canonical_label", ""),
            "baseline_diagnosis": diagnosis(baseline_record),
            "agent_diagnosis": diagnosis(agent_record),
            "baseline_top1": baseline_correct,
            "agent_top1": agent_correct,
            "baseline_topk": baseline_topk,
            "agent_topk": agent_topk,
            "baseline_malignant": baseline_malignant,
            "agent_malignant": agent_malignant,
            "agent_selected_skills_count": agent_skills,
            "agent_has_evidence": has_evidence(agent_record),
            "agent_confidence_or_uncertainty": confidence_label(agent_record),
            "baseline_record_path": str((baseline_target_dir / "records" / case_id / "case_execution_record.json")),
            "agent_record_path": str((find_case_artifact_dir(agent_target_dir, case_id) or agent_target_dir).joinpath("case_execution_record.json")),
            "baseline_case_dir": str(find_case_artifact_dir(baseline_target_dir, case_id) or baseline_target_dir),
            "agent_case_dir": str(find_case_artifact_dir(agent_target_dir, case_id) or agent_target_dir),
            "bucket_flags": {
                "agent_top1_win": agent_correct and not baseline_correct,
                "baseline_top1_win": baseline_correct and not agent_correct,
                "agent_malignant_win": agent_malignant and not baseline_malignant,
                "retrieval_helped": (agent_skills > 0 or has_evidence(agent_record))
                and ((agent_correct and not baseline_correct) or (agent_topk and not baseline_topk) or (agent_malignant and not baseline_malignant)),
                "both_wrong": (not agent_correct) and (not baseline_correct),
                "high_uncertainty": uncertainty in {"high", "moderate", "low"},
            },
        }
        candidates.append(candidate)
    return candidates


def copy_case_bundle(case: dict, bucket_dir: Path) -> dict:
    case_dir = bucket_dir / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    baseline_case_dir = Path(case["baseline_case_dir"])
    agent_case_dir = Path(case["agent_case_dir"])
    baseline_record = baseline_case_dir / "case_execution_record.json"
    agent_record = agent_case_dir / "case_execution_record.json"
    if baseline_record.exists():
        shutil.copy2(baseline_record, case_dir / "baseline_case_execution_record.json")
    if agent_record.exists():
        shutil.copy2(agent_record, case_dir / "agent_case_execution_record.json")
    for filename in ("evidence_package.json", "reflection.json", "state.json"):
        candidate = agent_case_dir / filename
        if candidate.exists():
            shutil.copy2(candidate, case_dir / filename)

    summary_path = case_dir / "case_summary.json"
    summary_path.write_text(json.dumps(case, indent=2))
    return {**case, "copied_case_dir": str(case_dir), "case_summary_path": str(summary_path)}


def main() -> None:
    args = parse_args()
    run_roots = [Path(path) for path in args.run_root] if args.run_root else discover_default_run_roots()
    run_roots = [path for path in run_roots if path.exists() and (path / "result_manifest.json").exists()]
    if not run_roots:
        raise SystemExit("No comparison run roots found. Pass --run-root or generate final comparison outputs first.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: List[dict] = []
    total_runs = len(run_roots)
    for run_index, run_root in enumerate(run_roots, start=1):
        percent = (run_index / total_runs * 100.0) if total_runs else 100.0
        print(f"[progress qual-runs {run_index}/{total_runs} ({percent:.1f}%)] run_root={run_root}", flush=True)
        all_candidates.extend(collect_candidates(run_root))

    selected_cases: List[dict] = []
    total_buckets = len(BUCKETS)
    for bucket_index, bucket in enumerate(BUCKETS, start=1):
        bucket_percent = (bucket_index / total_buckets * 100.0) if total_buckets else 100.0
        print(f"[progress qual-buckets {bucket_index}/{total_buckets} ({bucket_percent:.1f}%)] bucket={bucket}", flush=True)
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        bucket_candidates = [case for case in all_candidates if case["bucket_flags"].get(bucket)]
        bucket_candidates = sorted(bucket_candidates, key=rank_key)
        selected_for_bucket = bucket_candidates[: args.max_per_bucket]
        total_bucket_cases = len(selected_for_bucket)
        for case_index, case in enumerate(selected_for_bucket, start=1):
            case_percent = (case_index / total_bucket_cases * 100.0) if total_bucket_cases else 100.0
            print(
                f"[progress qual-cases {case_index}/{total_bucket_cases} ({case_percent:.1f}%)] bucket={bucket} case_id={case['case_id']}",
                flush=True,
            )
            copied = copy_case_bundle(case, bucket_dir)
            copied["bucket"] = bucket
            selected_cases.append(copied)

    (output_dir / "qualitative_case_study_summary.json").write_text(
        json.dumps({"run_roots": [str(path) for path in run_roots], "selected_cases": selected_cases}, indent=2)
    )

    with (output_dir / "qualitative_case_study_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bucket",
                "eval_id",
                "case_id",
                "ground_truth",
                "baseline_diagnosis",
                "agent_diagnosis",
                "baseline_top1",
                "agent_top1",
                "baseline_topk",
                "agent_topk",
                "baseline_malignant",
                "agent_malignant",
                "agent_selected_skills_count",
                "agent_has_evidence",
                "agent_confidence_or_uncertainty",
                "copied_case_dir",
            ],
        )
        writer.writeheader()
        for case in selected_cases:
            writer.writerow({key: case.get(key) for key in writer.fieldnames})

    print(json.dumps({"output_dir": str(output_dir), "selected_case_count": len(selected_cases)}, indent=2))


if __name__ == "__main__":
    main()
