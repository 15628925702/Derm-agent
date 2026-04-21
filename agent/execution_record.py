from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.controller_training import build_controller_training_example_from_record
from agent.evaluation import build_agent_vs_baseline_delta, evaluate_diagnosis_output, is_malignant_label
from agent.label_space import canonicalize_label, label_space_snapshot
from agent.state import CaseInput, CaseState


EXECUTION_RECORD_VERSION = "v3"
DEFAULT_RECORD_JSONL_NAME = "case_execution_records.jsonl"


@dataclass
class CaseExecutionRecord:
    record_version: str
    timestamp: str
    case_id: str
    dataset_name: str
    input_summary: dict[str, Any] = field(default_factory=dict)
    qwen_initial: dict[str, Any] = field(default_factory=dict)
    image_read_audit: dict[str, Any] = field(default_factory=dict)
    skill_retrieval: dict[str, Any] = field(default_factory=dict)
    planner_decision: dict[str, Any] = field(default_factory=dict)
    retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    selected_skills: list[str] = field(default_factory=list)
    skill_outputs: dict[str, Any] = field(default_factory=dict)
    evidence_bundle: dict[str, Any] = field(default_factory=dict)
    qwen_final: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    reflection_summary: dict[str, Any] = field(default_factory=dict)
    writeback_ops: dict[str, Any] = field(default_factory=dict)
    cognition_snapshot: dict[str, Any] = field(default_factory=dict)
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    state_versions: dict[str, Any] = field(default_factory=dict)
    controller_training_example: dict[str, Any] = field(default_factory=dict)
    baseline_qwen: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_case_execution_record(
    *,
    case_input: CaseInput,
    state: CaseState,
    evidence_bundle: dict[str, Any],
    planner_retrieval_bundle: dict[str, Any] | None,
    cognition_before: dict[str, Any] | None,
    cognition_after: dict[str, Any] | None,
    writeback_enabled: bool,
    state_versions: dict[str, Any] | None = None,
    baseline_qwen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_skills = _selected_skills(state)
    dataset_name = _infer_dataset_name(case_input)
    label_space = label_space_snapshot(dataset_name=dataset_name, label_space_id=case_input.label_space_id, metadata=case_input.metadata)
    agent_eval = evaluate_diagnosis_output(
        state.final_diagnosis,
        case_input.reference_label or case_input.label,
        dataset_name=dataset_name,
        label_space_id=case_input.label_space_id,
        metadata=case_input.metadata,
    )
    baseline_eval = (
        evaluate_diagnosis_output(
            baseline_qwen,
            case_input.reference_label or case_input.label,
            dataset_name=dataset_name,
            label_space_id=case_input.label_space_id,
            metadata=case_input.metadata,
        )
        if baseline_qwen
        else None
    )
    ground_truth_raw = case_input.reference_label or case_input.label
    planned_writeback_bundle = state.reflection.get("writeback_bundle", {}) if isinstance(state.reflection, dict) else {}
    persisted_writeback_bundle = (
        planned_writeback_bundle if writeback_enabled and state.reflection.get("write_experience") else {}
    )
    reflection_extract = state.reflection.get("reflection_extract", {}) if isinstance(state.reflection, dict) else {}
    skill_assessments = state.reflection.get("skill_assessments", []) if isinstance(state.reflection, dict) else []

    record = CaseExecutionRecord(
        record_version=EXECUTION_RECORD_VERSION,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        case_id=case_input.case_id,
        dataset_name=dataset_name,
        input_summary={
            "image_path": case_input.image_path,
            "metadata_path": case_input.source_metadata_path or "",
            "clinical_metadata": case_input.clinical_metadata(),
            "workflow_context": deepcopy(case_input.workflow_context or {}),
            "image_exists": Path(case_input.image_path).exists(),
            "label_space": label_space,
            "label_space_id": case_input.label_space_id or label_space.get("label_space_id", ""),
        },
        qwen_initial=deepcopy(state.perception),
        image_read_audit=deepcopy(state.image_read_audit),
        skill_retrieval=deepcopy(state.skill_retrieval_bundle),
        planner_decision=deepcopy(state.planner_output),
        retrieval_bundle={
            "before_skills": deepcopy(planner_retrieval_bundle or {}),
            "after_skills": deepcopy(state.retrieval_bundle),
        },
        selected_skills=selected_skills,
        skill_outputs=deepcopy(state.skill_outputs),
        evidence_bundle=deepcopy(evidence_bundle),
        qwen_final=deepcopy(state.final_diagnosis),
        ground_truth={
            "raw_label": ground_truth_raw,
            "canonical_label": canonicalize_label(
                ground_truth_raw,
                dataset_name=dataset_name,
                label_space_id=case_input.label_space_id,
                metadata=case_input.metadata,
            ),
            "malignant_flag": is_malignant_label(
                ground_truth_raw,
                dataset_name=dataset_name,
                label_space_id=case_input.label_space_id,
                metadata=case_input.metadata,
            ),
            "label_space": label_space,
        },
        evaluation={
            "correct": agent_eval.get("correct"),
            "topk_hit": agent_eval.get("topk_hit"),
            "malignant_recall_hit": agent_eval.get("malignant_recall_hit"),
            "baseline_correct": baseline_eval.get("correct") if baseline_eval else None,
            "baseline_topk_hit": baseline_eval.get("topk_hit") if baseline_eval else None,
            "baseline_malignant_recall_hit": baseline_eval.get("malignant_recall_hit") if baseline_eval else None,
            "agent_vs_baseline_delta": build_agent_vs_baseline_delta(agent_eval, baseline_eval),
        },
        reflection_summary={
            "reasoning_summary": state.reflection.get("reasoning_summary"),
            "experience_type": state.reflection.get("experience_type"),
            "detected_errors": deepcopy(state.reflection.get("detected_errors", [])),
            "case_outcome": deepcopy(state.reflection.get("case_outcome", {})),
            "skill_assessments": deepcopy(skill_assessments),
            "skill_judgement_version": _infer_skill_judgement_version(skill_assessments),
            "skill_outcome_counts": _summarize_skill_outcomes(skill_assessments),
            "harmful_skills": _collect_skill_names_by_impact(skill_assessments, impact="harmful"),
            "reflection_extract": deepcopy(reflection_extract),
        },
        writeback_ops=_build_writeback_ops(
            reflection=state.reflection,
            planned_writeback_bundle=planned_writeback_bundle,
            persisted_writeback_bundle=persisted_writeback_bundle,
            writeback_enabled=writeback_enabled,
        ),
        cognition_snapshot={
            "before": deepcopy(cognition_before or {}),
            "after": deepcopy(cognition_after or cognition_before or {}),
        },
        policy_snapshot=deepcopy(state.policy_snapshot),
        state_versions=deepcopy(state_versions or {}),
        baseline_qwen=deepcopy(baseline_qwen) if baseline_qwen else None,
    )
    payload = record.to_dict()
    payload["controller_training_example"] = build_controller_training_example_from_record(payload)
    return payload


def enrich_execution_record_with_baseline(
    record: dict[str, Any],
    *,
    baseline_qwen: dict[str, Any],
) -> dict[str, Any]:
    enriched = deepcopy(record)
    ground_truth_label = enriched.get("ground_truth", {}).get("raw_label")
    dataset_name = str(enriched.get("dataset_name", "")).strip() or None
    metadata = dict(enriched.get("input_summary", {}).get("clinical_metadata", {}) or {})
    label_space_id = str(enriched.get("input_summary", {}).get("label_space_id", "")).strip() or None
    baseline_eval = evaluate_diagnosis_output(
        baseline_qwen,
        ground_truth_label,
        dataset_name=dataset_name,
        label_space_id=label_space_id,
        metadata=metadata,
    )
    agent_eval = {
        "correct": enriched.get("evaluation", {}).get("correct"),
        "topk_hit": enriched.get("evaluation", {}).get("topk_hit"),
        "malignant_recall_hit": enriched.get("evaluation", {}).get("malignant_recall_hit"),
    }
    enriched["baseline_qwen"] = deepcopy(baseline_qwen)
    enriched.setdefault("evaluation", {})
    enriched["evaluation"]["baseline_correct"] = baseline_eval.get("correct")
    enriched["evaluation"]["baseline_topk_hit"] = baseline_eval.get("topk_hit")
    enriched["evaluation"]["baseline_malignant_recall_hit"] = baseline_eval.get("malignant_recall_hit")
    enriched["evaluation"]["agent_vs_baseline_delta"] = build_agent_vs_baseline_delta(agent_eval, baseline_eval)
    enriched["controller_training_example"] = build_controller_training_example_from_record(enriched)
    return enriched


def build_baseline_case_execution_record(
    *,
    case_input: CaseInput,
    baseline_qwen: dict[str, Any],
    policy_snapshot: dict[str, Any] | None = None,
    evaluation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = _infer_dataset_name(case_input)
    label_space = label_space_snapshot(dataset_name=dataset_name, label_space_id=case_input.label_space_id, metadata=case_input.metadata)
    baseline_eval = evaluate_diagnosis_output(
        baseline_qwen,
        case_input.reference_label or case_input.label,
        dataset_name=dataset_name,
        label_space_id=case_input.label_space_id,
        metadata=case_input.metadata,
    )
    ground_truth_raw = case_input.reference_label or case_input.label
    baseline_run_mode = str((evaluation_context or {}).get("run_mode", "baseline")).strip().lower() or "baseline"
    baseline_data_split = str((evaluation_context or {}).get("data_split", "unknown")).strip().lower() or "unknown"
    record = CaseExecutionRecord(
        record_version=EXECUTION_RECORD_VERSION,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        case_id=case_input.case_id,
        dataset_name=dataset_name,
        input_summary={
            "image_path": case_input.image_path,
            "metadata_path": case_input.source_metadata_path or "",
            "clinical_metadata": case_input.clinical_metadata(),
            "image_exists": Path(case_input.image_path).exists(),
            "label_space": label_space,
            "label_space_id": case_input.label_space_id or label_space.get("label_space_id", ""),
        },
        qwen_initial={},
        image_read_audit={},
        skill_retrieval={},
        planner_decision={
            "selected_skills": [],
            "selection_reasons": {},
            "ordering": [],
            "decision_trace": [],
            "planner_type": "direct_baseline",
            "planner_version": "v1",
            "controller_family": "none",
            "controller_training_ready": False,
            "policy_id": "",
        },
        retrieval_bundle={
            "before_skills": {},
            "after_skills": {},
        },
        selected_skills=[],
        skill_outputs={},
        evidence_bundle={
            "serialized_evidence_text": "",
            "risk_flags": [],
            "uncertainty_summary": {},
            "contradiction_summary": {},
            "planner_rationale": {},
        },
        qwen_final=deepcopy(baseline_qwen),
        ground_truth={
            "raw_label": ground_truth_raw,
            "canonical_label": canonicalize_label(
                ground_truth_raw,
                dataset_name=dataset_name,
                label_space_id=case_input.label_space_id,
                metadata=case_input.metadata,
            ),
            "malignant_flag": is_malignant_label(
                ground_truth_raw,
                dataset_name=dataset_name,
                label_space_id=case_input.label_space_id,
                metadata=case_input.metadata,
            ),
            "label_space": label_space,
        },
        evaluation={
            "correct": baseline_eval.get("correct"),
            "topk_hit": baseline_eval.get("topk_hit"),
            "malignant_recall_hit": baseline_eval.get("malignant_recall_hit"),
            "baseline_correct": baseline_eval.get("correct"),
            "baseline_topk_hit": baseline_eval.get("topk_hit"),
            "baseline_malignant_recall_hit": baseline_eval.get("malignant_recall_hit"),
            "agent_vs_baseline_delta": {
                "correct_delta": 0 if baseline_eval.get("correct") is not None else None,
                "topk_hit_delta": 0 if baseline_eval.get("topk_hit") is not None else None,
                "malignant_recall_hit_delta": 0 if baseline_eval.get("malignant_recall_hit") is not None else None,
            },
        },
        reflection_summary={},
        writeback_ops={
            "writeback_enabled": False,
            "write_experience": False,
            "raw_case_memory_written": False,
            "tactical_experience_count": 0,
            "abstract_experience_count": 0,
            "composite_skill_seed_count": 0,
            "planned_raw_case_memory": False,
            "planned_tactical_experience_count": 0,
            "planned_abstract_experience_count": 0,
            "skill_stats_update": [],
            "cognition_update": {},
            "planned_writeback_bundle": {},
            "persisted_writeback_bundle": {},
        },
        cognition_snapshot={
            "before": {},
            "after": {},
        },
        policy_snapshot=deepcopy(policy_snapshot or {}),
        state_versions={
            "run_mode": baseline_run_mode,
            "data_split": baseline_data_split,
            "experience_state": {},
            "cognition_state": {},
            "policy_state": {
                "split_name": str((policy_snapshot or {}).get("state_split", "global")),
                "split_aware_version": str((policy_snapshot or {}).get("state_version", "")),
                "policy_id": str((policy_snapshot or {}).get("policy_id", "")),
            },
        },
        baseline_qwen=deepcopy(baseline_qwen),
    )
    payload = record.to_dict()
    if evaluation_context:
        payload["evaluation_context"] = deepcopy(evaluation_context)
    return payload


def save_case_execution_record(record: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / str(record.get("case_id", "unknown_case"))
    case_dir.mkdir(parents=True, exist_ok=True)

    json_path = case_dir / "case_execution_record.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    jsonl_path = root / DEFAULT_RECORD_JSONL_NAME
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "json_path": str(json_path),
        "jsonl_path": str(jsonl_path),
    }


def _build_writeback_ops(
    *,
    reflection: dict[str, Any],
    planned_writeback_bundle: dict[str, Any],
    persisted_writeback_bundle: dict[str, Any],
    writeback_enabled: bool,
) -> dict[str, Any]:
    tactical_experiences = persisted_writeback_bundle.get("tactical_experiences", [])
    abstract_experiences = persisted_writeback_bundle.get("abstract_experiences", [])
    planned_tactical = planned_writeback_bundle.get("tactical_experiences", [])
    planned_abstract = planned_writeback_bundle.get("abstract_experiences", [])
    return {
        "writeback_enabled": writeback_enabled,
        "write_experience": bool(reflection.get("write_experience")),
        "raw_case_memory_written": bool(persisted_writeback_bundle.get("raw_case_memory", {}).get("case_id")),
        "tactical_experience_count": len(tactical_experiences) if isinstance(tactical_experiences, list) else 0,
        "abstract_experience_count": len(abstract_experiences) if isinstance(abstract_experiences, list) else 0,
        "composite_skill_seed_count": sum(
            1
            for record in abstract_experiences
            if isinstance(record, dict) and record.get("type") == "composite_skill_seed"
        )
        if isinstance(abstract_experiences, list)
        else 0,
        "planned_raw_case_memory": bool(planned_writeback_bundle.get("raw_case_memory", {}).get("case_id")),
        "planned_tactical_experience_count": len(planned_tactical) if isinstance(planned_tactical, list) else 0,
        "planned_abstract_experience_count": len(planned_abstract) if isinstance(planned_abstract, list) else 0,
        "skill_stats_update": deepcopy(persisted_writeback_bundle.get("skill_stats_update", [])),
        "cognition_update": deepcopy(reflection.get("cognition_update", {})),
        "planned_writeback_bundle": deepcopy(planned_writeback_bundle),
        "persisted_writeback_bundle": deepcopy(persisted_writeback_bundle),
    }


def _selected_skills(state: CaseState) -> list[str]:
    planner_output = state.planner_output or {}
    ordering = [str(item).strip() for item in planner_output.get("ordering", []) if str(item).strip()]
    if ordering:
        return ordering
    return [str(item).strip() for item in planner_output.get("selected_skills", []) if str(item).strip()]


def _infer_dataset_name(case_input: CaseInput) -> str:
    if case_input.dataset_name:
        return str(case_input.dataset_name).strip()

    path = Path(case_input.image_path)
    parts = list(path.parts)
    if "data" in parts:
        index = parts.index("data")
        if index + 1 < len(parts):
            return parts[index + 1]
    if case_input.source_metadata_path:
        metadata_parent = Path(case_input.source_metadata_path).parent.name.strip()
        if metadata_parent:
            return metadata_parent
    return path.parent.name.strip() or "unknown_dataset"


def _infer_skill_judgement_version(skill_assessments: Any) -> str:
    if not isinstance(skill_assessments, list):
        return "unknown"
    for item in skill_assessments:
        if not isinstance(item, dict):
            continue
        version = str(item.get("judgement_version", "")).strip()
        if version:
            return version
    return "legacy_or_unspecified"


def _summarize_skill_outcomes(skill_assessments: Any) -> dict[str, int]:
    counts = {
        "helpful": 0,
        "partially_helpful": 0,
        "harmful": 0,
        "unknown": 0,
    }
    if not isinstance(skill_assessments, list):
        return counts
    for item in skill_assessments:
        if not isinstance(item, dict):
            counts["unknown"] += 1
            continue
        impact = str(item.get("impact", "")).strip().lower()
        if impact in counts:
            counts[impact] += 1
        else:
            counts["unknown"] += 1
    return counts


def _collect_skill_names_by_impact(skill_assessments: Any, *, impact: str) -> list[str]:
    target = str(impact).strip().lower()
    if not target or not isinstance(skill_assessments, list):
        return []
    results: list[str] = []
    seen: set[str] = set()
    for item in skill_assessments:
        if not isinstance(item, dict):
            continue
        skill_name = str(item.get("skill_name", "")).strip()
        skill_impact = str(item.get("impact", "")).strip().lower()
        if not skill_name or skill_impact != target or skill_name in seen:
            continue
        seen.add(skill_name)
        results.append(skill_name)
    return results
