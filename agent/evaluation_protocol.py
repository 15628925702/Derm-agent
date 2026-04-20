from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.contamination_guard import build_split_state_version, infer_split_from_path, normalize_split_name
from agent.execution_record import build_baseline_case_execution_record, enrich_execution_record_with_baseline, save_case_execution_record
from agent.experiment_state import (
    build_experiment_state_manifest,
    load_split_payload,
    resolve_case_selection,
    resolve_split_state_paths,
    validate_selected_case_ids,
)
from agent.policy_evaluation import build_policy_summary, compare_policy_summaries
from agent.run_agent import run_agent
from cognition.cognition_state import CognitionState
from dataio.case_loader import discover_case_source, load_case_by_index
from integrations.openai_client import DermOpenAIClient
from memory.experience_schema import DEFAULT_EXPERIENCE_ROOT
from memory.experience_store import ExperienceStore
from memory.experience_bank import ExperienceBank
from agent.workflow_profiles import get_workflow_specialist_skills
from skills.registry import build_default_registry

EVALUATION_PROTOCOL_VERSION = "paper_eval_protocol_v1"
DEFAULT_EVAL_OUTPUT_ROOT = Path("/root/DermAgent/outputs/evaluation_protocol")
DEFAULT_COGNITION_PATH = Path("/root/DermAgent/state/cognition_state.json")
DEFAULT_SKILL_SPEC_ROOT = Path("/root/DermAgent/design/skill_specs")
DEFAULT_POLICY_PATH = Path("/root/DermAgent/state/policy/current_stable_policy.json")

FOUNDATIONAL_SKILLS = {
    "morphology_analysis_skill",
    "color_pattern_analysis_skill",
    "border_surface_analysis_skill",
    "distribution_analysis_skill",
    "lesion_description_structuring_skill",
    "metadata_consistency_skill",
}
# Default specialist skills for PAD-UFES-20.  Use register_specialist_skills()
# to override for other datasets.
SPECIALIST_SKILLS = {"mel_nev_specialist_skill", "ack_scc_specialist_skill", "benign_mimic_specialist_skill"}

_DATASET_SPECIALIST_SKILLS: dict[str, set[str]] = {
    "pad_ufes_20": {"mel_nev_specialist_skill", "ack_scc_specialist_skill"},
    "isic2019": {"mel_nev_specialist_skill", "ack_scc_specialist_skill"},
    "ham10000": {"mel_nev_specialist_skill", "benign_mimic_specialist_skill"},
    "scin": set(),
}


def register_specialist_skills(dataset_name: str, skills: set[str]) -> None:
    """Register dataset-specific specialist skills for ablation experiments."""
    _DATASET_SPECIALIST_SKILLS[str(dataset_name).strip().lower()] = skills


def get_specialist_skills(dataset_name: str | None = None) -> set[str]:
    if dataset_name:
        key = str(dataset_name).strip().lower()
        if key in _DATASET_SPECIALIST_SKILLS:
            return _DATASET_SPECIALIST_SKILLS[key]
    return SPECIALIST_SKILLS


def get_specialist_skills_for_case(*, dataset_name: str | None = None, workflow_context: dict[str, Any] | None = None) -> set[str]:
    workflow_skills = get_workflow_specialist_skills(workflow_context)
    if workflow_skills != SPECIALIST_SKILLS:
        return workflow_skills
    return get_specialist_skills(dataset_name)
UNCERTAINTY_ESCALATION_SKILLS = {
    "uncertainty_assessment_skill",
    "contradiction_check_skill",
    "information_gap_detection_skill",
    "escalation_recommendation_skill",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class EvaluationTargetSpec:
    target_id: str
    label: str
    target_type: str
    mode: str
    description: str
    policy_overrides: dict[str, Any] = field(default_factory=dict)
    execution_overrides: dict[str, Any] = field(default_factory=dict)
    experience_variant: str = "full"
    cognition_variant: str = "frozen"
    ablation_tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_ablation_target_specs(dataset_name: str | None = None) -> list[EvaluationTargetSpec]:
    registry = build_default_registry()
    all_skill_names = registry.list_names()
    non_foundational = [skill for skill in all_skill_names if skill not in FOUNDATIONAL_SKILLS]
    specialist_skills = get_specialist_skills(dataset_name)
    return [
        EvaluationTargetSpec(
            target_id="no_experience_retrieval",
            label="No Experience Retrieval",
            target_type="ablation",
            mode="agent",
            description="Disable raw/tactical/abstract experience retrieval while keeping the rest of DermAgent intact.",
            experience_variant="empty",
            ablation_tags=["experience"],
        ),
        EvaluationTargetSpec(
            target_id="no_skill_retrieval",
            label="No Skill Retrieval",
            target_type="ablation",
            mode="agent",
            description="Planner sees the full skill bank without skill retrieval scoring.",
            execution_overrides={"enable_skill_retrieval": False},
            ablation_tags=["skill_retrieval"],
        ),
        EvaluationTargetSpec(
            target_id="no_specialists",
            label="No Specialists",
            target_type="ablation",
            mode="agent",
            description="Disable specialist confusion-focused skills.",
            policy_overrides={"planner_policy": {"force_disable_skills": sorted(specialist_skills)}},
            ablation_tags=["specialist"],
        ),
        EvaluationTargetSpec(
            target_id="no_abstract_experience",
            label="No Abstract Experience",
            target_type="ablation",
            mode="agent",
            description="Keep raw+tactical experiences only, removing abstract prototypes/rules/confusion memories.",
            experience_variant="raw_tactical",
            ablation_tags=["experience", "abstract"],
        ),
        EvaluationTargetSpec(
            target_id="no_tactical_experience",
            label="No Tactical Experience",
            target_type="ablation",
            mode="agent",
            description="Keep raw+abstract experiences only, removing tactical condition-action traces.",
            experience_variant="raw_abstract",
            ablation_tags=["experience", "tactical"],
        ),
        EvaluationTargetSpec(
            target_id="no_cognition_bias",
            label="No Cognition Bias",
            target_type="ablation",
            mode="agent",
            description="Replace learned cognition with a blank state so planner/retrieval no longer benefit from prior stats.",
            cognition_variant="blank",
            ablation_tags=["cognition"],
        ),
        EvaluationTargetSpec(
            target_id="foundation_only",
            label="Foundation Only",
            target_type="ablation",
            mode="agent",
            description="Only allow the foundational observation skills, removing higher-level compare/risk/specialist layers.",
            policy_overrides={"planner_policy": {"force_disable_skills": sorted(non_foundational)}},
            ablation_tags=["skills", "foundation_only"],
        ),
        EvaluationTargetSpec(
            target_id="no_escalation_uncertainty_layer",
            label="No Escalation/Uncertainty Layer",
            target_type="ablation",
            mode="agent",
            description="Remove information-gap, contradiction, uncertainty, and escalation skills.",
            policy_overrides={"planner_policy": {"force_disable_skills": sorted(UNCERTAINTY_ESCALATION_SKILLS)}},
            ablation_tags=["skills", "uncertainty"],
        ),
    ]


def run_evaluation_suite(
    *,
    output_root: Path,
    data_root: Path,
    client: DermOpenAIClient,
    policy_config: dict[str, Any],
    baseline_client: DermOpenAIClient | None = None,
    target_specs: list[EvaluationTargetSpec],
    limit: int,
    case_offset: int,
    seed: int,
    suite_label: str,
    data_split: str = "test",
    split_json: Path | None = None,
    strict_frozen_eval: bool = True,
) -> dict[str, Any]:
    eval_id = f"{suite_label}_{compact_timestamp()}"
    run_root = output_root / eval_id
    run_root.mkdir(parents=True, exist_ok=True)

    split_payload, resolved_split_json = load_split_payload(split_json=split_json, data_root=data_root)
    case_data_root = resolve_case_data_root(data_root=data_root, split_payload=split_payload)
    case_selection = resolve_case_selection(
        data_split=data_split,
        split_payload=split_payload,
        split_json_path=resolved_split_json,
        limit=limit,
        case_offset=case_offset,
        strict=True,
    )
    policy_source_path = Path(str(policy_config.get("source_path", DEFAULT_POLICY_PATH) or DEFAULT_POLICY_PATH))
    state_paths = resolve_split_state_paths(
        data_split=case_selection.data_split,
        strict_frozen_eval=bool(strict_frozen_eval),
        policy_path=policy_source_path,
    )
    experiment_state_manifest = build_experiment_state_manifest(
        case_selection=case_selection,
        state_paths=state_paths,
        writeback_enabled=False,
        strict_frozen_eval=bool(strict_frozen_eval),
    )
    case_entries, cases, case_source = collect_cases(data_root=case_data_root, case_indices=case_selection.case_indices)
    validate_selected_case_ids(
        actual_case_ids=[case_input.case_id for case_input in cases],
        expected_case_ids=case_selection.case_ids,
        context=f"evaluation suite `{eval_id}`",
    )
    frozen_state = snapshot_frozen_state(
        run_root=run_root,
        client=client,
        policy_config=policy_config,
        state_paths=state_paths,
        strict_frozen_eval=bool(strict_frozen_eval),
    )
    resolved_baseline_client = baseline_client or client
    evaluation_manifest = build_evaluation_manifest(
        eval_id=eval_id,
        suite_label=suite_label,
        data_root=data_root,
        case_data_root=case_data_root,
        case_entries=case_entries,
        case_source=case_source,
        client=client,
        baseline_client=resolved_baseline_client,
        target_specs=target_specs,
        frozen_state=frozen_state,
        limit=limit,
        case_offset=case_offset,
        seed=seed,
        data_split=case_selection.data_split,
        case_selection=case_selection.to_dict(),
        experiment_state_manifest=experiment_state_manifest,
    )
    manifest_path = run_root / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(evaluation_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_spec = next((spec for spec in target_specs if spec.mode == "baseline"), None)
    baseline_records: list[dict[str, Any]] = []
    baseline_outputs: dict[str, dict[str, Any]] = {}
    target_summaries: dict[str, dict[str, Any]] = {}
    target_artifacts: dict[str, Any] = {}

    if baseline_spec:
        baseline_records, baseline_output_dir = run_baseline_target(
            cases=cases,
            client=resolved_baseline_client,
            target_spec=baseline_spec,
            run_root=run_root,
            policy_config=policy_config,
            eval_id=eval_id,
            manifest_path=manifest_path,
            data_split=data_split,
        )
        baseline_outputs = {record["case_id"]: dict(record.get("qwen_final", {})) for record in baseline_records}
        summary = build_policy_summary(baseline_records)
        target_summaries[baseline_spec.target_id] = summary
        target_artifacts[baseline_spec.target_id] = {
            "target_dir": str(baseline_output_dir),
            "records_jsonl_path": str(baseline_output_dir / "records" / "case_execution_records.jsonl"),
            "summary_path": str(baseline_output_dir / "summary.json"),
        }
        (baseline_output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for target_spec in target_specs:
        if target_spec.mode != "agent":
            continue
        records, target_output_dir = run_agent_target(
            cases=cases,
            client=client,
            target_spec=target_spec,
            run_root=run_root,
            policy_config=policy_config,
            eval_id=eval_id,
            manifest_path=manifest_path,
            frozen_state=frozen_state,
            baseline_outputs=baseline_outputs,
            data_split=data_split,
        )
        summary = build_policy_summary(records)
        target_summaries[target_spec.target_id] = summary
        target_artifacts[target_spec.target_id] = {
            "target_dir": str(target_output_dir),
            "records_jsonl_path": str(target_output_dir / "records" / "case_execution_records.jsonl"),
            "summary_path": str(target_output_dir / "summary.json"),
        }
        (target_output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_summary = {}
    if baseline_spec:
        baseline_summary = target_summaries.get(baseline_spec.target_id, {})

    comparisons: dict[str, Any] = {}
    for target_spec in target_specs:
        summary = target_summaries.get(target_spec.target_id, {})
        if target_spec.mode == "baseline":
            comparisons[target_spec.target_id] = {
                "vs_baseline": compare_policy_summaries(summary, summary),
            }
            continue
        comparisons[target_spec.target_id] = {
            "vs_baseline": compare_policy_summaries(baseline_summary, summary) if baseline_summary else {},
        }

    contamination_check = build_contamination_check(
        frozen_state=frozen_state,
        target_specs=target_specs,
        experiment_state_manifest=experiment_state_manifest,
    )

    result_manifest = {
        "eval_id": eval_id,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "created_at": utc_now(),
        "evaluation_manifest_path": str(manifest_path),
        "suite_label": suite_label,
        "frozen_evaluation_mode": True,
        "contamination_check": contamination_check,
        "target_results": [
            {
                "target": spec.to_dict(),
                "summary": target_summaries.get(spec.target_id, {}),
                "artifacts": target_artifacts.get(spec.target_id, {}),
            }
            for spec in target_specs
        ],
        "comparisons": comparisons,
    }
    result_manifest_path = run_root / "result_manifest.json"
    result_manifest_path.write_text(json.dumps(result_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "eval_id": eval_id,
        "run_root": str(run_root),
        "evaluation_manifest_path": str(manifest_path),
        "result_manifest_path": str(result_manifest_path),
        "evaluation_manifest": evaluation_manifest,
        "result_manifest": result_manifest,
    }


def collect_cases(*, data_root: Path, case_indices: list[int]) -> tuple[list[dict[str, Any]], list[Any], Any]:
    case_source = discover_case_source(data_root)
    case_entries: list[dict[str, Any]] = []
    cases: list[Any] = []
    for case_index in case_indices:
        case_input = load_case_by_index(case_index=case_index, data_root=data_root)
        cases.append(case_input)
        case_entries.append(
            {
                "case_index": case_index,
                "case_id": case_input.case_id,
                "dataset_name": case_input.dataset_name,
                "image_path": case_input.image_path,
                "ground_truth": case_input.reference_label or case_input.label,
            }
        )
    return case_entries, cases, case_source


def resolve_case_data_root(*, data_root: Path, split_payload: dict[str, Any]) -> Path:
    metadata_path = Path(str(split_payload.get("metadata_path", "")).strip())
    if metadata_path.exists():
        return metadata_path.parent
    return data_root


def build_evaluation_manifest(
    *,
    eval_id: str,
    suite_label: str,
    data_root: Path,
    case_data_root: Path,
    case_entries: list[dict[str, Any]],
    case_source: Any,
    client: DermOpenAIClient,
    baseline_client: DermOpenAIClient | None = None,
    target_specs: list[EvaluationTargetSpec],
    frozen_state: dict[str, Any],
    limit: int,
    case_offset: int,
    seed: int,
    data_split: str,
    case_selection: dict[str, Any],
    experiment_state_manifest: dict[str, Any],
) -> dict[str, Any]:
    dataset_name = case_entries[0].get("dataset_name") if case_entries else ""
    resolved_baseline_client = baseline_client or client
    return {
        "eval_id": eval_id,
        "created_at": utc_now(),
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "mode": suite_label,
        "fairness_constraints": {
            "same_model": client.runtime_manifest() == resolved_baseline_client.runtime_manifest(),
            "same_case_list": True,
            "frozen_evaluation_mode": True,
            "writeback_disabled": True,
            "online_contamination_guard": True,
        },
        "model_context": {
            "agent_client": client.runtime_manifest(),
            "baseline_client": resolved_baseline_client.runtime_manifest(),
        },
        "dataset": {
            "data_root": str(data_root),
            "case_data_root": str(case_data_root),
            "dataset_name": dataset_name,
            "metadata_path": str(case_source.metadata_path),
            "image_root": str(case_source.image_root),
            "case_selection": case_selection.get("selection_mode", "split_offset"),
            "case_indices": [entry["case_index"] for entry in case_entries],
            "case_ids": [entry["case_id"] for entry in case_entries],
            "case_offset": case_offset,
            "limit": limit,
            "seed": seed,
            "data_split": normalize_split_name(data_split, default="test"),
            "split_id": case_selection.get("split_id", ""),
            "split_json_path": case_selection.get("split_json_path", ""),
            "offset_within_split": case_selection.get("offset_within_split", 0),
        },
        "experiment_state": experiment_state_manifest,
        "run_targets": [spec.to_dict() for spec in target_specs],
        "frozen_state": frozen_state,
        "execution_config": {
            "enable_writeback": False,
            "persist_artifacts": True,
            "save_case_execution_records": True,
            "frozen_output_root": frozen_state.get("frozen_root", ""),
        },
        "ablation_plan": [spec.to_dict() for spec in target_specs if spec.target_type == "ablation"],
        "result_paths": {
            "manifests": [],
            "targets_root": str(Path(frozen_state["run_root"]) / "targets"),
        },
        "notes": [
            "Evaluation runs against frozen state snapshots, not live mutable state.",
            "Qwen remains the only final diagnosis stage in all non-baseline targets.",
            "Baseline targets may use a different model client than the agent path when explicitly configured.",
        ],
    }


def snapshot_frozen_state(
    *,
    run_root: Path,
    client: DermOpenAIClient,
    policy_config: dict[str, Any],
    state_paths: Any,
    strict_frozen_eval: bool,
) -> dict[str, Any]:
    frozen_root = run_root / "frozen_state"
    frozen_root.mkdir(parents=True, exist_ok=True)

    prompt_manifest = client.prompt_manifest()
    prompt_manifest_path = frozen_root / "prompt_manifest.json"
    prompt_manifest_path.write_text(json.dumps(prompt_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    experience_source_root = Path(str(state_paths.experience_root))
    cognition_source_path = Path(str(state_paths.cognition_path))
    policy_source_path = Path(str(state_paths.policy_path))

    experience_snapshot = snapshot_experience_bank(
        source_root=experience_source_root,
        target_root=frozen_root / "experience",
    )
    cognition_snapshot = snapshot_single_file(
        source_path=cognition_source_path,
        target_path=frozen_root / "cognition_state.json",
        component_id="cognition_state",
    )
    policy_path = frozen_root / "policy.json"
    policy_path.write_text(json.dumps(policy_config, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_hash = file_sha256(policy_path)

    skill_bank_snapshot = snapshot_skill_bank(frozen_root / "skill_bank")
    frozen_state_manifest = {
        "frozen_root": str(frozen_root),
        "run_root": str(run_root),
        "prompt_version": prompt_manifest["prompt_stack_version"],
        "prompt_manifest_path": str(prompt_manifest_path),
        "skill_bank_version": skill_bank_snapshot["skill_bank_version"],
        "skill_bank_manifest_path": skill_bank_snapshot["manifest_path"],
        "experience_bank_version": experience_snapshot["experience_bank_version"],
        "experience_state_split": experience_snapshot["split_name"],
        "experience_split_aware_version": experience_snapshot["split_aware_version"],
        "experience_manifest_path": experience_snapshot["manifest_path"],
        "cognition_version": cognition_snapshot["version_id"],
        "cognition_state_split": cognition_snapshot["split_name"],
        "cognition_split_aware_version": cognition_snapshot["split_aware_version"],
        "cognition_snapshot_path": cognition_snapshot["snapshot_path"],
        "policy_version": str(policy_config.get("state_version", "")).strip() or f"{policy_config.get('policy_id', 'policy')}:{policy_hash[:12]}",
        "policy_state_split": normalize_split_name(str(policy_config.get("state_split", "global")), default="global"),
        "policy_snapshot_path": str(policy_path),
        "policy_source_path": str(policy_source_path),
        "registry_version": skill_bank_snapshot["registry_version"],
        "strict_frozen_eval": bool(strict_frozen_eval),
        "requested_state_paths": {
            "experience_root": str(experience_source_root),
            "cognition_path": str(cognition_source_path),
            "policy_path": str(policy_source_path),
        },
        "state_snapshot_paths": {
            "experience_root": experience_snapshot["snapshot_root"],
            "cognition_path": cognition_snapshot["snapshot_path"],
            "policy_path": str(policy_path),
            "skill_bank_root": skill_bank_snapshot["snapshot_root"],
        },
        "source_hashes": {
            "experience_root_hash": experience_snapshot["combined_hash"],
            "cognition_hash": cognition_snapshot["sha256"],
            "policy_hash": policy_hash,
            "skill_bank_hash": skill_bank_snapshot["combined_hash"],
        },
    }
    manifest_path = frozen_root / "state_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(frozen_state_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen_state_manifest["state_snapshot_manifest_path"] = str(manifest_path)
    return frozen_state_manifest


def snapshot_experience_bank(*, source_root: Path, target_root: Path) -> dict[str, Any]:
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)
    manifest_path = target_root / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    hashes = gather_file_hashes(target_root)
    combined_hash = combined_hash_from_pairs(hashes)
    return {
        "snapshot_root": str(target_root),
        "source_root": str(source_root),
        "manifest_path": str(manifest_path),
        "experience_bank_version": f"experience_bank_{combined_hash[:12]}",
        "split_name": normalize_split_name(
            str(manifest_payload.get("state_partition", {}).get("split_name") or manifest_payload.get("state_split", "global")),
            default="global",
        ),
        "split_aware_version": str(
            manifest_payload.get("state_partition", {}).get("split_aware_version")
            or manifest_payload.get("split_aware_version", "")
        ).strip()
        or f"experience_bank:global:{combined_hash[:12]}",
        "combined_hash": combined_hash,
        "files": hashes,
    }


def snapshot_skill_bank(target_root: Path) -> dict[str, Any]:
    source_files = discover_skill_bank_files()
    copied_files: list[dict[str, Any]] = []
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in source_files:
        relative_path = source_path.relative_to(Path("/root/DermAgent"))
        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_files.append(
            {
                "source_path": str(source_path),
                "snapshot_path": str(target_path),
                "sha256": file_sha256(source_path),
            }
        )
    registry = build_default_registry()
    registry_payload = [
        {
            "skill_name": skill.name,
            "skill_id": skill.skill_id,
            "skill_type": skill.skill_type,
        }
        for skill in registry.list_skill_objects_structured()
    ]
    registry_version = hashlib.sha256(
        json.dumps(registry_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    manifest = {
        "files": copied_files,
        "registry_payload": registry_payload,
        "registry_version": f"registry_{registry_version}",
    }
    manifest_path = target_root / "skill_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    combined_hash = hashlib.sha256(
        json.dumps(copied_files + registry_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "snapshot_root": str(target_root),
        "manifest_path": str(manifest_path),
        "skill_bank_version": f"skill_bank_{combined_hash[:12]}",
        "registry_version": f"registry_{registry_version}",
        "combined_hash": combined_hash,
    }


def discover_skill_bank_files() -> list[Path]:
    paths = sorted(
        path
        for path in Path("/root/DermAgent/skills").glob("*.py")
        if path.is_file()
    )
    paths.extend(sorted(path for path in DEFAULT_SKILL_SPEC_ROOT.rglob("*.md") if path.is_file()))
    return paths


def snapshot_single_file(*, source_path: Path, target_path: Path, component_id: str = "snapshot_file") -> dict[str, Any]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    sha256 = file_sha256(target_path)
    split_name = "global"
    split_aware_version = f"{target_path.stem}_{sha256[:12]}"
    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        split_name = normalize_split_name(
            str(payload.get("state_split", "")).strip() or infer_split_from_path(target_path, default="global"),
            default="global",
        )
        split_aware_version = str(payload.get("state_version", "")).strip() or split_aware_version
    return {
        "source_path": str(source_path),
        "snapshot_path": str(target_path),
        "sha256": sha256,
        "version_id": f"{target_path.stem}_{sha256[:12]}",
        "split_name": split_name,
        "split_aware_version": split_aware_version if ":" in split_aware_version else f"{component_id}:{split_name}:{sha256[:12]}",
    }


def run_baseline_target(
    *,
    cases: list[Any],
    client: DermOpenAIClient,
    target_spec: EvaluationTargetSpec,
    run_root: Path,
    policy_config: dict[str, Any],
    eval_id: str,
    manifest_path: Path,
    data_split: str,
) -> tuple[list[dict[str, Any]], Path]:
    target_output_dir = run_root / "targets" / target_spec.target_id
    records_dir = target_output_dir / "records"
    records: list[dict[str, Any]] = []
    normalized_split = normalize_split_name(data_split, default="test")
    baseline_policy_snapshot = deepcopy(policy_config)
    baseline_policy_snapshot["state_split"] = normalized_split
    baseline_policy_snapshot["state_version"] = str(baseline_policy_snapshot.get("state_version", "")).strip() or build_split_state_version(
        component_id="policy_config",
        split_name=normalized_split,
        payload={
            "policy_id": baseline_policy_snapshot.get("policy_id"),
            "version": baseline_policy_snapshot.get("version"),
            "planner_policy": baseline_policy_snapshot.get("planner_policy", {}),
            "retrieval_policy": baseline_policy_snapshot.get("retrieval_policy", {}),
            "evidence_policy": baseline_policy_snapshot.get("evidence_policy", {}),
        },
    )
    total_cases = len(cases)
    for index, case_input in enumerate(cases, start=1):
        percent = (index / total_cases * 100.0) if total_cases else 100.0
        print(
            f"[progress baseline {index}/{total_cases} ({percent:.1f}%)] case_id={case_input.case_id}",
            flush=True,
        )
        baseline_qwen = client.baseline_diagnosis(case_input)
        record = build_baseline_case_execution_record(
            case_input=case_input,
            baseline_qwen=baseline_qwen,
            policy_snapshot=baseline_policy_snapshot,
            evaluation_context={
                "eval_id": eval_id,
                "target_id": target_spec.target_id,
                "target_label": target_spec.label,
                "target_type": target_spec.target_type,
                "mode": target_spec.mode,
                "evaluation_manifest_path": str(manifest_path),
                "frozen_evaluation_mode": True,
                "writeback_disabled": True,
                "run_mode": f"{normalized_split}_frozen_inference",
                "data_split": normalized_split,
            },
        )
        records.append(record)
        save_case_execution_record(record, records_dir)
    return records, target_output_dir


def run_agent_target(
    *,
    cases: list[Any],
    client: DermOpenAIClient,
    target_spec: EvaluationTargetSpec,
    run_root: Path,
    policy_config: dict[str, Any],
    eval_id: str,
    manifest_path: Path,
    frozen_state: dict[str, Any],
    baseline_outputs: dict[str, dict[str, Any]],
    data_split: str,
) -> tuple[list[dict[str, Any]], Path]:
    target_output_dir = run_root / "targets" / target_spec.target_id
    records_dir = target_output_dir / "records"
    frozen_experience_root = Path(frozen_state["state_snapshot_paths"]["experience_root"])
    frozen_cognition_path = Path(frozen_state["state_snapshot_paths"]["cognition_path"])
    experience_root = build_experience_variant_snapshot(
        base_root=frozen_experience_root,
        variant_name=target_spec.experience_variant,
        target_root=run_root / "experience_variants" / target_spec.target_id,
    )
    cognition_path = build_cognition_variant_snapshot(
        base_path=frozen_cognition_path,
        variant_name=target_spec.cognition_variant,
        target_path=run_root / "cognition_variants" / f"{target_spec.target_id}.json",
    )
    target_policy = build_target_policy(policy_config=policy_config, target_spec=target_spec)
    normalized_split = normalize_split_name(data_split, default="test")
    target_policy["state_split"] = normalized_split
    target_policy["state_version"] = str(target_policy.get("state_version", "")).strip() or build_split_state_version(
        component_id="policy_config",
        split_name=normalized_split,
        payload={
            "policy_id": target_policy.get("policy_id"),
            "version": target_policy.get("version"),
            "planner_policy": target_policy.get("planner_policy", {}),
            "retrieval_policy": target_policy.get("retrieval_policy", {}),
            "evidence_policy": target_policy.get("evidence_policy", {}),
        },
    )

    records: list[dict[str, Any]] = []
    total_cases = len(cases)
    for index, case_input in enumerate(cases, start=1):
        percent = (index / total_cases * 100.0) if total_cases else 100.0
        print(
            f"[progress agent {index}/{total_cases} ({percent:.1f}%)] target={target_spec.target_id} case_id={case_input.case_id}",
            flush=True,
        )
        baseline_qwen = dict(baseline_outputs.get(case_input.case_id, {}))
        cognition_state = CognitionState.load(cognition_path)
        cognition_state.state_split = normalized_split
        agent_state, _ = run_agent(
            case_input=case_input,
            client=client,
            experience_bank=ExperienceBank(root=experience_root),
            cognition=cognition_state,
            policy_config=target_policy,
            output_dir=target_output_dir / "artifacts",
            enable_writeback=False,
            run_mode=f"{normalized_split}_frozen_inference",
            data_split=normalized_split,
            execution_overrides=target_spec.execution_overrides,
        )
        enriched_record = enrich_execution_record_with_baseline(
            agent_state.execution_record,
            baseline_qwen=baseline_qwen,
        )
        enriched_record["evaluation_context"] = {
            "eval_id": eval_id,
            "target_id": target_spec.target_id,
            "target_label": target_spec.label,
            "target_type": target_spec.target_type,
            "mode": target_spec.mode,
            "evaluation_manifest_path": str(manifest_path),
            "frozen_evaluation_mode": True,
            "writeback_disabled": True,
            "run_mode": f"{normalized_split}_frozen_inference",
            "data_split": normalized_split,
            "experience_variant": target_spec.experience_variant,
            "cognition_variant": target_spec.cognition_variant,
            "execution_overrides": dict(target_spec.execution_overrides),
            "policy_overrides": deepcopy(target_spec.policy_overrides),
        }
        save_case_execution_record(enriched_record, records_dir)
        records.append(enriched_record)
    return records, target_output_dir


def build_target_policy(*, policy_config: dict[str, Any], target_spec: EvaluationTargetSpec) -> dict[str, Any]:
    policy = deepcopy(policy_config)
    base_planner_policy = dict(policy.get("planner_policy", {}) or {})
    override_planner_policy = dict(target_spec.policy_overrides.get("planner_policy", {}) or {})
    merged_force_disable = list(
        dict.fromkeys(
            list(base_planner_policy.get("force_disable_skills", []) or [])
            + list(override_planner_policy.get("force_disable_skills", []) or [])
        )
    )
    merged_force_select = list(
        dict.fromkeys(
            list(base_planner_policy.get("force_select_skills", []) or [])
            + list(override_planner_policy.get("force_select_skills", []) or [])
        )
    )
    policy["planner_policy"] = {
        **base_planner_policy,
        **override_planner_policy,
        "force_disable_skills": merged_force_disable,
        "force_select_skills": merged_force_select,
    }
    policy["retrieval_policy"] = {
        **dict(policy.get("retrieval_policy", {}) or {}),
        **dict(target_spec.policy_overrides.get("retrieval_policy", {}) or {}),
    }
    if target_spec.policy_overrides.get("policy_id"):
        policy["policy_id"] = target_spec.policy_overrides["policy_id"]
    else:
        policy["policy_id"] = f"{policy.get('policy_id', 'policy')}__{target_spec.target_id}"
    return policy


def build_experience_variant_snapshot(*, base_root: Path, variant_name: str, target_root: Path) -> Path:
    normalized = (variant_name or "full").strip().lower()
    if normalized == "full":
        return base_root
    store = ExperienceStore(target_root)
    selected_layers = experience_layers_for_variant(normalized)
    raw_records = ExperienceStore(base_root).load_raw_case_memories() if "raw" in selected_layers else []
    tactical_records = ExperienceStore(base_root).load_tactical_experiences() if "tactical" in selected_layers else []
    abstract_records = ExperienceStore(base_root).load_abstract_experiences() if "abstract" in selected_layers else []
    store._write_jsonl(store.raw_case_path, raw_records)
    store._write_jsonl(store.tactical_path, tactical_records)
    store._write_jsonl(store.abstract_path, abstract_records)
    store.refresh_metadata()
    return target_root


def experience_layers_for_variant(variant_name: str) -> set[str]:
    mapping = {
        "empty": set(),
        "raw_only": {"raw"},
        "tactical_only": {"tactical"},
        "abstract_only": {"abstract"},
        "raw_tactical": {"raw", "tactical"},
        "raw_abstract": {"raw", "abstract"},
        "tactical_abstract": {"tactical", "abstract"},
    }
    return mapping.get(variant_name, {"raw", "tactical", "abstract"})


def build_cognition_variant_snapshot(*, base_path: Path, variant_name: str, target_path: Path) -> Path:
    normalized = (variant_name or "frozen").strip().lower()
    if normalized == "frozen":
        return base_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    blank_state = CognitionState(
        self_capability_summary=CognitionState.load(base_path).self_capability_summary,
        known_confusion_patterns={},
        preferred_skills=[],
        retrieval_preferences={"top_k": 3, "prioritize_confusion_cases": True},
        failure_statistics={"total_cases": 0, "failed_cases": 0, "confusion_cases": 0, "hard_cases": 0},
        skill_statistics={},
    )
    blank_state.save(target_path)
    return target_path


def build_contamination_check(
    *,
    frozen_state: dict[str, Any],
    target_specs: list[EvaluationTargetSpec],
    experiment_state_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "used_snapshot_isolation": True,
        "writeback_disabled": True,
        "state_reads_from_snapshot": True,
        "live_state_mutation_allowed_to_affect_eval": False,
        "strict_frozen_eval": bool(experiment_state_manifest.get("strict_frozen_eval", False)),
        "expected_data_split": experiment_state_manifest.get("data_split", ""),
        "case_selection": dict(experiment_state_manifest.get("case_selection", {})),
        "resolved_state_paths": dict(experiment_state_manifest.get("state_paths", {})),
        "split_aware_versions_present": bool(
            frozen_state.get("experience_split_aware_version")
            and frozen_state.get("cognition_split_aware_version")
            and frozen_state.get("policy_version")
        ),
        "frozen_state_hashes": dict(frozen_state.get("source_hashes", {})),
        "target_execution_modes": {
            spec.target_id: {
                "mode": spec.mode,
                "experience_variant": spec.experience_variant,
                "cognition_variant": spec.cognition_variant,
                "execution_overrides": dict(spec.execution_overrides),
            }
            for spec in target_specs
        },
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gather_file_hashes(root: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        pairs.append(
            {
                "relative_path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
            }
        )
    return pairs


def combined_hash_from_pairs(pairs: list[dict[str, str]]) -> str:
    payload = json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
