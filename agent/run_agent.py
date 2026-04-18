from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from agent.contamination_guard import (
    build_split_state_version,
    enforce_writeback_policy,
    infer_split_from_path,
    normalize_split_name,
)
from agent.experiment_state import resolve_split_state_root
from agent.image_read_audit import build_image_read_audit
from agent.execution_record import build_case_execution_record, save_case_execution_record
from agent.evidence_package import EvidencePackage
from agent.planner import PlannerInput, build_default_planner
from agent.policy_config import load_stable_policy, snapshot_policy
from agent.reflection import apply_cognition_update, build_reflection
from agent.skill_retriever import SkillRetrievalQuery, build_default_skill_retriever
from agent.state import CaseInput, CaseState
from cognition.cognition_state import CognitionState
from integrations.openai_client import DermOpenAIClient
from memory.experience_bank import ExperienceBank
from skills.registry import build_default_registry

try:
    from agent.retrieval_scorer import LearnedRetrievalScorer
except Exception:  # pragma: no cover - optional dependency fallback
    LearnedRetrievalScorer = None  # type: ignore[assignment]


def run_agent(
    case_input: CaseInput,
    client: DermOpenAIClient | None = None,
    experience_bank: ExperienceBank | None = None,
    cognition: CognitionState | None = None,
    policy_config: dict | None = None,
    output_dir: str | Path = "/root/DermAgent/outputs",
    enable_writeback: bool = True,
    run_mode: str = "training",
    data_split: str = "train",
    strict_frozen_writeback_guard: bool = True,
    execution_overrides: dict | None = None,
) -> tuple[CaseState, EvidencePackage]:
    qwen_client = client or DermOpenAIClient()
    normalized_split = normalize_split_name(data_split, default="train")
    split_state_root = resolve_split_state_root() / normalized_split
    cognition_path = split_state_root / "cognition_state.json"
    bank = experience_bank or ExperienceBank(root=split_state_root / "experience")
    cognition_state = cognition or CognitionState.load(cognition_path)
    cognition_state.state_split = normalized_split
    cognition_before = cognition_state.to_dict()
    policy_snapshot = snapshot_policy(policy_config or load_stable_policy())
    policy_snapshot["state_split"] = normalize_split_name(
        str(policy_snapshot.get("state_split", "")).strip() or normalized_split,
        default=normalized_split,
    )
    policy_snapshot["state_version"] = str(policy_snapshot.get("state_version", "")).strip() or build_split_state_version(
        component_id="policy_config",
        split_name=str(policy_snapshot.get("state_split", normalized_split)),
        payload={
            "policy_id": policy_snapshot.get("policy_id"),
            "version": policy_snapshot.get("version"),
            "planner_policy": policy_snapshot.get("planner_policy", {}),
            "retrieval_policy": policy_snapshot.get("retrieval_policy", {}),
            "evidence_policy": policy_snapshot.get("evidence_policy", {}),
        },
    )
    execution_config = _normalize_execution_overrides(execution_overrides)
    retrieval_policy = dict(policy_snapshot.get("retrieval_policy", {}) or {})
    planner_policy = {
        **dict(policy_snapshot.get("planner_policy", {}) or {}),
        "_policy_id": str(policy_snapshot.get("policy_id", "")).strip(),
    }
    retrieval_reranker = _load_retrieval_reranker(retrieval_policy)
    effective_writeback = enforce_writeback_policy(
        run_mode=run_mode,
        enable_writeback=enable_writeback,
        strict=bool(strict_frozen_writeback_guard),
    )

    state = CaseState(case_input=case_input)
    state.policy_snapshot = policy_snapshot
    state.perception = qwen_client.initial_perception(case_input)
    if not state.uncertainty:
        perception_uncertainty = state.perception.get("uncertainty", {})
        state.uncertainty = {
            "uncertainty_level": str(perception_uncertainty.get("level", "unknown")).lower(),
            "reasons": list(perception_uncertainty.get("reasons", [])),
            "missing_information": [],
        }

    if execution_config["enable_experience_retrieval"]:
        state.retrieval_bundle = bank.retrieve_bundle(
            case_state=state,
            top_k_merged=int(retrieval_policy.get("top_k_experience", cognition_state.retrieval_preferences.get("top_k", 3))),
            top_k_tactical=max(3, int(retrieval_policy.get("top_k_tactical", cognition_state.retrieval_preferences.get("top_k", 3)))),
            top_k_abstract=max(3, int(retrieval_policy.get("top_k_abstract", cognition_state.retrieval_preferences.get("top_k", 3)))),
            dataset_name=case_input.dataset_name,
        )
        state.retrieval_bundle = _rerank_experience_bundle_if_enabled(
            state=state,
            bundle=state.retrieval_bundle,
            reranker=retrieval_reranker,
            fail_open=bool(retrieval_policy.get("retrieval_reranker_fail_open", True)),
        )
    else:
        state.retrieval_bundle = _build_empty_retrieval_bundle(bank=bank, state=state)
    planner_retrieval_bundle = dict(state.retrieval_bundle)
    state.retrieved_experience = list(state.retrieval_bundle.get("aggregator_summary", []))

    registry = build_default_registry()
    all_skill_objects = registry.list_skill_objects_structured()
    if execution_config["enable_skill_retrieval"]:
        skill_retriever = build_default_skill_retriever()
        skill_retrieval_bundle = skill_retriever.retrieve(
            SkillRetrievalQuery(
                perception=state.perception,
                metadata=state.clinical_metadata,
                cognition=cognition_state,
            ),
            all_skill_objects,
        )
        skill_retrieval_bundle_dict = skill_retrieval_bundle.to_dict()
        skill_retrieval_bundle_dict = _rerank_skill_bundle_if_enabled(
            state=state,
            cognition=cognition_state,
            bundle=skill_retrieval_bundle_dict,
            reranker=retrieval_reranker,
            fail_open=bool(retrieval_policy.get("retrieval_reranker_fail_open", True)),
        )
        state.skill_retrieval_bundle = _limit_skill_candidates(
            skill_retrieval_bundle_dict,
            top_k=int(retrieval_policy.get("top_k_skill_candidates", len(all_skill_objects) or 14)),
        )
        retrieved_skill_names = list(state.skill_retrieval_bundle.get("candidate_skill_names", []))
        available_skills = [skill for skill in all_skill_objects if skill.name in set(retrieved_skill_names)] or all_skill_objects
    else:
        state.skill_retrieval_bundle = _build_disabled_skill_retrieval_bundle(all_skill_objects)
        available_skills = all_skill_objects
    planner = build_default_planner(policy_config=planner_policy)
    planner_output = planner.plan(
        PlannerInput(
            perception=state.perception,
            metadata=state.clinical_metadata,
            cognition=cognition_state,
            retrieved_experience_summary=state.retrieved_experience,
            retrieved_experience_bundle=state.retrieval_bundle,
            skill_retrieval_bundle=state.skill_retrieval_bundle,
            policy_config=planner_policy,
            available_skills=available_skills,
            workflow_context=case_input.workflow_context,
            dataset_name=case_input.dataset_name,
        )
    )
    state.planner_output = planner_output.to_dict()
    selected_skills = list(planner_output.ordering or planner_output.selected_skills)
    registry.run_many(selected_skills, state, qwen_client)
    if execution_config["enable_experience_retrieval"]:
        state.retrieval_bundle = bank.retrieve_bundle(
            case_state=state,
            workflow_context=case_input.workflow_context,
            top_k_merged=int(retrieval_policy.get("top_k_experience", cognition_state.retrieval_preferences.get("top_k", 3))),
            top_k_tactical=max(3, int(retrieval_policy.get("top_k_tactical", cognition_state.retrieval_preferences.get("top_k", 3)))),
            top_k_abstract=max(3, int(retrieval_policy.get("top_k_abstract", cognition_state.retrieval_preferences.get("top_k", 3)))),
            dataset_name=case_input.dataset_name,
        )
        state.retrieval_bundle = _rerank_experience_bundle_if_enabled(
            state=state,
            bundle=state.retrieval_bundle,
            reranker=retrieval_reranker,
            fail_open=bool(retrieval_policy.get("retrieval_reranker_fail_open", True)),
        )
    else:
        state.retrieval_bundle = _build_empty_retrieval_bundle(bank=bank, state=state)
    state.retrieved_experience = list(state.retrieval_bundle.get("aggregator_summary", []))
    state.notes = _build_notes(state)

    evidence_package = EvidencePackage.from_state(state)
    state.final_diagnosis = qwen_client.final_diagnosis(case_input, evidence_package)
    if execution_config["enable_image_read_audit"]:
        counterfactual_state = _run_text_only_counterfactual(
            case_input=case_input,
            client=qwen_client,
            experience_bank=bank,
            cognition_state=cognition_state,
            policy_snapshot=policy_snapshot,
            output_dir=output_dir,
            run_mode=run_mode,
            data_split=normalized_split,
            strict_frozen_writeback_guard=bool(strict_frozen_writeback_guard),
            execution_overrides=execution_overrides,
        )
        state.image_read_audit = build_image_read_audit(
            case_input=case_input,
            primary_state=state,
            counterfactual_state=counterfactual_state,
        )
    state.reflection = build_reflection(state, cognition_state)

    if effective_writeback:
        if state.reflection.get("write_experience"):
            written_bundle = bank.writeback(state.reflection.get("writeback_bundle", {}))
            state.reflection["writeback_bundle"] = written_bundle
            if isinstance(written_bundle.get("reflection_extract"), dict):
                state.reflection["reflection_extract"] = written_bundle["reflection_extract"]
        apply_cognition_update(cognition_state, state.reflection, state)
        cognition_state.save(cognition_path)
    cognition_after = cognition_state.to_dict()
    state_versions = _build_state_versions(
        bank=bank,
        cognition_state=cognition_state,
        policy_snapshot=policy_snapshot,
        run_mode=run_mode,
        data_split=normalized_split,
    )

    state.execution_record = build_case_execution_record(
        case_input=case_input,
        state=state,
        evidence_bundle=evidence_package.to_dict(),
        planner_retrieval_bundle=planner_retrieval_bundle,
        cognition_before=cognition_before,
        cognition_after=cognition_after,
        writeback_enabled=effective_writeback,
        state_versions=state_versions,
    )

    _write_debug_artifacts(state, evidence_package, output_dir)
    return state, evidence_package


def _build_notes(state: CaseState) -> list[str]:
    notes = [
        "Agent generated structured evidence only.",
        "Qwen remains the sole final diagnostic decision maker.",
    ]
    planner_selected = state.planner_output.get("selected_skills", [])
    if planner_selected:
        notes.append(f"Planner selected skills: {', '.join(planner_selected)}.")
    policy_id = str(state.policy_snapshot.get("policy_id", "")).strip()
    if policy_id:
        notes.append(f"Policy in use: {policy_id}.")
    skill_candidates = state.skill_retrieval_bundle.get("candidate_skill_names", [])
    if skill_candidates:
        notes.append(f"Skill retrieval candidates: {', '.join(skill_candidates)}.")
    reranker = state.skill_retrieval_bundle.get("reranker") or state.retrieval_bundle.get("reranker")
    if isinstance(reranker, dict) and reranker.get("applied"):
        notes.append("Learned retrieval reranker applied (fallback-safe).")
    if state.perception.get("notes"):
        notes.extend(str(item) for item in state.perception.get("notes", []))
    if state.retrieved_experience:
        notes.append("Retrieved prior reasoning experience for context.")
    if state.uncertainty.get("uncertainty_level") == "high" or state.perception.get("uncertainty", {}).get("level") == "high":
        notes.append("High uncertainty detected in MVP skeleton.")
    return notes


def _build_state_versions(
    *,
    bank: ExperienceBank,
    cognition_state: CognitionState,
    policy_snapshot: dict[str, object],
    run_mode: str,
    data_split: str,
) -> dict[str, object]:
    experience_manifest = bank.store.load_manifest()
    experience_partition = dict(experience_manifest.get("state_partition", {}))
    experience_split = normalize_split_name(
        str(
            experience_partition.get("split_name")
            or experience_manifest.get("state_split")
            or infer_split_from_path(bank.store.root, default="global")
        ),
        default="global",
    )
    experience_version = str(
        experience_partition.get("split_aware_version")
        or experience_manifest.get("split_aware_version")
        or build_split_state_version(
            component_id="experience_bank",
            split_name=experience_split,
            payload={
                "raw_case_count": int(experience_manifest.get("raw_case_count", 0)),
                "tactical_count": int(experience_manifest.get("tactical_count", 0)),
                "abstract_count": int(experience_manifest.get("abstract_count", 0)),
            },
        )
    )
    cognition_split = normalize_split_name(
        str(getattr(cognition_state, "state_split", "")).strip() or infer_split_from_path("/root/DermAgent/state/cognition_state.json"),
        default="global",
    )
    cognition_version = str(getattr(cognition_state, "state_version", "")).strip() or build_split_state_version(
        component_id="cognition_state",
        split_name=cognition_split,
        payload={
            "failure_statistics": dict(cognition_state.failure_statistics or {}),
            "known_confusion_patterns": dict(cognition_state.known_confusion_patterns or {}),
            "preferred_skills": list(cognition_state.preferred_skills or []),
        },
    )
    policy_split = normalize_split_name(str(policy_snapshot.get("state_split", "global")), default="global")
    policy_version = str(policy_snapshot.get("state_version", "")).strip() or build_split_state_version(
        component_id="policy_config",
        split_name=policy_split,
        payload={
            "policy_id": policy_snapshot.get("policy_id"),
            "version": policy_snapshot.get("version"),
            "planner_policy": policy_snapshot.get("planner_policy", {}),
            "retrieval_policy": policy_snapshot.get("retrieval_policy", {}),
            "evidence_policy": policy_snapshot.get("evidence_policy", {}),
        },
    )
    return {
        "run_mode": str(run_mode).strip().lower(),
        "data_split": normalize_split_name(data_split, default="train"),
        "experience_state": {
            "split_name": experience_split,
            "split_aware_version": experience_version,
            "root": str(bank.store.root),
        },
        "cognition_state": {
            "split_name": cognition_split,
            "split_aware_version": cognition_version,
        },
        "policy_state": {
            "split_name": policy_split,
            "split_aware_version": policy_version,
            "policy_id": str(policy_snapshot.get("policy_id", "")),
        },
    }


def _write_debug_artifacts(state: CaseState, evidence_package: EvidencePackage, output_dir: str | Path) -> None:
    import json

    target_dir = Path(output_dir) / state.case_input.case_id
    target_dir.mkdir(parents=True, exist_ok=True)

    with (target_dir / "state.json").open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2)

    with (target_dir / "evidence_package.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence_package.to_dict(), handle, ensure_ascii=False, indent=2)

    with (target_dir / "reflection.json").open("w", encoding="utf-8") as handle:
        json.dump(state.reflection, handle, ensure_ascii=False, indent=2)

    if state.execution_record:
        save_case_execution_record(state.execution_record, output_dir)


def _normalize_execution_overrides(overrides: dict | None) -> dict[str, bool]:
    source = dict(overrides or {})
    return {
        "enable_experience_retrieval": bool(source.get("enable_experience_retrieval", True)),
        "enable_skill_retrieval": bool(source.get("enable_skill_retrieval", True)),
        "enable_image_read_audit": bool(source.get("enable_image_read_audit", False)),
    }


def _run_text_only_counterfactual(
    *,
    case_input: CaseInput,
    client: DermOpenAIClient,
    experience_bank: ExperienceBank,
    cognition_state: CognitionState,
    policy_snapshot: dict[str, object],
    output_dir: str | Path,
    run_mode: str,
    data_split: str,
    strict_frozen_writeback_guard: bool,
    execution_overrides: dict | None,
) -> CaseState:
    from agent.image_read_audit import build_text_only_case_input

    overrides = dict(execution_overrides or {})
    overrides["enable_image_read_audit"] = False
    counterfactual_state, _ = run_agent(
        case_input=build_text_only_case_input(case_input),
        client=client,
        experience_bank=ExperienceBank(root=experience_bank.store.root),
        cognition=deepcopy(cognition_state),
        policy_config=deepcopy(policy_snapshot),
        output_dir=Path(output_dir) / "_image_read_audit_text_only",
        enable_writeback=False,
        run_mode=f"{str(run_mode).strip()}_image_audit_text_only",
        data_split=str(data_split),
        strict_frozen_writeback_guard=strict_frozen_writeback_guard,
        execution_overrides=overrides,
    )
    return counterfactual_state


def _build_empty_retrieval_bundle(*, bank: ExperienceBank, state: CaseState) -> dict[str, object]:
    query = bank.retriever.build_query(
        case_state=state,
        top_k_raw=0,
        top_k_tactical=0,
        top_k_abstract=0,
        top_k_merged=0,
    )
    return {
        "query": query.to_dict(),
        "raw_case_results": [],
        "tactical_results": [],
        "abstract_results": [],
        "planner_summary": [],
        "skill_summary": [],
        "aggregator_summary": [],
        "merged_results": [],
    }


def _build_disabled_skill_retrieval_bundle(all_skill_objects: list) -> dict[str, object]:
    candidate_skill_names = [skill.name for skill in all_skill_objects]
    candidate_skill_ids = [skill.skill_id for skill in all_skill_objects]
    return {
        "candidate_skill_ids": candidate_skill_ids,
        "candidate_skill_names": candidate_skill_names,
        "skill_id_to_name": {skill.skill_id: skill.name for skill in all_skill_objects},
        "match_reasons": {},
        "trigger_hits": {},
        "retrieval_scores": {skill.name: 0.0 for skill in all_skill_objects},
        "decision_trace": [],
        "query_summary": {},
        "retriever_type": "disabled",
        "retriever_version": "v1",
    }


def _limit_skill_candidates(bundle: dict[str, object], *, top_k: int) -> dict[str, object]:
    if top_k <= 0:
        return bundle
    names = list(bundle.get("candidate_skill_names", []))[:top_k]
    ids = list(bundle.get("candidate_skill_ids", []))[:top_k]
    allowed_names = set(names)
    allowed_ids = set(ids)
    limited = dict(bundle)
    limited["candidate_skill_names"] = names
    limited["candidate_skill_ids"] = ids
    limited["skill_id_to_name"] = {
        skill_id: skill_name
        for skill_id, skill_name in dict(bundle.get("skill_id_to_name", {})).items()
        if skill_id in allowed_ids and skill_name in allowed_names
    }
    for key in ("match_reasons", "trigger_hits", "retrieval_scores"):
        limited[key] = {
            name: value
            for name, value in dict(bundle.get(key, {})).items()
            if name in allowed_names
        }
    if "reranker" in bundle:
        limited["reranker"] = dict(bundle.get("reranker", {}))
    return limited


def _load_retrieval_reranker(retrieval_policy: dict[str, object]) -> LearnedRetrievalScorer | None:
    if not bool(retrieval_policy.get("enable_learned_retrieval_reranker", False)):
        return None
    if LearnedRetrievalScorer is None:
        return None
    checkpoint_path = str(retrieval_policy.get("retrieval_reranker_checkpoint_path", "")).strip()
    if not checkpoint_path:
        return None
    try:
        return LearnedRetrievalScorer(
            checkpoint_path=checkpoint_path,
            blend_weight=float(retrieval_policy.get("retrieval_reranker_blend_weight", 2.5)),
        )
    except Exception:
        return None


def _rerank_experience_bundle_if_enabled(
    *,
    state: CaseState,
    bundle: dict[str, object],
    reranker: LearnedRetrievalScorer | None,
    fail_open: bool,
) -> dict[str, object]:
    if reranker is None:
        return bundle
    try:
        return reranker.rerank_experience_bundle(dict(bundle), case_state=state)
    except Exception as exc:
        fallback = dict(bundle)
        fallback["reranker"] = {
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}",
            "fail_open": fail_open,
        }
        if fail_open:
            return fallback
        raise


def _rerank_skill_bundle_if_enabled(
    *,
    state: CaseState,
    cognition: CognitionState,
    bundle: dict[str, object],
    reranker: LearnedRetrievalScorer | None,
    fail_open: bool,
) -> dict[str, object]:
    if reranker is None:
        return bundle
    try:
        return reranker.rerank_skill_bundle(dict(bundle), case_state=state, cognition=cognition)
    except Exception as exc:
        fallback = dict(bundle)
        fallback["reranker"] = {
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}",
            "fail_open": fail_open,
        }
        if fail_open:
            return fallback
        raise
