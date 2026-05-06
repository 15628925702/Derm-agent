from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_protocol import (
    EvaluationTargetSpec,
    build_contamination_check,
    build_target_policy,
    default_ablation_target_specs,
    experience_layers_for_variant,
)
from agent.execution_record import build_baseline_case_execution_record
from agent.model_workflow_router import apply_model_workflow_to_case, execution_overrides_for_run_agent
from agent.state import CaseInput


def test_default_ablation_target_specs_cover_key_dimensions() -> None:
    target_ids = {spec.target_id for spec in default_ablation_target_specs()}
    assert "no_experience_retrieval" in target_ids
    assert "no_skill_retrieval" in target_ids
    assert "no_cognition_bias" in target_ids
    assert "foundation_only" in target_ids


def test_build_target_policy_merges_overrides_and_appends_target_id() -> None:
    policy = {
        "policy_id": "stable_default_policy",
        "planner_policy": {"score_threshold_default": 4, "force_disable_skills": ["skill_a"]},
        "retrieval_policy": {"top_k_skill_candidates": 12},
    }
    spec = EvaluationTargetSpec(
        target_id="no_specialists",
        label="No Specialists",
        target_type="ablation",
        mode="agent",
        description="Disable specialists.",
        policy_overrides={"planner_policy": {"force_disable_skills": ["mel_nev_specialist_skill"]}},
    )

    target_policy = build_target_policy(policy_config=policy, target_spec=spec)

    assert target_policy["policy_id"] == "stable_default_policy__no_specialists"
    assert target_policy["planner_policy"]["score_threshold_default"] == 4
    assert target_policy["planner_policy"]["force_disable_skills"] == ["skill_a", "mel_nev_specialist_skill"]
    assert target_policy["retrieval_policy"]["top_k_skill_candidates"] == 12


def test_experience_layers_for_variant_is_explicit() -> None:
    assert experience_layers_for_variant("empty") == set()
    assert experience_layers_for_variant("raw_tactical") == {"raw", "tactical"}
    assert experience_layers_for_variant("raw_abstract") == {"raw", "abstract"}
    assert experience_layers_for_variant("unknown_variant") == {"raw", "tactical", "abstract"}


def test_build_baseline_case_execution_record_has_zero_delta() -> None:
    case_input = CaseInput(
        case_id="case_1",
        image_path="/tmp/missing.png",
        metadata={"age": "45", "region": "arm"},
        label="NEV",
        dataset_name="toy",
        source_metadata_path="/tmp/metadata.csv",
    )
    record = build_baseline_case_execution_record(
        case_input=case_input,
        baseline_qwen={
            "final_diagnosis": "NEV",
            "differential_diagnoses": ["MEL", "SEK"],
            "rationale": "Toy output.",
            "confidence": "medium",
            "follow_up_considerations": [],
        },
    )

    assert record["selected_skills"] == []
    assert record["evaluation"]["agent_vs_baseline_delta"]["correct_delta"] == 0
    assert record["writeback_ops"]["writeback_enabled"] is False


def test_build_contamination_check_marks_snapshot_isolation() -> None:
    contamination = build_contamination_check(
        frozen_state={"source_hashes": {"policy_hash": "abc123"}},
        experiment_state_manifest={
            "strict_frozen_eval": True,
            "data_split": "test",
            "case_selection": {"case_ids": ["CASE_1"]},
            "state_paths": {"experience_root": "/tmp/split_states/test/experience"},
        },
        target_specs=[
            EvaluationTargetSpec(
                target_id="full_dermagent",
                label="Full DermAgent",
                target_type="full_agent",
                mode="agent",
                description="Full agent.",
                execution_overrides={"enable_skill_retrieval": False},
            )
        ],
    )

    assert contamination["used_snapshot_isolation"] is True
    assert contamination["live_state_mutation_allowed_to_affect_eval"] is False
    assert contamination["strict_frozen_eval"] is True
    assert contamination["target_execution_modes"]["full_dermagent"]["execution_overrides"]["enable_skill_retrieval"] is False


def test_case_specific_model_overlay_is_merged_from_target_model_name() -> None:
    case_input = CaseInput(
        case_id="case_2",
        image_path="/tmp/missing.png",
        metadata={"region": "forearm"},
        label="BCC",
        dataset_name="pad_ufes_20",
        workflow_context={
            "workflow_profile": "clinical_full_taxonomy_lesion_workflow",
            "workflow_capabilities": ["clinical_metadata_reasoning"],
        },
    )
    target_execution_overrides = {"model_name": "Qwen2.5-VL-7B-Instruct"}

    case_overrides = apply_model_workflow_to_case(
        case_input,
        target_execution_overrides["model_name"],
        dataset_name=case_input.dataset_name,
    )
    merged_execution_overrides = {
        **target_execution_overrides,
        **case_overrides,
        **execution_overrides_for_run_agent(case_overrides),
    }

    assert case_input.workflow_context is not None
    assert case_input.workflow_context["workflow_profile"] == "clinical_full_taxonomy_lesion_workflow"
    assert case_input.workflow_context["model_workflow_profile"] == "clinical_malignant_guard_workflow"
    assert merged_execution_overrides["model_name"] == "Qwen2.5-VL-7B-Instruct"
    assert merged_execution_overrides["force_conservative_fusion"] is True
