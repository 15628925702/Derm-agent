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
    assert contamination["target_execution_modes"]["full_dermagent"]["execution_overrides"]["enable_skill_retrieval"] is False
