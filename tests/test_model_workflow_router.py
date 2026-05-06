from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.model_workflow_router import (
    apply_model_workflow_to_case,
    dataset_environment_overrides_for_model_dataset,
    execution_overrides_for_run_agent,
    get_model_dataset_workflow_profile,
    get_model_workflow_overrides,
    merge_model_workflow_policy_overrides,
)
from agent.policy_config import load_stable_policy
from agent.run_agent import run_agent
from agent.state import CaseInput
from agent.workflow_profiles import uses_legacy_agent_final_path


class SkinVLFallbackStubClient:
    def initial_perception(self, case_input: CaseInput) -> dict:
        return {
            "image_summary": "",
            "ddx_candidates": [],
            "uncertainty": {"level": "unknown", "reasons": []},
            "notes": [],
        }

    def run_skill_prompt(self, case_input: CaseInput, skill_name: str, prompt: str, output_schema: str) -> dict:
        return {"skill_name": skill_name}

    def baseline_diagnosis(self, case_input: CaseInput) -> dict:
        return {
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Basal Cell Carcinoma"],
            "rationale": "baseline",
            "confidence": "medium",
            "follow_up_considerations": [],
        }

    def final_diagnosis(self, case_input: CaseInput, evidence_package) -> dict:
        return {
            "final_diagnosis": "source_id raw_case_memory retrieval_score",
            "differential_diagnoses": ["Nevus"],
            "rationale": "malformed agent output",
            "confidence": "medium",
            "follow_up_considerations": [],
        }


def test_skinvl_model_workflow_layers_on_dataset_workflow() -> None:
    case_input = CaseInput(
        case_id="case_1",
        image_path="/tmp/missing.jpg",
        metadata={},
        dataset_name="ham10000",
        workflow_context={
            "workflow_profile": "sparse_lesion_workflow",
            "workflow_capabilities": ["sparse_lesion_reasoning"],
        },
    )

    overrides = apply_model_workflow_to_case(case_input, "SkinVL-MM", dataset_name="ham10000")

    assert case_input.workflow_context is not None
    assert case_input.workflow_context["workflow_profile"] == "sparse_lesion_workflow"
    assert case_input.workflow_context["dataset_workflow_profile"] == "sparse_lesion_workflow"
    assert case_input.workflow_context["model_workflow_profile"] == "direct_baseline_workflow"
    assert case_input.workflow_context["disable_legacy_final_path"] is True
    assert uses_legacy_agent_final_path(case_input.workflow_context) is False
    assert "sparse_lesion_reasoning" in case_input.workflow_context["workflow_capabilities"]
    assert "direct_prediction" in case_input.workflow_context["workflow_capabilities"]
    assert overrides["skip_specialist_skills"] is True
    assert overrides["skip_experience_retrieval"] is True
    assert execution_overrides_for_run_agent(overrides) == {"enable_experience_retrieval": False}


def test_default_models_do_not_change_workflow() -> None:
    assert get_model_workflow_overrides("Qwen2.5-VL-7B-Instruct") == {}


def test_skinvl_policy_overlay_disables_specialists_only() -> None:
    policy = {
        "policy_id": "stable_default_policy",
        "planner_policy": {"force_disable_skills": ["already_disabled_skill"]},
    }
    overrides = get_model_workflow_overrides("skinvl-local")

    routed_policy = merge_model_workflow_policy_overrides(policy, overrides)

    assert routed_policy["policy_id"] == "stable_default_policy__SkinVL-MM"
    assert routed_policy["planner_policy"]["force_disable_skills"] == [
        "already_disabled_skill",
        "mel_nev_specialist_skill",
        "ack_scc_specialist_skill",
        "benign_mimic_specialist_skill",
    ]


def test_qwen_model_dataset_cell_blocks_legacy_model_overlay() -> None:
    case_input = CaseInput(
        case_id="case_2",
        image_path="/tmp/missing.jpg",
        metadata={},
        dataset_name="pad_ufes_20",
        workflow_context={
            "workflow_profile": "clinical_full_taxonomy_lesion_workflow",
            "workflow_capabilities": ["clinical_metadata_reasoning"],
        },
    )

    overrides = apply_model_workflow_to_case(
        case_input,
        "Qwen2.5-VL-7B-Instruct",
        dataset_name="pad_ufes_20",
    )

    assert case_input.workflow_context is not None
    assert case_input.workflow_context["workflow_profile"] == "clinical_full_taxonomy_lesion_workflow"
    assert case_input.workflow_context["dataset_workflow_profile"] == "clinical_full_taxonomy_lesion_workflow"
    assert "model_workflow_profile" not in case_input.workflow_context
    assert case_input.workflow_context["workflow_cell_id"] == "qwen__pad20__dataset_best"
    assert case_input.workflow_context["workflow_routing_priority"] == "model_dataset"
    assert overrides["workflow_routing_priority"] == "model_dataset"


def test_qwen_scin_cell_declares_grouped_label_space_and_environment() -> None:
    cell = get_model_dataset_workflow_profile("Qwen2.5-VL-7B-Instruct", "scin")

    assert cell["workflow_cell_id"] == "qwen__scin__grouped_best"
    assert cell["label_space_id"] == "scin_grouped"
    assert dataset_environment_overrides_for_model_dataset("Qwen2.5-VL-7B-Instruct", "scin") == {
        "DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped"
    }


def test_model_dataset_label_space_override_is_applied_to_case() -> None:
    case_input = CaseInput(
        case_id="case_scin",
        image_path="/tmp/missing.jpg",
        metadata={"label_space_id": "scin_full"},
        dataset_name="scin",
        label_space_id="scin_full",
        workflow_context={
            "workflow_profile": "default_workflow",
            "workflow_capabilities": [],
        },
    )

    overrides = apply_model_workflow_to_case(case_input, "Qwen2.5-VL-7B-Instruct", dataset_name="scin")

    assert case_input.label_space_id == "scin_grouped"
    assert case_input.metadata["label_space_id"] == "scin_grouped"
    assert case_input.workflow_context is not None
    assert case_input.workflow_context["label_space_id"] == "scin_grouped"
    assert case_input.workflow_context["workflow_cell_id"] == "qwen__scin__grouped_best"
    assert overrides["block_model_workflow_profile"] is True


def test_llama_archive_workflow_adds_conservative_overlay() -> None:
    case_input = CaseInput(
        case_id="case_3",
        image_path="/tmp/missing.jpg",
        metadata={},
        dataset_name="isic2019",
        workflow_context={
            "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
            "workflow_capabilities": ["image_archive_reasoning"],
        },
    )

    overrides = apply_model_workflow_to_case(
        case_input,
        "Llama-3.2-11B-Vision-Instruct",
        dataset_name="isic2019",
    )

    assert case_input.workflow_context is not None
    assert case_input.workflow_context["workflow_profile"] == "image_archive_full_taxonomy_lesion_workflow"
    assert case_input.workflow_context["model_workflow_profile"] == "conservative_archive_workflow"
    assert case_input.workflow_context["disable_legacy_final_path"] is True
    assert overrides["fallback_on_malformed_final"] is True


def test_skinvl_run_agent_route_preserves_dataset_workflow_and_falls_back(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DERMAGENT_SPLIT_STATE_ROOT", str(tmp_path / "split_states"))
    monkeypatch.setenv("DERMAGENT_POLICY_ROOT", str(tmp_path / "policy"))
    image_path = tmp_path / "case.jpg"
    image_path.write_bytes(b"fake")
    case_input = CaseInput(
        case_id="SKINVL_ROUTE_CASE",
        image_path=str(image_path),
        metadata={"region": "forearm", "itch": "True"},
        label="BCC",
        dataset_name="pad_ufes_20",
        label_space_id="derm_six",
        workflow_context={
            "workflow_profile": "clinical_full_taxonomy_lesion_workflow",
            "workflow_capabilities": ["clinical_metadata_reasoning", "legacy_agent_final_reasoning"],
        },
    )
    overrides = apply_model_workflow_to_case(case_input, "SkinVL-MM", dataset_name=case_input.dataset_name)
    policy = merge_model_workflow_policy_overrides(load_stable_policy().to_dict(), overrides)

    state, _ = run_agent(
        case_input=case_input,
        client=SkinVLFallbackStubClient(),
        policy_config=policy,
        output_dir=tmp_path / "outputs",
        enable_writeback=False,
        run_mode="debug",
        data_split="train",
        execution_overrides={**overrides, **execution_overrides_for_run_agent(overrides)},
    )

    assert state.case_input.workflow_context is not None
    assert state.case_input.workflow_context["workflow_profile"] == "clinical_full_taxonomy_lesion_workflow"
    assert state.case_input.workflow_context["dataset_workflow_profile"] == "clinical_full_taxonomy_lesion_workflow"
    assert state.case_input.workflow_context["model_workflow_profile"] == "direct_baseline_workflow"
    assert state.final_diagnosis["final_diagnosis"] == "Basal Cell Carcinoma"
    assert "fallback_to_baseline" in state.final_diagnosis["fusion_decision"]["reasons"]
    assert state.retrieved_experience == []
    assert not [skill for skill in state.planner_output.get("selected_skills", []) if skill.endswith("specialist_skill")]
