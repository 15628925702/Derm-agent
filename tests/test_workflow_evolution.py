from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.model_workflow_router import get_model_dataset_workflow_profile
from workflow_evolution.proposal_generator import generate_workflow_evolution_proposal


def test_workflow_evolution_proposal_is_pending_and_disabled(tmp_path: Path) -> None:
    report_path = tmp_path / "compare.json"
    report_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_1",
                        "input_summary": {
                            "label_space_id": "isic2019_full",
                            "clinical_metadata": {
                                "workflow_context": {
                                    "workflow_profile": "image_archive_full_taxonomy_lesion_workflow",
                                    "workflow_cell_id": "baseline_cell",
                                    "label_space_id": "isic2019_full",
                                }
                            },
                        },
                        "selected_skills": ["morphology_analysis_skill", "contradiction_check_skill"],
                        "baseline_qwen": {"final_diagnosis": "Melanoma"},
                        "qwen_final": {
                            "final_diagnosis": "Nevus",
                            "fusion_decision": {"reasons": ["fallback_to_baseline"]},
                        },
                        "ground_truth": {"canonical_label": "NV", "malignant_flag": False},
                        "evaluation": {
                            "baseline_correct": False,
                            "correct": True,
                            "baseline_topk_hit": False,
                            "topk_hit": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    proposal = generate_workflow_evolution_proposal(
        report_paths=[report_path],
        model_name="Hulu-Med-7B",
        dataset_name="isic2019",
        output_dir=tmp_path / "proposals",
    )

    assert proposal["review_status"] == "pending_review"
    assert proposal["default_enabled"] is False
    assert proposal["activation"]["enabled"] is False
    assert proposal["proposed_workflow_cell"]["disable_legacy_final_path"] is True
    assert proposal["source_evidence"]["baseline_top1"] == 0
    assert proposal["source_evidence"]["agent_top1"] == 1


def test_approved_workflow_evolution_is_ignored_until_env_enabled(monkeypatch, tmp_path: Path) -> None:
    approved_dir = tmp_path / "approved"
    approved_dir.mkdir()
    proposal = {
        "proposal_type": "workflow_evolution_candidate",
        "proposal_id": "skinvl__sd198__candidate",
        "created_at": "2026-05-06T00:00:00Z",
        "model_name": "SkinVL-MM",
        "dataset_name": "sd198",
        "review_status": "approved",
        "activation": {"enabled": True},
        "proposed_workflow_cell": {
            "workflow_cell_id": "skinvl__sd198__evolved_candidate_v1",
            "label_space_id": "sd198_grouped",
            "workflow_profile": "coarse_taxonomy_workflow",
            "workflow_capabilities": ["coarse_taxonomy_reasoning", "grouped_label_reasoning"],
            "force_conservative_fusion": True,
            "disable_legacy_final_path": True,
        },
    }
    (approved_dir / "candidate.json").write_text(json.dumps(proposal), encoding="utf-8")
    monkeypatch.setenv("DERMAGENT_WORKFLOW_EVOLUTION_APPROVED_DIR", str(approved_dir))
    monkeypatch.delenv("DERMAGENT_ENABLE_WORKFLOW_EVOLUTION", raising=False)

    assert get_model_dataset_workflow_profile("SkinVL-MM", "sd198") == {}

    monkeypatch.setenv("DERMAGENT_ENABLE_WORKFLOW_EVOLUTION", "1")
    cell = get_model_dataset_workflow_profile("SkinVL-MM", "sd198")

    assert cell["workflow_cell_id"] == "skinvl__sd198__evolved_candidate_v1"
    assert cell["workflow_evolution_source"] == "approved_manual_proposal"
    assert cell["workflow_routing_priority"] == "model_dataset"
