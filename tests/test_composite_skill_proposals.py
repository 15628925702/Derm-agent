from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.composite_skill_proposal_generator import generate_composite_skill_proposals, save_composite_skill_proposals


def test_generate_composite_skill_proposals_from_batch_critique_and_seeds(tmp_path) -> None:
    experience_root = tmp_path / "experience"
    (experience_root / "indexes").mkdir(parents=True)
    for file_name in ("raw_case_memory.jsonl", "tactical_experience.jsonl", "abstract_experience.jsonl"):
        (experience_root / file_name).write_text("", encoding="utf-8")
    (experience_root / "manifest.json").write_text("{}", encoding="utf-8")
    for file_name in ("case_id_to_raw.json", "tactical_by_case.json", "abstract_by_type.json", "confusion_memory_index.json"):
        (experience_root / "indexes" / file_name).write_text("{}", encoding="utf-8")

    abstract_seed = {
        "abs_id": "abs_composite_seed_1",
        "type": "composite_skill_seed",
        "concept": "compare_then_audit_uncertainty",
        "pattern_summary": {
            "trigger_pattern": {
                "decision_pattern": "compare_then_audit_uncertainty",
                "uncertainty_level": "medium",
                "confusion_pair": "",
                "region": "ARM",
            },
            "seed_skills": [
                "morphology_analysis_skill",
                "color_pattern_analysis_skill",
                "uncertainty_assessment_skill",
            ],
        },
        "supporting_cases": ["case_a", "case_b"],
        "counter_cases": ["case_z"],
        "derived_rule": {},
        "version": "v2",
        "seed_id": "seed_1",
        "composite_skill_seed": {
            "seed_id": "seed_1",
            "trigger_pattern": {
                "decision_pattern": "compare_then_audit_uncertainty",
                "uncertainty_level": "medium",
                "confusion_pair": "",
                "region": "ARM",
            },
            "skill_sequence": [
                "morphology_analysis_skill",
                "color_pattern_analysis_skill",
                "uncertainty_assessment_skill",
            ],
            "supporting_cases": ["case_a", "case_b"],
            "success_count": 2,
            "notes": ["Candidate only."],
            "promotion_interface": {
                "future_skill_id": "composite_workflow_seed_1",
            },
        },
        "provenance": {},
    }
    (experience_root / "abstract_experience.jsonl").write_text(json.dumps(abstract_seed) + "\n", encoding="utf-8")

    batch_critique = {
        "batch_id": "batch_x",
        "success_clusters": [
            {
                "cluster_id": "succ_1",
                "case_ids": ["case_a", "case_b"],
                "support_count": 2,
                "common_skill_sequence": [
                    "morphology_analysis_skill",
                    "color_pattern_analysis_skill",
                    "uncertainty_assessment_skill",
                ],
                "representative_signals": ["region:arm", "uncertainty:medium"],
            }
        ],
        "failure_clusters": [
            {
                "cluster_id": "fail_1",
                "failure_type": "contradiction_heavy_case",
                "case_ids": ["case_z"],
                "common_skill_sequence": [
                    "morphology_analysis_skill",
                    "contradiction_check_skill",
                ],
            }
        ],
        "rule_candidates": [
            {
                "candidate_id": "rule_1",
                "pattern_summary": {
                    "trigger_type": "risk_review",
                    "decision_pattern": "structured_skill_sequence",
                },
            }
        ],
        "composite_skill_seed_candidates": [
            {
                "seed_id": "seed_1",
                "trigger_pattern": {
                    "decision_pattern": "compare_then_audit_uncertainty",
                    "uncertainty_level": "medium",
                    "confusion_pair": "",
                    "region": "ARM",
                },
                "skill_sequence": [
                    "morphology_analysis_skill",
                    "color_pattern_analysis_skill",
                    "uncertainty_assessment_skill",
                ],
                "supporting_cases": ["case_a", "case_b"],
                "success_count": 2,
                "notes": ["Repeated successful sequence."],
            }
        ],
        "refinement_inputs": {
            "success_patterns_for_policy_reuse": [],
        },
    }
    batch_critique_path = tmp_path / "batch_critique.json"
    batch_critique_path.write_text(json.dumps(batch_critique), encoding="utf-8")

    refinement_candidate = {
        "candidate_id": "ref_1",
        "target_skill_name": "uncertainty_assessment_skill",
        "proposed_update_type": "watchout_update",
        "proposed_change_summary": {"reason": "Need stronger contradiction watch-outs."},
    }
    refinement_candidates_path = tmp_path / "skill_refinement_candidates.jsonl"
    refinement_candidates_path.write_text(json.dumps(refinement_candidate) + "\n", encoding="utf-8")

    proposals, summary = generate_composite_skill_proposals(
        batch_critique_path=batch_critique_path,
        refinement_candidates_path=refinement_candidates_path,
        experience_root=experience_root,
        min_supporting_cases=2,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["review_status"] == "pending_review"
    assert proposal["supporting_cases"] == ["case_a", "case_b"]
    assert proposal["counter_cases"] == ["case_z"]
    assert proposal["proposal_artifacts"]["manual_integration_targets"]["registry_update_required"] is True
    assert proposal["evidence_refs"]["abstract_experience_ids"] == ["abs_composite_seed_1"]
    assert proposal["evidence_refs"]["refinement_candidate_ids"] == ["ref_1"]
    assert summary["proposal_count"] == 1

    output_dir = tmp_path / "proposals" / "composite_skills"
    output_paths = save_composite_skill_proposals(proposals, summary, output_dir=output_dir)
    assert Path(output_paths["proposals_path"]).exists()
    assert Path(output_paths["summary_path"]).exists()
    assert any(path.name.startswith(proposal["proposal_id"]) for path in output_dir.glob("*.json"))
