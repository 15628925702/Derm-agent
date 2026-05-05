from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ablations import (
    DEFAULT_MATRIX_CONFIG_PATH,
    MATRIX_SCHEMA_VERSION,
    build_matrix_target_specs,
    load_matrix_config,
)


def test_ablation_matrix_config_schema_and_required_targets() -> None:
    payload = load_matrix_config(DEFAULT_MATRIX_CONFIG_PATH)
    assert payload["schema_version"] == MATRIX_SCHEMA_VERSION
    target_ids = {str(item.get("target_id", "")) for item in payload.get("targets", [])}
    required = {
        "direct_qwen_baseline",
        "first_stage_basic_skills_only",
        "skill_bank_without_experience",
        "experience_without_advanced_skills",
        "full_without_cognition_update",
        "full_without_abstract_experiences",
        "full_without_specialist_skills",
        "full_without_learnable_controller",
        "full_with_rule_controller",
        "full_with_learned_controller",
    }
    assert required.issubset(target_ids)


def test_build_matrix_target_specs_skips_learned_when_checkpoint_missing() -> None:
    matrix = load_matrix_config(DEFAULT_MATRIX_CONFIG_PATH)
    target_specs, resolved_targets, skipped_targets = build_matrix_target_specs(
        matrix_config=matrix,
        include_optional=False,
        selected_target_ids=set(),
        policy={},
        checkpoint_catalog={
            "learned_controller_checkpoint": "",
            "retrieval_scorer_checkpoint": "",
            "evidence_calibrator_checkpoint": "",
        },
        strict_checkpoints=False,
    )
    target_ids = {spec.target_id for spec in target_specs}
    assert "direct_qwen_baseline" in target_ids
    assert "full_without_cognition_update" in target_ids
    assert "full_with_learned_controller" not in target_ids
    assert any(item.get("target_id") == "full_with_learned_controller" for item in skipped_targets)
    assert any(item.get("target_id") == "full_with_rule_controller" for item in resolved_targets)


def test_build_matrix_target_specs_includes_learned_when_checkpoint_available(tmp_path) -> None:
    dummy_ckpt = tmp_path / "dummy_controller.pt"
    dummy_ckpt.write_bytes(b"ckpt")
    matrix = load_matrix_config(DEFAULT_MATRIX_CONFIG_PATH)
    target_specs, _, skipped_targets = build_matrix_target_specs(
        matrix_config=matrix,
        include_optional=False,
        selected_target_ids={"direct_qwen_baseline", "full_without_cognition_update", "full_with_learned_controller"},
        policy={},
        checkpoint_catalog={
            "learned_controller_checkpoint": str(dummy_ckpt),
            "retrieval_scorer_checkpoint": "",
            "evidence_calibrator_checkpoint": "",
        },
        strict_checkpoints=True,
    )
    target_ids = {spec.target_id for spec in target_specs}
    assert "full_with_learned_controller" in target_ids
    assert not any(item.get("target_id") == "full_with_learned_controller" for item in skipped_targets)
