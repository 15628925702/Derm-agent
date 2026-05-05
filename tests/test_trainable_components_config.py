from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = PROJECT_ROOT / "configs" / "trainable_components.yaml"


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_config_exists_and_has_required_top_level_fields() -> None:
    assert CONFIG_PATH.exists()
    config = _load_config()
    required = {
        "schema_version",
        "protocol",
        "frozen_components",
        "training_data_sources",
        "trainable_components",
        "version_registry",
        "experiment_modes",
    }
    assert required.issubset(set(config.keys()))


def test_frozen_boundary_includes_qwen_and_skill_semantics() -> None:
    config = _load_config()
    frozen_ids = {item.get("component_id") for item in config.get("frozen_components", [])}
    assert "qwen_backbone" in frozen_ids
    assert "qwen_service_runtime" in frozen_ids
    assert "skill_semantic_definitions" in frozen_ids


def test_trainable_components_cover_stage4_targets() -> None:
    config = _load_config()
    component_ids = {item.get("component_id") for item in config.get("trainable_components", [])}
    expected = {
        "controller_planner_scorer",
        "retrieval_reranker",
        "skill_selection_policy_optimizer",
        "evidence_calibrator",
        "skill_helpfulness_predictor",
        "hard_case_prioritization_scorer",
    }
    assert expected.issubset(component_ids)


def test_each_trainable_component_has_io_signal_and_versioning() -> None:
    config = _load_config()
    for item in config.get("trainable_components", []):
        assert item.get("trainable") is True
        assert isinstance(item.get("input_contract"), dict)
        assert isinstance(item.get("output_contract"), dict)
        assert isinstance(item.get("training_signal"), dict)
        versioning = item.get("versioning")
        assert isinstance(versioning, dict)
        assert versioning.get("stable_pointer")
        assert versioning.get("candidates_dir")
        assert versioning.get("rollback_target")
