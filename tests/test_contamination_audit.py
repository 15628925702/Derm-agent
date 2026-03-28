from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.contamination_guard import enforce_writeback_policy
from scripts.audit_experiment_state import audit_evaluation_manifests, audit_execution_records, audit_script_writeback_explicitness


def test_enforce_writeback_policy_blocks_frozen_mode() -> None:
    with pytest.raises(ValueError):
        enforce_writeback_policy(
            run_mode="test_frozen_inference",
            enable_writeback=True,
            strict=True,
        )


def test_audit_execution_records_flags_frozen_writeback() -> None:
    records = [
        {
            "case_id": "CASE_X",
            "state_versions": {
                "run_mode": "test_frozen_inference",
                "data_split": "test",
                "experience_state": {"split_name": "test", "split_aware_version": "experience_bank:test:abc"},
                "cognition_state": {"split_name": "test", "split_aware_version": "cognition_state:test:def"},
                "policy_state": {"split_name": "test", "split_aware_version": "policy_config:test:ghi"},
            },
            "writeback_ops": {
                "writeback_enabled": True,
                "raw_case_memory_written": True,
                "tactical_experience_count": 1,
                "abstract_experience_count": 1,
            },
        }
    ]

    report = audit_execution_records(records, split_map={})
    issue_types = {item.get("issue_type") for item in report["issues"]}
    assert "frozen_mode_writeback_enabled" in issue_types
    assert "frozen_mode_persisted_writeback" in issue_types


def test_audit_script_writeback_explicitness_has_no_implicit_calls() -> None:
    report = audit_script_writeback_explicitness(PROJECT_ROOT / "scripts")
    assert report["implicit_run_agent_calls_without_enable_writeback"] == 0


def test_audit_evaluation_manifests_flags_case_selection_mismatch(tmp_path: Path) -> None:
    eval_root = tmp_path / "evaluation"
    run_root = eval_root / "eval_1"
    run_root.mkdir(parents=True)
    manifest_path = run_root / "evaluation_manifest.json"
    manifest_path.write_text(
        """
{
  "fairness_constraints": {
    "frozen_evaluation_mode": true
  },
  "execution_config": {
    "enable_writeback": false
  },
  "dataset": {
    "data_split": "test",
    "case_ids": ["CASE_A"]
  },
  "experiment_state": {
    "strict_frozen_eval": true,
    "writeback_enabled": false,
    "case_selection": {
      "case_ids": ["CASE_B"]
    },
    "state_paths": {
      "experience_root": "/root/DermAgent/state/split_states/test/experience",
      "cognition_path": "/root/DermAgent/state/split_states/test/cognition_state.json"
    }
  },
  "frozen_state": {
    "experience_split_aware_version": "experience_bank:test:abc",
    "cognition_split_aware_version": "cognition_state:test:def",
    "policy_version": "policy:test:ghi",
    "experience_state_split": "test",
    "cognition_state_split": "test"
  }
}
        """.strip(),
        encoding="utf-8",
    )

    report = audit_evaluation_manifests(eval_root, split_map={"test": {"CASE_A"}})
    issue_types = {item.get("issue_type") for item in report["issues"]}

    assert "manifest_case_selection_mismatch" in issue_types
