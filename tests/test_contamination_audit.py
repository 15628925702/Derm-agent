from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.contamination_guard import enforce_writeback_policy
from scripts.audit_experiment_state import audit_execution_records, audit_script_writeback_explicitness


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
