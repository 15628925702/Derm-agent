from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import policy_config


def test_policy_root_override_creates_isolated_store(tmp_path: Path, monkeypatch) -> None:
    custom_root = tmp_path / "policy_medgemma"
    monkeypatch.setenv("DERMAGENT_POLICY_ROOT", str(custom_root))

    paths = policy_config.ensure_policy_store()

    assert paths["root"] == custom_root
    assert paths["stable_path"] == custom_root / "current_stable_policy.json"
    assert paths["manifest_path"] == custom_root / "manifest.json"
    assert (custom_root / "current_stable_policy.json").exists()
    assert (custom_root / "manifest.json").exists()


def test_policy_root_defaults_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("DERMAGENT_POLICY_ROOT", raising=False)

    assert str(policy_config.CURRENT_STABLE_POLICY_PATH) == "/root/DermAgent/state/policy/current_stable_policy.json"
