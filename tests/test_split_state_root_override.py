from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import experiment_state


def test_split_state_root_override(monkeypatch, tmp_path: Path) -> None:
    custom_root = tmp_path / "split_states_medgemma"
    monkeypatch.setenv("DERMAGENT_SPLIT_STATE_ROOT", str(custom_root))

    resolved = experiment_state.resolve_split_state_root()

    assert resolved == custom_root


def test_ensure_split_state_paths_uses_override(monkeypatch, tmp_path: Path) -> None:
    custom_root = tmp_path / "split_states_medgemma"
    monkeypatch.setenv("DERMAGENT_SPLIT_STATE_ROOT", str(custom_root))

    resolved = experiment_state.ensure_split_state_paths(data_split="train")

    assert resolved.split_state_root == str(custom_root / "train")
    assert Path(resolved.experience_root).exists()
    assert Path(resolved.cognition_path).exists()
