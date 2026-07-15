from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(os.environ.get("DERMAGENT_REPO_ROOT", Path(__file__).resolve().parent)).resolve()


def workspace_root() -> Path:
    return Path(os.environ.get("DERMAGENT_WORKSPACE_ROOT", repo_root().parent)).resolve()


def data_root() -> Path:
    return Path(os.environ.get("DERMAGENT_DATA_ROOT", repo_root() / "data")).resolve()


def models_root() -> Path:
    return Path(os.environ.get("DERMAGENT_MODELS_ROOT", workspace_root() / "models")).resolve()


def state_root() -> Path:
    return Path(os.environ.get("DERMAGENT_STATE_ROOT", repo_root() / "state")).resolve()


def outputs_root() -> Path:
    return Path(os.environ.get("DERMAGENT_OUTPUTS_ROOT", repo_root() / "outputs")).resolve()
