from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from project_paths import repo_root


DEFAULT_APPROVED_WORKFLOW_DIR = repo_root() / "state" / "workflow_evolution" / "approved"


def workflow_evolution_enabled() -> bool:
    return str(os.getenv("DERMAGENT_ENABLE_WORKFLOW_EVOLUTION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def approved_workflow_dir() -> Path:
    return Path(os.getenv("DERMAGENT_WORKFLOW_EVOLUTION_APPROVED_DIR", DEFAULT_APPROVED_WORKFLOW_DIR)).resolve()


def maybe_apply_active_workflow_evolution(
    *,
    model_key: str,
    dataset_key: str,
    static_cell_config: dict[str, Any] | None,
) -> dict[str, Any]:
    base = deepcopy(static_cell_config or {})
    if not workflow_evolution_enabled():
        return base

    candidate = _load_active_workflow_cell(model_key=model_key, dataset_key=dataset_key)
    if not candidate:
        return base

    merged = {**base, **candidate}
    merged["workflow_evolution_source"] = "approved_manual_proposal"
    return merged


def _load_active_workflow_cell(*, model_key: str, dataset_key: str) -> dict[str, Any]:
    root = approved_workflow_dir()
    if not root.exists():
        return {}

    model_norm = _norm(model_key)
    dataset_norm = _norm(dataset_key)
    newest: tuple[str, dict[str, Any]] | None = None
    for path in sorted(root.glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        if str(payload.get("proposal_type", "")).strip() != "workflow_evolution_candidate":
            continue
        if str(payload.get("review_status", "")).strip().lower() != "approved":
            continue
        activation = dict(payload.get("activation", {}) or {})
        if not bool(activation.get("enabled", False)):
            continue
        if _norm(payload.get("model_name")) != model_norm or _norm(payload.get("dataset_name")) != dataset_norm:
            continue
        cell = dict(payload.get("proposed_workflow_cell", {}) or {})
        if not cell:
            continue
        created_at = str(payload.get("created_at", "")).strip()
        if newest is None or created_at > newest[0]:
            newest = (created_at, cell)
    return deepcopy(newest[1]) if newest else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")
