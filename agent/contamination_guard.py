from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


SPLIT_ALIAS = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
    "global": "global",
    "unknown": "unknown",
}

FROZEN_MODE_TOKENS = ("frozen", "_frozen_", "frozen_inference")


def normalize_split_name(value: str | None, *, default: str = "global") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return SPLIT_ALIAS.get(text, text)


def infer_split_from_path(path: str | Path | None, *, default: str = "global") -> str:
    if path is None:
        return default
    parts = [part.strip().lower() for part in Path(path).parts if str(part).strip()]
    if "split_states" in parts:
        index = parts.index("split_states")
        if index + 1 < len(parts):
            return normalize_split_name(parts[index + 1], default=default)
    for token in parts:
        if token in {"train", "training"}:
            return "train"
        if token in {"val", "validation", "valid"}:
            return "val"
        if token in {"test", "testing"}:
            return "test"
    return default


def build_split_state_version(
    *,
    component_id: str,
    split_name: str,
    payload: dict[str, Any],
) -> str:
    normalized_split = normalize_split_name(split_name, default="global")
    digest = hashlib.sha256(
        json.dumps(
            {
                "component_id": str(component_id).strip(),
                "split_name": normalized_split,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{component_id}:{normalized_split}:{digest}"


def build_state_partition(
    *,
    component_id: str,
    source_path: str | Path,
    split_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized_split = normalize_split_name(split_name, default="global")
    return {
        "component_id": str(component_id).strip(),
        "source_path": str(source_path),
        "split_name": normalized_split,
        "split_aware_version": build_split_state_version(
            component_id=component_id,
            split_name=normalized_split,
            payload=payload,
        ),
    }


def is_frozen_mode(run_mode: str | None) -> bool:
    text = str(run_mode or "").strip().lower()
    return any(token in text for token in FROZEN_MODE_TOKENS)


def enforce_writeback_policy(
    *,
    run_mode: str,
    enable_writeback: bool,
    strict: bool = True,
) -> bool:
    frozen = is_frozen_mode(run_mode)
    if not frozen:
        return bool(enable_writeback)
    if bool(enable_writeback):
        if strict:
            raise ValueError(
                f"Writeback is forbidden in frozen mode `{run_mode}`. "
                "Set enable_writeback=False for validation/test/evaluation inference."
            )
        return False
    return False


def find_run_agent_calls_without_explicit_writeback(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if call_name != "run_agent":
                continue
            has_enable_writeback = any(keyword.arg == "enable_writeback" for keyword in node.keywords if keyword.arg)
            if has_enable_writeback:
                continue
            findings.append(
                {
                    "file": str(path),
                    "line": int(getattr(node, "lineno", 0) or 0),
                    "column": int(getattr(node, "col_offset", 0) or 0),
                    "issue": "run_agent_without_explicit_enable_writeback",
                }
            )
    return findings


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
