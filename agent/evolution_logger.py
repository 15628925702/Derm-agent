from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EVOLUTION_LOG_PATH = Path("/root/DermAgent/state/cognition_evolution_log.jsonl")


def compute_cognition_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}

    before_confusion = before.get("known_confusion_patterns", {})
    after_confusion = after.get("known_confusion_patterns", {})
    new_pairs = {k: v for k, v in after_confusion.items() if k not in before_confusion}
    incremented_pairs = {
        k: {"before": before_confusion[k], "after": v}
        for k, v in after_confusion.items()
        if k in before_confusion and v != before_confusion[k]
    }
    if new_pairs or incremented_pairs:
        diff["confusion_patterns"] = {}
        if new_pairs:
            diff["confusion_patterns"]["new"] = new_pairs
        if incremented_pairs:
            diff["confusion_patterns"]["incremented"] = incremented_pairs

    before_skills = before.get("preferred_skills", [])
    after_skills = after.get("preferred_skills", [])
    if before_skills != after_skills:
        diff["preferred_skills"] = {"before": before_skills, "after": after_skills}

    before_stats = before.get("skill_statistics", {})
    after_stats = after.get("skill_statistics", {})
    skill_stat_changes: dict[str, Any] = {}
    for skill, after_vals in after_stats.items():
        before_vals = before_stats.get(skill, {})
        changes = {}
        for metric in ("helpful_rate", "harmful_rate", "call_count", "success_count"):
            bv = before_vals.get(metric)
            av = after_vals.get(metric)
            if av != bv:
                changes[metric] = {"before": bv, "after": av}
        if changes:
            skill_stat_changes[skill] = changes
    if skill_stat_changes:
        diff["skill_statistics"] = skill_stat_changes

    before_fail = before.get("failure_statistics", {})
    after_fail = after.get("failure_statistics", {})
    fail_changes = {k: {"before": before_fail.get(k), "after": v} for k, v in after_fail.items() if v != before_fail.get(k)}
    if fail_changes:
        diff["failure_statistics"] = fail_changes

    return diff


def append_evolution_log(
    diff: dict[str, Any],
    generation: int,
    log_path: Path = DEFAULT_EVOLUTION_LOG_PATH,
) -> None:
    if not diff:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "generation": generation,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "diff": diff,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
