from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SNAPSHOT_DIR = Path("/root/DermAgent/state/skill_fitness_snapshots")


def save_skill_fitness_snapshot(
    cognition_dict: dict[str, Any],
    output_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = int(cognition_dict.get("evolution_generation", 0))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"snapshot_gen{generation:04d}_{timestamp}.json"

    skill_statistics = cognition_dict.get("skill_statistics", {})
    fitness_records = []
    for skill_name, stats in skill_statistics.items():
        fitness_records.append({
            "skill_name": skill_name,
            "call_count": stats.get("call_count", 0),
            "helpful_rate": stats.get("helpful_rate"),
            "harmful_rate": stats.get("harmful_rate"),
            "average_evidence_strength": stats.get("average_evidence_strength"),
            "success_count": stats.get("success_count", 0),
            "harmful_count": stats.get("harmful_count", 0),
        })
    fitness_records.sort(key=lambda x: (x.get("helpful_rate") or 0.0), reverse=True)

    snapshot = {
        "generation": generation,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_skills_tracked": len(fitness_records),
        "failure_statistics": cognition_dict.get("failure_statistics", {}),
        "skill_fitness": fitness_records,
    }

    out_path = output_dir / filename
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
