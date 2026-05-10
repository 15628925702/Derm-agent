from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cognition.cognition_state import CognitionState
from memory.experience_store import ExperienceStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge isolated bootstrap shard states into one split state root.")
    parser.add_argument("--shard-root", type=Path, required=True, help="Directory containing shard_*/split_states.")
    parser.add_argument("--output-split-state-root", type=Path, required=True)
    parser.add_argument("--source-split", type=str, default="train")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def merge_cognition_states(paths: list[Path], output_path: Path, split_name: str) -> None:
    merged = CognitionState()
    merged.state_split = split_name
    preferred_skills: list[str] = []
    workflow_preferences: dict[str, dict[str, Any]] = {}
    known_confusions: Counter[str] = Counter()
    failure_stats: Counter[str] = Counter()
    skill_stats: dict[str, Counter[str]] = {}
    skill_float_sums: dict[str, Counter[str]] = {}

    for path in paths:
        if not path.exists():
            continue
        state = CognitionState.load(path)
        preferred_skills.extend(state.preferred_skills)
        known_confusions.update({k: int(v or 0) for k, v in state.known_confusion_patterns.items()})
        failure_stats.update({k: int(v or 0) for k, v in state.failure_statistics.items()})
        for name, payload in state.skill_statistics.items():
            counts = skill_stats.setdefault(name, Counter())
            floats = skill_float_sums.setdefault(name, Counter())
            for key, value in payload.items():
                if key.endswith("_rate") or key in {"average_evidence_strength"}:
                    continue
                if isinstance(value, int):
                    counts[key] += value
                elif isinstance(value, float):
                    floats[key] += value
            if "evidence_strength_sum" in payload:
                floats["evidence_strength_sum"] += float(payload.get("evidence_strength_sum") or 0.0)
        for key, value in state.workflow_preferences.items():
            workflow_preferences[key] = value

    merged.preferred_skills = list(dict.fromkeys(preferred_skills)) or merged.preferred_skills
    merged.known_confusion_patterns = dict(known_confusions)
    merged.failure_statistics.update(dict(failure_stats))
    merged.workflow_preferences = workflow_preferences
    merged.skill_statistics = {}
    for name, counts in skill_stats.items():
        payload = dict(counts)
        payload.update(skill_float_sums.get(name, {}))
        call_count = int(payload.get("call_count", 0) or 0)
        success = int(payload.get("success_count", 0) or 0)
        partial = int(payload.get("partially_helpful_count", 0) or 0)
        failure = int(payload.get("failure_count", 0) or 0)
        helpful = success + partial
        payload["helpful_count"] = helpful
        payload["helpful_rate"] = helpful / call_count if call_count else 0.0
        payload["failure_rate"] = failure / call_count if call_count else 0.0
        obs = int(payload.get("evidence_strength_observation_count", 0) or 0)
        strength = float(payload.get("evidence_strength_sum", 0.0) or 0.0)
        payload["average_evidence_strength"] = strength / obs if obs else 0.0
        merged.skill_statistics[name] = payload
    merged.save(output_path)


def main() -> int:
    args = parse_args()
    shard_dirs = sorted(path for path in args.shard_root.glob("shard_*") if path.is_dir())
    output_train_root = args.output_split_state_root / args.source_split
    output_experience = output_train_root / "experience"
    output_store = ExperienceStore(output_experience, split_name=args.source_split)

    raw_records: list[dict[str, Any]] = []
    tactical_records: list[dict[str, Any]] = []
    abstract_records: list[dict[str, Any]] = []
    cognition_paths: list[Path] = []
    shard_manifests: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        split_root = shard_dir / "split_states" / args.source_split
        experience = split_root / "experience"
        raw_records.extend(read_jsonl(experience / "raw_case_memory.jsonl"))
        tactical_records.extend(read_jsonl(experience / "tactical_experience.jsonl"))
        abstract_records.extend(read_jsonl(experience / "abstract_experience.jsonl"))
        cognition_paths.append(split_root / "cognition_state.json")
        done_path = shard_dir / "DONE.json"
        if done_path.exists():
            shard_manifests.append(json.loads(done_path.read_text(encoding="utf-8")))

    for path in (output_store.raw_case_path, output_store.tactical_path, output_store.abstract_path):
        path.write_text("", encoding="utf-8")
    if raw_records:
        output_store._upsert_jsonl(path=output_store.raw_case_path, records=raw_records, key_field="case_id")
    output_store.upsert_tactical_experiences(tactical_records)
    output_store.upsert_abstract_experiences(abstract_records)
    output_store.refresh_metadata()
    merge_cognition_states(cognition_paths, output_train_root / "cognition_state.json", args.source_split)

    payload = {
        "shard_root": str(args.shard_root),
        "output_split_state_root": str(args.output_split_state_root),
        "source_split": args.source_split,
        "shard_count": len(shard_dirs),
        "raw_case_records": len(output_store.load_raw_case_memories()),
        "tactical_records": len(output_store.load_tactical_experiences()),
        "abstract_records": len(output_store.load_abstract_experiences()),
        "shards": shard_manifests,
    }
    manifest_path = args.manifest or args.output_split_state_root / "merge_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
