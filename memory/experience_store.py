from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from agent.contamination_guard import build_state_partition, infer_split_from_path, normalize_split_name
from memory.experience_schema import DEFAULT_EXPERIENCE_ROOT


class ExperienceStore:
    def __init__(self, root: Path | None = None, *, split_name: str | None = None) -> None:
        self.root = root or DEFAULT_EXPERIENCE_ROOT
        inferred_split = split_name or infer_split_from_path(self.root, default="global")
        self.state_split = normalize_split_name(inferred_split, default="global")
        self.index_dir = self.root / "indexes"
        self.manifest_path = self.root / "manifest.json"
        self.raw_case_path = self.root / "raw_case_memory.jsonl"
        self.tactical_path = self.root / "tactical_experience.jsonl"
        self.abstract_path = self.root / "abstract_experience.jsonl"
        self.case_index_path = self.index_dir / "case_id_to_raw.json"
        self.tactical_index_path = self.index_dir / "tactical_by_case.json"
        self.abstract_index_path = self.index_dir / "abstract_by_type.json"
        self.confusion_index_path = self.index_dir / "confusion_memory_index.json"
        self.initialize()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        for target in (self.raw_case_path, self.tactical_path, self.abstract_path):
            if not target.exists():
                target.write_text("", encoding="utf-8")
        if not self.manifest_path.exists():
            self._write_json(
                self.manifest_path,
                {
                    "version": "experience_v2",
                    "state_split": self.state_split,
                    "split_aware_version": "",
                    "raw_case_count": 0,
                    "tactical_count": 0,
                    "abstract_count": 0,
                    "files": {
                        "raw_case_memory": str(self.raw_case_path),
                        "tactical_experience": str(self.tactical_path),
                        "abstract_experience": str(self.abstract_path),
                    },
                },
            )
        for target in (
            self.case_index_path,
            self.tactical_index_path,
            self.abstract_index_path,
            self.confusion_index_path,
        ):
            if not target.exists():
                self._write_json(target, {})
        self.refresh_metadata()

    def load_manifest(self) -> dict[str, Any]:
        return self._read_json(self.manifest_path, {})

    def load_raw_case_memories(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.raw_case_path)

    def load_tactical_experiences(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.tactical_path)

    def load_abstract_experiences(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.abstract_path)

    def upsert_raw_case_memory(self, record: dict[str, Any]) -> dict[str, Any]:
        self._upsert_jsonl(
            path=self.raw_case_path,
            records=[record],
            key_field="case_id",
        )
        self.refresh_metadata()
        return record

    def upsert_tactical_experiences(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if records:
            self._upsert_jsonl(
                path=self.tactical_path,
                records=records,
                key_field="exp_id",
            )
            self.refresh_metadata()
        return records

    def upsert_abstract_experiences(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if records:
            self._upsert_jsonl(
                path=self.abstract_path,
                records=records,
                key_field="abs_id",
                merge_fn=self._merge_abstract_record,
            )
            self.refresh_metadata()
        return records

    def refresh_metadata(self) -> None:
        raw_records = self.load_raw_case_memories()
        tactical_records = self.load_tactical_experiences()
        abstract_records = self.load_abstract_experiences()
        state_partition = self._build_state_partition(
            raw_records=raw_records,
            tactical_records=tactical_records,
            abstract_records=abstract_records,
        )
        self._write_json(
            self.manifest_path,
            {
                "version": "experience_v2",
                "state_split": self.state_split,
                "split_aware_version": state_partition["split_aware_version"],
                "state_partition": state_partition,
                "raw_case_count": len(raw_records),
                "tactical_count": len(tactical_records),
                "abstract_count": len(abstract_records),
                "files": {
                    "raw_case_memory": str(self.raw_case_path),
                    "tactical_experience": str(self.tactical_path),
                    "abstract_experience": str(self.abstract_path),
                },
            },
        )
        self._write_json(
            self.case_index_path,
            {record.get("case_id", ""): record.get("case_id", "") for record in raw_records if record.get("case_id")},
        )

        tactical_by_case: dict[str, list[str]] = {}
        for record in tactical_records:
            case_id = str(record.get("case_id", "")).strip()
            exp_id = str(record.get("exp_id", "")).strip()
            if not case_id or not exp_id:
                continue
            tactical_by_case.setdefault(case_id, []).append(exp_id)
        self._write_json(self.tactical_index_path, tactical_by_case)

        abstract_by_type: dict[str, list[str]] = {}
        confusion_index: dict[str, list[str]] = {}
        for record in abstract_records:
            exp_type = str(record.get("type", "")).strip()
            abs_id = str(record.get("abs_id", "")).strip()
            if exp_type and abs_id:
                abstract_by_type.setdefault(exp_type, []).append(abs_id)
            if exp_type == "confusion_memory":
                confusion_pair = str(record.get("pattern_summary", {}).get("confusion_pair", "")).strip()
                if confusion_pair and abs_id:
                    confusion_index.setdefault(confusion_pair, []).append(abs_id)
        self._write_json(self.abstract_index_path, abstract_by_type)
        self._write_json(self.confusion_index_path, confusion_index)

    def _upsert_jsonl(
        self,
        *,
        path: Path,
        records: list[dict[str, Any]],
        key_field: str,
        merge_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        existing_records = self._read_jsonl(path)
        by_key: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []
        for record in existing_records:
            key = str(record.get(key_field, "")).strip()
            if not key:
                continue
            if key not in by_key:
                ordered_keys.append(key)
            by_key[key] = record

        for record in records:
            key = str(record.get(key_field, "")).strip()
            if not key:
                continue
            if key in by_key and merge_fn is not None:
                by_key[key] = merge_fn(by_key[key], record)
            else:
                by_key[key] = record
            if key not in ordered_keys:
                ordered_keys.append(key)

        merged_records = [by_key[key] for key in ordered_keys]
        self._write_jsonl(path, merged_records)

    @staticmethod
    def _merge_abstract_record(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        supporting_cases = list(dict.fromkeys(existing.get("supporting_cases", []) + new.get("supporting_cases", [])))
        counter_cases = list(dict.fromkeys(existing.get("counter_cases", []) + new.get("counter_cases", [])))
        merged = dict(existing)
        merged.update(new)
        merged["supporting_cases"] = supporting_cases
        merged["counter_cases"] = counter_cases
        existing_provenance = existing.get("provenance", {})
        new_provenance = new.get("provenance", {})
        if isinstance(existing_provenance, dict) or isinstance(new_provenance, dict):
            provenance = dict(existing_provenance) if isinstance(existing_provenance, dict) else {}
            if isinstance(new_provenance, dict):
                provenance.update(new_provenance)
            for field_name in ("source_case_ids", "source_exp_ids", "source_abs_ids", "source_hard_case_ids"):
                combined = list(dict.fromkeys(list(existing_provenance.get(field_name, [])) + list(new_provenance.get(field_name, []))))
                if combined:
                    provenance[field_name] = combined
            merged["provenance"] = provenance
        existing_seed = existing.get("composite_skill_seed", {})
        new_seed = new.get("composite_skill_seed", {})
        if existing_seed or new_seed or merged.get("type") == "composite_skill_seed":
            merged_seed = dict(existing_seed) if isinstance(existing_seed, dict) else {}
            if isinstance(new_seed, dict):
                merged_seed.update(new_seed)
            merged_seed["seed_id"] = (
                merged_seed.get("seed_id")
                or new.get("seed_id")
                or existing.get("seed_id")
            )
            merged_seed["trigger_pattern"] = merged_seed.get("trigger_pattern") or merged.get("pattern_summary", {}).get(
                "trigger_pattern", {}
            )
            merged_seed["skill_sequence"] = list(
                dict.fromkeys(
                    list(existing_seed.get("skill_sequence", []))
                    + list(new_seed.get("skill_sequence", []))
                )
            )
            merged_seed["supporting_cases"] = supporting_cases
            merged_seed["success_count"] = len(supporting_cases)
            merged_seed["notes"] = list(
                dict.fromkeys(list(existing_seed.get("notes", [])) + list(new_seed.get("notes", [])))
            )
            existing_interface = existing_seed.get("promotion_interface", {})
            new_interface = new_seed.get("promotion_interface", {})
            promotion_interface = dict(existing_interface) if isinstance(existing_interface, dict) else {}
            if isinstance(new_interface, dict):
                promotion_interface.update(new_interface)
            merged_seed["promotion_interface"] = promotion_interface
            merged["seed_id"] = merged_seed.get("seed_id")
            merged["composite_skill_seed"] = merged_seed
            pattern_summary = dict(merged.get("pattern_summary", {}))
            pattern_summary["trigger_pattern"] = merged_seed.get("trigger_pattern", pattern_summary.get("trigger_pattern", {}))
            pattern_summary["seed_skills"] = merged_seed.get("skill_sequence", pattern_summary.get("seed_skills", []))
            merged["pattern_summary"] = pattern_summary
            derived_rule = dict(merged.get("derived_rule", {}))
            promotion_payload = dict(derived_rule.get("promotion_interface", {}))
            promotion_payload.update(promotion_interface)
            derived_rule["promotion_interface"] = promotion_payload
            merged["derived_rule"] = derived_rule
        return merged

    def _build_state_partition(
        self,
        *,
        raw_records: list[dict[str, Any]],
        tactical_records: list[dict[str, Any]],
        abstract_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "raw_case_count": len(raw_records),
            "tactical_count": len(tactical_records),
            "abstract_count": len(abstract_records),
            "record_hash": hashlib.sha256(
                json.dumps(
                    {
                        "raw_case_ids": sorted(
                            str(record.get("case_id", "")).strip()
                            for record in raw_records
                            if str(record.get("case_id", "")).strip()
                        ),
                        "tactical_ids": sorted(
                            str(record.get("exp_id", "")).strip()
                            for record in tactical_records
                            if str(record.get("exp_id", "")).strip()
                        ),
                        "abstract_ids": sorted(
                            str(record.get("abs_id", "")).strip()
                            for record in abstract_records
                            if str(record.get("abs_id", "")).strip()
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:12],
        }
        return build_state_partition(
            component_id="experience_bank",
            source_path=self.root,
            split_name=self.state_split,
            payload=payload,
        )

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
