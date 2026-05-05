from __future__ import annotations

from typing import Any

from memory.experience_store import ExperienceStore
from memory.experience_transform import legacy_record_to_bundle


class ExperienceWriter:
    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def writeback(self, bundle: dict[str, Any]) -> dict[str, Any]:
        updated_bundle = dict(bundle)
        raw_case_memory = bundle.get("raw_case_memory")
        tactical_experiences = bundle.get("tactical_experiences", [])
        abstract_experiences = bundle.get("abstract_experiences", [])

        if isinstance(raw_case_memory, dict) and raw_case_memory.get("case_id"):
            self.store.upsert_raw_case_memory(raw_case_memory)
        if isinstance(tactical_experiences, list):
            self.store.upsert_tactical_experiences(
                [record for record in tactical_experiences if isinstance(record, dict) and record.get("exp_id")]
            )
        if isinstance(abstract_experiences, list):
            target_records = [record for record in abstract_experiences if isinstance(record, dict) and record.get("abs_id")]
            self.store.upsert_abstract_experiences(target_records)
            persisted_by_id = {
                record.get("abs_id"): record
                for record in self.store.load_abstract_experiences()
                if isinstance(record, dict) and record.get("abs_id")
            }
            updated_bundle["abstract_experiences"] = [
                persisted_by_id[record["abs_id"]]
                for record in target_records
                if record.get("abs_id") in persisted_by_id
            ]
            updated_bundle["reflection_extract"] = self._refresh_reflection_extract(
                current=updated_bundle.get("reflection_extract", {}),
                abstract_experiences=updated_bundle["abstract_experiences"],
            )
        self.store.refresh_metadata()
        return updated_bundle

    def write_legacy_record(self, record: dict[str, Any]) -> dict[str, Any]:
        bundle = legacy_record_to_bundle(record)
        self.writeback(bundle)
        return bundle

    @staticmethod
    def _refresh_reflection_extract(current: dict[str, Any], abstract_experiences: list[dict[str, Any]]) -> dict[str, Any]:
        updated = dict(current)
        composite_records = [
            record
            for record in abstract_experiences
            if isinstance(record, dict) and record.get("type") == "composite_skill_seed"
        ]
        updated["abstract_experience_candidate_count"] = len(abstract_experiences)
        updated["composite_skill_seed_count"] = len(composite_records)
        updated["composite_skill_seed_ids"] = [
            record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id")
            for record in composite_records
            if record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id")
        ]
        updated["composite_skill_seed_ready_for_promotion"] = [
            record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id")
            for record in composite_records
            if int(record.get("composite_skill_seed", {}).get("success_count", 0)) >= 2
            and (record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id"))
        ]
        return updated
