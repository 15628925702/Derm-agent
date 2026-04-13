from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.state import CaseState
from memory.experience_retriever import ExperienceRetriever
from memory.experience_schema import (
    DEFAULT_EXPERIENCE_ROOT,
    LEGACY_EXPERIENCE_JSONL_PATH,
    LEGACY_EXPERIENCE_JSON_PATH,
    ExperienceRecord,
)
from memory.experience_store import ExperienceStore
from memory.experience_transform import (
    abstract_to_retrieval_packet,
    legacy_record_to_retrieval_packet,
    raw_case_to_retrieval_packet,
    tactical_to_retrieval_packet,
)
from memory.experience_writer import ExperienceWriter


DEFAULT_EXPERIENCE_PATH = LEGACY_EXPERIENCE_JSON_PATH


@dataclass
class _LegacyExperienceRecordCompat:
    case_id: str
    experience_type: str
    perception_summary: str
    skills_used: list[str]
    key_evidence: dict[str, Any]
    error_type: str
    confusion_pair: str | None
    learning_points: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceBank:
    def __init__(self, path: Path | None = None, root: Path | None = None) -> None:
        self.path = path or DEFAULT_EXPERIENCE_PATH
        if root is not None:
            experience_root = root
        elif path is not None and path.suffix == "":
            experience_root = path
        else:
            experience_root = DEFAULT_EXPERIENCE_ROOT
        self.store = ExperienceStore(experience_root)
        self.retriever = ExperienceRetriever(self.store)
        self.writer = ExperienceWriter(self.store)
        self._migrate_legacy_records_if_needed()

    def initialize_empty_bank(self) -> Path:
        self.store.initialize()
        return self.store.manifest_path

    def load(self) -> dict[str, Any]:
        return {
            "version": self.store.load_manifest().get("version", "experience_v2"),
            "records": self.all_records(),
        }

    def save(self, records: list[dict[str, Any]]) -> Path:
        for record in records:
            self.insert_record(record)
        return self.store.manifest_path

    def all_records(self) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        packets.extend(abstract_to_retrieval_packet(record) for record in self.store.load_abstract_experiences())
        packets.extend(tactical_to_retrieval_packet(record) for record in self.store.load_tactical_experiences())
        packets.extend(raw_case_to_retrieval_packet(record) for record in self.store.load_raw_case_memories())
        return packets

    def writeback(self, bundle: dict[str, Any]) -> dict[str, Any]:
        return self.writer.writeback(bundle)

    def insert_raw_case_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload["experience_type"] = "raw_case_experience"
        self.writer.write_legacy_record(payload)
        return payload

    def insert_hard_case_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload["experience_type"] = "hard_case_experience"
        self.writer.write_legacy_record(payload)
        return payload

    def insert_confusion_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload["experience_type"] = "confusion_experience"
        self.writer.write_legacy_record(payload)
        return payload

    def insert_record(self, record: ExperienceRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, ExperienceRecord):
            payload = record.to_dict()
        elif isinstance(record, _LegacyExperienceRecordCompat):
            payload = record.to_dict()
        else:
            payload = dict(record)
        self.writer.write_legacy_record(payload)
        return payload

    def retrieve(
        self,
        case_state: CaseState | None = None,
        top_k: int = 3,
        perception: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.retriever.retrieve(case_state=case_state, top_k=top_k, perception=perception)

    def retrieve_bundle(
        self,
        *,
        case_state: CaseState | None = None,
        perception: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        risk_flags: list[str] | None = None,
        uncertainty: dict[str, Any] | None = None,
        ddx_candidates: list[str] | None = None,
        morphology_clues: list[str] | None = None,
        confusion_pair: str | None = None,
        top_k_raw: int = 2,
        top_k_tactical: int = 4,
        top_k_abstract: int = 4,
        top_k_merged: int = 6,
    ) -> dict[str, Any]:
        return self.retriever.retrieve_bundle(
            case_state=case_state,
            perception=perception,
            metadata=metadata,
            risk_flags=risk_flags,
            uncertainty=uncertainty,
            ddx_candidates=ddx_candidates,
            morphology_clues=morphology_clues,
            confusion_pair=confusion_pair,
            top_k_raw=top_k_raw,
            top_k_tactical=top_k_tactical,
            top_k_abstract=top_k_abstract,
            top_k_merged=top_k_merged,
        )

    def retrieve_similar(self, case_state: CaseState, top_k: int = 3) -> list[dict[str, Any]]:
        return self.retrieve(case_state=case_state, top_k=top_k)

    def _migrate_legacy_records_if_needed(self) -> None:
        manifest = self.store.load_manifest()
        has_new_records = any(
            int(manifest.get(key, 0)) > 0 for key in ("raw_case_count", "tactical_count", "abstract_count")
        )
        if has_new_records:
            return

        migrated_records: list[dict[str, Any]] = []
        if LEGACY_EXPERIENCE_JSON_PATH.exists():
            try:
                with LEGACY_EXPERIENCE_JSON_PATH.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                migrated_records.extend(
                    record for record in payload.get("records", []) if isinstance(record, dict) and record.get("case_id")
                )
            except json.JSONDecodeError:
                pass

        if LEGACY_EXPERIENCE_JSONL_PATH.exists():
            with LEGACY_EXPERIENCE_JSONL_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and payload.get("case_id"):
                        migrated_records.append(payload)

        seen: set[str] = set()
        for record in migrated_records:
            packet = legacy_record_to_retrieval_packet(record)
            record_key = json.dumps(packet, ensure_ascii=False, sort_keys=True)
            if record_key in seen:
                continue
            seen.add(record_key)
            self.writer.write_legacy_record(record)
