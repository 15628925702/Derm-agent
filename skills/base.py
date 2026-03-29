from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from typing import Any, Iterable

from agent.confusion_clusters import (
    cluster_guidance_snapshot,
    cluster_pairs,
    cluster_related_keywords,
    detect_confusion_clusters,
)
from agent.state import CaseState
from integrations.openai_client import DermOpenAIClient
from skills.schema import SkillExecutionContext, SkillObject


STANDARD_OUTPUT_FIELDS = (
    ("referenced_experiences", "list[str]", "Source ids of retrieved experiences explicitly relied on during this skill execution."),
    ("evidence_strength", "str", "Overall strength of the evidence produced by this skill: low, medium, or high."),
    (
        "recommendation_type",
        "str",
        "What kind of reusable signal this skill is contributing, such as descriptive_evidence, comparative_support, risk_signal, uncertainty_signal, or conflict_signal.",
    ),
)


class BaseSkill(ABC):
    name = "base_skill"
    description = "Atomic clinical reasoning action."
    output_fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()
    skill_object: SkillObject | None = None
    _active_execution_context: SkillExecutionContext | None = None
    _active_reference_ids: list[str] = []
    _logger = logging.getLogger(__name__)

    def execute(self, state: CaseState, client: DermOpenAIClient) -> dict[str, Any]:
        self.get_skill_object().stats.call_count += 1
        try:
            execution_context = self.build_execution_context(state)
            self._active_execution_context = execution_context
            self._active_reference_ids = self._collect_reference_ids(execution_context)
            prompt = self.compose_prompt(state, execution_context)
            raw_output = client.run_skill_prompt(
                case_input=state.case_input,
                skill_name=self.name,
                prompt=prompt,
                output_schema=self.output_schema_text(),
            )
            normalized_output = self.normalize_output(raw_output)
            state.skill_outputs[self.name] = normalized_output
            self.after_execute(state, normalized_output)
            return normalized_output
        except json.JSONDecodeError as exc:
            self.get_skill_object().stats.failure_count += 1
            fallback_output = self.build_fallback_output(state, error=exc)
            state.skill_outputs[self.name] = fallback_output
            state.notes.append(
                f"{self.name} degraded to fallback output after malformed JSON during skill execution."
            )
            self._logger.warning(
                "Skill %s degraded to fallback output after JSON parsing failure: %s",
                self.name,
                exc,
            )
            return fallback_output
        except Exception:
            self.get_skill_object().stats.failure_count += 1
            raise
        finally:
            self._active_execution_context = None
            self._active_reference_ids = []

    @abstractmethod
    def build_prompt(self, state: CaseState) -> str:
        """Build a doctor-style prompt for this reasoning action."""

    def compose_prompt(self, state: CaseState, execution_context: SkillExecutionContext) -> str:
        core_prompt = self.build_prompt(state).rstrip()
        execution_block = self.execution_context_text(execution_context)
        return f"{core_prompt}\n{execution_block}\n"

    def get_skill_object(self) -> SkillObject:
        if self.skill_object is None:
            raise ValueError(f"{self.__class__.__name__} must define skill_object")
        return self.skill_object

    def normalize_output(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for field_name in self.output_fields:
            normalized[field_name] = self.normalize_field(field_name, raw_output.get(field_name))
        for field_name, _, _ in STANDARD_OUTPUT_FIELDS:
            normalized[field_name] = self.normalize_field(field_name, raw_output.get(field_name))
        return normalized

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if value in (None, ""):
            if (
                field_name in self.list_fields
                or field_name.endswith("_evidence")
                or field_name.endswith("_signals")
                or field_name == "reasons"
            ):
                return []
            if field_name == "referenced_experiences":
                return list(self._active_reference_ids)
            if field_name == "missing_information":
                return []
            if field_name == "evidence_strength":
                return self._default_evidence_strength()
            if field_name == "recommendation_type":
                return self._default_recommendation_type()
            return "unknown"
        if field_name == "referenced_experiences":
            if isinstance(value, list):
                candidates = [str(item).strip() for item in value if str(item).strip()]
            else:
                candidates = [str(value).strip()]
            if self._active_reference_ids:
                filtered = [item for item in candidates if item in self._active_reference_ids]
                return filtered or list(self._active_reference_ids)
            return candidates
        if field_name == "evidence_strength":
            normalized = str(value).strip().lower()
            if normalized not in {"low", "medium", "high"}:
                return self._default_evidence_strength()
            return normalized
        if field_name == "recommendation_type":
            normalized = str(value).strip().lower()
            if normalized:
                return normalized
            return self._default_recommendation_type()
        if field_name in self.list_fields and not isinstance(value, list):
            return [str(value)]
        return value

    def after_execute(self, state: CaseState, output: dict[str, Any]) -> None:
        """Optional state updates after skill execution."""

    def build_fallback_output(self, state: CaseState, *, error: Exception) -> dict[str, Any]:
        fallback = self.normalize_output({})
        fallback["execution_status"] = "fallback_json_parse_error"
        fallback["execution_error"] = error.__class__.__name__
        fallback["referenced_experiences"] = list(self._active_reference_ids)
        fallback["evidence_strength"] = "low"
        fallback["recommendation_type"] = self._default_recommendation_type()
        return fallback

    def output_schema_text(self) -> str:
        output_schema = self.get_skill_object().output_schema
        lines = [f"- {field.name}: {field.description}" for field in output_schema]
        lines.extend(f"- {name}: {description}" for name, _, description in STANDARD_OUTPUT_FIELDS)
        return "\n".join(lines)

    def to_skill_dict(self) -> dict[str, Any]:
        return self.get_skill_object().to_dict()

    def workflow_text(self) -> str:
        return self.get_skill_object().workflow_text

    def build_execution_context(self, state: CaseState) -> SkillExecutionContext:
        retrieval_bundle = state.retrieval_bundle or {}
        selected_raw_cases = retrieval_bundle.get("raw_case_results", [])[:2]
        selected_tactical_experiences = retrieval_bundle.get("skill_summary", retrieval_bundle.get("tactical_results", []))[:3]
        selected_abstract_experiences = retrieval_bundle.get("abstract_results", [])[:3]
        related_abstract_experiences = self.select_related_abstract_experiences(state, selected_abstract_experiences)
        return SkillExecutionContext(
            perception=state.perception,
            metadata=state.clinical_metadata,
            selected_raw_cases=[dict(item) for item in selected_raw_cases],
            selected_tactical_experiences=[dict(item) for item in selected_tactical_experiences],
            selected_abstract_experiences=[dict(item) for item in selected_abstract_experiences],
            related_abstract_experiences=[dict(item) for item in related_abstract_experiences],
        )

    def select_related_abstract_experiences(
        self,
        state: CaseState,
        abstract_experiences: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return []

    def active_confusion_clusters(self, state: CaseState) -> list[str]:
        perception = state.perception or {}
        related_packets = []
        if self._active_execution_context:
            related_packets = list(self._active_execution_context.related_abstract_experiences)
        related_text = " ".join(
            str(packet.get("confusion_pair", "")).strip()
            or str(packet.get("perception_summary", "")).strip()
            for packet in related_packets
        )
        retrieval_query = (state.retrieval_bundle or {}).get("query", {})
        return detect_confusion_clusters(
            ddx_candidates=[str(item) for item in perception.get("ddx_candidates", []) if str(item).strip()],
            confusion_pair=str(retrieval_query.get("confusion_pair", "")).strip() or None,
            known_confusion_text=related_text,
            image_summary=str(perception.get("image_summary", "")),
            notes=[str(item) for item in perception.get("notes", []) if str(item).strip()],
        )

    def confusion_cluster_prompt_payload(self, state: CaseState, *, max_items: int = 2) -> dict[str, Any]:
        cluster_names = self.active_confusion_clusters(state)
        return {
            "active_clusters": cluster_names,
            "related_confusion_pairs": list(cluster_pairs(cluster_names)),
            "related_keywords": list(cluster_related_keywords(cluster_names)),
            "guidance": cluster_guidance_snapshot(cluster_names, max_items=max_items),
        }

    def execution_context_text(self, execution_context: SkillExecutionContext) -> str:
        payload = {
            "selected_raw_cases": [self._compact_experience_packet(item) for item in execution_context.selected_raw_cases],
            "selected_tactical_experiences": [
                self._compact_experience_packet(item) for item in execution_context.selected_tactical_experiences
            ],
            "selected_abstract_experiences": [
                self._compact_experience_packet(item) for item in execution_context.selected_abstract_experiences
            ],
            "related_abstract_experiences": [
                self._compact_experience_packet(item) for item in execution_context.related_abstract_experiences
            ],
        }
        return (
            "Retrieved experience may be used only as auxiliary evidence, not as a final diagnosis shortcut.\n"
            "If you rely on any retrieved experience, cite its `source_id` values in `referenced_experiences`.\n"
            "If related abstract experiences are provided, use them as explicit confusion/prototype/rule references rather than generic memory.\n"
            "Return JSON only, and include the standard meta fields `referenced_experiences`, `evidence_strength`, and `recommendation_type`.\n"
            f"Skill execution context: {json.dumps(payload, ensure_ascii=False)}"
        )

    def _collect_reference_ids(self, execution_context: SkillExecutionContext) -> list[str]:
        reference_ids: list[str] = []
        for packet in (
            execution_context.selected_raw_cases
            + execution_context.selected_tactical_experiences
            + execution_context.selected_abstract_experiences
            + execution_context.related_abstract_experiences
        ):
            source_id = str(packet.get("source_id", "")).strip()
            if source_id and source_id not in reference_ids:
                reference_ids.append(source_id)
        return reference_ids

    def _default_evidence_strength(self) -> str:
        if self.get_skill_object().skill_type in {"observation", "pattern"}:
            return "medium"
        if self.get_skill_object().skill_type == "risk_uncertainty":
            return "high"
        return "medium"

    def _default_recommendation_type(self) -> str:
        skill_type = self.get_skill_object().skill_type
        if skill_type in {"observation", "pattern"}:
            return "descriptive_evidence"
        if skill_type == "risk_uncertainty":
            return "risk_signal"
        if skill_type in {"reasoning", "specialist"}:
            return "comparative_support"
        return "supporting_evidence"

    @staticmethod
    def filter_related_abstract_experiences(
        abstract_experiences: list[dict[str, Any]],
        *,
        confusion_pairs: tuple[str, ...] = (),
        keywords: tuple[str, ...] = (),
        allowed_types: tuple[str, ...] = ("confusion_memory", "prototype", "rule"),
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        lowered_pairs = {item.lower() for item in confusion_pairs if item}
        target_pair_tags = [BaseSkill._confusion_family_tags(item) for item in lowered_pairs]
        lowered_keywords = tuple(item.lower() for item in keywords if item)
        for packet in abstract_experiences:
            experience_type = str(packet.get("experience_type", "")).strip().lower()
            normalized_allowed_types = set(allowed_types)
            if "rule" in normalized_allowed_types:
                normalized_allowed_types.add("rule_candidate")
            if allowed_types and experience_type not in normalized_allowed_types:
                continue
            confusion_pair = str(packet.get("confusion_pair", "")).strip().lower()
            packet_text = " ".join(
                [
                    str(packet.get("perception_summary", "")),
                    str(packet.get("confusion_pair", "")),
                    " ".join(str(item) for item in packet.get("learning_points", [])),
                    str(packet.get("source_id", "")),
                ]
            ).lower()
            packet_tags = BaseSkill._confusion_family_tags(packet_text)
            matched = False
            if lowered_pairs and confusion_pair and confusion_pair in lowered_pairs:
                matched = True
            elif lowered_pairs and packet_tags and any(len(packet_tags.intersection(target_tags)) >= 2 for target_tags in target_pair_tags):
                matched = True
            elif lowered_keywords and all(keyword in packet_text for keyword in lowered_keywords):
                matched = True
            elif lowered_keywords and any(keyword in packet_text for keyword in lowered_keywords):
                matched = True
            if not matched:
                continue
            source_id = str(packet.get("source_id", "")).strip() or json.dumps(packet, ensure_ascii=False, sort_keys=True)
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            matches.append(dict(packet))
            if len(matches) >= top_k:
                break
        return matches

    @staticmethod
    def _confusion_family_tags(value: str) -> set[str]:
        text = str(value).strip().lower()
        tags: set[str] = set()
        if any(term in text for term in ("mel", "melanoma")):
            tags.add("mel")
        if any(term in text for term in ("nev", "naevus", "mole")):
            tags.add("nev")
        if any(term in text for term in ("ack", "actinic keratos")):
            tags.add("ack")
        if any(term in text for term in ("scc", "squamous")):
            tags.add("scc")
        if any(term in text for term in ("bcc", "basal cell")):
            tags.add("bcc")
        if any(term in text for term in ("seborrheic keratos", "sek")):
            tags.add("sek")
        if any(term in text for term in ("lichen", "psoriasis", "dermatitis", "eczema", "rosacea")):
            tags.add("inflammatory")
        return tags

    def active_related_abstract_map(self) -> dict[str, dict[str, Any]]:
        if not self._active_execution_context:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for packet in self._active_execution_context.related_abstract_experiences:
            source_id = str(packet.get("source_id", "")).strip()
            if source_id:
                result[source_id] = packet
        return result

    def perception_snapshot(self, state: CaseState) -> dict[str, Any]:
        perception = state.perception or {}
        uncertainty = perception.get("uncertainty", {}) if isinstance(perception.get("uncertainty", {}), dict) else {}
        notes = perception.get("notes", [])
        if not isinstance(notes, list):
            notes = [notes] if notes else []
        return {
            "image_summary": self._compact_value(perception.get("image_summary", "")),
            "ddx_candidates": [self._compact_value(item) for item in perception.get("ddx_candidates", [])[:4]],
            "uncertainty": {
                "level": str(uncertainty.get("level", "unknown")).lower(),
                "reasons": [self._compact_value(item) for item in uncertainty.get("reasons", [])[:3]],
            },
            "notes": [self._compact_value(item) for item in notes[:4]],
        }

    def metadata_snapshot(self, state: CaseState, *, max_fields: int = 16) -> dict[str, Any]:
        metadata = state.clinical_metadata or {}
        snapshot: dict[str, Any] = {}
        for key, value in metadata.items():
            compact_value = self._compact_value(value)
            if compact_value in ("", "unknown", "none", "false", None):
                continue
            snapshot[str(key)] = compact_value
            if len(snapshot) >= max_fields:
                break
        return snapshot

    def skill_outputs_snapshot(
        self,
        state: CaseState,
        *,
        preferred_skills: Iterable[str] | None = None,
        max_skills: int = 5,
    ) -> dict[str, Any]:
        ordered_names: list[str] = []
        for skill_name in list(preferred_skills or []):
            if skill_name in state.skill_outputs and skill_name != self.name and skill_name not in ordered_names:
                ordered_names.append(skill_name)
        for skill_name in state.skill_outputs:
            if skill_name != self.name and skill_name not in ordered_names:
                ordered_names.append(skill_name)
        ordered_names = ordered_names[:max_skills]

        snapshot: dict[str, Any] = {}
        for skill_name in ordered_names:
            raw_output = state.skill_outputs.get(skill_name, {})
            if not isinstance(raw_output, dict):
                continue
            compact_output: dict[str, Any] = {}
            for field_name, value in raw_output.items():
                if field_name == "referenced_experiences":
                    continue
                compact_value = self._compact_value(value)
                if compact_value in ("", "unknown", [], None):
                    continue
                compact_output[field_name] = compact_value
            if compact_output:
                snapshot[skill_name] = compact_output
        return snapshot

    @staticmethod
    def _compact_experience_packet(packet: dict[str, Any]) -> dict[str, Any]:
        summary = str(packet.get("perception_summary", "")).strip()
        if len(summary) > 220:
            summary = summary[:217] + "..."
        learning_points = [str(item).strip() for item in packet.get("learning_points", []) if str(item).strip()]
        return {
            "source_id": packet.get("source_id"),
            "source_layer": packet.get("source_layer"),
            "experience_type": packet.get("experience_type"),
            "confusion_pair": packet.get("confusion_pair"),
            "perception_summary": summary,
            "learning_points": learning_points[:2],
            "retrieval_score": packet.get("retrieval_score"),
        }

    @staticmethod
    def _compact_value(value: Any, *, max_length: int = 180) -> Any:
        if isinstance(value, list):
            compact_items = [BaseSkill._compact_value(item, max_length=max_length) for item in value[:4]]
            return [item for item in compact_items if item not in ("", "unknown", None)]
        if isinstance(value, dict):
            compact_dict: dict[str, Any] = {}
            for key, inner_value in list(value.items())[:8]:
                compact_inner = BaseSkill._compact_value(inner_value, max_length=max_length)
                if compact_inner in ("", "unknown", [], None):
                    continue
                compact_dict[str(key)] = compact_inner
            return compact_dict
        text = str(value).strip()
        if not text:
            return ""
        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text
