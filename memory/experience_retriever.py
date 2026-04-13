from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.confusion_clusters import cluster_match_bonus, detect_confusion_clusters
from agent.state import CaseState
from memory.experience_store import ExperienceStore
from memory.experience_transform import (
    abstract_to_retrieval_packet,
    raw_case_to_retrieval_packet,
    tactical_to_retrieval_packet,
)


@dataclass
class RetrievalQuery:
    ddx_candidates: list[str] = field(default_factory=list)
    morphology_clues: list[str] = field(default_factory=list)
    metadata_patterns: list[str] = field(default_factory=list)
    risk_patterns: list[str] = field(default_factory=list)
    confusion_pair: str | None = None
    confusion_clusters: list[str] = field(default_factory=list)
    uncertainty_level: str = "unknown"
    image_summary: str = ""
    top_k_raw: int = 2
    top_k_tactical: int = 4
    top_k_abstract: int = 4
    top_k_merged: int = 6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceRetriever:
    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def build_query(
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
    ) -> RetrievalQuery:
        query_perception = perception or (case_state.perception if case_state else {})
        query_metadata = metadata or (case_state.clinical_metadata if case_state else {})
        query_risk_flags = risk_flags or (case_state.risk_flags if case_state else [])
        query_uncertainty = uncertainty or (case_state.uncertainty if case_state else {})
        resolved_ddx = (
            [str(item) for item in ddx_candidates]
            if ddx_candidates is not None
            else [str(item) for item in query_perception.get("ddx_candidates", [])]
        )
        resolved_confusion_pair = confusion_pair or self._detect_confusion_pair([item.lower() for item in resolved_ddx])
        resolved_morphology = morphology_clues or self._extract_morphology_clues(case_state=case_state, perception=query_perception)
        resolved_confusion_clusters = detect_confusion_clusters(
            ddx_candidates=[str(item).lower() for item in resolved_ddx],
            confusion_pair=resolved_confusion_pair,
            image_summary=str(query_perception.get("image_summary", "")),
            notes=[str(item) for item in query_perception.get("notes", []) if str(item).strip()],
        )

        return RetrievalQuery(
            ddx_candidates=dedupe_strings(resolved_ddx),
            morphology_clues=dedupe_strings(resolved_morphology),
            metadata_patterns=dedupe_strings(self._extract_metadata_patterns(query_metadata)),
            risk_patterns=dedupe_strings(self._extract_risk_patterns(query_perception, query_metadata, query_risk_flags)),
            confusion_pair=resolved_confusion_pair,
            confusion_clusters=resolved_confusion_clusters,
            uncertainty_level=str(
                query_uncertainty.get("uncertainty_level", query_perception.get("uncertainty", {}).get("level", "unknown"))
            ).lower(),
            image_summary=str(query_perception.get("image_summary", "")),
            top_k_raw=top_k_raw,
            top_k_tactical=top_k_tactical,
            top_k_abstract=top_k_abstract,
            top_k_merged=top_k_merged,
        )

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
        query = self.build_query(
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

        abstract_results = self._retrieve_abstract(query)
        tactical_results = self._retrieve_tactical(query)
        raw_case_results = self._retrieve_raw(query)
        merged_results = self._merge_results(
            abstract_results=abstract_results,
            tactical_results=tactical_results,
            raw_case_results=raw_case_results,
            top_k=query.top_k_merged,
        )

        planner_summary = self._build_summary_slice(
            primary=tactical_results,
            secondary=abstract_results,
            tertiary=raw_case_results,
            top_k=query.top_k_tactical,
        )
        skill_summary = self._build_summary_slice(
            primary=tactical_results,
            secondary=abstract_results,
            tertiary=raw_case_results,
            top_k=query.top_k_tactical,
        )
        aggregator_summary = merged_results[: query.top_k_merged]

        return {
            "query": query.to_dict(),
            "raw_case_results": raw_case_results,
            "tactical_results": tactical_results,
            "abstract_results": abstract_results,
            "planner_summary": planner_summary,
            "skill_summary": skill_summary,
            "aggregator_summary": aggregator_summary,
            "merged_results": merged_results,
        }

    def retrieve(
        self,
        *,
        case_state: CaseState | None = None,
        top_k: int = 3,
        perception: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        bundle = self.retrieve_bundle(
            case_state=case_state,
            perception=perception,
            top_k_merged=top_k,
            top_k_tactical=max(top_k, 3),
            top_k_abstract=max(top_k, 3),
            top_k_raw=max(2, min(3, top_k)),
        )
        return list(bundle.get("merged_results", []))[:top_k]

    def _retrieve_tactical(self, query: RetrievalQuery) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.store.load_tactical_experiences():
            score = self._score_tactical(record, query)
            if score <= 0:
                continue
            packet = tactical_to_retrieval_packet(record)
            packet["retrieval_score"] = score
            scored.append((score, packet))
        return self._deduplicate_sorted_packets(scored, top_k=query.top_k_tactical)

    def _retrieve_abstract(self, query: RetrievalQuery) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.store.load_abstract_experiences():
            score = self._score_abstract(record, query)
            if score <= 0:
                continue
            packet = abstract_to_retrieval_packet(record)
            packet["retrieval_score"] = score
            scored.append((score, packet))
        return self._deduplicate_sorted_packets(scored, top_k=query.top_k_abstract)

    def _retrieve_raw(self, query: RetrievalQuery) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.store.load_raw_case_memories():
            score = self._score_raw(record, query)
            if score <= 0:
                continue
            packet = raw_case_to_retrieval_packet(record)
            packet["retrieval_score"] = score
            scored.append((score, packet))
        return self._deduplicate_sorted_packets(scored, top_k=query.top_k_raw)

    @staticmethod
    def _score_tactical(record: dict[str, Any], query: RetrievalQuery) -> int:
        score = 0
        condition = record.get("condition", {})
        observed_state = condition.get("observed_state", {})
        text = json.dumps(record, ensure_ascii=False).lower()

        for candidate_text in query.ddx_candidates:
            lowered = candidate_text.lower()
            if lowered and lowered in text:
                score += 3

        image_summary = query.image_summary.lower()
        if image_summary and image_summary in text:
            score += 2

        for clue in query.morphology_clues[:4]:
            lowered = clue.lower()
            if lowered and lowered in text:
                score += 2

        for pattern in query.metadata_patterns[:4]:
            lowered = pattern.lower()
            if lowered and lowered in text:
                score += 1

        for risk_flag in query.risk_patterns:
            if risk_flag in observed_state.get("risk_flags", []) or risk_flag.lower() in text:
                score += 2

        uncertainty_level = query.uncertainty_level
        if uncertainty_level and uncertainty_level != "unknown" and observed_state.get("uncertainty_level") == uncertainty_level:
            score += 1

        if query.confusion_pair:
            query_pair = _normalize_confusion_pair(query.confusion_pair)
            record_pair = _normalize_confusion_pair(str(condition.get("confusion_pair", "")))
            if query_pair and record_pair and query_pair == record_pair:
                score += 6
            elif query_pair and record_pair and _is_same_confusion_family(query_pair, record_pair):
                score += 3
        if query.confusion_clusters:
            score += int(round(cluster_match_bonus(cluster_names=query.confusion_clusters, text=text, subtype="tactical_experience")))

        if condition.get("trigger_type") == "confusion_pair":
            score += 2
        if record.get("reusable_scope", {}).get("priority") == "high":
            score += 1
        return score

    @staticmethod
    def _score_abstract(record: dict[str, Any], query: RetrievalQuery) -> int:
        score = 0
        text = json.dumps(record, ensure_ascii=False).lower()
        for candidate_text in query.ddx_candidates:
            lowered = candidate_text.lower()
            if lowered and lowered in text:
                score += 3

        for clue in query.morphology_clues[:4]:
            lowered = clue.lower()
            if lowered and lowered in text:
                score += 1

        for pattern in query.metadata_patterns[:4]:
            lowered = pattern.lower()
            if lowered and lowered in text:
                score += 1

        for risk_flag in query.risk_patterns:
            if risk_flag.lower() in text:
                score += 2

        uncertainty_level = query.uncertainty_level
        if uncertainty_level and uncertainty_level != "unknown" and uncertainty_level in text:
            score += 1

        if query.confusion_pair:
            query_pair = _normalize_confusion_pair(query.confusion_pair)
            record_pair = _normalize_confusion_pair(str(record.get("pattern_summary", {}).get("confusion_pair", "")))
            if query_pair and record_pair and query_pair == record_pair:
                score += 6
            elif query_pair and record_pair and _is_same_confusion_family(query_pair, record_pair):
                score += 3
            elif query_pair and query_pair in text:
                score += 3

        record_type = str(record.get("type", "")).strip()
        if record_type == "confusion_memory":
            score += 3
            if query.confusion_pair:
                query_tags = _confusion_family_tags(str(query.confusion_pair))
                record_tags = _confusion_family_tags(text)
                if len(query_tags.intersection(record_tags)) >= 2:
                    score += 2
        elif record_type in {"rule", "rule_candidate", "composite_skill_seed"}:
            score += 2
        elif record_type == "prototype":
            score += 1
        if query.confusion_clusters:
            score += int(round(cluster_match_bonus(cluster_names=query.confusion_clusters, text=text, subtype=record_type)))
        return score

    @staticmethod
    def _score_raw(record: dict[str, Any], query: RetrievalQuery) -> int:
        score = 0
        text = json.dumps(record, ensure_ascii=False).lower()
        for candidate_text in query.ddx_candidates:
            lowered = candidate_text.lower()
            if lowered and lowered in text:
                score += 2

        image_summary = query.image_summary.lower()
        if image_summary and image_summary in text:
            score += 1

        for clue in query.morphology_clues[:3]:
            lowered = clue.lower()
            if lowered and lowered in text:
                score += 1

        for pattern in query.metadata_patterns[:4]:
            lowered = pattern.lower()
            if lowered and lowered in text:
                score += 1

        for risk_flag in query.risk_patterns:
            if risk_flag.lower() in text:
                score += 1

        uncertainty_level = query.uncertainty_level
        if uncertainty_level and uncertainty_level != "unknown" and uncertainty_level in text:
            score += 1
        if query.confusion_pair and query.confusion_pair.lower() in text:
            score += 1
        if query.confusion_clusters:
            score += int(round(cluster_match_bonus(cluster_names=query.confusion_clusters, text=text, subtype="raw_case_memory")))
        return score

    @staticmethod
    def _merge_results(
        *,
        abstract_results: list[dict[str, Any]],
        tactical_results: list[dict[str, Any]],
        raw_case_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        combined.extend(abstract_results)
        combined.extend(tactical_results)
        combined.extend(raw_case_results)
        scored = [(int(packet.get("retrieval_score", 0)), _layer_priority(packet), packet) for packet in combined]
        return ExperienceRetriever._deduplicate_sorted_packets(scored, top_k=top_k)

    @staticmethod
    def _build_summary_slice(
        *,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        tertiary: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for packet in primary:
            scored.append((int(packet.get("retrieval_score", 0)) + 2, _layer_priority(packet), packet))
        for packet in secondary:
            scored.append((int(packet.get("retrieval_score", 0)) + 1, _layer_priority(packet), packet))
        for packet in tertiary:
            scored.append((int(packet.get("retrieval_score", 0)), _layer_priority(packet), packet))
        return ExperienceRetriever._deduplicate_sorted_packets(scored, top_k=top_k)

    @staticmethod
    def _deduplicate_sorted_packets(
        scored_packets: list[tuple[int, int | dict[str, Any], dict[str, Any] | None]] | list[tuple[int, dict[str, Any]]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        normalized: list[tuple[int, int, dict[str, Any]]] = []
        for item in scored_packets:
            if len(item) == 2:
                score, packet = item  # type: ignore[misc]
                normalized.append((int(score), _layer_priority(packet), packet))
            else:
                score, priority, packet = item  # type: ignore[misc]
                normalized.append((int(score), int(priority), packet))
        normalized.sort(key=lambda item: (item[0], item[1], str(item[2].get("source_id", ""))), reverse=True)

        deduplicated: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for score, _, packet in normalized:
            source_id = str(packet.get("source_id", "")).strip() or json.dumps(packet, ensure_ascii=False, sort_keys=True)
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            packet = dict(packet)
            packet["retrieval_score"] = score
            deduplicated.append(packet)
            if len(deduplicated) >= top_k:
                break
        return deduplicated

    @staticmethod
    def _extract_morphology_clues(
        *,
        case_state: CaseState | None,
        perception: dict[str, Any],
    ) -> list[str]:
        clues: list[str] = []
        image_summary = str(perception.get("image_summary", "")).strip()
        if image_summary:
            clues.append(image_summary)
        if case_state:
            skill_fields = {
                "morphology_analysis_skill": ("lesion_type", "size_range", "elevation", "count"),
                "color_pattern_analysis_skill": (
                    "primary_color",
                    "color_variation",
                    "pigmentation_pattern",
                    "asymmetry_color",
                ),
                "border_surface_analysis_skill": (
                    "border_clarity",
                    "border_irregularity",
                    "surface_texture",
                    "scaling_presence",
                ),
                "lesion_description_structuring_skill": (
                    "primary_lesion_morphology",
                    "color",
                    "border",
                    "surface",
                    "size_count",
                    "distribution",
                ),
            }
            for skill_name, allowed_fields in skill_fields.items():
                output = case_state.skill_outputs.get(skill_name, {})
                if isinstance(output, dict):
                    for field_name in allowed_fields:
                        value = output.get(field_name)
                        if isinstance(value, list):
                            for item in value[:3]:
                                text = str(item).strip()
                                if text and text.lower() != "unknown":
                                    clues.append(text)
                        else:
                            text = str(value).strip()
                            if text and text.lower() != "unknown":
                                clues.append(text)
        return clues

    @staticmethod
    def _extract_metadata_patterns(metadata: dict[str, Any]) -> list[str]:
        patterns: list[str] = []
        for field in ("region", "age", "diameter_1", "diameter_2", "grew", "changed", "bleed", "itch", "hurt", "elevation"):
            raw_value = metadata.get(field, "")
            value = str(raw_value).strip()
            if _is_meaningful_metadata_value(raw_value, value):
                patterns.append(f"{field}:{value}")
                patterns.append(value)
        return patterns

    @staticmethod
    def _extract_risk_patterns(
        perception: dict[str, Any],
        metadata: dict[str, Any],
        risk_flags: list[str],
    ) -> list[str]:
        patterns = list(risk_flags)
        for candidate in perception.get("ddx_candidates", []):
            candidate_text = str(candidate).lower()
            if any(keyword in candidate_text for keyword in ("mel", "melanoma", "bcc", "scc", "ack", "carcinoma")):
                patterns.append("malignancy_possible")
        for field in ("changed", "bleed", "hurt"):
            raw_value = metadata.get(field, "")
            value = str(raw_value).strip()
            if _is_meaningful_metadata_value(raw_value, value):
                patterns.append(f"{field}:{value}")
        return patterns

    @staticmethod
    def _detect_confusion_pair(ddx_candidates: list[str]) -> str | None:
        if ExperienceRetriever._has_confusion_pair(ddx_candidates, ("mel", "melanoma"), ("nev", "nevus", "naevus", "mole")):
            return "melanoma->nev"
        if ExperienceRetriever._has_confusion_pair(ddx_candidates, ("scc", "squamous cell", "squamous"), ("bcc", "basal cell")):
            return "scc->bcc"
        if ExperienceRetriever._has_confusion_pair(
            ddx_candidates,
            ("ack", "actinic keratosis", "actinic keratos"),
            ("bcc", "basal cell"),
        ):
            return "ack->bcc"
        if ExperienceRetriever._has_confusion_pair(
            ddx_candidates,
            ("seborrheic keratosis", "sek"),
            ("bcc", "basal cell"),
        ):
            return "sek->bcc"
        if ExperienceRetriever._has_confusion_pair(
            ddx_candidates,
            ("lichen simplex", "lichen planus", "psoriasis", "dermatitis", "eczema"),
            ("ack", "actinic keratosis", "actinic keratos"),
        ):
            return "inflammatory->ack"
        if ExperienceRetriever._has_confusion_pair(
            ddx_candidates,
            ("ack", "actinic keratosis", "actinic keratos"),
            ("scc", "squamous cell", "squamous"),
        ):
            return "ack->scc"
        return None

    @staticmethod
    def _has_confusion_pair(
        ddx_candidates: list[str],
        left_keywords: tuple[str, ...],
        right_keywords: tuple[str, ...],
    ) -> bool:
        has_left = any(any(keyword in candidate for keyword in left_keywords) for candidate in ddx_candidates)
        has_right = any(any(keyword in candidate for keyword in right_keywords) for candidate in ddx_candidates)
        return has_left and has_right


def _layer_priority(packet: dict[str, Any]) -> int:
    layer = str(packet.get("source_layer", ""))
    if layer == "abstract_experience":
        return 3
    if layer == "tactical_experience":
        return 2
    if layer == "raw_case_memory":
        return 1
    return 0


def _normalize_confusion_pair(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace(" vs ", "->").replace("_vs_", "->").replace(" vs.", "->")
    text = text.replace("→", "->")
    while "-->" in text:
        text = text.replace("-->", "->")
    if "->" not in text:
        return text
    left, right = [part.strip() for part in text.split("->", 1)]
    return f"{left}->{right}"


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


def _is_same_confusion_family(left: str, right: str) -> bool:
    left_tags = _confusion_family_tags(left)
    right_tags = _confusion_family_tags(right)
    if not left_tags or not right_tags:
        return False
    return len(left_tags.intersection(right_tags)) >= 2


def dedupe_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _is_meaningful_metadata_value(raw_value: Any, text: str) -> bool:
    if raw_value in (None, "", False):
        return False
    lowered = text.lower()
    if lowered in {"", "false", "none", "unknown", "nan"}:
        return False
    return True
