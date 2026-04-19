from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.confusion_clusters import cluster_priority_bonus, detect_confusion_clusters, get_metadata_fields, get_confusion_cluster_definitions
from cognition.cognition_state import CognitionState
from skills.schema import SkillObject


FOUNDATIONAL_SKILLS = {
    "morphology_analysis_skill",
    "color_pattern_analysis_skill",
    "border_surface_analysis_skill",
    "distribution_analysis_skill",
    "lesion_description_structuring_skill",
    "metadata_consistency_skill",
}

MALIGNANT_KEYWORDS = (
    "mel",
    "melanoma",
    "bcc",
    "basal cell",
    "scc",
    "squamous",
    "ack",
    "actinic keratosis",
    "malignant",
    "carcinoma",
)

SIGNAL_KEYWORD_MAP = {
    "foundational_observation": ("morphology", "color", "border", "surface", "distribution", "description"),
    "high_uncertainty": ("uncertainty", "ambiguity", "conflict", "contradiction", "missing"),
    "multiple_ddx": ("differential", "candidate", "compare", "comparison", "pair", "exclude", "unlikely"),
    "temporal_metadata": ("history", "temporal", "change", "growth", "progression", "bleeding", "symptom"),
    "malignancy_possible": ("risk", "malignan", "alarm", "concern"),
    "location_or_size_metadata": ("location", "distribution", "site", "diameter", "metadata", "consistency"),
    "mel_nev_confusion": ("mel", "nev", "specialist", "compare", "confusion"),
    "ack_scc_confusion": ("ack", "scc", "specialist", "compare", "confusion"),
    "ack_sek_confusion": ("ack", "seborrheic", "sek", "waxy", "stuck-on", "compare", "confusion"),
    "keratinocyte_bcc_confusion": ("bcc", "basal cell", "scc", "ack", "actinic", "seborrheic", "keratin"),
    "ham_benign_mimic_confusion": ("bkl", "benign keratosis", "nevus", "vascular", "dermatofibroma", "compare", "mimic"),
    "contradiction_rich": ("contradiction", "conflict", "audit", "inconsisten"),
    "information_gap": ("missing", "gap", "underdetermined", "need more information"),
    "escalation_needed": ("escalat", "dermoscopy", "biopsy", "further check", "closer exam"),
}


@dataclass
class SkillRetrievalQuery:
    perception: dict[str, Any]
    metadata: dict[str, Any]
    cognition: CognitionState
    dataset_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillRetrievalDecision:
    skill_id: str
    skill_name: str
    selected: bool
    score: float
    trigger_hits: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillRetrievalBundle:
    candidate_skill_ids: list[str]
    candidate_skill_names: list[str]
    skill_id_to_name: dict[str, str]
    match_reasons: dict[str, list[str]]
    trigger_hits: dict[str, list[str]]
    retrieval_scores: dict[str, float]
    decision_trace: list[dict[str, Any]]
    query_summary: dict[str, Any]
    retriever_type: str = "rule_metadata_hybrid"
    retriever_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseSkillRetriever(ABC):
    retriever_type = "base"
    retriever_version = "v0"

    @abstractmethod
    def retrieve(self, query: SkillRetrievalQuery, skills: list[SkillObject]) -> SkillRetrievalBundle:
        """Return a candidate skill set from the skill bank."""


class RuleMetadataHybridSkillRetriever(BaseSkillRetriever):
    retriever_type = "rule_metadata_hybrid"
    retriever_version = "v1"

    def __init__(self, *, enable_embedding: bool = False) -> None:
        self.enable_embedding = enable_embedding

    def retrieve(self, query: SkillRetrievalQuery, skills: list[SkillObject]) -> SkillRetrievalBundle:
        signal_profile = _build_signal_profile(query)
        query_text = _build_query_text(query, signal_profile)
        decisions = [self._evaluate_skill(skill, query, signal_profile, query_text) for skill in skills]

        selected = [decision for decision in decisions if decision.selected]
        if not selected:
            fallback = sorted(decisions, key=lambda item: (item.score, item.skill_name), reverse=True)[: max(6, min(8, len(decisions)))]
            for item in fallback:
                item.selected = True
                if "Fallback candidate because no skill exceeded retrieval threshold." not in item.reasons:
                    item.reasons.append("Fallback candidate because no skill exceeded retrieval threshold.")
            selected = fallback

        selected.sort(key=lambda item: (item.score, item.skill_name), reverse=True)
        candidate_skill_ids = [item.skill_id for item in selected]
        candidate_skill_names = [item.skill_name for item in selected]
        return SkillRetrievalBundle(
            candidate_skill_ids=candidate_skill_ids,
            candidate_skill_names=candidate_skill_names,
            skill_id_to_name={item.skill_id: item.skill_name for item in selected},
            match_reasons={item.skill_name: _dedupe(item.reasons) for item in selected},
            trigger_hits={item.skill_name: _dedupe(item.trigger_hits) for item in selected},
            retrieval_scores={item.skill_name: round(float(item.score), 4) for item in selected},
            decision_trace=[item.to_dict() for item in sorted(decisions, key=lambda item: (item.selected, item.score), reverse=True)],
            query_summary=_query_summary(query, signal_profile),
            retriever_type=self.retriever_type,
            retriever_version=self.retriever_version,
        )

    def _evaluate_skill(
        self,
        skill: SkillObject,
        query: SkillRetrievalQuery,
        signal_profile: dict[str, Any],
        query_text: str,
    ) -> SkillRetrievalDecision:
        score = 0.0
        reasons: list[str] = []
        trigger_hits: list[str] = []
        matched_fields: list[str] = []
        skill_text = _skill_text(skill)

        if skill.name in FOUNDATIONAL_SKILLS:
            score += 3.0
            reasons.append("Included as part of the core structured observation layer.")
            trigger_hits.append("foundational_observation")
            matched_fields.append("foundation")

        stats = query.cognition.skill_statistics.get(skill.name, {})
        helpful_rate = float(stats.get("helpful_rate", 0.0))
        failure_rate = float(stats.get("failure_rate", 0.0))
        if helpful_rate >= 0.5:
            score += 1.5
            reasons.append(f"Boosted by cognition helpful_rate={helpful_rate:.2f}.")
            matched_fields.append("cognition.skill_statistics.helpful_rate")
        if failure_rate >= 0.75:
            score -= 1.5
            reasons.append(f"Penalized by cognition failure_rate={failure_rate:.2f}.")
            matched_fields.append("cognition.skill_statistics.failure_rate")

        active_clusters = [str(item) for item in signal_profile.get("active_confusion_clusters", []) if str(item).strip()]
        cluster_bonus = float(cluster_priority_bonus(active_clusters, skill.name))
        if cluster_bonus > 0:
            score += cluster_bonus
            reasons.append(
                "Boosted by active confusion cluster(s): "
                + ", ".join(active_clusters)
                + f" (bonus={cluster_bonus:.1f})."
            )
            matched_fields.append("confusion_cluster.priority")

        for signal_name, active in signal_profile.items():
            if not active:
                continue
            keywords = SIGNAL_KEYWORD_MAP.get(signal_name, ())
            trigger_matched = False
            for trigger in skill.triggers:
                trigger_text = f"{trigger.condition} {trigger.rationale}".lower()
                if any(keyword in trigger_text or keyword in skill_text for keyword in keywords):
                    score += 2.0
                    trigger_hits.append(signal_name)
                    reasons.append(f"Matched active trigger profile `{signal_name}`.")
                    matched_fields.append(f"signal:{signal_name}")
                    trigger_matched = True
                    break
            if not trigger_matched and any(keyword in skill_text for keyword in keywords):
                score += 1.0
                trigger_hits.append(signal_name)
                reasons.append(f"Matched active case pattern `{signal_name}` through skill metadata.")
                matched_fields.append(f"metadata:{signal_name}")

        metadata_hits = _metadata_match_count(skill_text, query.metadata)
        if metadata_hits > 0:
            score += float(metadata_hits)
            reasons.append(f"Matched {metadata_hits} metadata clues from the current case.")
            matched_fields.append("metadata")

        morphology_hits = _morphology_match_count(skill_text, signal_profile.get("morphology_clues", []))
        if morphology_hits > 0:
            score += min(2.0, 0.5 * morphology_hits)
            reasons.append(f"Matched {morphology_hits} morphology clues from the current perception.")
            matched_fields.append("morphology")

        if self.enable_embedding:
            embedding_score = _token_similarity(query_text, skill_text)
            if embedding_score > 0.0:
                score += embedding_score
                reasons.append(f"Boosted by optional lexical-embedding similarity={embedding_score:.2f}.")
                matched_fields.append("optional_embedding")

        selected = _should_select_skill(skill.name, score, signal_profile)
        return SkillRetrievalDecision(
            skill_id=skill.skill_id,
            skill_name=skill.name,
            selected=selected,
            score=score,
            trigger_hits=_dedupe(trigger_hits),
            reasons=_dedupe(reasons),
            matched_fields=_dedupe(matched_fields),
        )


def build_default_skill_retriever(*, enable_embedding: bool = False) -> BaseSkillRetriever:
    return RuleMetadataHybridSkillRetriever(enable_embedding=enable_embedding)


def _build_signal_profile(query: SkillRetrievalQuery) -> dict[str, Any]:
    perception = query.perception
    metadata = query.metadata
    ddx_candidates = [str(item).lower() for item in perception.get("ddx_candidates", []) if str(item).strip()]
    image_summary = str(perception.get("image_summary", "")).lower()
    notes = " ".join(str(item) for item in perception.get("notes", []))
    uncertainty_level = str(perception.get("uncertainty", {}).get("level", "unknown")).lower()
    morphology_clues = _extract_morphology_clues(image_summary, metadata, notes)
    confusion_pair = detect_confusion_pair(ddx_candidates, dataset_name=query.dataset_name)
    known_confusion_text = " ".join(str(key).strip().lower() for key in query.cognition.known_confusion_patterns.keys())
    known_confusion_match = bool(
        confusion_pair
        and (
            confusion_pair in query.cognition.known_confusion_patterns
            or confusion_pair.lower() in known_confusion_text
        )
    )
    active_confusion_clusters = detect_confusion_clusters(
        ddx_candidates=ddx_candidates,
        confusion_pair=confusion_pair,
        known_confusion_text=known_confusion_text,
        image_summary=image_summary,
        notes=[str(item) for item in query.perception.get("notes", []) if str(item).strip()],
        dataset_name=query.dataset_name,
    )
    keratinocyte_precursor_present = any(
        any(term in candidate for term in ("ack", "actinic keratos", "scc", "squamous", "seborrheic", "sek"))
        for candidate in ddx_candidates
    )
    keratinocyte_bcc_confusion = (
        has_confusion_pair(ddx_candidates, ("scc", "squamous cell", "squamous"), ("bcc", "basal cell"))
        or has_confusion_pair(ddx_candidates, ("ack", "actinic keratosis", "actinic keratos"), ("bcc", "basal cell"))
        or has_confusion_pair(ddx_candidates, ("seborrheic keratosis", "sek"), ("bcc", "basal cell"))
        or (
            keratinocyte_precursor_present
            and any(
                pattern in known_confusion_text
                for pattern in (
                    "squamous cell carcinoma->bcc",
                    "scc->bcc",
                    "actinic keratosis->bcc",
                    "ack->bcc",
                    "seborrheic keratosis->bcc",
                    "sek->bcc",
                )
            )
            and (uncertainty_level in {"high", "medium"} or has_malignancy_possibility(ddx_candidates))
        )
    )
    ham_benign_mimic_confusion = bool(
        any(c in active_confusion_clusters for c in ("bkl_nv", "mel_bkl"))
        or has_confusion_pair(ddx_candidates, ("bkl", "benign keratosis", "seborrheic keratosis"), ("nv", "nevus"))
        or has_confusion_pair(ddx_candidates, ("bkl", "benign keratosis", "seborrheic keratosis"), ("mel", "melanoma"))
        or has_confusion_pair(ddx_candidates, ("df", "dermatofibroma"), ("mel", "melanoma", "nv", "nevus"))
        or has_confusion_pair(ddx_candidates, ("vasc", "vascular"), ("mel", "melanoma", "nv", "nevus"))
    )

    contradiction_rich = any(term in image_summary for term in ("contradict", "conflict", "irregular", "asymmetry"))
    information_gap = any(term in notes.lower() for term in ("missing", "unknown", "not available")) or uncertainty_level in {"medium", "high"}
    escalation_needed = uncertainty_level == "high" or has_malignancy_possibility(ddx_candidates)

    return {
        "ddx_candidates": ddx_candidates,
        "morphology_clues": morphology_clues,
        "uncertainty_level": uncertainty_level,
        "confusion_pair": confusion_pair,
        "foundational_observation": True,
        "high_uncertainty": uncertainty_level == "high",
        "multiple_ddx": len(ddx_candidates) >= 2,
        "temporal_metadata": any(
            str(metadata.get(field, "")).strip()
            for field in get_metadata_fields(query.dataset_name, "temporal")
        ),
        "location_or_size_metadata": any(
            str(metadata.get(field, "")).strip()
            for field in get_metadata_fields(query.dataset_name, "location_size")
        ),
        "malignancy_possible": has_malignancy_possibility(ddx_candidates),
        "mel_nev_confusion": any(c in active_confusion_clusters for c in ("mel_nev", "mel_nv")),
        "ack_scc_confusion": any(c in active_confusion_clusters for c in ("ack_bcc_scc", "ack_scc")),
        "ack_sek_confusion": "ack_sek" in active_confusion_clusters,
        "keratinocyte_bcc_confusion": keratinocyte_bcc_confusion,
        "ham_benign_mimic_confusion": ham_benign_mimic_confusion,
        "contradiction_rich": contradiction_rich,
        "information_gap": information_gap,
        "escalation_needed": escalation_needed,
        "known_confusion_match": known_confusion_match,
        "active_confusion_clusters": active_confusion_clusters,
    }


def _query_summary(query: SkillRetrievalQuery, signal_profile: dict[str, Any]) -> dict[str, Any]:
    metadata_patterns = [
        field_name
        for field_name in (
            get_metadata_fields(query.dataset_name, "location_size")
            + get_metadata_fields(query.dataset_name, "temporal")
        )
        if str(query.metadata.get(field_name, "")).strip()
    ]
    risk_patterns = []
    if signal_profile.get("malignancy_possible"):
        risk_patterns.append("malignancy_possible")
    if signal_profile.get("escalation_needed"):
        risk_patterns.append("escalation_needed")
    return {
        "ddx_candidates": list(signal_profile.get("ddx_candidates", [])),
        "morphology_clues": list(signal_profile.get("morphology_clues", [])),
        "metadata_patterns": metadata_patterns,
        "risk_patterns": risk_patterns,
        "uncertainty_level": signal_profile.get("uncertainty_level", "unknown"),
        "confusion_pair": signal_profile.get("confusion_pair"),
        "confusion_clusters": list(signal_profile.get("active_confusion_clusters", [])),
    }


def _build_query_text(query: SkillRetrievalQuery, signal_profile: dict[str, Any]) -> str:
    parts = [
        str(query.perception.get("image_summary", "")),
        " ".join(str(item) for item in query.perception.get("ddx_candidates", [])),
        " ".join(f"{key}:{value}" for key, value in query.metadata.items() if str(value).strip()),
        " ".join(signal_profile.get("morphology_clues", [])),
        str(signal_profile.get("confusion_pair") or ""),
        str(signal_profile.get("uncertainty_level", "unknown")),
    ]
    return " ".join(part.lower() for part in parts if part).strip()


def _extract_morphology_clues(image_summary: str, metadata: dict[str, Any], notes: str) -> list[str]:
    source = f"{image_summary} {' '.join(f'{key}:{value}' for key, value in metadata.items() if str(value).strip())} {notes}".lower()
    clue_terms = (
        "flat",
        "raised",
        "papule",
        "plaque",
        "nodule",
        "macule",
        "brown",
        "black",
        "pink",
        "red",
        "irregular",
        "regular",
        "asymmetric",
        "symmetric",
        "border",
        "surface",
        "scale",
        "keratotic",
        "ulcer",
        "solitary",
        "multiple",
        "arm",
        "face",
        "trunk",
        "leg",
    )
    return [term for term in clue_terms if term in source]


def _metadata_match_count(skill_text: str, metadata: dict[str, Any]) -> int:
    score = 0
    if str(metadata.get("region", "")).strip() and any(term in skill_text for term in ("location", "distribution", "site", "metadata")):
        score += 1
    if any(str(metadata.get(field, "")).strip() for field in ("diameter_1", "diameter_2")) and any(term in skill_text for term in ("size", "diameter", "count")):
        score += 1
    if any(str(metadata.get(field, "")).strip() for field in ("grew", "changed", "bleed", "itch", "hurt", "elevation")) and any(
        term in skill_text for term in ("temporal", "history", "evolution", "symptom", "metadata")
    ):
        score += 1
    if str(metadata.get("age", "")).strip() and "metadata" in skill_text:
        score += 1
    return score


def _morphology_match_count(skill_text: str, morphology_clues: list[str]) -> int:
    return sum(1 for clue in morphology_clues if clue in skill_text)


def _skill_text(skill: SkillObject) -> str:
    trigger_text = " ".join(f"{trigger.condition} {trigger.rationale}" for trigger in skill.triggers)
    watch_out_text = " ".join(skill.watch_outs)
    step_text = " ".join(f"{step.title} {step.instruction}" for step in skill.steps)
    return f"{skill.name} {skill.description} {skill.skill_type} {trigger_text} {skill.workflow_text} {watch_out_text} {step_text}".lower()


def _should_select_skill(skill_name: str, score: float, signal_profile: dict[str, Any]) -> bool:
    if skill_name in FOUNDATIONAL_SKILLS:
        return score >= 3.0
    if skill_name == "temporal_evolution_skill":
        return bool(signal_profile["temporal_metadata"] or score >= 3.0)
    if skill_name == "malignancy_risk_assessment_skill":
        return bool(signal_profile["malignancy_possible"] or score >= 3.5)
    if skill_name == "differential_compare_skill":
        return bool(signal_profile["multiple_ddx"] or score >= 3.5)
    if skill_name == "exclusion_reasoning_skill":
        return bool(
            signal_profile["multiple_ddx"]
            or signal_profile["information_gap"]
            or signal_profile.get("active_confusion_clusters")
            or score >= 3.5
        )
    if skill_name == "information_gap_detection_skill":
        return bool(signal_profile["high_uncertainty"] or signal_profile["information_gap"] or score >= 3.5)
    if skill_name == "mel_nev_specialist_skill":
        return bool(signal_profile["mel_nev_confusion"] or (signal_profile["known_confusion_match"] and score >= 2.5))
    if skill_name == "ack_scc_specialist_skill":
        return bool(
            signal_profile["ack_scc_confusion"]
            or signal_profile.get("ack_sek_confusion", False)
            or signal_profile.get("keratinocyte_bcc_confusion", False)
            or (signal_profile["known_confusion_match"] and score >= 2.5)
        )
    if skill_name == "benign_mimic_specialist_skill":
        return bool(
            signal_profile.get("ham_benign_mimic_confusion", False)
            or (signal_profile["known_confusion_match"] and score >= 2.5)
        )
    if skill_name == "uncertainty_assessment_skill":
        return bool(signal_profile["high_uncertainty"] or signal_profile["information_gap"] or score >= 3.5)
    if skill_name == "contradiction_check_skill":
        return bool(signal_profile["contradiction_rich"] or signal_profile["multiple_ddx"] or score >= 3.5)
    if skill_name == "escalation_recommendation_skill":
        return bool(signal_profile["escalation_needed"] or signal_profile["high_uncertainty"] or score >= 3.5)
    return score >= 3.5


def detect_confusion_pair(ddx_candidates: list[str], dataset_name: str | None = None) -> str | None:
    clusters = detect_confusion_clusters(ddx_candidates=ddx_candidates, dataset_name=dataset_name)
    if not clusters:
        return None
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in clusters:
        pairs = list(cluster_defs.get(cluster_name, {}).get("pairs", ()))
        if pairs:
            return str(pairs[0]).strip().lower()
    return None


def has_malignancy_possibility(ddx_candidates: list[str]) -> bool:
    return any(any(keyword in candidate for keyword in MALIGNANT_KEYWORDS) for candidate in ddx_candidates)


def has_confusion_pair(
    ddx_candidates: list[str],
    left_keywords: tuple[str, ...],
    right_keywords: tuple[str, ...],
) -> bool:
    has_left = any(any(keyword in candidate for keyword in left_keywords) for candidate in ddx_candidates)
    has_right = any(any(keyword in candidate for keyword in right_keywords) for candidate in ddx_candidates)
    return has_left and has_right


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    if union == 0:
        return 0.0
    return round(2.0 * (intersection / union), 4)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_]+", str(text).lower()) if token]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
