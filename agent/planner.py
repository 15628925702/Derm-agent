from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from cognition.cognition_state import CognitionState
from skills.schema import SkillObject

try:
    from agent.supervised_controller import LearnedControllerScorer
except Exception:  # pragma: no cover - fallback for minimal runtime environments without torch
    LearnedControllerScorer = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


FOUNDATIONAL_SKILLS = {
    "morphology_analysis_skill",
    "color_pattern_analysis_skill",
    "border_surface_analysis_skill",
    "distribution_analysis_skill",
    "lesion_description_structuring_skill",
    "metadata_consistency_skill",
}

ORDERING_HINTS = {
    "morphology_analysis_skill": 10,
    "color_pattern_analysis_skill": 20,
    "border_surface_analysis_skill": 30,
    "distribution_analysis_skill": 40,
    "lesion_description_structuring_skill": 45,
    "metadata_consistency_skill": 50,
    "temporal_evolution_skill": 60,
    "malignancy_risk_assessment_skill": 70,
    "differential_compare_skill": 80,
    "exclusion_reasoning_skill": 95,
    "information_gap_detection_skill": 98,
    "mel_nev_specialist_skill": 90,
    "ack_scc_specialist_skill": 91,
    "uncertainty_assessment_skill": 100,
    "contradiction_check_skill": 110,
    "escalation_recommendation_skill": 120,
}

SIGNAL_KEYWORD_MAP = {
    "high_uncertainty": ("uncertainty", "ambiguity", "conflict", "contradiction", "missing"),
    "multiple_ddx": ("differential", "candidate", "compare", "comparison", "pair", "exclude", "unlikely"),
    "temporal_metadata": ("history", "temporal", "change", "growth", "progression", "bleeding", "symptom"),
    "malignancy_possible": ("risk", "malignan", "alarm", "concern"),
    "location_or_size_metadata": ("location", "distribution", "site", "diameter", "metadata", "consistency"),
    "mel_nev_confusion": ("mel", "nev", "specialist", "compare", "confusion"),
    "ack_scc_confusion": ("ack", "scc", "specialist", "compare", "confusion"),
    "keratinocyte_bcc_confusion": ("bcc", "basal cell", "scc", "ack", "actinic", "seborrheic", "keratin"),
    "experience_compare_pattern": ("compare", "differential", "confusion", "specialist"),
    "experience_risk_pattern": ("risk", "alarm", "uncertainty"),
    "experience_gap_pattern": ("missing", "gap", "underdetermined", "what information", "uncertainty"),
    "experience_conflict_pattern": ("conflict", "contradiction", "inconsisten", "audit"),
    "experience_escalation_pattern": ("escalat", "dermoscopy", "biopsy", "closer exam", "further check", "urgent"),
}


@dataclass
class PlannerInput:
    perception: dict[str, Any]
    metadata: dict[str, Any]
    cognition: CognitionState
    retrieved_experience_summary: list[dict[str, Any]]
    available_skills: list[SkillObject]
    retrieved_experience_bundle: dict[str, Any] = field(default_factory=dict)
    skill_retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    policy_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "perception": self.perception,
            "metadata": self.metadata,
            "cognition": self.cognition.to_dict(),
            "retrieved_experience_summary": list(self.retrieved_experience_summary),
            "retrieved_experience_bundle": dict(self.retrieved_experience_bundle),
            "skill_retrieval_bundle": dict(self.skill_retrieval_bundle),
            "policy_config": dict(self.policy_config),
            "available_skills": [skill.to_dict() for skill in self.available_skills],
        }


@dataclass
class SkillSelectionDecision:
    skill_name: str
    selected: bool
    score: int
    reasons: list[str] = field(default_factory=list)
    ordering_hint: int = 999
    matched_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerOutput:
    available_skill_candidates: list[str]
    selected_skills: list[str]
    selection_reasons: dict[str, list[str]]
    ordering: list[str]
    decision_trace: list[dict[str, Any]]
    planner_type: str = "rule_based"
    planner_version: str = "v1"
    controller_family: str = "heuristic"
    controller_training_ready: bool = True
    policy_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseSkillPlanner(ABC):
    planner_type = "base"
    planner_version = "v0"

    @abstractmethod
    def plan(self, planner_input: PlannerInput) -> PlannerOutput:
        """Select skills without making the final diagnosis."""


class RuleBasedSkillPlanner(BaseSkillPlanner):
    planner_type = "rule_based"
    planner_version = "v1"

    def __init__(self, policy_config: dict[str, Any] | None = None) -> None:
        self.policy_config = _normalize_planner_policy(policy_config)
        self.learned_controller: LearnedControllerScorer | None = None
        self._load_learned_controller_if_needed(self.policy_config)

    def plan(self, planner_input: PlannerInput) -> PlannerOutput:
        policy = _normalize_planner_policy(planner_input.policy_config or self.policy_config)
        self._load_learned_controller_if_needed(policy)
        signals = _build_signal_profile(planner_input)
        learned_prediction = None
        if self.learned_controller is not None and str(policy.get("controller_family", "")).strip() == "learned_supervised":
            learned_prediction = self.learned_controller.score_from_planner_input(
                perception=planner_input.perception,
                metadata=planner_input.metadata,
                skill_retrieval_bundle=planner_input.skill_retrieval_bundle,
                retrieved_experience_bundle=planner_input.retrieved_experience_bundle,
                available_skill_names=[skill.name for skill in planner_input.available_skills],
                cognition=planner_input.cognition,
            )
        learned_selected_set = set(learned_prediction.selected_skills) if learned_prediction is not None else set()
        learned_rank_map = {
            skill_name: rank
            for rank, skill_name in enumerate(learned_prediction.ranked_skills)
        } if learned_prediction is not None else {}
        decisions: list[SkillSelectionDecision] = []
        for skill in planner_input.available_skills:
            probability = None
            if learned_prediction is not None:
                probability = float(learned_prediction.skill_probabilities.get(skill.name, 0.0))
            decisions.append(
                self._evaluate_skill(
                    skill,
                    planner_input,
                    signals,
                    policy,
                    learned_probability=probability,
                    learned_selected=skill.name in learned_selected_set,
                    learned_rank=learned_rank_map.get(skill.name),
                )
            )

        selected = [decision for decision in decisions if decision.selected]
        force_top_k = int(policy.get("learned_controller_force_top_k", 0) or 0)
        if (
            not selected
            and learned_prediction is not None
            and force_top_k > 0
            and str(policy.get("controller_family", "")).strip() == "learned_supervised"
        ):
            fallback_skills = set(learned_prediction.ranked_skills[:force_top_k])
            for decision in decisions:
                if decision.skill_name in fallback_skills:
                    decision.selected = True
                    decision.reasons = dedupe_reasons(
                        decision.reasons + [f"Forced selected by learned controller top-{force_top_k} fallback."]
                    )
                    decision.matched_fields = dedupe_reasons(
                        decision.matched_fields + ["learned_controller.force_top_k"]
                    )
            selected = [decision for decision in decisions if decision.selected]
        selected.sort(key=lambda item: (item.ordering_hint, -item.score, item.skill_name))
        ordering = [decision.skill_name for decision in selected]
        selection_reasons = {decision.skill_name: decision.reasons for decision in selected}

        return PlannerOutput(
            available_skill_candidates=[skill.name for skill in planner_input.available_skills],
            selected_skills=ordering,
            selection_reasons=selection_reasons,
            ordering=ordering,
            decision_trace=[decision.to_dict() for decision in sorted(decisions, key=lambda item: item.ordering_hint)],
            planner_type=self.planner_type,
            planner_version=self.planner_version,
            controller_family=str(policy.get("controller_family", "heuristic")),
            controller_training_ready=True,
            policy_id=str(policy.get("_policy_id", "")).strip(),
        )

    def _load_learned_controller_if_needed(self, policy: dict[str, Any]) -> None:
        controller_family = str(policy.get("controller_family", "heuristic")).strip()
        checkpoint_path = str(policy.get("controller_checkpoint_path", "")).strip()
        if controller_family != "learned_supervised" or not checkpoint_path:
            self.learned_controller = None
            return
        if LearnedControllerScorer is None:
            LOGGER.warning(
                "Learned controller requested but dependencies are unavailable. Falling back to rule-based planner."
            )
            self.learned_controller = None
            return
        if self.learned_controller and self.learned_controller.checkpoint_path == checkpoint_path:
            return
        try:
            self.learned_controller = LearnedControllerScorer(
                checkpoint_path=checkpoint_path,
                threshold=float(policy.get("learned_controller_select_threshold", 0.4)),
                top_k=int(policy.get("learned_controller_top_k", 0) or 0),
            )
        except Exception as exc:
            LOGGER.warning("Failed to load learned controller checkpoint `%s`: %s. Falling back to rule-based planner.", checkpoint_path, exc)
            self.learned_controller = None

    def _evaluate_skill(
        self,
        skill: SkillObject,
        planner_input: PlannerInput,
        signals: dict[str, Any],
        policy: dict[str, Any],
        *,
        learned_probability: float | None = None,
        learned_selected: bool = False,
        learned_rank: int | None = None,
    ) -> SkillSelectionDecision:
        score = 0
        reasons: list[str] = []
        matched_fields: list[str] = []
        skill_text = _skill_text(skill)
        skill_overrides = dict(policy.get("skill_overrides", {}) or {}).get(skill.name, {})
        force_disable = set(policy.get("force_disable_skills", []) or [])
        force_select = set(policy.get("force_select_skills", []) or [])

        if skill.name in force_disable or skill_overrides.get("enabled") is False:
            reasons.append("Disabled by current planner policy.")
            return SkillSelectionDecision(
                skill_name=skill.name,
                selected=False,
                score=-999,
                reasons=dedupe_reasons(reasons),
                ordering_hint=ORDERING_HINTS.get(skill.name, 999),
                matched_fields=["policy.force_disable"],
            )

        retrieval_scores = planner_input.skill_retrieval_bundle.get("retrieval_scores", {})
        retrieval_reasons = planner_input.skill_retrieval_bundle.get("match_reasons", {})
        retrieval_trigger_hits = planner_input.skill_retrieval_bundle.get("trigger_hits", {})
        retrieval_score = float(retrieval_scores.get(skill.name, 0.0))
        if retrieval_score > 0.0:
            retrieval_cap = int(policy.get("retrieval_score_cap", 3))
            score += min(retrieval_cap, int(round(retrieval_score)))
            reasons.append(f"Boosted by skill retrieval score={retrieval_score:.2f}.")
            matched_fields.append("skill_retrieval.retrieval_scores")
        if retrieval_reasons.get(skill.name):
            reasons.extend(str(item) for item in retrieval_reasons.get(skill.name, []))
            matched_fields.append("skill_retrieval.match_reasons")
        if retrieval_trigger_hits.get(skill.name):
            matched_fields.extend(f"skill_retrieval.trigger:{item}" for item in retrieval_trigger_hits.get(skill.name, []))

        if skill.name in FOUNDATIONAL_SKILLS:
            score += int(policy.get("foundational_bonus", 6))
            reasons.append("Selected as part of the foundational structured observation layer.")
            matched_fields.append("foundation")

        if skill.name in planner_input.cognition.preferred_skills:
            score += int(policy.get("preferred_skill_bonus", 3))
            reasons.append("Boosted by cognition.preferred_skills from prior cases.")
            matched_fields.append("cognition.preferred_skills")

        skill_stats = planner_input.cognition.skill_statistics.get(skill.name, {})
        helpful_rate = float(skill_stats.get("helpful_rate", 0.0))
        failure_rate = float(skill_stats.get("failure_rate", 0.0))
        if helpful_rate >= 0.5:
            score += int(policy.get("helpful_rate_bonus", 2))
            reasons.append(f"Boosted by cognition skill helpful_rate={helpful_rate:.2f}.")
            matched_fields.append("cognition.skill_statistics.helpful_rate")
        if failure_rate >= 0.75:
            score -= int(policy.get("failure_rate_penalty", 2))
            reasons.append(f"Penalized by cognition skill failure_rate={failure_rate:.2f}.")
            matched_fields.append("cognition.skill_statistics.failure_rate")

        enabled_signals = {
            **{
                "experience_compare_pattern": True,
                "experience_risk_pattern": True,
                "experience_gap_pattern": True,
                "experience_conflict_pattern": True,
                "experience_escalation_pattern": True,
            },
            **dict(policy.get("enable_signals", {}) or {}),
        }
        for trigger in skill.triggers:
            trigger_text = f"{trigger.condition} {trigger.rationale}".lower()
            for signal_name, active in signals.items():
                if not active:
                    continue
                if signal_name.startswith("experience_") and not enabled_signals.get(signal_name, True):
                    continue
                keywords = SIGNAL_KEYWORD_MAP.get(signal_name, ())
                if any(keyword in trigger_text or keyword in skill_text for keyword in keywords):
                    score += int(policy.get("signal_match_bonus", 2))
                    reasons.append(f"Matched active signal `{signal_name}` via skill trigger metadata.")
                    matched_fields.append(f"signal:{signal_name}")
                    break

        if signals["known_confusion_match"] and skill.name in {
            "differential_compare_skill",
            "exclusion_reasoning_skill",
            "uncertainty_assessment_skill",
            "mel_nev_specialist_skill",
            "ack_scc_specialist_skill",
        }:
            score += int(policy.get("known_confusion_bonus", 2))
            reasons.append("Boosted by cognition.known_confusion_patterns matching the current case.")
            matched_fields.append("cognition.known_confusion_patterns")

        if signals["experience_compare_pattern"] and skill.name in {
            "differential_compare_skill",
            "exclusion_reasoning_skill",
            "uncertainty_assessment_skill",
            "mel_nev_specialist_skill",
            "ack_scc_specialist_skill",
        }:
            score += int(policy.get("experience_compare_bonus", 2))
            reasons.append("Boosted by retrieved experience summary favoring compare/uncertainty style control.")
            matched_fields.append("retrieved_experience_summary")

        if signals["experience_risk_pattern"] and skill.name in {
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
            "escalation_recommendation_skill",
        }:
            score += int(policy.get("experience_risk_bonus", 2))
            reasons.append("Boosted by retrieved experience summary emphasizing risk and uncertainty framing.")
            matched_fields.append("retrieved_experience_summary")

        if signals["experience_gap_pattern"] and skill.name in {
            "information_gap_detection_skill",
            "uncertainty_assessment_skill",
            "contradiction_check_skill",
        }:
            score += int(policy.get("experience_gap_bonus", 2))
            reasons.append("Boosted by retrieved experience summary emphasizing missing information and unresolved gaps.")
            matched_fields.append("retrieved_experience_summary")

        if signals["experience_conflict_pattern"] and skill.name in {
            "information_gap_detection_skill",
            "contradiction_check_skill",
            "escalation_recommendation_skill",
        }:
            score += int(policy.get("experience_conflict_bonus", 2))
            reasons.append("Boosted by retrieved experience summary emphasizing conflicts or reasoning audits.")
            matched_fields.append("retrieved_experience_summary")

        if signals["experience_escalation_pattern"] and skill.name == "escalation_recommendation_skill":
            score += int(policy.get("experience_escalation_bonus", 2))
            reasons.append("Boosted by retrieved experience summary suggesting more cautious diagnostic checking.")
            matched_fields.append("retrieved_experience_summary")

        if learned_probability is not None:
            learned_weight = float(policy.get("learned_controller_weight", 4.0))
            learned_bonus = int(round(learned_probability * learned_weight))
            if learned_bonus > 0:
                score += learned_bonus
                reasons.append(f"Boosted by learned controller probability={learned_probability:.3f}.")
                matched_fields.append("learned_controller.probability")
        if learned_rank is not None:
            rank_bonus = max(0, int(policy.get("learned_controller_rank_bonus", 3)) - int(learned_rank))
            if rank_bonus > 0:
                score += rank_bonus
                reasons.append(f"Boosted by learned controller rank={learned_rank + 1}.")
                matched_fields.append("learned_controller.rank")

        min_score = int(skill_overrides.get("min_score", policy.get("score_threshold_default", 4)))
        selected = self._should_select(skill.name, score, signals, min_score=min_score)
        if (
            learned_selected
            and str(policy.get("controller_family", "")).strip() == "learned_supervised"
        ):
            selected = True
            reasons.append("Selected by learned controller sparse top-k policy.")
            matched_fields.append("learned_controller.sparse_selection")
        if skill.name in force_select:
            selected = True
            reasons.append("Forced selected by current planner policy.")
            matched_fields.append("policy.force_select")
        for forced_signal in skill_overrides.get("force_select_when", []):
            if signals.get(str(forced_signal), False):
                selected = True
                reasons.append(f"Forced selected by planner policy when `{forced_signal}` is active.")
                matched_fields.append(f"policy.force_select_when:{forced_signal}")

        if selected and not reasons:
            reasons.append("Selected by default rule-based planner policy.")

        return SkillSelectionDecision(
            skill_name=skill.name,
            selected=selected,
            score=score,
            reasons=dedupe_reasons(reasons),
            ordering_hint=ORDERING_HINTS.get(skill.name, 999),
            matched_fields=dedupe_reasons(matched_fields),
        )

    @staticmethod
    def _should_select(skill_name: str, score: int, signals: dict[str, Any], *, min_score: int = 4) -> bool:
        if skill_name in FOUNDATIONAL_SKILLS:
            return True
        if skill_name == "temporal_evolution_skill":
            return signals["temporal_metadata"] and score >= 2
        if skill_name == "malignancy_risk_assessment_skill":
            return signals["malignancy_possible"] or score >= min_score
        if skill_name == "differential_compare_skill":
            return signals["multiple_ddx"] or score >= min_score
        if skill_name == "exclusion_reasoning_skill":
            return signals["multiple_ddx"] or signals["high_uncertainty"] or score >= min_score
        if skill_name == "information_gap_detection_skill":
            return signals["high_uncertainty"] or signals["experience_gap_pattern"] or signals["multiple_ddx"] or score >= min_score
        if skill_name == "mel_nev_specialist_skill":
            return signals["mel_nev_confusion"]
        if skill_name == "ack_scc_specialist_skill":
            return (
                signals["ack_scc_confusion"]
                or signals.get("keratinocyte_bcc_confusion", False)
                or (signals.get("known_confusion_match", False) and score >= max(2, min_score - 1))
            )
        if skill_name == "uncertainty_assessment_skill":
            return signals["high_uncertainty"] or score >= min_score
        if skill_name == "contradiction_check_skill":
            return score >= min_score or signals["high_uncertainty"] or signals["multiple_ddx"]
        if skill_name == "escalation_recommendation_skill":
            return (
                signals["malignancy_possible"]
                or signals["high_uncertainty"]
                or signals["experience_conflict_pattern"]
                or signals["experience_escalation_pattern"]
                or score >= min_score
            )
        return score >= min_score


def build_default_planner(policy_config: dict[str, Any] | None = None) -> BaseSkillPlanner:
    return RuleBasedSkillPlanner(policy_config=policy_config)


def _build_signal_profile(planner_input: PlannerInput) -> dict[str, Any]:
    perception = planner_input.perception
    metadata = planner_input.metadata
    ddx_candidates = [str(item).lower() for item in perception.get("ddx_candidates", [])]
    uncertainty_level = str(perception.get("uncertainty", {}).get("level", "unknown")).lower()
    retrieved_source = planner_input.retrieved_experience_bundle.get("planner_summary") or planner_input.retrieved_experience_summary
    retrieved_chunks = []
    for record in retrieved_source:
        retrieved_chunks.extend(
            [
                str(record.get("experience_type", "")),
                str(record.get("perception_summary", "")),
                str(record.get("confusion_pair", "")),
                " ".join(str(item) for item in record.get("learning_points", [])),
            ]
        )
    retrieved_text = " ".join(retrieved_chunks).lower()
    current_confusion_pair = detect_confusion_pair(ddx_candidates)
    known_confusion_patterns = planner_input.cognition.known_confusion_patterns
    known_confusion_text = " ".join(str(key).strip().lower() for key in known_confusion_patterns.keys())
    known_confusion_match = bool(
        current_confusion_pair
        and (
            current_confusion_pair in known_confusion_patterns
            or current_confusion_pair.lower() in known_confusion_text
        )
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

    return {
        "high_uncertainty": uncertainty_level == "high",
        "multiple_ddx": len(ddx_candidates) >= 2,
        "temporal_metadata": any(str(metadata.get(field, "")).strip() for field in ("grew", "changed", "bleed", "itch", "hurt", "elevation")),
        "location_or_size_metadata": any(
            str(metadata.get(field, "")).strip() for field in ("region", "age", "diameter_1", "diameter_2")
        ),
        "malignancy_possible": has_malignancy_possibility(ddx_candidates),
        "mel_nev_confusion": has_confusion_pair(ddx_candidates, ("mel", "melanoma"), ("nev", "nevus", "naevus", "mole")),
        "ack_scc_confusion": has_confusion_pair(
            ddx_candidates,
            ("ack", "actinic keratosis", "actinic keratos"),
            ("scc", "squamous cell", "squamous"),
        ),
        "keratinocyte_bcc_confusion": keratinocyte_bcc_confusion,
        "experience_compare_pattern": any(
            pattern in retrieved_text for pattern in ("compare_then_audit_uncertainty", "confusion_memory", "differential")
        ),
        "experience_risk_pattern": any(
            pattern in retrieved_text for pattern in ("risk_then_uncertainty_audit", "risk", "alarm")
        ),
        "experience_gap_pattern": any(
            pattern in retrieved_text for pattern in ("missing", "gap", "underdetermined", "need more information", "uncertainty")
        ),
        "experience_conflict_pattern": any(
            pattern in retrieved_text for pattern in ("conflict", "contradiction", "audit", "inconsisten")
        ),
        "experience_escalation_pattern": any(
            pattern in retrieved_text for pattern in ("escalat", "dermoscopy", "biopsy", "closer exam", "further check", "urgent")
        ),
        "known_confusion_match": known_confusion_match,
    }


def detect_confusion_pair(ddx_candidates: list[str]) -> str | None:
    if has_confusion_pair(ddx_candidates, ("mel", "melanoma"), ("nev", "nevus", "naevus", "mole")):
        return "melanoma->nev"
    if has_confusion_pair(ddx_candidates, ("scc", "squamous cell", "squamous"), ("bcc", "basal cell")):
        return "scc->bcc"
    if has_confusion_pair(ddx_candidates, ("ack", "actinic keratosis", "actinic keratos"), ("bcc", "basal cell")):
        return "ack->bcc"
    if has_confusion_pair(ddx_candidates, ("seborrheic keratosis", "sek"), ("bcc", "basal cell")):
        return "sek->bcc"
    if has_confusion_pair(
        ddx_candidates,
        ("lichen simplex", "lichen planus", "psoriasis", "dermatitis", "eczema"),
        ("ack", "actinic keratosis", "actinic keratos"),
    ):
        return "inflammatory->ack"
    if has_confusion_pair(
        ddx_candidates,
        ("ack", "actinic keratosis", "actinic keratos"),
        ("scc", "squamous cell", "squamous"),
    ):
        return "ack->scc"
    return None


def has_malignancy_possibility(ddx_candidates: list[str]) -> bool:
    malignant_keywords = (
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
    return any(any(keyword in candidate for keyword in malignant_keywords) for candidate in ddx_candidates)


def has_confusion_pair(
    ddx_candidates: list[str],
    left_keywords: tuple[str, ...],
    right_keywords: tuple[str, ...],
) -> bool:
    has_left = any(any(keyword in candidate for keyword in left_keywords) for candidate in ddx_candidates)
    has_right = any(any(keyword in candidate for keyword in right_keywords) for candidate in ddx_candidates)
    return has_left and has_right


def _skill_text(skill: SkillObject) -> str:
    trigger_text = " ".join(f"{trigger.condition} {trigger.rationale}" for trigger in skill.triggers)
    return f"{skill.name} {skill.description} {skill.skill_type} {trigger_text} {skill.workflow_text}".lower()


def dedupe_reasons(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_planner_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(policy or {})
    return {
        "controller_family": "heuristic",
        "controller_checkpoint_path": "",
        "learned_controller_weight": 4.0,
        "learned_controller_rank_bonus": 3,
        "learned_controller_select_threshold": 0.4,
        "learned_controller_top_k": 0,
        "learned_controller_force_top_k": 0,
        "score_threshold_default": 4,
        "retrieval_score_cap": 3,
        "foundational_bonus": 6,
        "preferred_skill_bonus": 3,
        "helpful_rate_bonus": 2,
        "failure_rate_penalty": 2,
        "signal_match_bonus": 2,
        "known_confusion_bonus": 2,
        "experience_compare_bonus": 2,
        "experience_risk_bonus": 2,
        "experience_gap_bonus": 2,
        "experience_conflict_bonus": 2,
        "experience_escalation_bonus": 2,
        "enable_signals": {},
        "force_select_skills": [],
        "force_disable_skills": [],
        "skill_overrides": {},
        "_policy_id": str(source.get("_policy_id", "")).strip(),
        **source,
    }
