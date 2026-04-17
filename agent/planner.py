from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.confusion_clusters import cluster_priority_bonus, detect_confusion_clusters
from cognition.cognition_state import CognitionState
from skills.schema import SkillObject

try:
    from agent.supervised_controller import ControllerSelectionPolicy, LearnedControllerScorer
except Exception:  # pragma: no cover - fallback for minimal runtime environments without torch
    LearnedControllerScorer = None  # type: ignore[assignment]
    ControllerSelectionPolicy = None  # type: ignore[assignment]


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
    "ack_sek_confusion": ("ack", "seborrheic", "sek", "waxy", "stuck-on", "compare", "confusion"),
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
    workflow_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "perception": self.perception,
            "metadata": self.metadata,
            "cognition": self.cognition.to_dict(),
            "retrieved_experience_summary": list(self.retrieved_experience_summary),
            "retrieved_experience_bundle": dict(self.retrieved_experience_bundle),
            "skill_retrieval_bundle": dict(self.skill_retrieval_bundle),
            "policy_config": dict(self.policy_config),
            "workflow_context": dict(self.workflow_context or {}),
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
    controller_probability: float | None = None
    controller_selected: bool = False
    controller_rejected: bool = False
    helpfulness_penalty: float = 0.0
    adaptive_budget_retain: bool = False
    adaptive_budget_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerOutput:
    available_skill_candidates: list[str]
    selected_skills: list[str]
    selection_reasons: dict[str, list[str]]
    ordering: list[str]
    decision_trace: list[dict[str, Any]]
    rejected_skills: list[str] = field(default_factory=list)
    rejected_reasons: dict[str, list[str]] = field(default_factory=dict)
    selection_scores: dict[str, Any] = field(default_factory=dict)
    controller_decision_info: dict[str, Any] = field(default_factory=dict)
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
        budget_info = _compute_case_adaptive_budget(planner_input=planner_input, signals=signals, policy=policy)
        learned_prediction = None
        runtime_selection_policy = None
        if (
            self.learned_controller is not None
            and str(policy.get("controller_family", "")).strip() == "learned_supervised"
            and ControllerSelectionPolicy is not None
        ):
            runtime_selection_policy = self.learned_controller.derive_selection_policy(
                top_k=int(budget_info["final_budget"]),
                min_select=int(budget_info["min_select"]),
                max_select=int(budget_info["final_budget"]),
                threshold=float(policy.get("learned_controller_select_threshold", 0.4) or 0.4),
                top_k_buffer=0,
                preserve_top1=True,
            )
        if self.learned_controller is not None and str(policy.get("controller_family", "")).strip() == "learned_supervised":
            learned_prediction = self.learned_controller.score_from_planner_input(
                perception=planner_input.perception,
                metadata=planner_input.metadata,
                skill_retrieval_bundle=planner_input.skill_retrieval_bundle,
                retrieved_experience_bundle=planner_input.retrieved_experience_bundle,
                available_skill_names=[skill.name for skill in planner_input.available_skills],
                cognition=planner_input.cognition,
                selection_policy=runtime_selection_policy,
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

        if learned_prediction is not None and str(policy.get("controller_family", "")).strip() == "learned_supervised":
            self._apply_learned_controller_sparsification(
                decisions=decisions,
                learned_prediction=learned_prediction,
                policy=policy,
                signals=signals,
                budget_info=budget_info,
            )

        self._adjust_decisions_by_workflow_context(
            decisions=decisions,
            workflow_context=planner_input.workflow_context,
            policy=policy,
        )

        self._apply_case_budget_gate(
            decisions=decisions,
            policy=policy,
            signals=signals,
            budget_info=budget_info,
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
        rejected = [decision for decision in decisions if not decision.selected]
        rejected.sort(key=lambda item: (item.ordering_hint, -item.score, item.skill_name))
        selection_scores = {
            decision.skill_name: {
                "rule_score": int(decision.score),
                "controller_probability": (
                    round(float(decision.controller_probability), 6)
                    if decision.controller_probability is not None
                    else None
                ),
                "selected": bool(decision.selected),
                "controller_selected": bool(decision.controller_selected),
                "controller_rejected": bool(decision.controller_rejected),
                "helpfulness_penalty": round(float(decision.helpfulness_penalty), 6),
                "adaptive_budget_retain": bool(decision.adaptive_budget_retain),
                "adaptive_budget_reason": str(decision.adaptive_budget_reason),
            }
            for decision in decisions
        }
        controller_decision_info = {
            "controller_family": str(policy.get("controller_family", "heuristic")),
            "controller_mode": str(policy.get("learned_controller_mode", "union")),
            "threshold": float(policy.get("learned_controller_select_threshold", 0.4) or 0.4),
            "target_top_k": int(policy.get("learned_controller_top_k", 0) or 0),
            "force_top_k": int(policy.get("learned_controller_force_top_k", 0) or 0),
            "final_budget": int(budget_info["final_budget"]),
            "budget_min": int(budget_info["min_budget"]),
            "budget_max": int(budget_info["max_budget"]),
            "budget_tier": str(budget_info["budget_tier"]),
            "budget_reasons": list(budget_info["budget_reasons"]),
            "complexity_signals": dict(budget_info["complexity_signals"]),
        }
        if learned_prediction is not None:
            controller_decision_info["selection_info"] = learned_prediction.selection_info
            controller_decision_info["controller_selected_skills"] = list(learned_prediction.selected_skills)
            controller_decision_info["controller_rejected_skills"] = list(learned_prediction.rejected_skills)
            if runtime_selection_policy is not None:
                controller_decision_info["runtime_selection_policy"] = runtime_selection_policy.to_dict()

        return PlannerOutput(
            available_skill_candidates=[skill.name for skill in planner_input.available_skills],
            selected_skills=ordering,
            selection_reasons=selection_reasons,
            ordering=ordering,
            decision_trace=[decision.to_dict() for decision in sorted(decisions, key=lambda item: item.ordering_hint)],
            rejected_skills=[decision.skill_name for decision in rejected],
            rejected_reasons={decision.skill_name: decision.reasons for decision in rejected},
            selection_scores=selection_scores,
            controller_decision_info=controller_decision_info,
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
        harmful_rate = float(skill_stats.get("harmful_count", 0)) / max(1, int(skill_stats.get("call_count", 0) or 0))
        helpful_floor = float(policy.get("skill_penalty_helpful_floor", 0.2) or 0.2)
        harmful_threshold = float(policy.get("skill_penalty_harmful_threshold", 0.08) or 0.08)
        harmful_penalty = 0.0
        if harmful_rate > harmful_threshold:
            harmful_penalty += (harmful_rate - harmful_threshold) * float(
                policy.get("skill_harmful_rate_penalty_weight", 4.0) or 4.0
            )
        if helpful_rate < helpful_floor:
            harmful_penalty += (helpful_floor - helpful_rate) * float(
                policy.get("skill_low_helpful_rate_penalty_weight", 2.0) or 2.0
            )
        if harmful_penalty > 0.0:
            penalty_points = max(1, int(round(harmful_penalty)))
            score -= penalty_points
            reasons.append(
                "Penalized by skill helpfulness prior "
                f"(helpful_rate={helpful_rate:.2f}, harmful_rate={harmful_rate:.2f}, penalty={penalty_points})."
            )
            matched_fields.append("cognition.skill_statistics.helpfulness_penalty")

        active_clusters = [str(item) for item in signals.get("active_confusion_clusters", []) if str(item).strip()]
        cluster_bonus = float(cluster_priority_bonus(active_clusters, skill.name))
        if cluster_bonus > 0:
            rounded_bonus = max(1, int(round(cluster_bonus)))
            score += rounded_bonus
            reasons.append(
                "Boosted by active confusion cluster(s): "
                + ", ".join(active_clusters)
                + f" (bonus={rounded_bonus})."
            )
            matched_fields.append("confusion_cluster.priority")

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
            controller_probability=learned_probability,
            controller_selected=learned_selected,
            helpfulness_penalty=round(harmful_penalty, 6),
        )

    def _apply_learned_controller_sparsification(
        self,
        *,
        decisions: list[SkillSelectionDecision],
        learned_prediction: Any,
        policy: dict[str, Any],
        signals: dict[str, Any],
        budget_info: dict[str, Any],
    ) -> None:
        mode = str(policy.get("learned_controller_mode", "union")).strip().lower()
        if mode not in {"sparse_hybrid", "strict_topk"}:
            return
        learned_selected_set = {
            str(skill_name).strip()
            for skill_name in getattr(learned_prediction, "selected_skills", [])
            if str(skill_name).strip()
        }
        for decision in decisions:
            if not decision.selected:
                continue
            if decision.skill_name in FOUNDATIONAL_SKILLS:
                continue
            if "policy.force_select" in decision.matched_fields:
                continue
            if decision.skill_name in learned_selected_set:
                decision.controller_selected = True
                decision.reasons = dedupe_reasons(
                    decision.reasons + ["Retained by learned controller sparse selection."]
                )
                decision.matched_fields = dedupe_reasons(
                    decision.matched_fields + ["learned_controller.sparse_selection"]
                )
                continue
            if mode == "sparse_hybrid" and self._allow_sparse_rule_bypass(
                skill_name=decision.skill_name,
                score=decision.score,
                signals=signals,
                policy=policy,
                decision=decision,
                budget_info=budget_info,
            ):
                decision.adaptive_budget_retain = True
                decision.adaptive_budget_reason = "Retained by sparse hybrid safety guard within adaptive budget."
                decision.reasons = dedupe_reasons(
                    decision.reasons + ["Retained by sparse hybrid safety guard despite controller rejection."]
                )
                decision.matched_fields = dedupe_reasons(
                    decision.matched_fields + ["learned_controller.sparse_bypass"]
                )
                continue
            decision.selected = False
            decision.controller_rejected = True
            decision.reasons = dedupe_reasons(
                decision.reasons + ["Rejected by learned controller sparse pruning."]
            )
            decision.matched_fields = dedupe_reasons(
                decision.matched_fields + ["learned_controller.sparse_prune"]
            )

    @staticmethod
    def _allow_sparse_rule_bypass(
        *,
        skill_name: str,
        score: int,
        signals: dict[str, Any],
        policy: dict[str, Any],
        decision: SkillSelectionDecision,
        budget_info: dict[str, Any],
    ) -> bool:
        if int(budget_info.get("soft_bypass_remaining", 0) or 0) <= 0:
            return False
        protected_floor = int(policy.get("learned_controller_protected_min_score", 7) or 7)
        rule_floor = int(policy.get("learned_controller_sparse_rule_floor", 9) or 9)
        allowed = False
        if skill_name == "malignancy_risk_assessment_skill":
            allowed = signals["malignancy_possible"] and score >= protected_floor
        elif skill_name == "uncertainty_assessment_skill":
            allowed = signals["high_uncertainty"] and score >= protected_floor
        elif skill_name == "information_gap_detection_skill":
            allowed = signals["high_uncertainty"] and score >= protected_floor
        elif skill_name == "contradiction_check_skill":
            allowed = (signals["high_uncertainty"] or signals["multiple_ddx"]) and score >= protected_floor
        elif skill_name == "escalation_recommendation_skill":
            allowed = (signals["malignancy_possible"] or signals["high_uncertainty"]) and score >= max(protected_floor, 8)
        elif skill_name == "mel_nev_specialist_skill":
            allowed = signals["mel_nev_confusion"] and score >= protected_floor
        elif skill_name == "ack_scc_specialist_skill":
            allowed = (
                signals["ack_scc_confusion"]
                or signals.get("ack_sek_confusion", False)
                or signals.get("keratinocyte_bcc_confusion", False)
            ) and score >= protected_floor
        else:
            allowed = score >= rule_floor and (
                signals["malignancy_possible"]
                or signals["high_uncertainty"]
                or signals["multiple_ddx"]
            )
        if allowed:
            budget_info["soft_bypass_remaining"] = max(0, int(budget_info.get("soft_bypass_remaining", 0)) - 1)
            decision.adaptive_budget_retain = True
        return allowed

    def _adjust_decisions_by_workflow_context(
        self,
        *,
        decisions: list[SkillSelectionDecision],
        workflow_context: dict[str, Any] | None,
        policy: dict[str, Any],
    ) -> None:
        """根据 workflow_context 调整 skill 优先级和选择"""
        if not workflow_context:
            return

        preference = str(workflow_context.get("workflow_preference", "")).strip()
        available_tests = list(workflow_context.get("available_tests", []) or [])
        metadata_completeness = str(workflow_context.get("metadata_completeness", "")).strip()

        # 场景1: risk_first workflow → 提前 malignancy_risk_assessment
        if preference == "risk_first":
            for decision in decisions:
                if decision.skill_name == "malignancy_risk_assessment_skill":
                    decision.score += int(policy.get("workflow_risk_first_bonus", 5) or 5)
                    decision.ordering_hint = 5  # 提前到 morphology 之前
                    decision.reasons = dedupe_reasons(
                        decision.reasons + ["Prioritized by risk_first workflow preference."]
                    )
                    decision.matched_fields = dedupe_reasons(
                        decision.matched_fields + ["workflow_context.risk_first"]
                    )

        # 场景2: 没有 dermoscopy → 降低 morphology 系列权重
        if available_tests and "dermoscopy" not in available_tests:
            morphology_skills = {
                "morphology_analysis_skill",
                "border_surface_analysis_skill",
                "color_pattern_analysis_skill",
            }
            for decision in decisions:
                if decision.skill_name in morphology_skills:
                    penalty_factor = float(policy.get("workflow_no_dermoscopy_penalty", 0.7) or 0.7)
                    decision.score = int(decision.score * penalty_factor)
                    decision.reasons = dedupe_reasons(
                        decision.reasons + ["Downweighted due to lack of dermoscopy in available tests."]
                    )
                    decision.matched_fields = dedupe_reasons(
                        decision.matched_fields + ["workflow_context.no_dermoscopy"]
                    )

        # 场景3: metadata 不完整 → 提高 information_gap_detection 权重
        if metadata_completeness == "minimal":
            for decision in decisions:
                if decision.skill_name == "information_gap_detection_skill":
                    decision.score += int(policy.get("workflow_minimal_metadata_bonus", 3) or 3)
                    decision.selected = True  # 强制选择
                    decision.reasons = dedupe_reasons(
                        decision.reasons + ["Forced selected due to minimal metadata completeness."]
                    )
                    decision.matched_fields = dedupe_reasons(
                        decision.matched_fields + ["workflow_context.minimal_metadata"]
                    )
        elif metadata_completeness == "partial":
            for decision in decisions:
                if decision.skill_name == "information_gap_detection_skill":
                    decision.score += int(policy.get("workflow_partial_metadata_bonus", 2) or 2)
                    decision.reasons = dedupe_reasons(
                        decision.reasons + ["Boosted due to partial metadata completeness."]
                    )
                    decision.matched_fields = dedupe_reasons(
                        decision.matched_fields + ["workflow_context.partial_metadata"]
                    )

    def _apply_case_budget_gate(
        self,
        *,
        decisions: list[SkillSelectionDecision],
        policy: dict[str, Any],
        signals: dict[str, Any],
        budget_info: dict[str, Any],
    ) -> None:
        final_budget = int(budget_info.get("final_budget", 0) or 0)
        if final_budget <= 0:
            return
        max_foundational = int(policy.get("adaptive_budget_max_foundational", 5) or 5)
        selected = [decision for decision in decisions if decision.selected]
        foundational = [
            decision for decision in selected
            if decision.skill_name in FOUNDATIONAL_SKILLS
        ]
        non_foundational = [
            decision for decision in selected
            if decision.skill_name not in FOUNDATIONAL_SKILLS
        ]
        foundational.sort(key=lambda item: (item.score, -ORDERING_HINTS.get(item.skill_name, 999), item.skill_name), reverse=True)
        non_foundational.sort(
            key=lambda item: (
                item.adaptive_budget_retain,
                item.controller_selected,
                item.score,
                -float(item.controller_probability or 0.0),
                -ORDERING_HINTS.get(item.skill_name, 999),
                item.skill_name,
            ),
            reverse=True,
        )
        keep: list[SkillSelectionDecision] = []
        keep.extend(foundational[: min(max_foundational, final_budget)])
        remaining_budget = max(0, final_budget - len(keep))
        keep.extend(non_foundational[:remaining_budget])
        keep_names = {decision.skill_name for decision in keep}
        for decision in decisions:
            if not decision.selected:
                continue
            if decision.skill_name in keep_names:
                continue
            decision.selected = False
            decision.controller_rejected = True
            decision.reasons = dedupe_reasons(
                decision.reasons + [f"Rejected by adaptive controller budget (budget={final_budget})."]
            )
            decision.matched_fields = dedupe_reasons(
                decision.matched_fields + ["learned_controller.adaptive_budget_prune"]
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
            return signals["multiple_ddx"] or bool(signals.get("active_confusion_clusters")) or score >= min_score
        if skill_name == "exclusion_reasoning_skill":
            return (
                signals["multiple_ddx"]
                or signals["high_uncertainty"]
                or bool(signals.get("active_confusion_clusters"))
                or score >= min_score
            )
        if skill_name == "information_gap_detection_skill":
            return signals["high_uncertainty"] or signals["experience_gap_pattern"] or signals["multiple_ddx"] or score >= min_score
        if skill_name == "mel_nev_specialist_skill":
            return signals["mel_nev_confusion"]
        if skill_name == "ack_scc_specialist_skill":
            return (
                signals["ack_scc_confusion"]
                or signals.get("ack_sek_confusion", False)
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
    active_confusion_clusters = detect_confusion_clusters(
        ddx_candidates=ddx_candidates,
        confusion_pair=current_confusion_pair,
        known_confusion_text=known_confusion_text,
        image_summary=str(perception.get("image_summary", "")),
        notes=[str(item) for item in perception.get("notes", []) if str(item).strip()],
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
        "ddx_count": len(ddx_candidates),
        "uncertainty_level": uncertainty_level,
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
        "ack_sek_confusion": has_confusion_pair(
            ddx_candidates,
            ("ack", "actinic keratosis", "actinic keratos"),
            ("seborrheic keratosis", "sek"),
        )
        or "ack_sek" in active_confusion_clusters,
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
        "active_confusion_clusters": active_confusion_clusters,
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
        "learned_controller_mode": "union",
        "learned_controller_sparse_rule_floor": 9,
        "learned_controller_protected_min_score": 7,
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
        "skill_harmful_rate_penalty_weight": 4.0,
        "skill_low_helpful_rate_penalty_weight": 2.0,
        "skill_penalty_harmful_threshold": 0.08,
        "skill_penalty_helpful_floor": 0.2,
        "adaptive_budget_enabled": True,
        "adaptive_budget_min": 6,
        "adaptive_budget_max": 12,
        "adaptive_budget_default": 8,
        "adaptive_budget_high_uncertainty_bonus": 2,
        "adaptive_budget_medium_uncertainty_bonus": 1,
        "adaptive_budget_high_risk_bonus": 2,
        "adaptive_budget_medium_risk_bonus": 1,
        "adaptive_budget_confusion_bonus": 1,
        "adaptive_budget_known_confusion_bonus": 1,
        "adaptive_budget_contradiction_bonus": 1,
        "adaptive_budget_retrieval_ambiguity_bonus": 1,
        "adaptive_budget_ddx_bonus": 1,
        "adaptive_budget_max_foundational": 5,
        "adaptive_budget_max_soft_bypass": 2,
        "enable_signals": {},
        "force_select_skills": [],
        "force_disable_skills": [],
        "skill_overrides": {},
        "_policy_id": str(source.get("_policy_id", "")).strip(),
        **source,
    }


def _compute_case_adaptive_budget(
    *,
    planner_input: PlannerInput,
    signals: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    min_budget = max(3, int(policy.get("adaptive_budget_min", 6) or 6))
    max_budget = max(min_budget, int(policy.get("adaptive_budget_max", 12) or 12))
    default_budget = int(policy.get("adaptive_budget_default", min_budget) or min_budget)
    budget = min(max(default_budget, min_budget), max_budget)
    reasons: list[str] = []
    perception_uncertainty = str(planner_input.perception.get("uncertainty", {}).get("level", "unknown")).strip().lower()
    if bool(policy.get("adaptive_budget_enabled", True)):
        if perception_uncertainty == "high":
            budget += int(policy.get("adaptive_budget_high_uncertainty_bonus", 2) or 2)
            reasons.append("Expanded budget for high uncertainty.")
        elif perception_uncertainty == "medium":
            budget += int(policy.get("adaptive_budget_medium_uncertainty_bonus", 1) or 1)
            reasons.append("Expanded budget for medium uncertainty.")
        risk_level = _infer_case_risk_level(planner_input=planner_input, signals=signals)
        if risk_level == "high":
            budget += int(policy.get("adaptive_budget_high_risk_bonus", 2) or 2)
            reasons.append("Expanded budget for high malignant risk.")
        elif risk_level == "medium":
            budget += int(policy.get("adaptive_budget_medium_risk_bonus", 1) or 1)
            reasons.append("Expanded budget for medium malignant risk.")
        active_clusters = list(signals.get("active_confusion_clusters", []) or [])
        if active_clusters:
            budget += int(policy.get("adaptive_budget_confusion_bonus", 1) or 1)
            reasons.append(f"Expanded budget for active confusion clusters: {', '.join(active_clusters)}.")
        if bool(signals.get("known_confusion_match", False)):
            budget += int(policy.get("adaptive_budget_known_confusion_bonus", 1) or 1)
            reasons.append("Expanded budget for known confusion match.")
        contradiction_count = _estimate_contradiction_count(planner_input=planner_input, signals=signals)
        if contradiction_count >= 2:
            budget += int(policy.get("adaptive_budget_contradiction_bonus", 1) or 1)
            reasons.append(f"Expanded budget for contradiction-rich case (count={contradiction_count}).")
        retrieval_ambiguity = _estimate_retrieval_ambiguity(planner_input)
        if retrieval_ambiguity >= 0.45:
            budget += int(policy.get("adaptive_budget_retrieval_ambiguity_bonus", 1) or 1)
            reasons.append(
                f"Expanded budget for ambiguous skill retrieval spread (ambiguity={retrieval_ambiguity:.2f})."
            )
        ddx_count = int(signals.get("ddx_count", 0) or 0)
        if ddx_count >= 4:
            budget += int(policy.get("adaptive_budget_ddx_bonus", 1) or 1)
            reasons.append(f"Expanded budget for broad differential ({ddx_count} candidates).")
    final_budget = max(min_budget, min(max_budget, budget))
    if not reasons:
        reasons.append("Tightened to default budget because the case signals are relatively simple.")
    budget_tier = "tight"
    if final_budget >= max_budget - 1:
        budget_tier = "wide"
    elif final_budget >= default_budget + 1:
        budget_tier = "medium"
    min_select = max(2, min(final_budget, max(2, final_budget - 2)))
    return {
        "min_budget": min_budget,
        "max_budget": max_budget,
        "default_budget": default_budget,
        "final_budget": final_budget,
        "min_select": min_select,
        "budget_tier": budget_tier,
        "budget_reasons": reasons,
        "complexity_signals": {
            "uncertainty_level": perception_uncertainty,
            "risk_level": _infer_case_risk_level(planner_input=planner_input, signals=signals),
            "active_confusion_clusters": list(signals.get("active_confusion_clusters", []) or []),
            "known_confusion_match": bool(signals.get("known_confusion_match", False)),
            "contradiction_count": _estimate_contradiction_count(planner_input=planner_input, signals=signals),
            "retrieval_ambiguity": round(_estimate_retrieval_ambiguity(planner_input), 6),
            "ddx_count": int(signals.get("ddx_count", 0) or 0),
        },
        "soft_bypass_remaining": int(policy.get("adaptive_budget_max_soft_bypass", 2) or 2),
    }


def _infer_case_risk_level(*, planner_input: PlannerInput, signals: dict[str, Any]) -> str:
    if not bool(signals.get("malignancy_possible", False)):
        return "low"
    ddx = " ".join(str(item).strip().lower() for item in planner_input.perception.get("ddx_candidates", []) if str(item).strip())
    if any(term in ddx for term in ("mel", "melanoma", "bcc", "basal cell", "scc", "squamous")):
        return "high"
    if "ack" in ddx or "actinic keratos" in ddx or bool(signals.get("keratinocyte_bcc_confusion", False)):
        return "medium"
    return "medium"


def _estimate_contradiction_count(*, planner_input: PlannerInput, signals: dict[str, Any]) -> int:
    perception = planner_input.perception or {}
    image_summary = str(perception.get("image_summary", "")).lower()
    notes = " ".join(str(item).strip().lower() for item in perception.get("notes", []) if str(item).strip())
    count = 0
    for token in ("irregular", "asymmetry", "contradict", "conflict", "uncertain", "poorly defined"):
        if token in image_summary or token in notes:
            count += 1
    if bool(signals.get("high_uncertainty", False)):
        count += 1
    return count


def _estimate_retrieval_ambiguity(planner_input: PlannerInput) -> float:
    retrieval_scores = dict((planner_input.skill_retrieval_bundle or {}).get("retrieval_scores", {}) or {})
    if len(retrieval_scores) < 2:
        return 0.0
    ordered_scores = sorted((float(value or 0.0) for value in retrieval_scores.values()), reverse=True)
    top_score = ordered_scores[0]
    second_score = ordered_scores[1]
    if top_score <= 0.0:
        return 0.0
    margin = max(0.0, top_score - second_score)
    return max(0.0, min(1.0, 1.0 - (margin / max(top_score, 1e-6))))
