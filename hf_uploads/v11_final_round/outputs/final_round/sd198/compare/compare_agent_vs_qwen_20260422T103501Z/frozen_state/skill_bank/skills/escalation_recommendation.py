from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class EscalationRecommendationSkill(BaseSkill):
    name = "escalation_recommendation_skill"
    description = "Decide whether the reasoning state justifies more cautious or escalated diagnostic checking."
    output_fields = (
        "whether_escalation_needed",
        "escalation_reason",
        "suggested_next_check_type",
        "caution_flags",
    )
    list_fields = ("escalation_reason", "caution_flags")
    skill_object = make_skill_object(
        skill_id="skill.escalation_recommendation.v1",
        name=name,
        description=description,
        skill_type="risk_uncertainty",
        triggers=[
            SkillTrigger(
                condition="Use when malignancy concern, unresolved uncertainty, or unresolved contradiction suggests that the case may need more cautious diagnostic checking.",
                rationale="Doctors often decide not only what they think, but also whether the current evidence state warrants escalation to closer or more definitive evaluation.",
            )
        ],
        workflow_text=(
            "Evaluate whether the current reasoning state supports escalation to a more cautious diagnostic check. "
            "Use only the evidence state, risk signals, uncertainty, contradictions, and information gaps to decide "
            "whether escalation is needed, why it is needed, what type of next check would be most appropriate, "
            "and what caution flags should travel with that recommendation. "
            "This skill must not recommend treatment."
        ),
        steps=[
            SkillStep("review_risk_and_uncertainty", "Review Risk And Uncertainty", "Review malignancy risk, uncertainty, contradictions, and information gaps together.", ["risk", "uncertainty", "conflicts", "missing information"]),
            SkillStep("decide_need", "Decide Need", "Classify whether escalation is needed, should be considered, or is not currently needed.", ["evidence sufficiency", "residual danger"]),
            SkillStep("name_check_type", "Name Check Type", "Suggest the next diagnostic check type rather than a treatment action.", ["dermoscopy", "closer exam", "short interval review", "biopsy consideration"]),
            SkillStep("record_caution_flags", "Record Caution Flags", "List the specific reasons the final reasoner should stay cautious.", ["alarm signals", "ambiguity", "conflicts"]),
        ],
        watch_outs=[
            "Do not recommend treatment or procedural management as if this were a care plan.",
            "Do not escalate solely because uncertainty exists; connect escalation to risk, conflict, or meaningful unresolved gaps.",
            "Do not suppress caution when malignancy concern and missing information coexist.",
            "Do not present escalation as a final diagnosis.",
        ],
        output_schema=[
            SkillSchemaField("whether_escalation_needed", "str", "One of yes, consider, or no."),
            SkillSchemaField("escalation_reason", "list[str]", "Why escalation is needed or should be considered."),
            SkillSchemaField("suggested_next_check_type", "str", "Type of next diagnostic check that would best reduce risk or uncertainty."),
            SkillSchemaField("caution_flags", "list[str]", "Flags that the final reasoner should carry forward cautiously."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the escalation recommendation routine.\n"
            "When to use: use when risk, uncertainty, contradictions, or unresolved information gaps may justify more cautious diagnostic checking.\n"
            "What evidence to inspect: malignancy risk, uncertainty, information gaps, contradictions, exclusion reasoning, and metadata.\n"
            "Common pitfalls: recommending treatment instead of diagnostic checking, escalating for vague reasons, or ignoring caution flags when risk remains nontrivial.\n"
            "Do NOT output treatment advice, management plans, or a final diagnosis.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('malignancy_risk_assessment_skill', 'information_gap_detection_skill', 'uncertainty_assessment_skill', 'contradiction_check_skill', 'exclusion_reasoning_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name == "whether_escalation_needed":
            normalized = str(value).strip().lower()
            if normalized not in {"yes", "consider", "no"}:
                return "consider" if value else "unknown"
            return normalized
        return super().normalize_field(field_name, value)
