from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class ContactAtopicSpecialistSkill(BaseSkill):
    name = "contact_atopic_specialist_skill"
    description = "Compare contact dermatitis versus atopic dermatitis clues without making the final diagnosis."
    output_fields = (
        "pair_focus_summary",
        "contact_supporting_evidence",
        "atopic_supporting_evidence",
        "clues_against_contact",
        "clues_against_atopic",
        "missing_history_needed",
        "uncertainty_under_current_evidence",
        "recommended_next_questions",
        "provisional_pairwise_impression",
    )
    list_fields = (
        "contact_supporting_evidence",
        "atopic_supporting_evidence",
        "clues_against_contact",
        "clues_against_atopic",
        "missing_history_needed",
        "recommended_next_questions",
    )
    skill_object = make_skill_object(
        skill_id="skill.contact_atopic_specialist.v1",
        name=name,
        description=description,
        skill_type="specialist",
        triggers=[
            SkillTrigger(
                condition=(
                    "Trigger when contact dermatitis and atopic dermatitis are both plausible, "
                    "or when prior confusion evidence shows recurrent contact-versus-atopic ambiguity."
                ),
                rationale=(
                    "This specialist should sharpen inflammatory-family differentiation when the generic observation layer "
                    "is too weak and the case risks collapsing into an over-narrow contact-dermatitis answer."
                ),
            )
        ],
        workflow_text=(
            "Focus narrowly on the contact-dermatitis-versus-atopic-dermatitis pair. "
            "Use the current image, metadata, and prior skill outputs to name findings that support contact dermatitis, "
            "findings that support atopic dermatitis, and the most important missing history still needed to separate the pair. "
            "If the visible evidence is too weak for a narrow conclusion, say so explicitly. "
            "Do not make the final diagnosis, but do provide a provisional pairwise impression such as leans_contact, leans_atopic, or indeterminate."
        ),
        steps=[
            SkillStep(
                "pair_focus",
                "Focus the Pair",
                "Restrict reasoning to contact dermatitis versus atopic dermatitis.",
                ["recurrent inflammatory-family confusion", "pairwise comparison"],
            ),
            SkillStep(
                "extract_visible_clues",
                "Extract Visible Clues",
                "List visible clues that favor either contact-like or atopic-like interpretation.",
                ["erythema", "scale", "dryness", "distribution", "surface change"],
            ),
            SkillStep(
                "identify_missing_history",
                "Identify Missing History",
                "State which exposure, chronicity, itch, and recurrence details are still needed.",
                ["exposure history", "chronic relapsing course", "pruritus pattern"],
            ),
            SkillStep(
                "guard_against_over_narrowing",
                "Guard Against Over-Narrowing",
                "If the image is nonspecific, explicitly block overconfident contact-dermatitis narrowing.",
                ["weak subtype evidence", "insufficient context"],
            ),
        ],
        watch_outs=[
            "Do not output a final diagnosis.",
            "Do not invent exposure history or chronicity if the case does not show it.",
            "Do not treat generic erythema and scale as decisive contact-dermatitis evidence by default.",
            "If the image supports only broad inflammatory change, say the pair remains underdetermined.",
            "Keep the output pairwise and evidence-centered rather than disease-label centered.",
        ],
        output_schema=[
            SkillSchemaField("pair_focus_summary", "str", "Short clinician-style summary of the pairwise comparison focus."),
            SkillSchemaField("contact_supporting_evidence", "list[str]", "Visible or contextual clues that support contact dermatitis."),
            SkillSchemaField("atopic_supporting_evidence", "list[str]", "Visible or contextual clues that support atopic dermatitis."),
            SkillSchemaField("clues_against_contact", "list[str]", "Clues that make a narrow contact-dermatitis interpretation weaker."),
            SkillSchemaField("clues_against_atopic", "list[str]", "Clues that make an atopic-dermatitis interpretation weaker."),
            SkillSchemaField("missing_history_needed", "list[str]", "History or context still needed to separate the pair better."),
            SkillSchemaField("uncertainty_under_current_evidence", "str", "Residual uncertainty level under the current evidence: low / medium / high."),
            SkillSchemaField("recommended_next_questions", "list[str]", "Most useful follow-up questions or observations for this pair."),
            SkillSchemaField(
                "provisional_pairwise_impression",
                "str",
                "Pairwise impression only: leans_contact / leans_atopic / indeterminate.",
            ),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the contact-versus-atopic specialist routine.\n"
            "When to use: use when inflammatory-family differentiation is weak and the case risks collapsing to a generic contact-dermatitis answer.\n"
            "What evidence to inspect: erythema pattern, scale or xerosis, follicular context, chronic-looking versus acute-irritant pattern, and whether the image is too nonspecific for a narrow call.\n"
            "Important: if exposure history is missing, do not over-credit contact dermatitis.\n"
            "Important: if the image mainly shows broad inflammatory change, you must state that the pair remains underdetermined.\n"
            "Do NOT output the final diagnosis.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'distribution_analysis_skill', 'lesion_description_structuring_skill', 'metadata_consistency_skill', 'uncertainty_assessment_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name == "uncertainty_under_current_evidence":
            normalized = str(value).strip().lower()
            if normalized not in {"low", "medium", "high"}:
                return "medium" if value else "unknown"
            return normalized
        if field_name == "provisional_pairwise_impression":
            normalized = str(value).strip().lower()
            if normalized not in {"leans_contact", "leans_atopic", "indeterminate"}:
                return "indeterminate" if value else "unknown"
            return normalized
        return super().normalize_field(field_name, value)

    def normalize_output(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        normalized = super().normalize_output(raw_output)
        if normalized.get("recommendation_type") == "descriptive_evidence":
            normalized["recommendation_type"] = "comparative_support"
        return normalized
