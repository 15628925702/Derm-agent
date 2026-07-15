from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class UncertaintyAssessmentSkill(BaseSkill):
    name = "uncertainty_assessment_skill"
    description = "Quantify uncertainty and information gaps."
    output_fields = ("uncertainty_level", "reasons", "missing_information")
    skill_object = make_skill_object(
        skill_id="skill.uncertainty_assessment.v1",
        name=name,
        description=description,
        skill_type="risk_uncertainty",
        triggers=[
            SkillTrigger(
                condition="Use when visual ambiguity, missing metadata, or unresolved differential conflict remains high.",
                rationale="Explicit uncertainty improves Qwen's interpretability and prevents overconfident evidence use.",
            )
        ],
        workflow_text=(
            "Act as an uncertainty auditor for the current case. Identify what is ambiguous in the image, what metadata is "
            "missing or weak, and which unresolved cues still limit confident reasoning. Summarize this as a low/medium/high "
            "uncertainty state with explicit reasons and missing information."
        ),
        steps=[
            SkillStep("scan_ambiguity", "Scan Ambiguity", "Inspect perception and prior skills for ambiguous visual findings.", ["ambiguous morphology", "weak border/color evidence"]),
            SkillStep("list_missing_info", "List Missing Information", "Name metadata or observation gaps that matter clinically.", ["size, history, site, temporal info"]),
            SkillStep("rate_uncertainty", "Rate Uncertainty", "Assign low/medium/high uncertainty based on the total evidence deficit.", ["ambiguity burden"]),
        ],
        watch_outs=[
            "Uncertainty is not failure; it is structured self-knowledge.",
            "Do not understate missing evidence just because a lesion looks superficially simple.",
            "Keep uncertainty separate from final diagnosis.",
        ],
        output_schema=[
            SkillSchemaField("uncertainty_level", "str", "Low, medium, or high uncertainty."),
            SkillSchemaField("reasons", "list[str]", "Reasons that justify the uncertainty level."),
            SkillSchemaField("missing_information", "list[str]", "Key missing information that would improve reasoning."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the uncertainty assessment routine.\n"
            "When to use: use when evidence remains incomplete or internal ambiguity is clinically meaningful.\n"
            "What evidence to inspect: image ambiguity, missing metadata, unresolved differential conflict.\n"
            "Common pitfalls: hiding uncertainty behind confident but weak language.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Case perception so far: {self.perception_snapshot(state)}\n"
            f"Existing skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'malignancy_risk_assessment_skill', 'differential_compare_skill'), max_skills=5)}\n"
            f"Case metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name in {"reasons", "missing_information"}:
            if value in (None, ""):
                return []
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]
        if field_name == "uncertainty_level":
            if value not in {"low", "medium", "high"}:
                return "high" if value else "unknown"
        return super().normalize_field(field_name, value)

    def after_execute(self, state: CaseState, output: dict[str, Any]) -> None:
        state.uncertainty = output
