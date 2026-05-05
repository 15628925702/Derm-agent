from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class InformationGapDetectionSkill(BaseSkill):
    name = "information_gap_detection_skill"
    description = "Identify clinically meaningful missing information and explain how it limits reasoning."
    output_fields = (
        "missing_information",
        "why_it_matters",
        "impact_on_differential",
        "uncertainty_if_missing",
    )
    list_fields = ("missing_information", "why_it_matters", "impact_on_differential")
    skill_object = make_skill_object(
        skill_id="skill.information_gap_detection.v1",
        name=name,
        description=description,
        skill_type="risk_uncertainty",
        triggers=[
            SkillTrigger(
                condition="Use when current evidence is incomplete, conflicts remain unresolved, or uncertainty is still clinically important after observation and comparison.",
                rationale="Doctors explicitly ask what information is still missing and why that absence prevents cleaner differential narrowing.",
            )
        ],
        workflow_text=(
            "Act like a clinician identifying what information is still missing from the case. "
            "Name the missing information, explain why each missing item matters, describe how the missing data "
            "keeps the differential open, and estimate how much uncertainty will persist if those gaps remain unfilled. "
            "This skill must strengthen interpretability rather than diagnose the lesion."
        ),
        steps=[
            SkillStep("scan_current_state", "Scan Current State", "Review current observations, comparisons, uncertainty, and contradictions to see what cannot yet be resolved.", ["skill outputs", "uncertainty", "contradictions"]),
            SkillStep("name_missing_items", "Name Missing Items", "List the clinically useful information that is still absent or too weak.", ["history", "closer inspection", "measurement", "follow-up data"]),
            SkillStep("explain_relevance", "Explain Relevance", "State why each missing item matters for interpretation or narrowing the differential.", ["clinical impact", "decision relevance"]),
            SkillStep("estimate_uncertainty", "Estimate Residual Uncertainty", "Judge whether uncertainty remains low, medium, or high if these gaps stay unresolved.", ["residual ambiguity", "differential openness"]),
        ],
        watch_outs=[
            "Do not invent missing tests that are unrelated to the current lesion reasoning problem.",
            "Do not confuse direct contradictory evidence with missing evidence.",
            "Do not recommend treatment or management beyond identifying information needs.",
            "Do not hide why the missing information matters clinically.",
        ],
        output_schema=[
            SkillSchemaField("missing_information", "list[str]", "Clinically relevant information that is still missing."),
            SkillSchemaField("why_it_matters", "list[str]", "Why those missing items matter for reasoning."),
            SkillSchemaField("impact_on_differential", "list[str]", "How the missing information keeps differential candidates unresolved."),
            SkillSchemaField("uncertainty_if_missing", "str", "Low, medium, or high residual uncertainty if the information remains missing."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the information gap detection routine.\n"
            "When to use: use when the case remains underdetermined and the system needs to say exactly what information is still missing.\n"
            "What evidence to inspect: current observation skills, differential comparison, exclusion reasoning, uncertainty, contradiction checks, and metadata.\n"
            "Common pitfalls: inventing irrelevant missing data, confusing weak support with contradiction, and turning missing information into treatment advice.\n"
            "Do NOT output any disease diagnosis, treatment plan, or management recommendation.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('lesion_description_structuring_skill', 'differential_compare_skill', 'exclusion_reasoning_skill', 'uncertainty_assessment_skill', 'contradiction_check_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name == "uncertainty_if_missing":
            normalized = str(value).strip().lower()
            if normalized not in {"low", "medium", "high"}:
                return "medium" if value else "unknown"
            return normalized
        return super().normalize_field(field_name, value)
