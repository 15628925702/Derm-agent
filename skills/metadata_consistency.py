from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class MetadataConsistencySkill(BaseSkill):
    name = "metadata_consistency_skill"
    description = "Check consistency between image evidence and metadata."
    output_fields = ("consistency_score", "conflicts", "suspicious_points")
    list_fields = ("conflicts", "suspicious_points")
    skill_object = make_skill_object(
        skill_id="skill.metadata_consistency.v1",
        name=name,
        description=description,
        skill_type="reasoning",
        triggers=[
            SkillTrigger(
                condition="Use when metadata credibility or image-metadata coherence could alter confidence.",
                rationale="Clinical reasoning degrades when image evidence and case context do not align.",
            )
        ],
        workflow_text=(
            "Compare the visible lesion description with the structured case metadata and identify whether the two sources "
            "cohere. Flag direct contradictions as conflicts, weaker credibility concerns as suspicious points, and summarize "
            "overall reliability as a consistency score. This routine is about evidence reliability, not diagnosis."
            " Use hard-vs-soft mismatch triage so inflammatory or irritation-like context does not get overcounted as hard contradiction."
        ),
        steps=[
            SkillStep("review_metadata", "Review Metadata", "Identify the metadata items most relevant to visible lesion assessment.", ["region", "diameter", "symptoms", "elevation"]),
            SkillStep("compare_sources", "Compare Sources", "Check whether image impression and metadata tell a coherent story.", ["image summary", "history fields"]),
            SkillStep("flag_conflicts", "Flag Conflicts", "List direct contradictions and weaker suspicious points separately.", ["hard mismatch", "soft inconsistency", "inflammatory mimic versus true conflict"]),
            SkillStep("rate_reliability", "Rate Reliability", "Assign high, medium, or low consistency.", ["overall coherence"]),
        ],
        watch_outs=[
            "Single-image appearance cannot confirm all metadata fields.",
            "Do not mark uncertainty itself as a hard contradiction.",
            "Benign-looking images can still carry high-risk metadata and vice versa.",
            "Chronic irritation/inflammatory texture can mimic keratotic change; do not label this as hard conflict without direct contradiction.",
            "Soft inconsistencies should not automatically force low consistency_score.",
        ],
        output_schema=[
            SkillSchemaField("consistency_score", "str", "Overall image-metadata consistency score."),
            SkillSchemaField("conflicts", "list[str]", "Direct contradictions between image impression and metadata."),
            SkillSchemaField("suspicious_points", "list[str]", "Softer mismatches or reliability concerns."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the metadata consistency routine.\n"
            "When to use: use when context credibility and image coherence affect downstream confidence.\n"
            "What evidence to inspect: image summary, lesion descriptors, metadata history, and size/site fields.\n"
            "Common pitfalls: weak visibility is not the same thing as contradiction.\n"
            "Classify each issue as either hard conflict (direct mismatch) or soft inconsistency (context-limited ambiguity).\n"
            "In inflammatory-versus-actinic confusion, prefer suspicious_points unless there is explicit direct mismatch.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'temporal_evolution_skill'), max_skills=4)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name == "consistency_score" and value not in {"high", "medium", "low"}:
            return "medium" if value else "unknown"
        return super().normalize_field(field_name, value)
