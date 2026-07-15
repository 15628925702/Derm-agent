from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class ContradictionCheckSkill(BaseSkill):
    name = "contradiction_check_skill"
    description = "Detect internal inconsistencies and missing key evidence."
    output_fields = ("contradictions", "missing_links", "reasoning_gaps")
    list_fields = ("contradictions", "missing_links", "reasoning_gaps")
    skill_object = make_skill_object(
        skill_id="skill.contradiction_check.v1",
        name=name,
        description=description,
        skill_type="reasoning",
        triggers=[
            SkillTrigger(
                condition="Use when multiple evidence streams are active or when uncertainty remains despite several skills.",
                rationale="Clinical reasoning improves when contradictions and missing links are surfaced explicitly.",
            )
        ],
        workflow_text=(
            "Review the current reasoning state as a consistency auditor. Look for direct conflicts between image, metadata, "
            "and prior skill outputs; identify where the chain of reasoning lacks a necessary intermediate link; and surface "
            "broader unresolved gaps that keep the case underdetermined. This routine improves interpretability and safety."
        ),
        steps=[
            SkillStep("scan_conflicts", "Scan Conflicts", "Look for direct contradictions among perception, metadata, and skill outputs.", ["image-metadata mismatch", "skill-to-skill mismatch"]),
            SkillStep("trace_links", "Trace Missing Links", "Identify where a conclusion lacks needed supporting evidence.", ["unsupported inference"]),
            SkillStep("surface_gaps", "Surface Gaps", "Name broader reasoning gaps that remain unresolved.", ["missing key evidence", "ambiguous visual cue"]),
        ],
        watch_outs=[
            "Do not invent contradictions that are really just uncertainty.",
            "A reasoning gap is not always a factual conflict.",
            "Use this skill to expose problems, not to relabel the lesion.",
        ],
        output_schema=[
            SkillSchemaField("contradictions", "list[str]", "Direct conflicts between evidence sources."),
            SkillSchemaField("missing_links", "list[str]", "Missing support that prevents a clean reasoning chain."),
            SkillSchemaField("reasoning_gaps", "list[str]", "Broader unresolved gaps in the case reasoning state."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the contradiction check routine.\n"
            "When to use: use after multiple evidence sources are available or when reasoning still feels unstable.\n"
            "What evidence to inspect: perception, metadata, prior skill outputs, and evidence chain completeness.\n"
            "Common pitfalls: confusing ambiguity with contradiction and overclaiming weak conflicts.\n"
            "You must focus especially on:\n"
            "- image and metadata conflicts\n"
            "- internal conflicts inside Qwen initial perception\n"
            "- missing key evidence required for differential narrowing\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'malignancy_risk_assessment_skill', 'uncertainty_assessment_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )
