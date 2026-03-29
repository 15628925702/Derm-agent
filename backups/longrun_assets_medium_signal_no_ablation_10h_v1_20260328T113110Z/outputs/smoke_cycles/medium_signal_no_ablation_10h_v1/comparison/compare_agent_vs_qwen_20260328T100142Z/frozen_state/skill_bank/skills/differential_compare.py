from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class DifferentialCompareSkill(BaseSkill):
    name = "differential_compare_skill"
    description = "Compare top differential candidates without selecting the final diagnosis."
    output_fields = ("candidate_pairs", "supporting_evidence", "conflicting_evidence")
    list_fields = ("candidate_pairs", "supporting_evidence", "conflicting_evidence")
    skill_object = make_skill_object(
        skill_id="skill.differential_compare.v1",
        name=name,
        description=description,
        skill_type="reasoning",
        triggers=[
            SkillTrigger(
                condition="Use when initial perception surfaces two or more plausible candidate diagnoses.",
                rationale="Differential comparison is the physician step of weighing candidate-specific clues without final commitment.",
            )
        ],
        workflow_text=(
            "Take the leading candidate diagnoses from initial perception and compare them explicitly. Identify which observed "
            "features support one candidate over another and which unresolved facts keep the comparison open. This routine must "
            "support differential narrowing, not produce the final disease label."
        ),
        steps=[
            SkillStep("select_pairs", "Select Candidate Pairs", "Choose the most relevant pairwise comparisons from current ddx candidates.", ["ddx candidates"]),
            SkillStep("support_features", "Support Features", "List evidence that favors one side of each comparison.", ["supporting clues from prior skills"]),
            SkillStep("conflict_features", "Conflict Features", "List unresolved or contradictory features that prevent clean separation.", ["competing clues"]),
            SkillStep("preserve_openness", "Preserve Diagnostic Openness", "Keep the comparison open without selecting a winner.", ["remaining uncertainty"]),
        ],
        watch_outs=[
            "Do not collapse comparison into final diagnosis selection.",
            "Avoid using unsupported candidate names that were not already in scope.",
            "Be explicit about unresolved comparisons rather than hiding them.",
        ],
        output_schema=[
            SkillSchemaField("candidate_pairs", "list[str]", "Pairwise candidate comparisons such as 'A vs B'."),
            SkillSchemaField("supporting_evidence", "list[str]", "Clues that support one side of the candidate comparison."),
            SkillSchemaField("conflicting_evidence", "list[str]", "Clues that keep the candidate comparison unresolved."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the differential comparison routine.\n"
            "When to use: use after initial ddx generation when two or more plausible candidates remain.\n"
            "What evidence to inspect: current ddx list and previously generated structured clues.\n"
            "Common pitfalls: choosing a winner too early or adding unsupported candidate names.\n"
            "Do NOT choose a final diagnosis and do NOT output a single winning disease.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'temporal_evolution_skill', 'malignancy_risk_assessment_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )
