from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class DistributionAnalysisSkill(BaseSkill):
    name = "distribution_analysis_skill"
    description = "Analyze lesion location and distribution pattern."
    output_fields = ("body_location", "symmetry", "localized_vs_generalized", "clustering_pattern")
    skill_object = make_skill_object(
        skill_id="skill.distribution_analysis.v1",
        name=name,
        description=description,
        skill_type="observation",
        triggers=[
            SkillTrigger(
                condition="Use when lesion location and distribution context may shift likely differentials.",
                rationale="Body site and visible distribution pattern are common dermatology reasoning anchors.",
            )
        ],
        workflow_text=(
            "Determine where the lesion is located, whether the visible pattern appears isolated or generalized, and "
            "whether there is symmetry or clustering. This routine uses both image context and metadata location cues "
            "to produce structured distribution evidence without making a diagnostic claim."
        ),
        steps=[
            SkillStep("locate_site", "Locate Site", "Identify the anatomic location using image context and metadata.", ["body site metadata", "visual site cues"]),
            SkillStep("symmetry", "Assess Symmetry", "Judge whether the visible distribution appears symmetric or asymmetric.", ["distribution balance"]),
            SkillStep("extent", "Assess Extent", "Decide whether the pattern is localized or generalized.", ["single-site vs multi-site impression"]),
            SkillStep("cluster_pattern", "Cluster Pattern", "State whether the visible lesion pattern is solitary, clustered, scattered, or unknown.", ["grouping pattern"]),
        ],
        watch_outs=[
            "A single clinical image often under-represents true distribution.",
            "Do not infer generalized disease from one cropped image.",
            "Metadata body region may be more reliable than weak background cues.",
        ],
        output_schema=[
            SkillSchemaField("body_location", "str", "Most likely body location."),
            SkillSchemaField("symmetry", "str", "Whether distribution appears symmetric, asymmetric, or uncertain."),
            SkillSchemaField("localized_vs_generalized", "str", "Whether the visible pattern is localized, generalized, or uncertain."),
            SkillSchemaField("clustering_pattern", "str", "Whether the visible lesion pattern appears solitary, clustered, scattered, or unknown."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the distribution analysis routine.\n"
            "When to use: use when anatomic site or pattern extent matters for differential refinement.\n"
            "What evidence to inspect: site metadata, visible context, solitary versus grouped pattern, symmetry.\n"
            "Common pitfalls: single-image crops limit true distribution judgment.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )
