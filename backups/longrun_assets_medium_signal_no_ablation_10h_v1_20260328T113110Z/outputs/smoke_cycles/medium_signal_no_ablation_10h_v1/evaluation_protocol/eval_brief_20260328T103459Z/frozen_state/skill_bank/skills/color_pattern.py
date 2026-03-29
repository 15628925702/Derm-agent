from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class ColorPatternAnalysisSkill(BaseSkill):
    name = "color_pattern_analysis_skill"
    description = "Analyze pigmentation pattern and color asymmetry."
    output_fields = (
        "primary_color",
        "color_variation",
        "pigmentation_pattern",
        "asymmetry_color",
    )
    skill_object = make_skill_object(
        skill_id="skill.color_pattern_analysis.v1",
        name=name,
        description=description,
        skill_type="pattern",
        triggers=[
            SkillTrigger(
                condition="Use when pigment, color heterogeneity, or asymmetry may help narrow the differential.",
                rationale="Color pattern is a core dermatologist cue for benign-versus-concerning lesion assessment.",
            )
        ],
        workflow_text=(
            "Inspect the lesion's dominant color and the way pigment is distributed across the lesion. Determine "
            "whether the color is homogeneous or varied, whether the pattern is reticular, patchy, or mixed, and "
            "whether color asymmetry is present. This routine produces pigment evidence only and must not decide the disease."
        ),
        steps=[
            SkillStep("identify_primary_color", "Identify Primary Color", "State the dominant visible lesion color.", ["dominant pigment tone"]),
            SkillStep("judge_variation", "Judge Variation", "Assess whether color is uniform, mildly varied, or markedly varied.", ["color heterogeneity"]),
            SkillStep("pattern_type", "Pattern Type", "Describe the pigmentation arrangement.", ["reticular vs homogeneous vs patchy pattern"]),
            SkillStep("asymmetry_check", "Asymmetry Check", "Determine whether color distribution is symmetric.", ["left-right / center-edge pigment asymmetry"]),
        ],
        watch_outs=[
            "Lighting and image white balance can create false color irregularity.",
            "Do not equate multicolor appearance with a final melanoma diagnosis.",
            "Shadowing near lesion edges can mimic asymmetry.",
        ],
        output_schema=[
            SkillSchemaField("primary_color", "str", "Dominant lesion color."),
            SkillSchemaField("color_variation", "str", "Whether color is uniform, mild, marked, or unknown."),
            SkillSchemaField("pigmentation_pattern", "str", "Pattern of visible pigmentation."),
            SkillSchemaField("asymmetry_color", "str", "Whether color asymmetry is absent, present, or uncertain."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the color pattern analysis routine.\n"
            "When to use: use when pigment distribution can help separate plausible differentials.\n"
            "What evidence to inspect: dominant color, heterogeneity, pattern type, asymmetry.\n"
            "Common pitfalls: lighting, shadow, compression artifacts, and overcalling risk from one color cue.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Case perception so far: {self.perception_snapshot(state)}\n"
            f"Case metadata: {self.metadata_snapshot(state)}\n"
        )
