from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class BorderSurfaceAnalysisSkill(BaseSkill):
    name = "border_surface_analysis_skill"
    description = "Assess border clarity, irregularity, and surface texture."
    output_fields = (
        "border_clarity",
        "border_irregularity",
        "surface_texture",
        "scaling_presence",
    )
    skill_object = make_skill_object(
        skill_id="skill.border_surface_analysis.v1",
        name=name,
        description=description,
        skill_type="observation",
        triggers=[
            SkillTrigger(
                condition="Use when lesion margin and surface texture may alter malignancy concern or differential ordering.",
                rationale="Border and surface are foundational descriptive clues in dermatology.",
            )
        ],
        workflow_text=(
            "Inspect the lesion edge and outer contour, then study the visible surface quality. Decide whether the border "
            "is crisp or poorly defined, whether it is regular or irregular, and whether the surface is smooth, rough, "
            "keratotic, or ulcerated. This routine describes margin and surface evidence only."
        ),
        steps=[
            SkillStep("border_definition", "Border Definition", "Judge how clearly the lesion edge is demarcated.", ["edge sharpness"]),
            SkillStep("border_shape", "Border Shape", "Assess regularity versus irregularity of the lesion outline.", ["contour irregularity"]),
            SkillStep("surface_texture", "Surface Texture", "Describe the visible surface character.", ["smoothness", "keratosis", "ulceration"]),
            SkillStep("scaling_check", "Scaling Check", "State whether visible scale is present.", ["surface scale"]),
        ],
        watch_outs=[
            "Blur, hair, and shadow can falsely reduce border clarity.",
            "Do not infer malignancy from irregularity alone.",
            "Shiny highlights can obscure scale or ulceration.",
        ],
        output_schema=[
            SkillSchemaField("border_clarity", "str", "Clarity of lesion boundary."),
            SkillSchemaField("border_irregularity", "str", "Regularity of lesion border shape."),
            SkillSchemaField("surface_texture", "str", "Visible surface texture category."),
            SkillSchemaField("scaling_presence", "str", "Whether visible scaling is present, absent, or uncertain."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the border and surface analysis routine.\n"
            "When to use: use early when visible edge and surface cues help constrain the lesion description.\n"
            "What evidence to inspect: edge definition, contour pattern, texture, and scale.\n"
            "Common pitfalls: blur, glare, and occlusion can mimic poor definition or surface irregularity.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Case perception so far: {self.perception_snapshot(state)}\n"
            f"Case metadata: {self.metadata_snapshot(state)}\n"
        )
