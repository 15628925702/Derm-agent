from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class MorphologyAnalysisSkill(BaseSkill):
    name = "morphology_analysis_skill"
    description = "Extract primary lesion morphology."
    output_fields = ("lesion_type", "size_range", "elevation", "count")
    skill_object = make_skill_object(
        skill_id="skill.morphology_analysis.v1",
        name=name,
        description=description,
        skill_type="observation",
        triggers=[
            SkillTrigger(
                condition="Run in nearly all lesion cases before downstream comparison.",
                rationale="Morphology is the first dermatologist-style structural description needed for later reasoning.",
            )
        ],
        workflow_text=(
            "Examine the lesion as a dermatologist would at first glance: determine the dominant lesion form, "
            "estimate whether it is flat or raised, note the approximate size band, and decide whether the visible "
            "finding is solitary or part of a broader visible set. This routine should describe morphology only and "
            "must not collapse directly into a disease label."
        ),
        steps=[
            SkillStep("observe_form", "Observe Form", "Inspect the dominant visible lesion form.", ["lesion silhouette", "elevation cue"]),
            SkillStep("estimate_size", "Estimate Size", "Estimate coarse size band from image and metadata clues.", ["diameter fields", "relative image scale"]),
            SkillStep("judge_elevation", "Judge Elevation", "Decide whether the lesion appears flat, raised, or mixed.", ["surface height cue"]),
            SkillStep("count_pattern", "Count Pattern", "State whether the visible lesion is solitary or part of a cluster.", ["single lesion context"]),
        ],
        watch_outs=[
            "Do not infer disease identity from morphology alone.",
            "Do not overstate size when the image scale is uncertain.",
            "Flat pigmented lesions can still have subtle surface variation that should not be mistaken for nodularity.",
        ],
        output_schema=[
            SkillSchemaField("lesion_type", "str", "Primary lesion morphology class such as macule, papule, plaque, nodule, or unknown."),
            SkillSchemaField("size_range", "str", "Coarse size range estimate."),
            SkillSchemaField("elevation", "str", "Whether the lesion appears flat, raised, mixed, or unknown."),
            SkillSchemaField("count", "str", "Whether the lesion appears solitary, multiple, clustered, or unknown."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the morphology analysis routine.\n"
            "When to use: this skill is used early to establish the lesion's basic structural description.\n"
            "What evidence to inspect: lesion form, size cue, elevation cue, and visible lesion count.\n"
            "Common pitfalls: do not confuse diagnostic suspicion with morphology; avoid overconfident size claims.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Case perception so far: {self.perception_snapshot(state)}\n"
            f"Case metadata: {self.metadata_snapshot(state)}\n"
        )
