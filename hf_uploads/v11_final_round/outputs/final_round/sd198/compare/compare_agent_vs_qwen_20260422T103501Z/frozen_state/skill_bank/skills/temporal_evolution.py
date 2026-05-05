from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class TemporalEvolutionSkill(BaseSkill):
    name = "temporal_evolution_skill"
    description = "Summarize temporal evolution using metadata cues."
    output_fields = ("onset_type", "progression_speed", "recurrence", "stability")
    skill_object = make_skill_object(
        skill_id="skill.temporal_evolution.v1",
        name=name,
        description=description,
        skill_type="reasoning",
        triggers=[
            SkillTrigger(
                condition="Use when history fields indicate change, bleeding, growth, recurrence, or stability concerns.",
                rationale="Time-course is a core clinical reasoning dimension that cannot be recovered from morphology alone.",
            )
        ],
        workflow_text=(
            "Use history-bearing metadata to reconstruct the lesion's temporal behavior. Decide whether the process looks "
            "acute or chronic, whether change is stable, slow, or rapid, whether recurrence is suggested, and whether the "
            "overall trajectory appears stable or unstable. This routine translates history into structured temporal evidence."
        ),
        steps=[
            SkillStep("collect_history", "Collect History Cues", "Review metadata fields related to growth, change, bleeding, and symptoms.", ["grew", "changed", "bleed", "itch", "hurt"]),
            SkillStep("onset_frame", "Estimate Onset", "Infer whether the presentation feels acute or chronic.", ["age/history context"]),
            SkillStep("tempo", "Estimate Tempo", "Judge whether change appears stable, slow, or rapid.", ["progression cues"]),
            SkillStep("stability", "Stability Summary", "State recurrence and stability explicitly.", ["recurrence cue", "overall stability"]),
        ],
        watch_outs=[
            "Metadata history may be sparse or noisy.",
            "Absence of recorded change is not proof of true stability.",
            "Symptoms like itch can be nonspecific and should not dominate interpretation.",
        ],
        output_schema=[
            SkillSchemaField("onset_type", "str", "Whether onset appears acute, chronic, or unknown."),
            SkillSchemaField("progression_speed", "str", "Whether progression appears stable, slow, rapid, or unknown."),
            SkillSchemaField("recurrence", "str", "Whether recurrence appears yes, no, or uncertain."),
            SkillSchemaField("stability", "str", "Whether the lesion appears stable, unstable, or uncertain."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the temporal evolution routine.\n"
            "When to use: use when history or symptom metadata can clarify lesion behavior over time.\n"
            "What evidence to inspect: growth, change, bleeding, symptoms, and stability cues.\n"
            "Common pitfalls: sparse history is not equivalent to reassuring history.\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )
