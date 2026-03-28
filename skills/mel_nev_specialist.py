from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class MelNevSpecialistSkill(BaseSkill):
    name = "mel_nev_specialist_skill"
    description = "Analyze MEL vs NEV differentiation clues without making the final call."
    output_fields = (
        "differentiation_features",
        "supporting_evidence",
        "opposing_evidence",
        "further_observation_suggestions",
        "referenced_confusion_patterns",
        "critical_supporting_evidence",
        "counterexample_watchouts",
    )
    list_fields = output_fields
    skill_object = make_skill_object(
        skill_id="skill.mel_nev_specialist.v1",
        name=name,
        description=description,
        skill_type="specialist",
        triggers=[
            SkillTrigger(
                condition="Trigger when melanoma-like and nevus-like candidates are both active, or when retrieved confusion memory indicates recurrent MEL↔NEV ambiguity.",
                rationale="This specialist routine addresses a high-value pigmented-lesion confusion pair and should surface explicit negative evidence.",
            )
        ],
        workflow_text=(
            "Focus narrowly on the MEL-versus-NEV confusion pair. Inspect the lesion for features that would help an expert "
            "separate melanoma-like concern from nevus-like reassurance, list evidence for and against the more concerning side, "
            "state what critical evidence is still missing, and recommend what additional observation would most efficiently resolve the ambiguity. "
            "Do not make the final call."
        ),
        steps=[
            SkillStep("pair_focus", "Focus the Pair", "Restrict reasoning to the MEL vs NEV comparison only.", ["active ddx pair"]),
            SkillStep("differentiate", "Differentiate", "List features that best separate melanoma-like from nevus-like appearance.", ["pigment asymmetry", "border irregularity", "size/evolution clues"]),
            SkillStep("balance_evidence", "Balance Evidence", "Separate supporting and opposing evidence for the concerning side.", ["supporting vs opposing clues"]),
            SkillStep("next_observation", "Next Observation", "State what additional observation would most help resolve the pair.", ["dermoscopy", "better scale", "history detail"]),
        ],
        watch_outs=[
            "Do not decide the winner of the MEL vs NEV pair.",
            "Do not introduce a new disease label outside the active confusion pair.",
            "Specialist reasoning should sharpen the evidence, not replace Qwen's final judgment.",
        ],
        output_schema=[
            SkillSchemaField("differentiation_features", "list[str]", "Features that help distinguish MEL from NEV."),
            SkillSchemaField("supporting_evidence", "list[str]", "Evidence supporting the more concerning side of the pair."),
            SkillSchemaField("opposing_evidence", "list[str]", "Evidence arguing against the more concerning side of the pair."),
            SkillSchemaField("further_observation_suggestions", "list[str]", "Additional observations that would best resolve the confusion pair."),
            SkillSchemaField("referenced_confusion_patterns", "list[str]", "Confusion patterns from abstract experiences that were explicitly used."),
            SkillSchemaField("critical_supporting_evidence", "list[str]", "Most decision-relevant supporting clues for the pairwise distinction."),
            SkillSchemaField("counterexample_watchouts", "list[str]", "Counterexamples, prototype mismatches, or cautionary opposing patterns to keep in mind."),
        ],
    )

    def select_related_abstract_experiences(
        self,
        state: CaseState,
        abstract_experiences: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return self.filter_related_abstract_experiences(
            abstract_experiences,
            confusion_pairs=("melanoma->nev", "malignant melanoma->nev", "malignant melanoma_vs_nev"),
            keywords=("mel", "melanoma", "nev", "naevus", "pigmented"),
            allowed_types=("confusion_memory", "prototype", "rule"),
            top_k=3,
        )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the MEL-vs-NEV specialist routine.\n"
            "When to use: only when MEL and NEV are both active confusion candidates.\n"
            "What evidence to inspect: asymmetry, border irregularity, pigment complexity, size/evolution context.\n"
            "If related abstract experiences are provided, explicitly use confusion_memory, prototype, and rule patterns as auxiliary comparison references.\n"
            "Common pitfalls: turning pairwise analysis into a final verdict and omitting opposing evidence that favors benign nevus patterning.\n"
            "Do NOT output a final diagnosis, final winner, or definitive disease label.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'temporal_evolution_skill', 'malignancy_risk_assessment_skill', 'differential_compare_skill'), max_skills=6)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_output(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        normalized = super().normalize_output(raw_output)
        related_map = self.active_related_abstract_map()
        normalized["referenced_confusion_patterns"] = self._normalize_referenced_confusion_patterns(
            raw_output.get("referenced_confusion_patterns"),
            normalized.get("referenced_experiences", []),
            related_map,
        )
        normalized["referenced_experiences"] = self._augment_referenced_experiences(
            normalized.get("referenced_experiences", []),
            raw_output.get("referenced_confusion_patterns"),
            related_map,
        )
        if normalized.get("recommendation_type") == "descriptive_evidence":
            normalized["recommendation_type"] = "comparative_support"
        return normalized

    @staticmethod
    def _normalize_referenced_confusion_patterns(
        raw_value: Any,
        referenced_experiences: list[str],
        related_map: dict[str, dict[str, Any]],
    ) -> list[str]:
        values = raw_value if isinstance(raw_value, list) else [raw_value] if raw_value not in (None, "") else []
        patterns: list[str] = []
        valid_patterns = {
            str(packet.get("confusion_pair", "")).strip(): source_id
            for source_id, packet in related_map.items()
            if str(packet.get("confusion_pair", "")).strip()
        }
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            if text in related_map:
                pattern = str(related_map[text].get("confusion_pair", "")).strip() or text
                if pattern and pattern not in patterns:
                    patterns.append(pattern)
            elif text in valid_patterns and text not in patterns:
                patterns.append(text)
        for source_id in referenced_experiences:
            if source_id in related_map:
                pattern = str(related_map[source_id].get("confusion_pair", "")).strip()
                if pattern and pattern not in patterns:
                    patterns.append(pattern)
        return patterns

    @staticmethod
    def _augment_referenced_experiences(
        referenced_experiences: list[str],
        raw_confusion_patterns: Any,
        related_map: dict[str, dict[str, Any]],
    ) -> list[str]:
        combined = [str(item).strip() for item in referenced_experiences if str(item).strip()]
        values = raw_confusion_patterns if isinstance(raw_confusion_patterns, list) else [raw_confusion_patterns] if raw_confusion_patterns not in (None, "") else []
        pattern_to_id = {
            str(packet.get("confusion_pair", "")).strip(): source_id
            for source_id, packet in related_map.items()
            if str(packet.get("confusion_pair", "")).strip()
        }
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            if text in related_map and text not in combined:
                combined.append(text)
            elif text in pattern_to_id and pattern_to_id[text] not in combined:
                combined.append(pattern_to_id[text])
        return combined
