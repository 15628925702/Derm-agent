from __future__ import annotations

from typing import Any

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class AckSccSpecialistSkill(BaseSkill):
    name = "ack_scc_specialist_skill"
    description = "Analyze keratinocyte confusion clues (ACK/SCC/BCC) without making the final call."
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
        skill_id="skill.ack_scc_specialist.v1",
        name=name,
        description=description,
        skill_type="specialist",
        triggers=[
            SkillTrigger(
                condition="Trigger when ACK/SCC/BCC keratinocyte-line confusion is active, especially SCC↔BCC or ACK↔BCC in high-uncertainty/high-risk cases.",
                rationale="These clusters are repeatedly hard and require explicit pairwise specialist distinction rather than generic comparison.",
            )
        ],
        workflow_text=(
            "Focus narrowly on keratinocyte-line confusion (ACK/SCC/BCC). First define the active pair (e.g., SCC-vs-BCC or ACK-vs-BCC), "
            "then compare scale/keratinization versus pearly-translucent morphology, ulceration, and invasive-looking surface disruption. "
            "Explicitly list negative evidence that argues against over-calling SCC, and what missing evidence still prevents strong exclusion. "
            "Produce pairwise differentiation clues only; do not make the final diagnosis."
        ),
        steps=[
            SkillStep("pair_focus", "Focus the Pair", "Restrict reasoning to one active keratinocyte confusion pair at a time (ACK/SCC/BCC family).", ["active ddx pair"]),
            SkillStep("differentiate", "Differentiate", "List distinction clues with explicit positive and negative evidence.", ["keratinization", "scale", "pearly/translucent cues", "ulceration", "surface disruption"]),
            SkillStep("balance_evidence", "Balance Evidence", "Separate support, opposition, and missing evidence that blocks overconfident narrowing.", ["supporting vs opposing clues", "required missing evidence"]),
            SkillStep("next_observation", "Next Observation", "State what additional observation would most efficiently resolve the pair.", ["closer border/surface inspection", "vascular pattern clues", "history", "dermoscopy"]),
        ],
        watch_outs=[
            "Do not collapse ACK-vs-SCC analysis into a final label.",
            "Do not introduce unrelated candidate diseases.",
            "Surface roughness alone is not equivalent to invasive carcinoma.",
            "Do not equate hyperkeratosis by itself with SCC when BCC-like cues are present.",
            "If BCC is in play, list at least one opposing clue before escalating concern.",
        ],
        output_schema=[
            SkillSchemaField("differentiation_features", "list[str]", "Features that help distinguish ACK from SCC."),
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
            confusion_pairs=(
                "ack->scc",
                "scc->bcc",
                "squamous cell carcinoma->bcc",
                "ack->bcc",
                "actinic keratosis->bcc",
                "seborrheic keratosis->bcc",
            ),
            keywords=("ack", "scc", "bcc", "actinic", "keratin", "seborrheic"),
            allowed_types=("confusion_memory", "prototype", "rule"),
            top_k=2,
        )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the keratinocyte confusion specialist routine (ACK/SCC/BCC family).\n"
            "When to use: when ACK/SCC/BCC confusion is active or strongly suspected from retrieved confusion memory.\n"
            "What evidence to inspect: scale/keratinization, pearly-translucent cues, ulceration, vascular hints, and invasive-looking surface change.\n"
            "If related abstract experiences are provided, explicitly use confusion_memory, prototype, and rule patterns as auxiliary comparison references.\n"
            "Common pitfalls: equating roughness or keratin alone with SCC, and ignoring BCC-like counter-clues.\n"
            "Keep the response compact: use at most 3 short items per list field, and keep each item as a short phrase.\n"
            "Do NOT output a final diagnosis, final winner, or definitive disease label.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'border_surface_analysis_skill', 'temporal_evolution_skill', 'malignancy_risk_assessment_skill'), max_skills=4)}\n"
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
