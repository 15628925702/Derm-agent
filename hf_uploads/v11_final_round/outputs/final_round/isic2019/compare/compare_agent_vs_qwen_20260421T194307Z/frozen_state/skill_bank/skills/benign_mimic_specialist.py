from __future__ import annotations

from typing import Any

from agent.confusion_clusters import cluster_pairs, cluster_related_keywords
from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class BenignMimicSpecialistSkill(BaseSkill):
    name = "benign_mimic_specialist_skill"
    description = "Analyze benign-mimic versus malignant/keratinocyte confusion without making the final call."
    output_fields = (
        "differentiation_features",
        "supporting_evidence",
        "opposing_evidence",
        "required_missing_evidence",
        "uncertainty_under_current_evidence",
        "further_observation_suggestions",
        "cluster_watchouts",
        "referenced_confusion_patterns",
        "critical_supporting_evidence",
        "counterexample_watchouts",
    )
    list_fields = output_fields[:-1] + ("counterexample_watchouts",)
    skill_object = make_skill_object(
        skill_id="skill.benign_mimic_specialist.v1",
        name=name,
        description=description,
        skill_type="specialist",
        triggers=[
            SkillTrigger(
                condition="Trigger when benign keratosis, dermatofibroma, vascular lesion, or nevus-like mimics compete with melanoma or keratinocyte malignancy in the differential.",
                rationale="These benign mimics frequently absorb diagnostic mass in HAM10000-like label spaces and require explicit comparison against malignant and keratinocytic alternatives.",
            )
        ],
        workflow_text=(
            "Focus on benign mimic differentiation. Compare benign keratosis, dermatofibroma, vascular lesion, and nevus-like explanations "
            "against melanoma-like or keratinocyte-malignant explanations. Explicitly separate lesion-specific support for benign mimicry "
            "from lesion-specific support for malignancy or actinic keratinocytic change. Name what visible clues argue for benign mimicry, "
            "what clues still support malignant or keratinocytic concern, and what missing evidence prevents cleaner separation. "
            "Do not make the final diagnosis."
        ),
        steps=[
            SkillStep("scope_pair", "Scope Pair", "Restrict reasoning to the active benign-mimic versus malignant/keratinocytic confusion.", ["active ddx pair"]),
            SkillStep("compare_benign_clues", "Compare Benign Clues", "List image-supported clues for benign keratosis, nevus-like, vascular, or dermatofibroma-style mimicry.", ["surface texture", "uniformity", "vascular appearance", "keratotic surface"]),
            SkillStep("compare_malignant_clues", "Compare Malignant Clues", "List image-supported clues that still support melanoma or keratinocyte concern.", ["asymmetry", "border irregularity", "pigment complexity", "ulcer/crust"]),
            SkillStep("name_missing_checks", "Name Missing Checks", "State the highest-value missing evidence that would separate benign mimicry from malignancy.", ["dermoscopy", "surface magnification", "vascular detail", "history"]),
        ],
        watch_outs=[
            "Do not collapse comparison into the final diagnosis.",
            "Do not use generic age, site, or prevalence as supporting or opposing evidence.",
            "Do not treat color alone as sufficient support for melanoma.",
            "Do not treat keratotic surface alone as sufficient support for actinic keratosis or SCC.",
            "If benign mimicry is plausible, preserve at least one explicit opposing clue against malignancy escalation.",
        ],
        output_schema=[
            SkillSchemaField("differentiation_features", "list[str]", "Features that distinguish benign mimicry from melanoma or keratinocyte malignancy."),
            SkillSchemaField("supporting_evidence", "list[str]", "Lesion-specific evidence supporting the currently favored side of the benign-mimic comparison."),
            SkillSchemaField("opposing_evidence", "list[str]", "Lesion-specific evidence opposing the currently favored side."),
            SkillSchemaField("required_missing_evidence", "list[str]", "Missing checks that would best resolve the benign-mimic confusion."),
            SkillSchemaField("uncertainty_under_current_evidence", "str", "Low, medium, or high residual uncertainty under the current evidence."),
            SkillSchemaField("further_observation_suggestions", "list[str]", "Additional observations that would best resolve the confusion."),
            SkillSchemaField("cluster_watchouts", "list[str]", "Confusion-cluster watch-outs that should not be ignored."),
            SkillSchemaField("referenced_confusion_patterns", "list[str]", "Referenced abstract confusion patterns used during reasoning."),
            SkillSchemaField("critical_supporting_evidence", "list[str]", "Most decision-relevant supporting clues."),
            SkillSchemaField("counterexample_watchouts", "list[str]", "Counterexamples or prototype mismatches that should temper confidence."),
        ],
    )

    def select_related_abstract_experiences(
        self,
        state: CaseState,
        abstract_experiences: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        cluster_names = self.active_confusion_clusters(state)
        dataset_name = getattr(state.case_input, "dataset_name", None)
        return self.filter_related_abstract_experiences(
            abstract_experiences,
            confusion_pairs=cluster_pairs(cluster_names, dataset_name=dataset_name),
            keywords=cluster_related_keywords(cluster_names, dataset_name=dataset_name),
            allowed_types=("confusion_memory", "prototype", "rule"),
            top_k=3,
        )

    def build_prompt(self, state: CaseState) -> str:
        cluster_payload = self.confusion_cluster_prompt_payload(state, max_items=3)
        return (
            f"{self.workflow_text()}\n"
            "You are executing the benign-mimic specialist routine.\n"
            "When to use: use when benign keratosis, dermatofibroma, vascular lesion, or nevus-like mimicry competes with melanoma or keratinocyte malignancy.\n"
            "What evidence to inspect: keratotic or waxy surface, stuck-on character, symmetry, vascular appearance, pigment network, border irregularity, ulcer/crust, and whether the lesion truly looks melanocytic.\n"
            "If related abstract experiences are provided, explicitly use confusion_memory, prototype, and rule patterns as auxiliary references.\n"
            "Common pitfalls: over-calling melanoma from color alone, over-calling actinic keratosis from scale alone, or ignoring benign-mimic surface structure.\n"
            "Supporting evidence and opposing evidence must stay lesion-specific. Do not use age, prevalence, site, or generic risk background as supporting_evidence or opposing_evidence.\n"
            "Do NOT output a final diagnosis.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('lesion_description_structuring_skill', 'differential_compare_skill', 'mel_nev_specialist_skill', 'ack_scc_specialist_skill', 'malignancy_risk_assessment_skill', 'exclusion_reasoning_skill'), max_skills=6)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
            f"Active confusion cluster guidance: {cluster_payload}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name == "uncertainty_under_current_evidence":
            normalized = str(value).strip().lower()
            if normalized not in {"low", "medium", "high"}:
                return "medium" if value else "unknown"
            return normalized
        return super().normalize_field(field_name, value)
