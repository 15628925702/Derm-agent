from __future__ import annotations

from typing import Any

from agent.confusion_clusters import cluster_pairs, cluster_related_keywords
from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class ExclusionReasoningSkill(BaseSkill):
    name = "exclusion_reasoning_skill"
    description = "Model physician-style exclusion reasoning using negative evidence and missing evidence."
    output_fields = (
        "unlikely_candidates",
        "exclusion_evidence",
        "required_missing_evidence",
        "exclusion_confidence",
    )
    list_fields = ("unlikely_candidates", "exclusion_evidence", "required_missing_evidence")
    skill_object = make_skill_object(
        skill_id="skill.exclusion_reasoning.v1",
        name=name,
        description=description,
        skill_type="reasoning",
        triggers=[
            SkillTrigger(
                condition="Use when two or more active differential candidates remain and the case would benefit from explicit exclusion-style reasoning.",
                rationale="Doctors often narrow a differential by stating what is less likely, why it is less likely, and what missing evidence still limits strong exclusion.",
            )
        ],
        workflow_text=(
            "Perform physician-style exclusion reasoning over the currently active candidate set. "
            "Identify which already-mentioned candidates look less likely, state the negative or opposing evidence "
            "that weakens them, and separate this from evidence that is simply missing. "
            "For frequent hard clusters (AK/SCC/SEK versus BCC, inflammatory lesions versus ACK), explicitly name which classic clues are absent "
            "and which missing checks prevent confident exclusion. "
            "This skill must narrow reasoning transparently without producing the final diagnosis."
        ),
        steps=[
            SkillStep("scope_candidates", "Scope Candidates", "Limit exclusion reasoning to candidates already present in the current differential or active pairwise comparisons.", ["current ddx", "specialist comparisons"]),
            SkillStep("collect_negative_evidence", "Collect Negative Evidence", "List features that actively argue against a candidate rather than support it.", ["absent typical clues", "opposing morphology or pattern clues"]),
            SkillStep("separate_missing_evidence", "Separate Missing Evidence", "State what evidence is still missing and why that prevents strong exclusion.", ["needed dermoscopy", "missing history", "missing border/vascular detail", "missing evolution support"]),
            SkillStep("grade_exclusion", "Grade Exclusion", "Assign low, medium, or high exclusion confidence based on the current strength of the negative case.", ["strength of negative evidence", "dependency on missing evidence"]),
        ],
        watch_outs=[
            "Do not invent new disease labels outside the active candidate set.",
            "Do not confuse missing support with true contradictory evidence.",
            "Do not convert exclusion reasoning into a final winner selection.",
            "Do not overstate exclusion confidence when critical evidence is still missing.",
            "For BCC-related confusion, do not claim exclusion without at least one explicit opposing clue and one required missing check.",
            "For inflammatory-versus-ACK confusion, avoid treating scale alone as sufficient evidence for actinic keratosis.",
        ],
        output_schema=[
            SkillSchemaField("unlikely_candidates", "list[str]", "Candidates already in scope that currently look less likely."),
            SkillSchemaField("exclusion_evidence", "list[str]", "Negative or opposing evidence that weakens those candidates."),
            SkillSchemaField("required_missing_evidence", "list[str]", "Evidence that would be needed for stronger exclusion."),
            SkillSchemaField("exclusion_confidence", "str", "Low, medium, or high confidence in the current exclusion reasoning."),
        ],
    )

    def select_related_abstract_experiences(
        self,
        state: CaseState,
        abstract_experiences: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        cluster_names = self.active_confusion_clusters(state)
        return self.filter_related_abstract_experiences(
            abstract_experiences,
            confusion_pairs=cluster_pairs(cluster_names),
            keywords=cluster_related_keywords(cluster_names),
            allowed_types=("confusion_memory", "prototype", "rule"),
            top_k=3,
        )

    def build_prompt(self, state: CaseState) -> str:
        cluster_payload = self.confusion_cluster_prompt_payload(state, max_items=2)
        return (
            f"{self.workflow_text()}\n"
            "You are executing the exclusion reasoning routine.\n"
            "When to use: use when the differential remains open and physician-style narrowing by exclusion would improve interpretability.\n"
            "What evidence to inspect: current ddx, structured observation outputs, pairwise comparisons, specialist comparisons, uncertainty, and missing evidence.\n"
            "Common pitfalls: introducing new diagnoses, treating missing evidence as hard contradiction, and naming a final winner.\n"
            "Exclusion evidence must stay lesion-specific. Do not use age, body site, Fitzpatrick type, or generic risk background as exclusion_evidence unless directly tied to a visible lesion pattern.\n"
            "Hard-cluster requirement: if BCC is in active/related differential, explicitly state at least one exclusion_evidence item and one required_missing_evidence item for BCC.\n"
            "Hard-cluster requirement: if inflammatory descriptors are present with ACK candidate, separate inflammatory mimic clues from true actinic support.\n"
            "Hard-cluster requirement: for ACK/BCC/SCC, ACK/SEK, or MEL/NEV confusion, explicitly keep negative evidence separate from still-missing evidence.\n"
            "Do NOT output a final diagnosis, final winner, or definitive disease class.\n"
            "Only exclude or weaken candidates that are already present in the current case reasoning state.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('lesion_description_structuring_skill', 'differential_compare_skill', 'mel_nev_specialist_skill', 'ack_scc_specialist_skill', 'metadata_consistency_skill', 'uncertainty_assessment_skill'), max_skills=6)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
            f"Active confusion cluster guidance: {cluster_payload}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name == "exclusion_confidence":
            normalized = str(value).strip().lower()
            if normalized not in {"low", "medium", "high"}:
                return "medium" if value else "unknown"
            return normalized
        return super().normalize_field(field_name, value)
