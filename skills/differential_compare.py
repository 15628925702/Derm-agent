from __future__ import annotations

from typing import Any

from agent.confusion_clusters import cluster_pairs, cluster_related_keywords
from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class DifferentialCompareSkill(BaseSkill):
    name = "differential_compare_skill"
    description = "Compare top differential candidates without selecting the final diagnosis."
    output_fields = ("candidate_pairs", "supporting_evidence", "conflicting_evidence", "required_missing_evidence")
    list_fields = ("candidate_pairs", "supporting_evidence", "conflicting_evidence", "required_missing_evidence")
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
            SkillSchemaField("required_missing_evidence", "list[str]", "Missing evidence that would most efficiently resolve the active comparison."),
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
        cluster_payload = self.confusion_cluster_prompt_payload(state, max_items=2)
        return (
            f"{self.workflow_text()}\n"
            "You are executing the differential comparison routine.\n"
            "When to use: use after initial ddx generation when two or more plausible candidates remain.\n"
            "What evidence to inspect: current ddx list and previously generated structured clues.\n"
            "Common pitfalls: choosing a winner too early, adding unsupported candidate names, or hiding which missing evidence keeps the pair unresolved.\n"
            "If active confusion clusters are provided, explicitly compare supporting evidence, opposing evidence, and required missing evidence for that cluster rather than drifting into generic prose.\n"
            "Do NOT choose a final diagnosis and do NOT output a single winning disease.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'temporal_evolution_skill', 'malignancy_risk_assessment_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
            f"Active confusion cluster guidance: {cluster_payload}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name in {"supporting_evidence", "conflicting_evidence", "required_missing_evidence"} and isinstance(value, dict):
            flattened: list[str] = []
            for pair_name, items in value.items():
                if isinstance(items, list):
                    item_text = "; ".join(str(item).strip() for item in items[:2] if str(item).strip())
                else:
                    item_text = str(items).strip()
                if item_text:
                    flattened.append(f"{str(pair_name).strip()}: {item_text}")
            return flattened
        if field_name == "candidate_pairs" and isinstance(value, dict):
            return [str(key).strip() for key in value.keys() if str(key).strip()]
        return super().normalize_field(field_name, value)
