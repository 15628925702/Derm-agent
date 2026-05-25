from __future__ import annotations

from typing import Any

from agent.label_space import canonicalize_label
from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class XiangyaAcneDisambiguationSkill(BaseSkill):
    name = "xiangya_acne_disambiguation_skill"
    description = "Disambiguate Xiangya COMMON_ACNE from dermatitis and vitiligo without making an open-set diagnosis."
    output_fields = (
        "candidate_focus",
        "comedone_presence",
        "pustule_presence",
        "follicular_centered_papules_presence",
        "true_depigmentation_presence",
        "diffuse_eczematous_morphology_presence",
        "acne_supporting_findings",
        "evidence_against_dermatitis",
        "evidence_against_vitiligo",
        "canonical_label_recommendation",
        "recommendation_rationale",
        "confidence_under_current_evidence",
    )
    list_fields = (
        "candidate_focus",
        "acne_supporting_findings",
        "evidence_against_dermatitis",
        "evidence_against_vitiligo",
    )
    skill_object = make_skill_object(
        skill_id="skill.xiangya_acne_disambiguation.v1",
        name=name,
        description=description,
        skill_type="specialist",
        triggers=[
            SkillTrigger(
                condition=(
                    "Trigger for Xiangya 7-class cases when COMMON_ACNE may be confused with "
                    "ATOPIC_DERMATITIS, ECZEMA_DERMATITIS, or VITILIGO."
                ),
                rationale=(
                    "The Hulu-Med Xiangya workflow repeatedly drifts from acneiform lesions to dermatitis or vitiligo; "
                    "this skill must produce explicit positive and negative morphology evidence."
                ),
            )
        ],
        workflow_text=(
            "Focus narrowly on Xiangya 7-class acne disambiguation. Inspect whether the image shows comedones, pustules, "
            "or follicular-centered papules that support `COMMON_ACNE`; separately inspect whether there is true pigment loss "
            "supporting `VITILIGO`, or diffuse eczematous morphology supporting `ATOPIC_DERMATITIS`/`ECZEMA_DERMATITIS`. "
            "The output must use Xiangya canonical label IDs, especially `COMMON_ACNE`, and must preserve explicit evidence "
            "against dermatitis/vitiligo when acneiform morphology is present."
        ),
        steps=[
            SkillStep("scope_pair", "Scope Confusion", "Restrict the comparison to COMMON_ACNE vs dermatitis/vitiligo labels.", ["active Xiangya labels"]),
            SkillStep("find_acne_units", "Find Acne Units", "Look for comedones, pustules, follicular-centered papules, and acneiform clustering.", ["comedones", "pustules", "follicular papules"]),
            SkillStep("exclude_vitiligo", "Exclude Vitiligo", "Check whether pale areas are true depigmentation rather than lighting, scale, or erythema.", ["depigmentation", "hypopigmentation"]),
            SkillStep("exclude_dermatitis", "Exclude Dermatitis", "Check whether the morphology is diffuse eczematous dermatitis or discrete acneiform lesions.", ["eczema", "xerosis", "diffuse rash"]),
            SkillStep("canonical_recommendation", "Canonical Recommendation", "Recommend only one Xiangya canonical label ID or uncertain.", ["canonical label ID"]),
        ],
        watch_outs=[
            "Do not output open-set labels such as `Acne`; use `COMMON_ACNE` if acne is supported.",
            "Do not call vitiligo unless true depigmentation is visible.",
            "Do not call dermatitis from redness alone when follicular papules, pustules, or comedones are present.",
            "Do not use chronic history or atopy history unless it is actually available in metadata.",
            "This skill provides structured evidence for fusion; it is not the final diagnosis step.",
        ],
        output_schema=[
            SkillSchemaField("candidate_focus", "list[str]", "Active Xiangya labels being compared."),
            SkillSchemaField("comedone_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("pustule_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("follicular_centered_papules_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("true_depigmentation_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("diffuse_eczematous_morphology_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("acne_supporting_findings", "list[str]", "Image findings supporting COMMON_ACNE."),
            SkillSchemaField("evidence_against_dermatitis", "list[str]", "Findings arguing against ATOPIC_DERMATITIS or ECZEMA_DERMATITIS."),
            SkillSchemaField("evidence_against_vitiligo", "list[str]", "Findings arguing against VITILIGO."),
            SkillSchemaField(
                "canonical_label_recommendation",
                "str",
                "One of COMMON_ACNE, ATOPIC_DERMATITIS, ECZEMA_DERMATITIS, VITILIGO, or uncertain.",
            ),
            SkillSchemaField("recommendation_rationale", "str", "Short rationale for the canonical recommendation."),
            SkillSchemaField("confidence_under_current_evidence", "str", "low, medium, or high."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the Xiangya COMMON_ACNE disambiguation routine.\n"
            "When to use: Xiangya 7-class cases where acne may be confused with dermatitis or vitiligo.\n"
            "What evidence to inspect: comedones, pustules, follicular-centered papules, acneiform distribution, true pigment loss, diffuse eczematous morphology, xerosis, scale, and lichenification.\n"
            "Important: `Acne` is not an allowed final label ID in this dataset. If acne is the supported category, write `COMMON_ACNE` exactly in canonical_label_recommendation.\n"
            "Return compact JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'distribution_analysis_skill', 'lesion_description_structuring_skill', 'differential_compare_skill'), max_skills=6)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_output(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        normalized = super().normalize_output(raw_output)
        recommendation = str(normalized.get("canonical_label_recommendation", "")).strip()
        canonical = canonicalize_label(
            recommendation,
            label_space_id="xiangya_7class",
            dataset_name="xiangya_7class",
        )
        if canonical in {"COMMON_ACNE", "ATOPIC_DERMATITIS", "ECZEMA_DERMATITIS", "VITILIGO"}:
            normalized["canonical_label_recommendation"] = canonical
        else:
            normalized["canonical_label_recommendation"] = "uncertain"
        if normalized.get("recommendation_type") == "descriptive_evidence":
            normalized["recommendation_type"] = "comparative_support"
        return normalized

    def after_execute(self, state: CaseState, output: dict[str, Any]) -> None:
        derived = self._deterministic_acne_readout(state)
        if not derived:
            return
        output.update(derived)
        state.skill_outputs[self.name] = output

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name in {
            "comedone_presence",
            "pustule_presence",
            "follicular_centered_papules_presence",
            "true_depigmentation_presence",
            "diffuse_eczematous_morphology_presence",
        }:
            return self._normalize_presence(value)
        if field_name == "confidence_under_current_evidence":
            normalized = str(value).strip().lower()
            if normalized in {"low", "medium", "high"}:
                return normalized
            return "medium" if value else "unknown"
        return super().normalize_field(field_name, value)

    @staticmethod
    def _normalize_presence(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"present", "yes", "true", "seen", "visible", "positive"}:
            return "present"
        if text in {"absent", "no", "false", "not seen", "none", "negative"}:
            return "absent"
        return "uncertain"

    @staticmethod
    def _deterministic_acne_readout(state: CaseState) -> dict[str, Any]:
        perception = state.perception or {}
        metadata = state.clinical_metadata or {}
        notes = " ".join(str(item) for item in perception.get("notes", []) if str(item).strip()).lower()
        image_summary = str(perception.get("image_summary", "")).lower()
        uncertainty_reasons = " ".join(
            str(item)
            for item in (
                perception.get("uncertainty", {}).get("reasons", [])
                if isinstance(perception.get("uncertainty", {}), dict)
                else []
            )
            if str(item).strip()
        ).lower()
        combined = " ".join(
            [
                image_summary,
                notes,
                uncertainty_reasons,
                str(metadata.get("related_category", "")).lower(),
                str(metadata.get("presentation_mode_hint", "")).lower(),
                " ".join(str(item).lower() for item in perception.get("ddx_candidates", []) if str(item).strip()),
            ]
        )
        acne_signals = (
            "comedone",
            "comedones",
            "pustule",
            "pustules",
            "papule",
            "papules",
            "acneiform",
            "follicular",
        )
        dermatitis_signals = (
            "diffuse rash",
            "eczema",
            "eczemat",
            "atopic dermatitis",
            "xerosis",
            "flexural",
            "lichenif",
        )
        vitiligo_signals = (
            "depigment",
            "hypopigment",
            "white patch",
            "white patches",
        )
        acne_hits = sum(1 for term in acne_signals if term in combined)
        dermatitis_hits = sum(1 for term in dermatitis_signals if term in combined)
        vitiligo_hits = sum(1 for term in vitiligo_signals if term in combined)
        if acne_hits < 2 or (dermatitis_hits > acne_hits and vitiligo_hits >= acne_hits):
            return {}
        supporting = []
        if "comedone" in combined or "comedones" in combined:
            supporting.append("comedones present")
        if "pustule" in combined or "pustules" in combined:
            supporting.append("pustules present")
        if "papule" in combined or "papules" in combined:
            supporting.append("papules present")
        if "acneiform" in combined:
            supporting.append("acneiform morphology")
        if "follicular" in combined:
            supporting.append("follicular-centered pattern")
        evidence_against_dermatitis = []
        if "no significant xerosis" in combined:
            evidence_against_dermatitis.append("no significant xerosis")
        if "no clear signs of eczema" in combined:
            evidence_against_dermatitis.append("no clear eczema signs")
        if "atopic" in combined and "atypical distribution for atopic dermatitis" in combined:
            evidence_against_dermatitis.append("distribution atypical for atopic dermatitis")
        evidence_against_vitiligo = []
        if "no depigmentation" in combined or "no hypopigmentation" in combined:
            evidence_against_vitiligo.append("no true depigmentation")
        if vitiligo_hits == 0:
            evidence_against_vitiligo.append("no pigment-loss signal")
        return {
            "candidate_focus": ["COMMON_ACNE", "ATOPIC_DERMATITIS", "ECZEMA_DERMATITIS", "VITILIGO"],
            "comedone_presence": "present" if "comedone" in combined or "comedones" in combined else "uncertain",
            "pustule_presence": "present" if "pustule" in combined or "pustules" in combined else "uncertain",
            "follicular_centered_papules_presence": "present" if "follicular" in combined or "papule" in combined or "papules" in combined else "uncertain",
            "true_depigmentation_presence": "absent" if vitiligo_hits == 0 else "uncertain",
            "diffuse_eczematous_morphology_presence": "absent" if dermatitis_hits < 2 else "uncertain",
            "acne_supporting_findings": supporting[:4] or ["acneiform eruption pattern"],
            "evidence_against_dermatitis": evidence_against_dermatitis[:4] or ["no dominant diffuse eczematous pattern"],
            "evidence_against_vitiligo": evidence_against_vitiligo[:4] or ["no depigmented patch pattern"],
            "canonical_label_recommendation": "COMMON_ACNE",
            "recommendation_rationale": "Acneiform morphology with comedones/pustules/papules is stronger than dermatitis or vitiligo clues.",
            "confidence_under_current_evidence": "high" if acne_hits >= 3 else "medium",
        }
