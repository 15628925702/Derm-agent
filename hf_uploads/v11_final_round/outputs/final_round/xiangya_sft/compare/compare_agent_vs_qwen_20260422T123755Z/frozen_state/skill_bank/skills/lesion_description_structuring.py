from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class LesionDescriptionStructuringSkill(BaseSkill):
    name = "lesion_description_structuring_skill"
    description = "Organize image and metadata into a standardized dermatologist-style lesion description."
    output_fields = (
        "primary_lesion_morphology",
        "color",
        "border",
        "surface",
        "size_count",
        "distribution",
        "associated_context",
    )
    list_fields = ("color", "border", "surface", "size_count", "distribution", "associated_context")
    skill_object = make_skill_object(
        skill_id="skill.lesion_description_structuring.v1",
        name=name,
        description=description,
        skill_type="observation",
        triggers=[
            SkillTrigger(
                condition="Use after first-pass observation skills when the case needs a reusable standard lesion description before deeper reasoning.",
                rationale="Doctors often stabilize a case by restating the lesion in structured descriptive language before comparing or excluding diagnoses.",
            )
        ],
        workflow_text=(
            "Convert the current image and metadata into a standard dermatologist-style lesion description. "
            "Start from the primary lesion morphology, then organize color, border, surface, size/count, "
            "distribution, and relevant associated context into a reusable structured description. "
            "This skill must standardize description rather than decide disease identity."
        ),
        steps=[
            SkillStep("anchor_primary_form", "Anchor Primary Form", "Restate the dominant lesion morphology in standard clinical terms.", ["morphology", "elevation"]),
            SkillStep("slot_visible_features", "Slot Visible Features", "Organize color, border, surface, size/count, and distribution into stable descriptive buckets.", ["color variation", "border quality", "surface texture", "count", "distribution"]),
            SkillStep("add_context", "Add Context", "Attach metadata context that changes how the lesion should be read clinically.", ["site", "symptoms", "temporal hints"]),
            SkillStep("preserve_negatives", "Preserve Negatives", "Include important absent or not-clearly-seen findings when they help explain later reasoning.", ["negative evidence", "missing visible support"]),
        ],
        watch_outs=[
            "Do not turn a standardized description into a disease label.",
            "Do not copy noisy metadata into the description if it is not clinically useful.",
            "Do not hide uncertainty when the image does not support a specific surface or border claim.",
            "Do not collapse risk language into the descriptive layer.",
        ],
        output_schema=[
            SkillSchemaField("primary_lesion_morphology", "str", "Standardized primary lesion morphology phrase."),
            SkillSchemaField("color", "list[str]", "Standardized color description elements."),
            SkillSchemaField("border", "list[str]", "Standardized border description elements."),
            SkillSchemaField("surface", "list[str]", "Standardized surface description elements."),
            SkillSchemaField("size_count", "list[str]", "Structured size and lesion count description."),
            SkillSchemaField("distribution", "list[str]", "Structured distribution and site description."),
            SkillSchemaField("associated_context", "list[str]", "Relevant metadata context that should travel with the lesion description."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the lesion description structuring routine.\n"
            "When to use: use after first-pass visual observation to standardize the lesion description into reusable clinical slots.\n"
            "What evidence to inspect: morphology, color, border, surface, size/count, distribution, and clinically relevant metadata context.\n"
            "Common pitfalls: smuggling in diagnosis words, mixing risk language into description, and pretending weakly seen features are definite.\n"
            "Do NOT output any disease diagnosis, final class, or favored disease label.\n"
            "Do NOT output raw metadata dicts, identifiers, filenames, labels, or pathology outcome fields.\n"
            "Prefer standard reusable clinical description phrases that later reasoning can cite directly.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'distribution_analysis_skill', 'temporal_evolution_skill'), max_skills=5)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: object) -> object:
        if field_name in {"color", "border", "surface", "size_count", "distribution", "associated_context"}:
            cleaned: list[str] = []
            values = value if isinstance(value, list) else [value] if value not in (None, "") else []
            for item in values:
                text = str(item).strip()
                if not text:
                    continue
                lowered = text.lower()
                if any(token in lowered for token in ("diagnostic", "patient_id", "lesion_id", "img_id", "biopsed")):
                    continue
                if text.startswith("{") and text.endswith("}"):
                    text = text.replace("{", "").replace("}", "").replace("'", "")
                text = text.strip(" ,")
                if text and text not in cleaned:
                    cleaned.append(text)
            return cleaned
        return super().normalize_field(field_name, value)
