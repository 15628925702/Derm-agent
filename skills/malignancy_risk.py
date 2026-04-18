from __future__ import annotations

from typing import Any

from agent.label_space import resolve_label_space
from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


def _build_risk_calibration_note(state: CaseState) -> str:
    """Derive a risk calibration note from the label space — no per-dataset hardcoding."""
    case_input = state.case_input
    ls = resolve_label_space(
        label_space_id=getattr(case_input, "label_space_id", None),
        dataset_name=getattr(case_input, "dataset_name", None),
        metadata=getattr(case_input, "metadata", None),
    )
    nm = len(ls.malignant_labels)
    nb = len(ls.benign_labels)
    if nb > nm:
        return (
            f"Calibration: this label space has {nb} benign classes and {nm} malignant classes. "
            "Benign lesions are more numerous. Only assign high risk when morphological evidence is strong and specific — "
            "do not default to high risk for ambiguous lesions."
        )
    if nm > nb:
        return (
            f"Calibration: this label space has {nm} malignant classes and {nb} benign classes. "
            "Malignant and pre-malignant classes are relatively common."
        )
    return ""


class MalignancyRiskAssessmentSkill(BaseSkill):
    name = "malignancy_risk_assessment_skill"
    description = "Assess malignant risk without assigning disease identity."
    output_fields = ("risk_level", "risk_evidence", "alarm_signals", "benign_reassuring_features")
    skill_object = make_skill_object(
        skill_id="skill.malignancy_risk_assessment.v1",
        name=name,
        description=description,
        skill_type="risk_uncertainty",
        triggers=[
            SkillTrigger(
                condition="Use when the current differential contains malignancy possibilities or concerning lesion cues.",
                rationale="Risk stratification helps Qwen interpret whether the case deserves more cautious weighting.",
            )
        ],
        workflow_text=(
            "Review the lesion as a risk-focused dermatologist would: integrate morphology, border, color, symptoms, and "
            "evolution cues to decide whether the lesion currently looks low, medium, or high risk. The routine must name "
            "alarm signals explicitly AND benign-reassuring features explicitly to provide balanced risk assessment. "
            "Cannot convert risk into a final cancer diagnosis."
        ),
        steps=[
            SkillStep("collect_alarm_signals", "Collect Alarm Signals", "Gather visual and metadata cues that raise concern.", ["irregularity", "change", "bleeding", "symptoms"]),
            SkillStep("collect_benign_features", "Collect Benign Features", "Gather visual cues that suggest benign nature.", ["symmetry", "regular border", "uniform color", "stable size"]),
            SkillStep("weigh_risk", "Weigh Risk", "Balance concerning cues against benign features for low/medium/high risk judgement.", ["risk cue aggregation", "benign feature weighting"]),
            SkillStep("state_alarm_summary", "State Alarm Summary", "List explicit alarm signals, supporting evidence, and benign-reassuring features.", ["risk evidence", "alarm signals", "benign features"]),
        ],
        watch_outs=[
            "Risk level is not a final disease label.",
            "Do not suppress low-risk evidence when a single concerning cue appears.",
            "Symptoms can raise caution but are often nonspecific.",
            "MUST list benign-reassuring features when present - do not ignore them.",
        ],
        output_schema=[
            SkillSchemaField("risk_level", "str", "Coarse malignant risk level."),
            SkillSchemaField("risk_evidence", "list[str]", "Structured evidence supporting the current risk estimate."),
            SkillSchemaField("alarm_signals", "list[str]", "Explicit alarm features worth highlighting to Qwen."),
            SkillSchemaField("benign_reassuring_features", "list[str]", "Explicit benign-favoring features that argue against malignancy."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        prior_note = _build_risk_calibration_note(state)
        prior_section = f"Calibration note: {prior_note}\n" if prior_note else ""
        return (
            f"{self.workflow_text()}\n"
            "You are executing the malignancy risk assessment routine.\n"
            "When to use: use when malignancy remains plausible or when alarming visual/history cues are present.\n"
            "What evidence to inspect: border, color, evolution, bleeding, symptoms, and prior skill outputs.\n"
            "Common pitfalls: equating medium/high risk with a definitive malignant label, ignoring benign features.\n"
            f"{prior_section}"
            "\n"
            "Benign-Reassuring Features:\n"
            "When present, document benign characteristics in the benign_reassuring_features field:\n"
            "- Symmetry (bilateral or radial)\n"
            "- Regular border (smooth, well-defined)\n"
            "- Uniform color (single color, homogeneous)\n"
            "- Lack of structural irregularity\n"
            "- Small size (<6mm)\n"
            "- Smooth surface texture\n"
            "\n"
            "Risk assessment should consider both alarm_signals and benign_reassuring_features.\n"
            "However, strong malignant-specific features (ulceration, rapid growth, irregular vascular patterns) should not be downgraded simply because some benign features exist.\n"
            "\n"
            "Do NOT output any disease diagnosis or final class.\n"
            "Return JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Case perception so far: {self.perception_snapshot(state)}\n"
            f"Existing skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'temporal_evolution_skill', 'metadata_consistency_skill'), max_skills=5)}\n"
            f"Case metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name in {"risk_evidence", "alarm_signals", "benign_reassuring_features"}:
            if value in (None, ""):
                return []
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]
        if field_name == "risk_level":
            if value not in {"low", "medium", "high"}:
                return "medium" if value else "unknown"
        return super().normalize_field(field_name, value)

    def after_execute(self, state: CaseState, output: dict[str, Any]) -> None:
        risk_level = output.get("risk_level")
        if risk_level in {"medium", "high"}:
            state.risk_flags.append(f"malignancy_risk:{risk_level}")
