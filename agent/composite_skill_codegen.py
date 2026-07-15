from __future__ import annotations

import re
import textwrap
from typing import Any


def _slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _class_name(skill_id: str) -> str:
    parts = re.split(r"[_\-\s]+", skill_id.lower())
    return "".join(p.capitalize() for p in parts if p) + "Skill"


def generate_skill_code(proposal: dict[str, Any]) -> str:
    artifacts = proposal.get("proposal_artifacts", {})
    raw_name = artifacts.get("suggested_skill_name") or proposal.get("proposal_id", "composite_skill")
    raw_id = artifacts.get("suggested_skill_id") or _slugify(raw_name)

    skill_name = _slugify(raw_name) if not raw_name.endswith("_skill") else _slugify(raw_name)
    if not skill_name.endswith("_skill"):
        skill_name = skill_name + "_skill"
    skill_id_str = raw_id if raw_id.startswith("skill.") else f"skill.{_slugify(raw_id)}.v1"
    class_name = _class_name(skill_name.replace("_skill", ""))

    workflow_text = str(proposal.get("proposed_workflow_text", "Perform composite clinical reasoning."))
    skill_sequence = proposal.get("skill_sequence", [])
    trigger_pattern = proposal.get("trigger_pattern", {})
    intended_scope = proposal.get("intended_scope", {})
    risk_notes = proposal.get("risk_notes", [])
    expected_benefit = proposal.get("expected_benefit", {})
    source_type = proposal.get("_source_type", "composite")

    trigger_condition = str(trigger_pattern.get("decision_pattern") or trigger_pattern.get("condition") or "Trigger when composite reasoning pattern is detected.")
    trigger_rationale = str(intended_scope.get("use_case") or expected_benefit.get("planner") or "Bundles multiple evidence steps into a focused composite workflow.")

    applicable_when = str(intended_scope.get("applicable_when") or "")
    do_not_use_when = str(intended_scope.get("do_not_use_when") or "")

    watch_out_lines = list(risk_notes)
    if skill_sequence:
        watch_out_lines.append(f"Component skill sequence: {' → '.join(skill_sequence)}")
    if applicable_when:
        watch_out_lines.append(f"Applicable when: {applicable_when}")
    if do_not_use_when:
        watch_out_lines.append(f"Do not use when: {do_not_use_when}")

    watch_outs_repr = repr(watch_out_lines)

    description = str(artifacts.get("description") or intended_scope.get("use_case") or f"Composite skill: {skill_name}.")

    output_fields = ("composite_findings", "supporting_evidence", "opposing_evidence", "uncertainty_summary", "further_observation_suggestions")
    list_fields = output_fields

    workflow_escaped = workflow_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n"\n            "')

    source_tag = "composite_skill_proposal" if source_type == "composite" else "confusion_triggered_proposal"

    code = f'''\
from __future__ import annotations

from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class {class_name}(BaseSkill):
    name = "{skill_name}"
    description = "{description}"
    output_fields = (
        "composite_findings",
        "supporting_evidence",
        "opposing_evidence",
        "uncertainty_summary",
        "further_observation_suggestions",
    )
    list_fields = (
        "composite_findings",
        "supporting_evidence",
        "opposing_evidence",
        "further_observation_suggestions",
    )
    skill_object = make_skill_object(
        skill_id="{skill_id_str}",
        name=name,
        description=description,
        skill_type="composite",
        triggers=[
            SkillTrigger(
                condition="{trigger_condition}",
                rationale="{trigger_rationale}",
            )
        ],
        workflow_text=(
            "{workflow_escaped}"
        ),
        steps=[
            SkillStep(
                step_id="composite_reasoning",
                evidence_focus="Integrate evidence across component reasoning steps.",
            )
        ],
        watch_outs={watch_outs_repr},
        output_schema=[
            SkillSchemaField(name="composite_findings", field_type="list[str]", description="Key composite findings from integrated reasoning."),
            SkillSchemaField(name="supporting_evidence", field_type="list[str]", description="Evidence supporting the primary hypothesis."),
            SkillSchemaField(name="opposing_evidence", field_type="list[str]", description="Evidence opposing or complicating the primary hypothesis."),
            SkillSchemaField(name="uncertainty_summary", field_type="str", description="Summary of remaining uncertainty after composite reasoning."),
            SkillSchemaField(name="further_observation_suggestions", field_type="list[str]", description="Suggested follow-up observations or tests."),
            SkillSchemaField(name="evidence_strength", field_type="str", description="Overall evidence strength: high / medium / low."),
        ],
        source="{source_tag}",
    )

    def build_prompt(self, state: CaseState, execution_context: dict) -> str:
        perception = state.perception or {{}}
        metadata = state.case_input.metadata or {{}}
        prior_outputs = state.skill_outputs or {{}}

        prior_text = ""
        if prior_outputs:
            parts = []
            for sk, out in prior_outputs.items():
                if isinstance(out, dict):
                    parts.append(f"{{sk}}: {{out}}")
            if parts:
                prior_text = "\\nPrior skill outputs:\\n" + "\\n".join(parts[:6])

        return (
            f"{{self.skill_object.workflow_text}}\\n\\n"
            f"Perception summary: {{perception.get('image_summary', '')}}\\n"
            f"Metadata: {{metadata}}\\n"
            f"{{prior_text}}\\n\\n"
            "Return JSON with fields: composite_findings (list), supporting_evidence (list), "
            "opposing_evidence (list), uncertainty_summary (str), "
            "further_observation_suggestions (list), evidence_strength (str: high/medium/low)."
        )
'''
    return code
