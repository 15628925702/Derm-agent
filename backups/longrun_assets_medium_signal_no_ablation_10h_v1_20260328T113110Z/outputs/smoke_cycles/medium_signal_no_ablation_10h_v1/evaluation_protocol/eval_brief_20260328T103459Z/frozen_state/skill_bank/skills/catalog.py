from __future__ import annotations

from skills.schema import SkillObject, SkillSchemaField, SkillStats, SkillStep, SkillTrigger


DEFAULT_INPUT_SCHEMA = [
    SkillSchemaField(
        name="case_input.metadata",
        field_type="dict",
        description="Structured patient metadata and case context available before skill execution.",
    ),
    SkillSchemaField(
        name="case_input.image_path",
        field_type="str",
        description="Path to the lesion image that Qwen can inspect.",
    ),
    SkillSchemaField(
        name="perception",
        field_type="dict",
        description="Initial Qwen perception containing image summary, ddx candidates, uncertainty, and notes.",
    ),
    SkillSchemaField(
        name="skill_outputs",
        field_type="dict",
        description="Previously generated structured evidence from earlier skills in the same case.",
        required=False,
    ),
    SkillSchemaField(
        name="retrieval_bundle.raw_case_results",
        field_type="list[dict]",
        description="Retrieved raw case memories selected as auxiliary references for the current skill execution.",
        required=False,
    ),
    SkillSchemaField(
        name="retrieval_bundle.tactical_results",
        field_type="list[dict]",
        description="Retrieved tactical experiences selected as action-oriented support for the current skill execution.",
        required=False,
    ),
    SkillSchemaField(
        name="retrieval_bundle.abstract_results",
        field_type="list[dict]",
        description="Retrieved abstract experiences selected as confusion, rule, or prototype support for the current skill execution.",
        required=False,
    ),
    SkillSchemaField(
        name="execution_context.related_abstract_experiences",
        field_type="list[dict]",
        description="Abstract experiences specifically filtered as relevant to the current skill, such as confusion memories, prototypes, or rules for a specialist comparison.",
        required=False,
    ),
]


def make_skill_object(
    *,
    skill_id: str,
    name: str,
    description: str,
    skill_type: str,
    triggers: list[SkillTrigger],
    workflow_text: str,
    steps: list[SkillStep],
    watch_outs: list[str],
    output_schema: list[SkillSchemaField],
    version: str = "2.0.0",
    source: str = "design+phase2_refactor",
) -> SkillObject:
    return SkillObject(
        skill_id=skill_id,
        name=name,
        description=description,
        skill_type=skill_type,
        triggers=triggers,
        input_schema=list(DEFAULT_INPUT_SCHEMA),
        workflow_text=workflow_text,
        steps=steps,
        watch_outs=watch_outs,
        output_schema=output_schema,
        version=version,
        source=source,
        stats=SkillStats(),
    )
