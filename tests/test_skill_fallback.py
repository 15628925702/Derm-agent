from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.state import CaseInput, CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField


class _BrokenJsonClient:
    def run_skill_prompt(self, **_: object) -> dict[str, object]:
        raise json.JSONDecodeError("broken", '{"oops"', 1)


class _DummySkill(BaseSkill):
    name = "dummy_skill"
    description = "dummy"
    output_fields = ("supporting_evidence",)
    list_fields = output_fields
    skill_object = make_skill_object(
        skill_id="skill.dummy.v1",
        name=name,
        description=description,
        skill_type="reasoning",
        triggers=[],
        workflow_text="dummy workflow",
        steps=[],
        watch_outs=[],
        output_schema=[
            SkillSchemaField("supporting_evidence", "list[str]", "dummy evidence"),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return "dummy prompt"


def test_skill_json_failure_degrades_to_fallback_output() -> None:
    case_input = CaseInput(case_id="case_1", image_path="/tmp/missing.png", metadata={})
    state = CaseState(case_input=case_input)
    skill = _DummySkill()

    result = skill.execute(state, _BrokenJsonClient())

    assert result["execution_status"] == "fallback_json_parse_error"
    assert result["execution_error"] == "JSONDecodeError"
    assert result["evidence_strength"] == "low"
    assert state.skill_outputs["dummy_skill"]["execution_status"] == "fallback_json_parse_error"
    assert any("dummy_skill degraded to fallback output" in note for note in state.notes)
