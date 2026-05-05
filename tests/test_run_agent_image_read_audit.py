from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.run_agent import run_agent
from agent.state import CaseInput
from cognition.cognition_state import CognitionState
from memory.experience_bank import ExperienceBank


class StubClient:
    def __init__(self) -> None:
        self._perception_calls = 0
        self._final_calls = 0

    def initial_perception(self, case_input: CaseInput) -> dict:
        self._perception_calls += 1
        if "missing.png" in case_input.image_path:
            return {
                "image_summary": "text-only fallback description",
                "ddx_candidates": ["Nevus"],
                "uncertainty": {"level": "high", "reasons": ["no image attached"]},
                "notes": ["no image"],
            }
        return {
            "image_summary": "dark asymmetric lesion",
            "ddx_candidates": ["Malignant Melanoma", "Nevus"],
            "uncertainty": {"level": "medium", "reasons": ["single image"]},
            "notes": ["image present"],
        }

    def run_skill_prompt(self, case_input: CaseInput, skill_name: str, prompt: str, output_schema: str) -> dict:
        return {"skill_name": skill_name, "supporting_evidence": ["stub"]}

    def final_diagnosis(self, case_input: CaseInput, evidence_package) -> dict:
        self._final_calls += 1
        if "missing.png" in case_input.image_path:
            return {
                "final_diagnosis": "Nevus",
                "differential_diagnoses": ["Nevus"],
                "rationale": "No image, so the output stays conservative.",
                "confidence": "low",
                "follow_up_considerations": [],
            }
        return {
            "final_diagnosis": "Malignant Melanoma",
            "differential_diagnoses": ["Malignant Melanoma", "Nevus"],
            "rationale": "Image features support melanoma concern.",
            "confidence": "medium",
            "follow_up_considerations": [],
        }


def test_run_agent_records_image_read_audit_when_enabled(tmp_path: Path) -> None:
    case_input = CaseInput(
        case_id="CASE_A",
        image_path=str(tmp_path / "real.png"),
        metadata={"site": "arm"},
        label="Malignant Melanoma",
    )
    case_input_path = Path(case_input.image_path)
    case_input_path.write_bytes(b"fake-image")

    state, _ = run_agent(
        case_input=case_input,
        client=StubClient(),
        experience_bank=ExperienceBank(root=tmp_path / "experience"),
        cognition=CognitionState(),
        output_dir=tmp_path / "outputs",
        enable_writeback=False,
        run_mode="test",
        data_split="test",
        execution_overrides={"enable_image_read_audit": True},
    )

    assert state.image_read_audit["overall_verdict"] == "evidence_of_image_use"
    assert state.image_read_audit["result_impact"]["result_changed"] is True
    assert state.execution_record["image_read_audit"]["overall_verdict"] == "evidence_of_image_use"
