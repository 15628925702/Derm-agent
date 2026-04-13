from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.run_agent import run_agent
from agent.state import CaseInput


class StubClient:
    def initial_perception(self, case_input: CaseInput) -> dict:
        return {
            "image_summary": "brown lesion on arm",
            "ddx_candidates": ["Nevus", "Malignant Melanoma"],
            "uncertainty": {"level": "medium", "reasons": ["single image"]},
            "notes": ["stub perception"],
        }

    def run_skill_prompt(self, case_input: CaseInput, skill_name: str, prompt: str, output_schema: str) -> dict:
        return {}

    def final_diagnosis(self, case_input: CaseInput, evidence_package) -> dict:
        return {
            "final_diagnosis": "Nevus",
            "differential_diagnoses": ["Nevus", "Malignant Melanoma"],
            "rationale": "stub final diagnosis",
            "confidence": "medium",
            "follow_up_considerations": [],
        }


def test_run_agent_uses_split_state_root_override(monkeypatch, tmp_path: Path) -> None:
    custom_split_root = tmp_path / "isolated_split_states"
    custom_policy_root = tmp_path / "isolated_policy_root"
    monkeypatch.setenv("DERMAGENT_SPLIT_STATE_ROOT", str(custom_split_root))
    monkeypatch.setenv("DERMAGENT_POLICY_ROOT", str(custom_policy_root))

    image_path = tmp_path / "case.png"
    image_path.write_bytes(b"fake-image")
    case_input = CaseInput(
        case_id="CASE_ISOLATED",
        image_path=str(image_path),
        metadata={"site": "arm"},
        label="Nevus",
        dataset_name="toy_dataset",
    )

    state, _ = run_agent(
        case_input=case_input,
        client=StubClient(),
        output_dir=tmp_path / "outputs",
        enable_writeback=True,
        run_mode="toy_bootstrap",
        data_split="train",
    )

    cognition_path = custom_split_root / "train" / "cognition_state.json"
    experience_manifest = custom_split_root / "train" / "experience" / "manifest.json"
    stable_policy_path = custom_policy_root / "current_stable_policy.json"

    assert cognition_path.exists()
    assert experience_manifest.exists()
    assert stable_policy_path.exists()
    assert state.execution_record["state_versions"]["experience_state"]["root"].startswith(str(custom_split_root))
    assert state.execution_record["state_versions"]["experience_state"]["split_name"] == "train"
    assert state.execution_record["policy_snapshot"]["source_path"].startswith(str(custom_policy_root))

    cognition_payload = json.loads(cognition_path.read_text(encoding="utf-8"))
    assert cognition_payload["state_split"] == "train"
