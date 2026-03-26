from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.state import CaseInput, CaseState
from integrations.openai_client import DermOpenAIClient
from skills.registry import build_default_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or all clinical reasoning skills on a mock case.")
    parser.add_argument(
        "--skill",
        default="all",
        choices=[
            "all",
            "morphology_analysis_skill",
            "color_pattern_analysis_skill",
            "border_surface_analysis_skill",
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
        ],
        help="Skill name to run, or 'all'.",
    )
    return parser.parse_args()


def build_mock_state() -> CaseState:
    case_input = CaseInput(
        case_id="mock_case_001",
        image_path=str(PROJECT_ROOT / "data" / "pad_ufes_20" / "images" / "imgs_part_3" / "PAT_1516_1765_530.png"),
        metadata={
            "age": "55",
            "region": "ARM",
            "diameter_1": "6.0",
            "diameter_2": "5.0",
            "itch": "False",
            "grew": "True",
            "hurt": "False",
            "changed": "True",
            "bleed": "False",
            "elevation": "True",
        },
    )
    return CaseState(
        case_input=case_input,
        perception={
            "summary": "Single pigmented lesion with some asymmetry and irregular edge.",
            "body_location": "arm",
            "morphology": "pigmented papular lesion",
            "color": "brown with darker areas",
            "border": "irregular",
            "surface": "mostly smooth",
            "uncertainty": "moderate",
        },
    )


def main() -> int:
    args = parse_args()
    registry = build_default_registry()
    client = DermOpenAIClient()
    state = build_mock_state()

    if args.skill == "all":
        outputs = registry.run_many(registry.list_names(), state, client)
    else:
        outputs = {args.skill: registry.run_skill(args.skill, state, client)}

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
