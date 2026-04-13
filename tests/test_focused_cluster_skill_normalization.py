from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.ack_scc_specialist import AckSccSpecialistSkill
from skills.differential_compare import DifferentialCompareSkill
from skills.mel_nev_specialist import MelNevSpecialistSkill


def test_mel_nev_specialist_filters_non_visual_opposing_evidence() -> None:
    skill = MelNevSpecialistSkill()
    output = skill.normalize_output(
        {
            "supporting_evidence": [
                "Irregular border and asymmetry support melanoma concern.",
                "The patient's skin type increases the risk of melanoma.",
            ],
            "opposing_evidence": [
                "Benign-leaning symmetry weakens melanoma concern.",
                "The lesion is located on the back, which is a common site for melanoma.",
            ],
        }
    )
    assert output["supporting_evidence"] == ["Irregular border and asymmetry support melanoma concern."]
    assert output["opposing_evidence"] == ["Benign-leaning symmetry weakens melanoma concern."]


def test_ack_specialist_filters_generic_background_evidence() -> None:
    skill = AckSccSpecialistSkill()
    output = skill.normalize_output(
        {
            "supporting_evidence": [
                "Pearly rolled border supports BCC concern.",
                "Erythema could indicate inflammation or irritation.",
            ],
            "opposing_evidence": [
                "Lack of destructive ulceration weakens SCC concern.",
                "Sun-exposed location alone should not drive concern.",
            ],
        }
    )
    assert output["supporting_evidence"] == ["Pearly rolled border supports BCC concern."]
    assert output["opposing_evidence"] == ["Lack of destructive ulceration weakens SCC concern."]


def test_differential_compare_flattens_dict_outputs() -> None:
    skill = DifferentialCompareSkill()
    output = skill.normalize_output(
        {
            "candidate_pairs": {"A vs B": ["x"], "B vs C": ["y"]},
            "supporting_evidence": {"A vs B": ["rolled border", "pearly surface"]},
            "conflicting_evidence": {"A vs B": ["missing dermoscopy"]},
            "required_missing_evidence": {"A vs B": ["closer border inspection"]},
        }
    )
    assert output["candidate_pairs"] == ["A vs B", "B vs C"]
    assert output["supporting_evidence"] == ["A vs B: rolled border; pearly surface"]
    assert output["conflicting_evidence"] == ["A vs B: missing dermoscopy"]
    assert output["required_missing_evidence"] == ["A vs B: closer border inspection"]
