from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation import evaluate_diagnosis_output
from agent.label_space import (
    LabelAlias,
    LabelSpace,
    canonicalize_label,
    is_malignant_label,
    labels_match,
    register_label_space,
)
from agent.policy_evaluation import build_policy_summary


def test_default_derm_six_label_space_preserves_existing_behavior() -> None:
    assert canonicalize_label("melanoma", dataset_name="pad_ufes_20") == "MEL"
    assert canonicalize_label("Basal Cell Carcinoma", dataset_name="pad_ufes_20") == "BCC"
    assert canonicalize_label("nevus", dataset_name="pad_ufes_20") == "NEV"
    assert is_malignant_label("melanoma", dataset_name="pad_ufes_20") is True
    assert is_malignant_label("nevus", dataset_name="pad_ufes_20") is False
    assert labels_match("melanoma", "MEL", dataset_name="pad_ufes_20") is True


def test_registered_dataset_specific_label_space_is_used_by_evaluation_and_summary() -> None:
    register_label_space(
        LabelSpace(
            label_space_id="toy_binary",
            canonical_labels=("MALIGNANT", "BENIGN"),
            aliases=(
                LabelAlias("MALIGNANT", ("malignant", "cancer", "high risk lesion")),
                LabelAlias("BENIGN", ("benign", "non cancer", "low risk lesion")),
            ),
            malignant_labels=("MALIGNANT",),
            benign_labels=("BENIGN",),
        ),
        dataset_names=["ddi_toy"],
    )

    evaluation = evaluate_diagnosis_output(
        {
            "final_diagnosis": "high risk lesion",
            "differential_diagnoses": ["benign"],
        },
        "malignant",
        dataset_name="ddi_toy",
    )

    assert evaluation["ground_truth_canonical"] == "MALIGNANT"
    assert evaluation["final_canonical_label"] == "MALIGNANT"
    assert evaluation["correct"] is True
    assert evaluation["malignant_recall_hit"] is True
    assert evaluation["label_space"]["label_space_id"] == "toy_binary"

    summary = build_policy_summary(
        [
            {
                "dataset_name": "ddi_toy",
                "ground_truth": {
                    "canonical_label": "MALIGNANT",
                    "malignant_flag": True,
                },
                "evaluation": {
                    "correct": True,
                    "topk_hit": True,
                    "malignant_recall_hit": True,
                },
                "reflection_summary": {"case_outcome": {"confusion_pair": "malignant->benign"}},
                "skill_retrieval": {"query_summary": {"confusion_pair": "malignant->benign"}},
            }
        ]
    )

    assert summary["dataset_names"] == ["ddi_toy"]
    assert summary["num_with_ground_truth"] == 1
