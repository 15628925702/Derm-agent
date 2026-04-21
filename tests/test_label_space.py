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


def test_scin_label_families_allow_name_granularity_matches() -> None:
    assert labels_match("Contact Dermatitis", "Allergic Contact Dermatitis", dataset_name="scin") is True
    assert labels_match("Acute dermatitis", "Acute dermatitis, NOS", dataset_name="scin") is True

    evaluation = evaluate_diagnosis_output(
        {
            "final_diagnosis": "Contact Dermatitis",
            "differential_diagnoses": ["Irritant Contact Dermatitis"],
        },
        "Allergic Contact Dermatitis",
        dataset_name="scin",
        label_space_id="scin_full",
    )

    assert evaluation["correct"] is True
    assert evaluation["topk_hit"] is True


def test_scin_grouped_label_space_maps_broad_model_outputs() -> None:
    assert canonicalize_label("Contact Dermatitis", dataset_name="scin", label_space_id="scin_grouped") == "DERMATITIS_ECZEMA"
    assert canonicalize_label("Herpes Zoster", dataset_name="scin", label_space_id="scin_grouped") == "INFECTION_VIRAL_FUNGAL"
    assert canonicalize_label("Leukocytoclastic Vasculitis", dataset_name="scin", label_space_id="scin_grouped") == "VASCULAR_PURPURIC"
    assert canonicalize_label("Nevus", dataset_name="scin", label_space_id="scin_grouped") == "PIGMENT_KERATOSIS_NEVUS"


def test_sd198_full_label_space_accepts_raw_and_humanized_labels() -> None:
    assert canonicalize_label("Basal_Cell_Carcinoma", dataset_name="sd198") == "BASAL CELL CARCINOMA"
    assert canonicalize_label("Basal Cell Carcinoma", dataset_name="sd198") == "BASAL CELL CARCINOMA"
    assert canonicalize_label("Malignant_Melanoma", dataset_name="sd198") == "MALIGNANT MELANOMA"
    assert is_malignant_label("Basal_Cell_Carcinoma", dataset_name="sd198") is True
    assert is_malignant_label("Acne_Vulgaris", dataset_name="sd198") is False


def test_sd198_grouped_label_space_maps_to_coarse_categories() -> None:
    assert canonicalize_label("Basal Cell Carcinoma", dataset_name="sd198", label_space_id="sd198_grouped") == "MALIGNANT_SKIN_CANCER"
    assert canonicalize_label("Acne Vulgaris", dataset_name="sd198", label_space_id="sd198_grouped") == "ACNE_FOLLICULITIS_ROSACEA"
    assert canonicalize_label("Allergic Contact Dermatitis", dataset_name="sd198", label_space_id="sd198_grouped") == "DERMATITIS_ECZEMA"
    assert canonicalize_label("Onychomycosis", dataset_name="sd198", label_space_id="sd198_grouped") == "INFECTION_INFESTATION"
    assert is_malignant_label("Basal Cell Carcinoma", dataset_name="sd198", label_space_id="sd198_grouped") is True
    assert is_malignant_label("Acne Vulgaris", dataset_name="sd198", label_space_id="sd198_grouped") is False


def test_xiangya_sft_grouped_label_space_maps_common_chinese_labels() -> None:
    assert canonicalize_label("接触性皮炎", dataset_name="xiangya_sft") == "CONTACT_DERMATITIS"
    assert canonicalize_label("特应性皮炎（AD）", dataset_name="xiangya_sft") == "ATOPIC_DERMATITIS"
    assert canonicalize_label("口周皮炎", dataset_name="xiangya_sft") == "PERIORAL_DERMATITIS"
    assert canonicalize_label("疱疹性湿疹", dataset_name="xiangya_sft") == "HERPETIC_ECZEMA"
    assert is_malignant_label("接触性皮炎", dataset_name="xiangya_sft") is False
