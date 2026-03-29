from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.skill_retriever import SkillRetrievalQuery, build_default_skill_retriever
from cognition.cognition_state import CognitionState
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


def test_skill_retriever_returns_foundational_and_reasoning_candidates_for_multi_ddx_case() -> None:
    retriever = build_default_skill_retriever()
    skills = [
        _skill(
            skill_id="skill.morphology_analysis.v1",
            name="morphology_analysis_skill",
            skill_type="observation",
            trigger_condition="Use when morphology needs to be structured from the current lesion perception.",
            trigger_rationale="Morphology is a foundational clinical observation step.",
        ),
        _skill(
            skill_id="skill.lesion_description_structuring.v1",
            name="lesion_description_structuring_skill",
            skill_type="observation",
            trigger_condition="Use when the lesion description should be standardized with morphology, color, border, surface, distribution, and context.",
            trigger_rationale="Structured lesion description improves downstream comparison.",
        ),
        _skill(
            skill_id="skill.differential_compare.v1",
            name="differential_compare_skill",
            skill_type="reasoning",
            trigger_condition="Use when multiple differential candidates need explicit pairwise comparison.",
            trigger_rationale="Comparison keeps the differential open.",
        ),
        _skill(
            skill_id="skill.exclusion_reasoning.v1",
            name="exclusion_reasoning_skill",
            skill_type="reasoning",
            trigger_condition="Use when some candidates look unlikely and negative evidence matters.",
            trigger_rationale="Exclusion reasoning surfaces negative evidence.",
        ),
        _skill(
            skill_id="skill.uncertainty_assessment.v1",
            name="uncertainty_assessment_skill",
            skill_type="risk_uncertainty",
            trigger_condition="Use when uncertainty, ambiguity, or missing information remains.",
            trigger_rationale="Uncertainty should be explicit.",
        ),
        _skill(
            skill_id="skill.contradiction_check.v1",
            name="contradiction_check_skill",
            skill_type="reasoning",
            trigger_condition="Use when conflicts or contradictions may exist across perception and metadata.",
            trigger_rationale="Contradictions should be audited.",
        ),
    ]
    bundle = retriever.retrieve(
        SkillRetrievalQuery(
            perception={
                "image_summary": "A flat irregular brown lesion with slightly asymmetric border.",
                "ddx_candidates": ["Mole", "Seborrheic Keratosis", "Atypical Mole"],
                "uncertainty": {"level": "medium"},
                "notes": ["No change documented over time."],
            },
            metadata={
                "region": "ARM",
                "age": "8",
                "changed": "False",
                "bleed": "False",
            },
            cognition=CognitionState(),
        ),
        skills,
    )

    candidate_names = set(bundle.candidate_skill_names)
    assert "morphology_analysis_skill" in candidate_names
    assert "lesion_description_structuring_skill" in candidate_names
    assert "differential_compare_skill" in candidate_names
    assert "exclusion_reasoning_skill" in candidate_names
    assert "uncertainty_assessment_skill" in candidate_names
    assert "contradiction_check_skill" in candidate_names
    assert bundle.candidate_skill_ids
    assert bundle.retrieval_scores["morphology_analysis_skill"] > 0
    assert bundle.match_reasons["morphology_analysis_skill"]


def test_skill_retriever_prefers_specialist_skill_for_mel_nev_confusion() -> None:
    retriever = build_default_skill_retriever()
    skills = [
        _skill(
            skill_id="skill.mel_nev_specialist.v1",
            name="mel_nev_specialist_skill",
            skill_type="specialist",
            trigger_condition="Use when melanoma versus nevus confusion is active.",
            trigger_rationale="Specialist comparison helps melanoma-nevus confusion.",
        ),
        _skill(
            skill_id="skill.malignancy_risk_assessment.v1",
            name="malignancy_risk_assessment_skill",
            skill_type="risk_uncertainty",
            trigger_condition="Use when malignant risk or alarm signals are possible.",
            trigger_rationale="Risk framing should be explicit.",
        ),
        _skill(
            skill_id="skill.uncertainty_assessment.v1",
            name="uncertainty_assessment_skill",
            skill_type="risk_uncertainty",
            trigger_condition="Use when uncertainty remains high.",
            trigger_rationale="Uncertainty should be explicit.",
        ),
    ]
    bundle = retriever.retrieve(
        SkillRetrievalQuery(
            perception={
                "image_summary": "A pigmented lesion with irregular border and color variation.",
                "ddx_candidates": ["Melanoma", "Atypical Nevus"],
                "uncertainty": {"level": "high"},
                "notes": ["Concern for melanoma versus nevus confusion."],
            },
            metadata={
                "region": "BACK",
                "changed": "True",
                "bleed": "False",
            },
            cognition=CognitionState(known_confusion_patterns={"melanoma->nev": 3}),
        ),
        skills,
    )

    candidate_names = set(bundle.candidate_skill_names)
    assert "mel_nev_specialist_skill" in candidate_names
    assert "malignancy_risk_assessment_skill" in candidate_names
    assert "uncertainty_assessment_skill" in candidate_names
    assert any(hit in {"mel_nev_confusion", "known_confusion_match"} for hit in bundle.trigger_hits["mel_nev_specialist_skill"])


def test_skill_retriever_boosts_ack_sek_cluster_and_exclusion_reasoning() -> None:
    retriever = build_default_skill_retriever()
    skills = [
        _skill(
            skill_id="skill.ack_scc_specialist.v1",
            name="ack_scc_specialist_skill",
            skill_type="specialist",
            trigger_condition="Use when ACK/SCC/BCC/SEK confusion is active.",
            trigger_rationale="Specialist comparison helps recurrent keratinocyte confusion clusters.",
        ),
        _skill(
            skill_id="skill.exclusion_reasoning.v1",
            name="exclusion_reasoning_skill",
            skill_type="reasoning",
            trigger_condition="Use when negative evidence and missing evidence matter for the active differential.",
            trigger_rationale="Explicit exclusion reasoning improves hard-cluster interpretability.",
        ),
        _skill(
            skill_id="skill.malignancy_risk_assessment.v1",
            name="malignancy_risk_assessment_skill",
            skill_type="risk_uncertainty",
            trigger_condition="Use when malignant risk or alarm signals are possible.",
            trigger_rationale="Risk framing should be explicit.",
        ),
    ]
    bundle = retriever.retrieve(
        SkillRetrievalQuery(
            perception={
                "image_summary": "A rough scaly lesion with uncertain stuck-on versus actinic surface pattern.",
                "ddx_candidates": ["Actinic Keratosis", "Seborrheic Keratosis"],
                "uncertainty": {"level": "high"},
                "notes": ["Need better texture detail for ACK versus SEK confusion."],
            },
            metadata={"region": "FACE", "changed": "True"},
            cognition=CognitionState(),
        ),
        skills,
    )

    assert "ack_scc_specialist_skill" in bundle.candidate_skill_names
    assert "exclusion_reasoning_skill" in bundle.candidate_skill_names
    assert "ack_sek_confusion" in bundle.trigger_hits["ack_scc_specialist_skill"]
    assert bundle.retrieval_scores["ack_scc_specialist_skill"] >= bundle.retrieval_scores["malignancy_risk_assessment_skill"]


def _skill(
    *,
    skill_id: str,
    name: str,
    skill_type: str,
    trigger_condition: str,
    trigger_rationale: str,
):
    return make_skill_object(
        skill_id=skill_id,
        name=name,
        description=name,
        skill_type=skill_type,
        triggers=[SkillTrigger(condition=trigger_condition, rationale=trigger_rationale)],
        workflow_text="Doctor-style structured reasoning workflow.",
        steps=[SkillStep(step_id="s1", title="Step", instruction="Inspect relevant evidence.")],
        watch_outs=["Do not give the final diagnosis."],
        output_schema=[SkillSchemaField(name="evidence", field_type="str", description="Structured evidence output.")],
    )
