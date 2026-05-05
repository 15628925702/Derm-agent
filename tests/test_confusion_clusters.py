from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.confusion_clusters import cluster_ordering_hints, detect_confusion_clusters, preferred_abstract_section
from agent.evidence_package import EvidencePackage
from agent.state import CaseInput, CaseState
from skills.differential_compare import DifferentialCompareSkill


def test_detect_confusion_clusters_recognizes_ack_sek_and_ack_bcc_scc() -> None:
    clusters = detect_confusion_clusters(
        ddx_candidates=["actinic keratosis", "seborrheic keratosis", "basal cell carcinoma"],
        image_summary="rough keratotic plaque with uncertain pearly border",
        notes=["Need better surface detail for ACK versus SEK and BCC confusion."],
    )
    assert "ack_sek" in clusters
    assert "ack_bcc_scc" in clusters


def test_detect_confusion_clusters_does_not_confuse_back_with_ack() -> None:
    clusters = detect_confusion_clusters(
        ddx_candidates=["Malignant Melanoma", "Atypical Mole", "Seborrheic Keratosis"],
        image_summary="A dark lesion on the back with irregular border.",
        notes=["Posterior trunk location only should not trigger unrelated keratinocyte clusters."],
    )
    assert "ack_sek" not in clusters
    assert "ack_bcc_scc" not in clusters


def test_preferred_abstract_section_sends_cluster_confusion_memory_to_comparison() -> None:
    record = {
        "experience_type": "confusion_memory",
        "confusion_pair": "scc->bcc",
        "perception_summary": "crusted lesion where pearly border still mattered",
        "learning_points": ["do not let crust alone force SCC"],
    }
    assert preferred_abstract_section(record=record, cluster_names=["ack_bcc_scc"]) == "comparison"


def test_cluster_ordering_hints_prioritize_cluster_specific_skills() -> None:
    hints = cluster_ordering_hints(["ack_bcc_scc"])
    assert hints["ack_scc_specialist_skill"] >= 2.0
    assert hints["exclusion_reasoning_skill"] >= 1.2


def test_differential_compare_payload_uses_ham10000_cluster_metadata() -> None:
    state = CaseState(
        case_input=CaseInput(
            case_id="HAM_CASE_001",
            image_path="/tmp/ham_case_001.jpg",
            metadata={},
            dataset_name="ham10000",
        ),
        perception={
            "ddx_candidates": ["Benign Keratosis", "Nevus"],
            "image_summary": "Pigmented lesion with waxy surface and possible stuck-on appearance.",
            "notes": ["Need BKL versus nevus comparison."],
        },
    )

    payload = DifferentialCompareSkill().confusion_cluster_prompt_payload(state)

    assert "bkl_nv" in payload["active_clusters"]
    assert "bkl->nv" in payload["related_confusion_pairs"]
    assert "benign keratosis" in payload["related_keywords"]
    assert any(item.get("cluster_id") == "bkl_nv" for item in payload["guidance"])


def test_evidence_package_confusion_summary_uses_dataset_specific_clusters() -> None:
    state = CaseState(
        case_input=CaseInput(
            case_id="HAM_CASE_002",
            image_path="/tmp/ham_case_002.jpg",
            metadata={},
            dataset_name="ham10000",
        ),
        perception={
            "ddx_candidates": ["Benign Keratosis", "Nevus"],
            "image_summary": "Brown lesion with keratotic surface texture.",
            "notes": ["Dataset-specific BKL/NV confusion should be preserved."],
        },
    )

    evidence = EvidencePackage.from_state(state)
    summary = evidence.confusion_cluster_summary

    assert "bkl_nv" in summary["active_clusters"]
    assert any(item.get("cluster_id") == "bkl_nv" for item in summary["guidance"])
