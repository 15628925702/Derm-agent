from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.confusion_clusters import cluster_ordering_hints, detect_confusion_clusters, preferred_abstract_section


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
