from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.label_space import canonicalize_label, is_malignant_label, label_space_snapshot


def test_ham10000_full_label_space_maps_original_labels() -> None:
    assert canonicalize_label("mel", dataset_name="HAM10000") == "MEL"
    assert canonicalize_label("nv", dataset_name="HAM10000") == "NV"
    assert canonicalize_label("bkl", dataset_name="HAM10000") == "BKL"
    assert canonicalize_label("akiec", dataset_name="HAM10000") == "AKIEC"
    assert is_malignant_label("mel", dataset_name="HAM10000") is True
    assert is_malignant_label("bcc", dataset_name="HAM10000") is True
    assert is_malignant_label("nv", dataset_name="HAM10000") is False
    assert label_space_snapshot(dataset_name="HAM10000")["label_space_id"] == "ham10000_full"
