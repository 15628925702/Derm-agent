from __future__ import annotations

from typing import Any


DEFAULT_LABEL_SPACE = ("MEL", "BCC", "NEV")

KEYWORD_MAP = {
    "MEL": (
        "mel",
        "melanoma",
        "malignant melanoma",
        "suspicious for melanoma",
        "suspicious for malignant melanoma",
        "lentigo maligna",
        "acral melanoma",
    ),
    "BCC": (
        "bcc",
        "basal cell",
        "basal cell carcinoma",
        "suspicious for basal cell carcinoma",
    ),
    "NEV": (
        "nev",
        "nevus",
        "naevus",
        "mole",
        "melanocytic nevus",
        "atypical nevus",
        "benign nevus",
    ),
}

PRIORITY = ("MEL", "BCC", "NEV")


def canonicalize_prediction(text: str, label_space: list[str] | tuple[str, ...]) -> dict[str, Any]:
    allowed = tuple(str(label).strip().upper() for label in label_space if str(label).strip())
    normalized_text = str(text or "").strip()
    lowered = normalized_text.lower()
    if not normalized_text:
        return {
            "raw_prediction": normalized_text,
            "canonical_label": "OTHER",
            "matched": False,
            "mapping_rule": "fallback",
            "confidence": "low",
        }

    upper = normalized_text.upper()
    if upper in allowed:
        return {
            "raw_prediction": normalized_text,
            "canonical_label": upper,
            "matched": True,
            "mapping_rule": "exact",
            "confidence": "high",
        }

    matched_labels: list[str] = []
    for label in PRIORITY:
        if label not in allowed:
            continue
        for keyword in KEYWORD_MAP.get(label, ()):
            if keyword in lowered:
                matched_labels.append(label)
                break

    if matched_labels:
        canonical_label = matched_labels[0]
        confidence = "medium" if len(matched_labels) == 1 else "low"
        return {
            "raw_prediction": normalized_text,
            "canonical_label": canonical_label,
            "matched": True,
            "mapping_rule": "keyword",
            "confidence": confidence,
        }

    return {
        "raw_prediction": normalized_text,
        "canonical_label": "OTHER",
        "matched": False,
        "mapping_rule": "fallback",
        "confidence": "low",
    }
