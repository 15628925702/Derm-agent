from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.sd198_label_catalog import (
    SD198_FULL_LABELS,
    SD198_GROUPED_CANONICAL_LABELS,
    SD198_GROUPED_KEYWORDS,
    SD198_GROUPED_MALIGNANT_LABELS,
    SD198_MALIGNANT_LABELS,
    SD198_RAW_TO_CANONICAL,
)
from agent.scin_full_label_catalog import (
    SCIN_FULL_LABELS,
    SCIN_GROUPED_CANONICAL_LABELS,
    SCIN_MALIGNANT_LABELS,
)


DEFAULT_LABEL_SPACE_ID = "derm_six"


@dataclass(frozen=True)
class LabelAlias:
    canonical: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LabelSpace:
    label_space_id: str
    canonical_labels: tuple[str, ...]
    aliases: tuple[LabelAlias, ...]
    malignant_labels: tuple[str, ...] = field(default_factory=tuple)
    benign_labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def canonical_set(self) -> set[str]:
        return {str(label).strip().upper() for label in self.canonical_labels if str(label).strip()}

    @property
    def malignant_set(self) -> set[str]:
        return {str(label).strip().upper() for label in self.malignant_labels if str(label).strip()}

    @property
    def benign_set(self) -> set[str]:
        return {str(label).strip().upper() for label in self.benign_labels if str(label).strip()}


DERM_SIX_LABEL_SPACE = LabelSpace(
    label_space_id=DEFAULT_LABEL_SPACE_ID,
    canonical_labels=("BCC", "ACK", "NEV", "SEK", "SCC", "MEL"),
    aliases=(
        LabelAlias("BCC", ("bcc", "basal cell", "basal cell carcinoma", "suspicious for basal cell carcinoma")),
        LabelAlias("ACK", ("ack", "actinic keratos", "actinic keratosis")),
        LabelAlias("NEV", ("nev", "nevus", "naevus", "nevi", "mole", "melanocytic nevus", "atypical nevus", "atypical naevus")),
        LabelAlias("SEK", ("sek", "seborrheic keratos", "seborrhoeic keratos", "seborrheic keratosis")),
        LabelAlias("SCC", ("scc", "squamous cell", "squamous cell carcinoma")),
        LabelAlias("MEL", ("mel", "melanoma", "malignant melanoma", "lentigo maligna", "acral melanoma", "suspicious for melanoma")),
    ),
    malignant_labels=("BCC", "ACK", "SCC", "MEL"),
    benign_labels=("NEV", "SEK"),
)

ISIC2019_FULL_LABEL_SPACE = LabelSpace(
    label_space_id="isic2019_full",
    canonical_labels=("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"),
    aliases=(
        LabelAlias("MEL", ("mel", "melanoma", "malignant melanoma")),
        LabelAlias("NV", ("nv", "nevus", "naevus", "nevi", "mole")),
        LabelAlias("BCC", ("bcc", "basal cell", "basal cell carcinoma")),
        LabelAlias("AK", ("ak", "actinic keratos", "actinic keratosis")),
        LabelAlias("BKL", ("bkl", "seborrheic keratos", "seborrhoeic keratos", "lichenoid keratosis")),
        LabelAlias("DF", ("df", "dermatofibroma")),
        LabelAlias("VASC", ("vasc", "vascular lesion", "angioma", "hemangioma")),
        LabelAlias("SCC", ("scc", "squamous cell", "squamous cell carcinoma")),
        LabelAlias("UNK", ("unk", "unknown")),
    ),
    malignant_labels=("MEL", "BCC", "AK", "SCC"),
    benign_labels=("NV", "BKL", "DF", "VASC", "UNK"),
)

HAM10000_BINARY_LABEL_SPACE = LabelSpace(
    label_space_id="ham10000_binary",
    canonical_labels=("MALIGNANT", "BENIGN", "UNKNOWN"),
    aliases=(
        LabelAlias("MALIGNANT", ("malignant", "mel", "melanoma", "bcc", "basal cell", "akiec", "actinic keratosis", "ak")),
        LabelAlias("BENIGN", ("benign", "nv", "nevus", "naevus", "bkl", "seborrheic keratosis", "df", "dermatofibroma", "vasc", "vascular")),
        LabelAlias("UNKNOWN", ("unknown", "unk")),
    ),
    malignant_labels=("MALIGNANT",),
    benign_labels=("BENIGN",),
)

HAM10000_FULL_LABEL_SPACE = LabelSpace(
    label_space_id="ham10000_full",
    canonical_labels=("MEL", "BCC", "NV", "BKL", "DF", "VASC", "AKIEC"),
    aliases=(
        LabelAlias("MEL", ("mel", "melanoma", "malignant melanoma")),
        LabelAlias("BCC", ("bcc", "basal cell", "basal cell carcinoma")),
        LabelAlias("NV", ("nv", "nevus", "naevus", "nevi", "mole")),
        LabelAlias("BKL", ("bkl", "seborrheic keratosis", "seborrhoeic keratosis", "benign keratosis")),
        LabelAlias("DF", ("df", "dermatofibroma")),
        LabelAlias("VASC", ("vasc", "vascular lesion", "angioma", "hemangioma")),
        LabelAlias("AKIEC", ("akiec", "actinic keratosis", "bowen disease", "intraepithelial carcinoma")),
    ),
    malignant_labels=("MEL", "BCC", "AKIEC"),
    benign_labels=("NV", "BKL", "DF", "VASC"),
)

LABEL_SPACES: dict[str, LabelSpace] = {
    DEFAULT_LABEL_SPACE_ID: DERM_SIX_LABEL_SPACE,
    "isic2019_full": ISIC2019_FULL_LABEL_SPACE,
    "ham10000_binary": HAM10000_BINARY_LABEL_SPACE,
    "ham10000_full": HAM10000_FULL_LABEL_SPACE,
    "scin_full": None,  # type: ignore[dict-item]
    "scin_grouped": None,  # type: ignore[dict-item]
    "sd198_full": None,  # type: ignore[dict-item]
    "sd198_grouped": None,  # type: ignore[dict-item]
}

DATASET_LABEL_SPACE_ALIASES: dict[str, str] = {
    "pad_ufes_20": DEFAULT_LABEL_SPACE_ID,
    "ham10000": "ham10000_full",
    "ham10000_binary": "ham10000_binary",
    "ham10000_full": "ham10000_full",
    "ham10000_aligned": DEFAULT_LABEL_SPACE_ID,
    "isic2019": DEFAULT_LABEL_SPACE_ID,
    "isic2019_aligned": DEFAULT_LABEL_SPACE_ID,
    "isic2019_full": "isic2019_full",
    "scin": "scin_full",
    "scin_full": "scin_full",
    "scin_grouped": "scin_grouped",
    "sd198": "sd198_full",
    "sd198_full": "sd198_full",
    "sd198_grouped": "sd198_grouped",
}

_SCIN_EQUIVALENCE_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "CONTACT DERMATITIS",
            "ALLERGIC CONTACT DERMATITIS",
            "IRRITANT CONTACT DERMATITIS",
            "CONTACT DERMATITIS, NOS",
        }
    ),
    frozenset(
        {
            "ACUTE DERMATITIS",
            "ACUTE DERMATITIS, NOS",
        }
    ),
)


def register_label_space(label_space: LabelSpace, *, dataset_names: list[str] | None = None) -> None:
    LABEL_SPACES[label_space.label_space_id] = label_space
    for name in dataset_names or []:
        normalized = str(name).strip().lower()
        if normalized:
            DATASET_LABEL_SPACE_ALIASES[normalized] = label_space.label_space_id


def resolve_label_space(
    *,
    label_space_id: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LabelSpace:
    explicit_id = str(label_space_id or "").strip().lower()
    if explicit_id:
        if explicit_id == "scin_full":
            return _get_scin_full_label_space()
        if explicit_id == "scin_grouped":
            return _get_scin_grouped_label_space()
        if explicit_id == "sd198_full":
            return _get_sd198_full_label_space()
        if explicit_id == "sd198_grouped":
            return _get_sd198_grouped_label_space()
        return LABEL_SPACES.get(explicit_id, DERM_SIX_LABEL_SPACE)

    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key:
        mapped_id = DATASET_LABEL_SPACE_ALIASES.get(dataset_key)
        if mapped_id:
            if mapped_id == "scin_full":
                return _get_scin_full_label_space()
            if mapped_id == "scin_grouped":
                return _get_scin_grouped_label_space()
            if mapped_id == "sd198_full":
                return _get_sd198_full_label_space()
            if mapped_id == "sd198_grouped":
                return _get_sd198_grouped_label_space()
            return LABEL_SPACES.get(mapped_id, DERM_SIX_LABEL_SPACE)

    metadata_payload = dict(metadata or {})
    metadata_label_space = str(metadata_payload.get("label_space_id", "")).strip().lower()
    if metadata_label_space:
        if metadata_label_space == "scin_full":
            return _get_scin_full_label_space()
        if metadata_label_space == "scin_grouped":
            return _get_scin_grouped_label_space()
        if metadata_label_space == "sd198_full":
            return _get_sd198_full_label_space()
        if metadata_label_space == "sd198_grouped":
            return _get_sd198_grouped_label_space()
        return LABEL_SPACES.get(metadata_label_space, DERM_SIX_LABEL_SPACE)

    return DERM_SIX_LABEL_SPACE


def canonicalize_label(
    raw_label: str | None,
    *,
    label_space_id: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    if raw_label is None:
        return None
    text = str(raw_label).strip()
    if not text:
        return None

    label_space = resolve_label_space(label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    upper_text = text.upper()
    if upper_text in label_space.canonical_set:
        return upper_text

    lowered = text.lower()
    normalized = _normalize_label_text(text)
    best_match: tuple[int, str] | None = None
    for alias in label_space.aliases:
        canonical_upper = alias.canonical.upper()
        if canonical_upper == upper_text:
            return canonical_upper
        for keyword in alias.keywords:
            keyword_text = str(keyword).strip()
            if not keyword_text:
                continue
            keyword_lower = keyword_text.lower()
            keyword_normalized = _normalize_label_text(keyword_text)
            matched = False
            if keyword_lower and keyword_lower in lowered:
                matched = True
            elif keyword_normalized and keyword_normalized in normalized:
                matched = True
            if matched:
                score = len(keyword_normalized or keyword_lower)
                if best_match is None or score > best_match[0]:
                    best_match = (score, canonical_upper)
    if best_match is not None:
        return best_match[1]
    return None


def is_malignant_label(
    label: str | None,
    *,
    label_space_id: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool | None:
    canonical = canonicalize_label(
        label,
        label_space_id=label_space_id,
        dataset_name=dataset_name,
        metadata=metadata,
    )
    if canonical is None:
        return None
    label_space = resolve_label_space(label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    if canonical in label_space.malignant_set:
        return True
    if label_space.benign_set and canonical in label_space.benign_set:
        return False
    return None


def labels_match(
    left: str | None,
    right: str | None,
    *,
    label_space_id: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool | None:
    if left is None or right is None:
        return None
    if str(left) == str(right):
        return True
    if _dataset_equivalent_labels_from_raw(left, right, dataset_name=dataset_name, label_space_id=label_space_id):
        return True
    left_canonical = canonicalize_label(left, label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    right_canonical = canonicalize_label(right, label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    if left_canonical and right_canonical:
        if left_canonical == right_canonical:
            return True
        if _dataset_equivalent_labels(left_canonical, right_canonical, dataset_name=dataset_name, label_space_id=label_space_id):
            return True
    return False


def label_space_snapshot(
    *,
    label_space_id: str | None = None,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_space = resolve_label_space(label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    return {
        "label_space_id": label_space.label_space_id,
        "canonical_labels": list(label_space.canonical_labels),
        "malignant_labels": list(label_space.malignant_labels),
        "benign_labels": list(label_space.benign_labels),
    }


def _dataset_equivalent_labels(
    left_canonical: str,
    right_canonical: str,
    *,
    dataset_name: str | None = None,
    label_space_id: str | None = None,
) -> bool:
    dataset_key = str(dataset_name or "").strip().lower()
    label_space_key = str(label_space_id or "").strip().lower()
    if dataset_key == "scin" or label_space_key == "scin_full":
        left_upper = str(left_canonical).strip().upper()
        right_upper = str(right_canonical).strip().upper()
        for family in _SCIN_EQUIVALENCE_FAMILIES:
            if left_upper in family and right_upper in family:
                return True
    return False


def _dataset_equivalent_labels_from_raw(
    left: str,
    right: str,
    *,
    dataset_name: str | None = None,
    label_space_id: str | None = None,
) -> bool:
    dataset_key = str(dataset_name or "").strip().lower()
    label_space_key = str(label_space_id or "").strip().lower()
    if dataset_key != "scin" and label_space_key != "scin_full":
        return False

    left_upper = _normalize_scin_family_label(left)
    right_upper = _normalize_scin_family_label(right)
    for family in _SCIN_EQUIVALENCE_FAMILIES:
        if left_upper in family and right_upper in family:
            return True
    return False


def _normalize_scin_family_label(text: str) -> str:
    normalized = str(text).strip().upper()
    if normalized in {
        "CONTACT DERMATITIS",
        "ALLERGIC CONTACT DERMATITIS",
        "IRRITANT CONTACT DERMATITIS",
        "CONTACT DERMATITIS, NOS",
    }:
        return normalized
    if normalized in {
        "ACUTE DERMATITIS",
        "ACUTE DERMATITIS, NOS",
    }:
        return normalized
    return normalized


def _normalize_label_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower())
    return " ".join(normalized.split())


def _build_keyword_variants(label: str) -> tuple[str, ...]:
    raw = " ".join(str(label).strip().split())
    normalized = _normalize_label_text(raw)
    variants = [raw, normalized]
    if " and/or " in raw.lower():
        variants.append(raw.replace("and/or", "and"))
        variants.append(raw.replace("and/or", "/"))
    if " - " in raw:
        variants.append(raw.replace(" - ", " "))
        variants.append(raw.replace(" - ", ": "))
    if "/" in raw:
        variants.append(raw.replace("/", " / "))
        variants.append(raw.replace("/", " "))
    return tuple(dict.fromkeys(item for item in variants if item))


def _looks_malignant_scin_label(label: str) -> bool:
    return label in SCIN_MALIGNANT_LABELS


@lru_cache(maxsize=1)
def _get_sd198_full_label_space() -> LabelSpace:
    labels = tuple(SD198_FULL_LABELS)
    alias_pairs: list[tuple[str, tuple[str, ...]]] = []
    for raw_label, canonical_label in SD198_RAW_TO_CANONICAL.items():
        alias_pairs.append(
            (
                canonical_label,
                tuple(
                    dict.fromkeys(
                        _build_keyword_variants(canonical_label)
                        + _build_keyword_variants(raw_label)
                    )
                ),
            )
        )
    aliases = tuple(
        LabelAlias(canonical, keywords)
        for canonical, keywords in sorted(alias_pairs, key=lambda item: (-len(_normalize_label_text(item[0])), item[0].lower()))
    )
    malignant_labels = tuple(label for label in sorted(labels) if label in SD198_MALIGNANT_LABELS)
    benign_labels = tuple(label for label in sorted(labels) if label not in SD198_MALIGNANT_LABELS)
    return LabelSpace(
        label_space_id="sd198_full",
        canonical_labels=tuple(sorted(labels)),
        aliases=aliases,
        malignant_labels=malignant_labels,
        benign_labels=benign_labels,
    )


@lru_cache(maxsize=1)
def _get_sd198_grouped_label_space() -> LabelSpace:
    aliases: list[LabelAlias] = []
    for grouped_label in SD198_GROUPED_CANONICAL_LABELS:
        keywords = [grouped_label.lower()]
        keywords.extend(SD198_GROUPED_KEYWORDS.get(grouped_label, ()))
        aliases.append(LabelAlias(grouped_label, tuple(dict.fromkeys(keywords))))
    return LabelSpace(
        label_space_id="sd198_grouped",
        canonical_labels=tuple(SD198_GROUPED_CANONICAL_LABELS),
        aliases=tuple(aliases),
        malignant_labels=tuple(sorted(SD198_GROUPED_MALIGNANT_LABELS)),
        benign_labels=tuple(
            label for label in SD198_GROUPED_CANONICAL_LABELS if label not in SD198_GROUPED_MALIGNANT_LABELS
        ),
    )


@lru_cache(maxsize=1)
def _get_scin_full_label_space() -> LabelSpace:
    labels = tuple(SCIN_FULL_LABELS)
    aliases = tuple(
        LabelAlias(label, _build_keyword_variants(label))
        for label in sorted(labels, key=lambda item: (-len(_normalize_label_text(item)), item.lower()))
    )
    malignant_labels = tuple(label for label in sorted(labels) if _looks_malignant_scin_label(label))
    return LabelSpace(
        label_space_id="scin_full",
        canonical_labels=tuple(sorted(labels)),
        aliases=aliases,
        malignant_labels=malignant_labels,
        benign_labels=(),
    )


def _scin_grouped_label_for_text(text: str) -> str:
    normalized = str(text).strip().lower()
    if not normalized:
        return "OTHER"

    dermatitis_family = (
        "eczema",
        "dermatitis",
        "lichen simplex chronicus",
        "lichenified eczema",
        "seborrheic dermatitis",
        "photodermatitis",
        "intertrigo",
        "pruritic dermatitis",
    )
    bite_urticaria_follicular_family = (
        "urticaria",
        "insect bite",
        "folliculitis",
        "hypersensitivity",
        "prurigo",
        "miliaria",
    )
    infection_family = (
        "herpes zoster",
        "herpes simplex",
        "tinea",
        "impetigo",
        "viral exanthem",
        "candida",
        "molluscum",
        "scabies",
        "cellulitis",
        "infection",
    )
    vascular_family = (
        "vasculitis",
        "purpuric",
        "purpura",
        "hemangioma",
        "ecchym",
        "petech",
        "erythema ab igne",
        "vascular",
        "stasis dermatitis",
    )
    acne_family = (
        "acne",
        "rosacea",
        "perioral dermatitis",
        "keratosis pilaris",
        "hidradenitis",
        "comedone",
    )
    pigment_keratosis_family = (
        "nevus",
        "nevus",
        "melasma",
        "post-inflammatory",
        "seborrheic keratos",
        "sk/isk",
        "dermatofibroma",
        "lentigo",
        "pigment",
        "xanthoma",
        "vitiligo",
    )
    malignant_family = (
        "melanoma",
        "basal cell carcinoma",
        "scc/sccis",
        "squamous cell carcinoma",
        "actinic keratosis",
        "skin cancer",
        "lymphoma",
        "leukemia cutis",
        "metastasis",
        "sarcoma",
        "paget disease",
    )

    if any(keyword in normalized for keyword in malignant_family):
        return "MALIGNANT_PREMALIGNANT"
    if any(keyword in normalized for keyword in vascular_family):
        return "VASCULAR_PURPURIC"
    if any(keyword in normalized for keyword in infection_family):
        return "INFECTION_VIRAL_FUNGAL"
    if any(keyword in normalized for keyword in acne_family):
        return "ACNE_ROSACEA_FOLLICULAR"
    if any(keyword in normalized for keyword in pigment_keratosis_family):
        return "PIGMENT_KERATOSIS_NEVUS"
    if any(keyword in normalized for keyword in bite_urticaria_follicular_family):
        return "URTICARIA_BITE_FOLLICULITIS"
    if any(keyword in normalized for keyword in dermatitis_family):
        return "DERMATITIS_ECZEMA"
    return "OTHER"


@lru_cache(maxsize=1)
def _get_scin_grouped_label_space() -> LabelSpace:
    aliases: list[LabelAlias] = []
    grouped_to_keywords: dict[str, list[str]] = {label: [] for label in SCIN_GROUPED_CANONICAL_LABELS}
    for raw_label in SCIN_FULL_LABELS:
        grouped = _scin_grouped_label_for_text(raw_label)
        grouped_to_keywords.setdefault(grouped, []).append(raw_label)
    grouped_to_keywords["DERMATITIS_ECZEMA"].extend(
        [
            "Contact Dermatitis",
            "Allergic Contact Dermatitis",
            "Irritant Contact Dermatitis",
            "Acute dermatitis",
            "Acute dermatitis, NOS",
            "Acute and chronic dermatitis",
            "Eczema",
            "Psoriasis",
        ]
    )
    grouped_to_keywords["INFECTION_VIRAL_FUNGAL"].extend(
        [
            "Herpes Zoster",
            "Herpes Simplex",
            "Tinea",
            "Impetigo",
            "Cellulitis",
            "Candida",
            "Molluscum Contagiosum",
        ]
    )
    grouped_to_keywords["VASCULAR_PURPURIC"].extend(
        [
            "Leukocytoclastic Vasculitis",
            "Hemangioma",
            "Pigmented purpuric eruption",
            "Purpura",
            "Erythema ab igne",
            "Vasculitis",
        ]
    )
    grouped_to_keywords["ACNE_ROSACEA_FOLLICULAR"].extend(
        [
            "Acne",
            "Rosacea",
            "Folliculitis",
            "Perioral Dermatitis",
            "Keratosis pilaris",
        ]
    )
    grouped_to_keywords["PIGMENT_KERATOSIS_NEVUS"].extend(
        [
            "Nevus",
            "Melanocytic Nevus",
            "Seborrheic Keratosis",
            "SK/ISK",
            "Dermatofibroma",
        ]
    )
    grouped_to_keywords["MALIGNANT_PREMALIGNANT"].extend(
        [
            "Basal Cell Carcinoma",
            "Squamous Cell Carcinoma",
            "SCC/SCCIS",
            "Actinic Keratosis",
            "Melanoma",
            "Skin cancer",
        ]
    )
    for grouped_label in SCIN_GROUPED_CANONICAL_LABELS:
        keywords = [grouped_label.lower()]
        keywords.extend(item.lower() for item in grouped_to_keywords.get(grouped_label, []))
        aliases.append(LabelAlias(grouped_label, tuple(dict.fromkeys(keywords))))
    return LabelSpace(
        label_space_id="scin_grouped",
        canonical_labels=SCIN_GROUPED_CANONICAL_LABELS,
        aliases=tuple(aliases),
        malignant_labels=("MALIGNANT_PREMALIGNANT",),
        benign_labels=tuple(label for label in SCIN_GROUPED_CANONICAL_LABELS if label != "MALIGNANT_PREMALIGNANT"),
    )
