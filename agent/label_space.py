from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
}


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
        return LABEL_SPACES.get(explicit_id, DERM_SIX_LABEL_SPACE)

    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key:
        mapped_id = DATASET_LABEL_SPACE_ALIASES.get(dataset_key)
        if mapped_id:
            return LABEL_SPACES.get(mapped_id, DERM_SIX_LABEL_SPACE)

    metadata_payload = dict(metadata or {})
    metadata_label_space = str(metadata_payload.get("label_space_id", "")).strip().lower()
    if metadata_label_space:
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
    for alias in label_space.aliases:
        if alias.canonical.upper() == upper_text:
            return alias.canonical.upper()
        for keyword in alias.keywords:
            if str(keyword).strip().lower() in lowered:
                return alias.canonical.upper()
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
    left_canonical = canonicalize_label(left, label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    right_canonical = canonicalize_label(right, label_space_id=label_space_id, dataset_name=dataset_name, metadata=metadata)
    if left_canonical and right_canonical:
        return left_canonical == right_canonical
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
