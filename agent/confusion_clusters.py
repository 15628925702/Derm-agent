from __future__ import annotations

import re
from typing import Any


TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "ack": ("ack", "actinic keratosis", "actinic keratos"),
    "bcc": ("bcc", "basal cell", "basal cell carcinoma"),
    "scc": ("scc", "squamous", "squamous cell", "squamous cell carcinoma"),
    "sek": ("sek", "seborrheic keratosis", "seborrheic keratos"),
    "mel": ("mel", "melanoma", "malignant melanoma"),
    "nev": ("nev", "nevus", "naevus", "mole"),
    # HAM10000 / ISIC aliases
    "nv": ("nv", "nevus", "naevus", "mole", "melanocytic nevus"),
    "bkl": ("bkl", "benign keratosis", "seborrheic keratosis", "lichenoid keratosis"),
    "df": ("df", "dermatofibroma"),
    "vasc": ("vasc", "vascular lesion", "angioma", "hemangioma"),
    "akiec": ("akiec", "actinic keratosis", "bowen", "intraepithelial carcinoma"),
}

# Registry for dataset-specific confusion cluster definitions.
# Keys are dataset_name strings; values are dicts in the same format as
# CONFUSION_CLUSTER_DEFINITIONS.  Populated via register_confusion_clusters().
_DATASET_CLUSTER_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}


def register_confusion_clusters(dataset_name: str, clusters: dict[str, dict[str, Any]]) -> None:
    """Register dataset-specific confusion cluster definitions."""
    _DATASET_CLUSTER_REGISTRY[str(dataset_name).strip().lower()] = clusters


def get_confusion_cluster_definitions(dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
    """Return cluster definitions for the given dataset, falling back to the default PAD-UFES-20 set."""
    if dataset_name:
        key = str(dataset_name).strip().lower()
        if key in _DATASET_CLUSTER_REGISTRY:
            return _DATASET_CLUSTER_REGISTRY[key]
    return CONFUSION_CLUSTER_DEFINITIONS


# Default PAD-UFES-20 cluster definitions (kept for backward compatibility).
CONFUSION_CLUSTER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "ack_bcc_scc": {
        "label": "ACK / BCC / SCC",
        "pairs": (
            "ack->bcc",
            "actinic keratosis->bcc",
            "scc->bcc",
            "squamous cell carcinoma->bcc",
            "ack->scc",
            "actinic keratosis->scc",
        ),
        "term_groups": (
            ("ack", "bcc"),
            ("scc", "bcc"),
            ("ack", "scc"),
        ),
        "keywords": (
            "ack",
            "actinic keratosis",
            "bcc",
            "basal cell",
            "scc",
            "squamous",
            "keratin",
            "scale",
            "pearly",
            "translucent",
            "ulcer",
            "crust",
        ),
        "supporting_clues": (
            "Separate keratotic scale from pearly/translucent or rolled-border cues.",
            "Keep focal ulcer/crust from being treated as automatic SCC support when BCC-like morphology remains plausible.",
            "Use distribution and sun-exposed context only as supporting context, not decisive proof.",
        ),
        "opposing_clues": (
            "Absence of clear pearly/rolled-border or translucency weakens BCC confidence.",
            "Absence of thicker hyperkeratosis, deeper surface destruction, or more invasive texture weakens SCC-style escalation.",
            "Roughness alone should not over-call ACK/SCC when other malignant surface clues are unclear.",
        ),
        "missing_evidence": (
            "Missing closer surface/border inspection for pearly or rolled-edge detail.",
            "Missing vessel/ulcer depth detail that would better separate superficial scale from invasive disruption.",
            "Missing dermoscopic or magnified texture clues for keratin versus translucency.",
        ),
        "watch_outs": (
            "Do not equate crust, irritation, or roughness by itself with SCC.",
            "Do not exclude BCC without naming at least one explicit opposing clue and one still-missing check.",
            "Do not let sun-exposed location substitute for lesion-specific evidence.",
        ),
        "priority_skills": {
            "ack_scc_specialist_skill": 2.0,
            "exclusion_reasoning_skill": 1.5,
            "differential_compare_skill": 1.0,
            "lesion_description_structuring_skill": 0.8,
        },
    },
    "ack_sek": {
        "label": "ACK / SEK",
        "pairs": (
            "ack->sek",
            "actinic keratosis->sek",
            "sek->ack",
            "seborrheic keratosis->ack",
        ),
        "term_groups": (
            ("ack", "sek"),
            ("ack", "seborrheic keratosis"),
        ),
        "keywords": (
            "ack",
            "actinic keratosis",
            "sek",
            "seborrheic keratosis",
            "waxy",
            "stuck-on",
            "verrucous",
            "scale",
            "keratotic",
            "face",
        ),
        "supporting_clues": (
            "Explicitly contrast stuck-on/waxy/verrucous clues against flatter actinic roughness.",
            "State whether the lesion reads as sharply demarcated keratotic plaque versus more actinic field-type change.",
            "Preserve any negative clues that argue against classic seborrheic keratosis patterning.",
        ),
        "opposing_clues": (
            "Scale alone is not enough to favor ACK when waxy or stuck-on cues dominate.",
            "Seborrheic-looking keratin does not exclude ACK if actinic roughness and malignant-risk context remain active.",
            "If lesion lacks clear stuck-on or waxy character, do not over-reassure toward SEK.",
        ),
        "missing_evidence": (
            "Missing closer surface texture and border-demarcation detail.",
            "Missing confirmation of waxy/stuck-on appearance versus flatter actinic roughness.",
            "Missing broader field-change context or adjacent sun-damaged skin comparison.",
        ),
        "watch_outs": (
            "Do not use keratin or scale as a shortcut to either ACK or SEK.",
            "Do not hide uncertainty when stuck-on/waxy morphology is not clearly seen.",
            "Do not ignore malignant-risk context just because seborrheic terminology appears in the initial ddx.",
        ),
        "priority_skills": {
            "ack_scc_specialist_skill": 1.8,
            "exclusion_reasoning_skill": 1.4,
            "differential_compare_skill": 1.0,
            "lesion_description_structuring_skill": 0.9,
        },
    },
    "mel_nev": {
        "label": "Melanoma / Nevus",
        "pairs": (
            "melanoma->nev",
            "malignant melanoma->nev",
            "superficial spreading melanoma->nev",
        ),
        "term_groups": (
            ("mel", "nev"),
            ("melanoma", "mole"),
        ),
        "keywords": (
            "mel",
            "melanoma",
            "nev",
            "nevus",
            "naevus",
            "mole",
            "pigmented",
            "asymmetry",
            "irregular border",
            "variegated",
        ),
        "supporting_clues": (
            "Separate melanoma-like asymmetry, irregular border, and pigment complexity from nevus-like symmetry or pattern stability.",
            "Preserve negative melanoma clues if symmetry or simpler pigment pattern is more convincing.",
            "Use temporal evolution only when actually supported, not as a default risk filler.",
        ),
        "opposing_clues": (
            "Absence of convincing asymmetry or pigment complexity weakens melanoma-style escalation.",
            "Benign-leaning symmetry or simpler border pattern is active opposing evidence, not just weak support.",
            "Do not convert darker color alone into melanoma support without structural irregularity.",
        ),
        "missing_evidence": (
            "Missing dermoscopic/pattern detail for pigment network or additional structure.",
            "Missing clearer evolution history when growth/change is not explicit.",
            "Missing close border inspection to support true irregularity rather than image uncertainty.",
        ),
        "watch_outs": (
            "Do not turn pigmented concern into a final melanoma verdict.",
            "Do not ignore nevus-like opposing evidence when uncertainty remains high.",
            "Do not over-read image noise or shadow as true color variegation.",
        ),
        "priority_skills": {
            "mel_nev_specialist_skill": 2.0,
            "exclusion_reasoning_skill": 1.5,
            "differential_compare_skill": 1.0,
            "lesion_description_structuring_skill": 0.8,
        },
    },
}


def has_confusion_pair(
    ddx_candidates: list[str],
    left_keywords: tuple[str, ...],
    right_keywords: tuple[str, ...],
) -> bool:
    lowered = [str(item).strip().lower() for item in ddx_candidates if str(item).strip()]
    has_left = any(any(keyword in candidate for keyword in left_keywords) for candidate in lowered)
    has_right = any(any(keyword in candidate for keyword in right_keywords) for candidate in lowered)
    return has_left and has_right


def detect_primary_confusion_pair(ddx_candidates: list[str]) -> str | None:
    lowered = [str(item).strip().lower() for item in ddx_candidates if str(item).strip()]
    if has_confusion_pair(lowered, ("mel", "melanoma"), ("nev", "nevus", "naevus", "mole")):
        return "melanoma->nev"
    if has_confusion_pair(lowered, ("scc", "squamous cell", "squamous"), ("bcc", "basal cell")):
        return "scc->bcc"
    if has_confusion_pair(lowered, ("ack", "actinic keratosis", "actinic keratos"), ("bcc", "basal cell")):
        return "ack->bcc"
    if has_confusion_pair(lowered, ("seborrheic keratosis", "sek"), ("bcc", "basal cell")):
        return "sek->bcc"
    if has_confusion_pair(lowered, ("ack", "actinic keratosis", "actinic keratos"), ("seborrheic keratosis", "sek")):
        return "ack->sek"
    if has_confusion_pair(
        lowered,
        ("lichen simplex", "lichen planus", "psoriasis", "dermatitis", "eczema"),
        ("ack", "actinic keratosis", "actinic keratos"),
    ):
        return "inflammatory->ack"
    if has_confusion_pair(lowered, ("ack", "actinic keratosis", "actinic keratos"), ("scc", "squamous cell", "squamous")):
        return "ack->scc"
    return None


def detect_confusion_clusters(
    *,
    ddx_candidates: list[str] | None = None,
    confusion_pair: str | None = None,
    known_confusion_text: str = "",
    image_summary: str = "",
    notes: list[str] | None = None,
    dataset_name: str | None = None,
) -> list[str]:
    ddx = [str(item).strip().lower() for item in (ddx_candidates or []) if str(item).strip()]
    pair = str(confusion_pair or "").strip().lower()
    evidence_text = " ".join(ddx + [pair, str(known_confusion_text or "").lower(), str(image_summary or "").lower()])
    if notes:
        evidence_text += " " + " ".join(str(item).strip().lower() for item in notes if str(item).strip())
    active: list[str] = []
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name, definition in cluster_defs.items():
        pairs = {str(item).strip().lower() for item in definition.get("pairs", ()) if str(item).strip()}
        if pair and pair in pairs:
            active.append(cluster_name)
            continue
        if any(candidate in str(known_confusion_text or "").lower() for candidate in pairs):
            active.append(cluster_name)
            continue
        term_groups = definition.get("term_groups", ())
        if any(all(_contains_term(evidence_text, term) for term in group) for group in term_groups):
            active.append(cluster_name)
    return list(dict.fromkeys(active))


def cluster_guidance_snapshot(cluster_names: list[str], *, max_items: int = 2, dataset_name: str | None = None) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in cluster_names:
        definition = cluster_defs.get(cluster_name)
        if not definition:
            continue
        snapshots.append(
            {
                "cluster_id": cluster_name,
                "label": definition.get("label"),
                "supporting_clues": list(definition.get("supporting_clues", ()))[:max_items],
                "opposing_clues": list(definition.get("opposing_clues", ()))[:max_items],
                "required_missing_evidence": list(definition.get("missing_evidence", ()))[:max_items],
                "watch_outs": list(definition.get("watch_outs", ()))[:max_items],
            }
        )
    return snapshots


def cluster_priority_bonus(cluster_names: list[str], skill_name: str, dataset_name: str | None = None) -> float:
    bonus = 0.0
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in cluster_names:
        definition = cluster_defs.get(cluster_name, {})
        bonus = max(bonus, float(dict(definition.get("priority_skills", {})).get(skill_name, 0.0) or 0.0))
    return bonus


def cluster_ordering_hints(cluster_names: list[str], dataset_name: str | None = None) -> dict[str, float]:
    hints: dict[str, float] = {}
    if not cluster_names:
        return hints

    hints["lesion_description_structuring_skill"] = 0.6
    hints["differential_compare_skill"] = 0.9
    hints["exclusion_reasoning_skill"] = 1.2

    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in cluster_names:
        definition = cluster_defs.get(cluster_name, {})
        for skill_name, bonus in dict(definition.get("priority_skills", {})).items():
            normalized_name = str(skill_name).strip()
            if not normalized_name:
                continue
            hints[normalized_name] = max(float(hints.get(normalized_name, 0.0) or 0.0), float(bonus or 0.0))
    return hints


def cluster_related_keywords(cluster_names: list[str], dataset_name: str | None = None) -> tuple[str, ...]:
    keywords: list[str] = []
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in cluster_names:
        definition = cluster_defs.get(cluster_name, {})
        for item in definition.get("keywords", ()):
            text = str(item).strip().lower()
            if text and text not in keywords:
                keywords.append(text)
    return tuple(keywords)


def cluster_pairs(cluster_names: list[str], dataset_name: str | None = None) -> tuple[str, ...]:
    pairs: list[str] = []
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in cluster_names:
        definition = cluster_defs.get(cluster_name, {})
        for item in definition.get("pairs", ()):
            text = str(item).strip().lower()
            if text and text not in pairs:
                pairs.append(text)
    return tuple(pairs)


def cluster_match_bonus(
    *,
    cluster_names: list[str],
    text: str,
    subtype: str = "",
    dataset_name: str | None = None,
) -> float:
    lowered = str(text or "").lower()
    subtype_lower = str(subtype or "").strip().lower()
    bonus = 0.0
    cluster_defs = get_confusion_cluster_definitions(dataset_name)
    for cluster_name in cluster_names:
        definition = cluster_defs.get(cluster_name, {})
        pairs = {str(item).strip().lower() for item in definition.get("pairs", ()) if str(item).strip()}
        keywords = {str(item).strip().lower() for item in definition.get("keywords", ()) if str(item).strip()}
        if any(pair in lowered for pair in pairs):
            bonus = max(bonus, 2.0 if subtype_lower in {"confusion_memory", "prototype"} else 1.2)
            continue
        hit_count = sum(1 for keyword in keywords if _contains_term(lowered, keyword))
        if hit_count >= 3:
            bonus = max(bonus, 1.4 if subtype_lower in {"confusion_memory", "prototype"} else 0.9)
        elif hit_count >= 2:
            bonus = max(bonus, 0.8)
    return bonus


def preferred_abstract_section(
    *,
    record: dict[str, Any],
    cluster_names: list[str],
    dataset_name: str | None = None,
) -> str:
    subtype = str(record.get("experience_type", record.get("source_subtype", ""))).strip().lower()
    text = " ".join(
        [
            str(record.get("confusion_pair", "")),
            str(record.get("perception_summary", "")),
            " ".join(str(item) for item in record.get("learning_points", [])),
        ]
    )
    if subtype in {"confusion_memory", "prototype"} and cluster_match_bonus(cluster_names=cluster_names, text=text, subtype=subtype, dataset_name=dataset_name) > 0:
        return "comparison"
    if subtype in {"rule", "rule_candidate"} and any(term in text.lower() for term in ("compare", "exclude", "confusion", "prototype")):
        return "comparison"
    return "risk"


register_confusion_clusters("ham10000", {
    "mel_nv": {
        "label": "Melanoma / Nevus",
        "pairs": (
            "melanoma->nv",
            "malignant melanoma->nv",
            "mel->nv",
        ),
        "term_groups": (
            ("mel", "nv"),
            ("melanoma", "nv"),
        ),
        "keywords": (
            "mel", "melanoma", "nv", "nevus", "naevus", "mole",
            "pigmented", "asymmetry", "irregular border", "variegated",
            "atypical network", "regression",
        ),
        "supporting_clues": (
            "Contrast irregular pigment network, asymmetry, and color variation against uniform nevus pattern.",
            "Dermoscopic atypical network or regression structures strongly support melanoma over nevus.",
            "Preserve negative melanoma clues if symmetry or simpler pigment pattern is more convincing.",
        ),
        "opposing_clues": (
            "Symmetric, uniform pigmentation and regular border weakens melanoma confidence.",
            "Absence of atypical network, regression, or blue-white veil argues against melanoma.",
            "Do not convert darker color alone into melanoma support without structural irregularity.",
        ),
        "missing_evidence": (
            "Missing dermoscopic detail for atypical pigment network or additional structures.",
            "Missing evolution history when growth or change is not documented.",
            "Missing close border inspection to confirm true irregularity.",
        ),
        "watch_outs": (
            "Do not call melanoma based on size or darkness alone.",
            "NV is the dominant class (67%) — do not over-escalate to melanoma without clear structural evidence.",
            "Do not ignore nevus-like opposing evidence when uncertainty remains high.",
        ),
        "priority_skills": {
            "mel_nev_specialist_skill": 2.0,
            "malignancy_risk_assessment_skill": 1.5,
            "differential_compare_skill": 1.0,
            "lesion_description_structuring_skill": 0.8,
        },
    },
    "bkl_nv": {
        "label": "Benign Keratosis / Nevus",
        "pairs": (
            "bkl->nv",
            "benign keratosis->nv",
            "seborrheic keratosis->nv",
        ),
        "term_groups": (
            ("bkl", "nv"),
            ("benign keratosis", "nv"),
        ),
        "keywords": (
            "bkl", "benign keratosis", "seborrheic keratosis", "lichenoid keratosis",
            "nv", "nevus", "mole",
            "waxy", "stuck-on", "milia-like cysts", "comedo-like openings",
        ),
        "supporting_clues": (
            "Milia-like cysts and comedo-like openings are strong BKL indicators over nevus.",
            "Waxy or stuck-on surface texture with sharp demarcation favors BKL.",
            "Absence of pigment network argues against melanocytic nevus.",
        ),
        "opposing_clues": (
            "Presence of pigment network or globules weakens BKL and supports nevus.",
            "Smooth, dome-shaped lesion without keratotic surface argues against BKL.",
        ),
        "missing_evidence": (
            "Missing close surface texture detail to confirm milia-like cysts or comedo openings.",
            "Missing dermoscopic detail to confirm or exclude pigment network.",
        ),
        "watch_outs": (
            "Do not confuse flat seborrheic keratosis with melanocytic nevus based on color alone.",
            "BKL and NV together account for ~78% of HAM10000 — careful differentiation is critical.",
        ),
        "priority_skills": {
            "differential_compare_skill": 2.0,
            "lesion_description_structuring_skill": 1.5,
            "border_surface_analysis_skill": 1.2,
            "exclusion_reasoning_skill": 1.0,
        },
    },
    "mel_bkl": {
        "label": "Melanoma / Benign Keratosis",
        "pairs": (
            "melanoma->bkl",
            "mel->bkl",
        ),
        "term_groups": (
            ("mel", "bkl"),
        ),
        "keywords": (
            "mel", "melanoma", "bkl", "benign keratosis", "seborrheic keratosis",
            "pigmented", "dark", "irregular",
        ),
        "supporting_clues": (
            "Irregular pigmentation without keratotic surface features favors melanoma over BKL.",
            "Atypical vascular structures or regression areas support melanoma.",
        ),
        "opposing_clues": (
            "Milia-like cysts, comedo openings, or stuck-on appearance strongly argue against melanoma.",
            "Sharp, well-demarcated border with keratotic surface favors BKL.",
        ),
        "missing_evidence": (
            "Missing dermoscopic detail to confirm or exclude keratotic surface structures.",
        ),
        "watch_outs": (
            "Pigmented BKL can mimic melanoma — do not escalate without explicit keratotic surface exclusion.",
        ),
        "priority_skills": {
            "malignancy_risk_assessment_skill": 2.0,
            "mel_nev_specialist_skill": 1.5,
            "exclusion_reasoning_skill": 1.2,
        },
    },
})


def _contains_term(text: str, term: str) -> bool:
    source = str(text or "").lower()
    query = str(term or "").strip().lower()
    if not source or not query:
        return False
    candidates = TERM_ALIASES.get(query, (query,))
    for candidate in candidates:
        pattern = rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])"
        if re.search(pattern, source) is not None:
            return True
    return False
