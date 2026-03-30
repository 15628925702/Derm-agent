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
}


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
) -> list[str]:
    ddx = [str(item).strip().lower() for item in (ddx_candidates or []) if str(item).strip()]
    pair = str(confusion_pair or "").strip().lower()
    evidence_text = " ".join(ddx + [pair, str(known_confusion_text or "").lower(), str(image_summary or "").lower()])
    if notes:
        evidence_text += " " + " ".join(str(item).strip().lower() for item in notes if str(item).strip())
    active: list[str] = []
    for cluster_name, definition in CONFUSION_CLUSTER_DEFINITIONS.items():
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


def cluster_guidance_snapshot(cluster_names: list[str], *, max_items: int = 2) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for cluster_name in cluster_names:
        definition = CONFUSION_CLUSTER_DEFINITIONS.get(cluster_name)
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


def cluster_priority_bonus(cluster_names: list[str], skill_name: str) -> float:
    bonus = 0.0
    for cluster_name in cluster_names:
        definition = CONFUSION_CLUSTER_DEFINITIONS.get(cluster_name, {})
        bonus = max(bonus, float(dict(definition.get("priority_skills", {})).get(skill_name, 0.0) or 0.0))
    return bonus


def cluster_ordering_hints(cluster_names: list[str]) -> dict[str, float]:
    hints: dict[str, float] = {}
    if not cluster_names:
        return hints

    # Keep description and exclusion-oriented evidence near the front when a
    # confusion cluster is active so later ordering logic can down-rank pure
    # risk items without losing the main comparison signal.
    hints["lesion_description_structuring_skill"] = 0.6
    hints["differential_compare_skill"] = 0.9
    hints["exclusion_reasoning_skill"] = 1.2

    for cluster_name in cluster_names:
        definition = CONFUSION_CLUSTER_DEFINITIONS.get(cluster_name, {})
        for skill_name, bonus in dict(definition.get("priority_skills", {})).items():
            normalized_name = str(skill_name).strip()
            if not normalized_name:
                continue
            hints[normalized_name] = max(float(hints.get(normalized_name, 0.0) or 0.0), float(bonus or 0.0))
    return hints


def cluster_related_keywords(cluster_names: list[str]) -> tuple[str, ...]:
    keywords: list[str] = []
    for cluster_name in cluster_names:
        definition = CONFUSION_CLUSTER_DEFINITIONS.get(cluster_name, {})
        for item in definition.get("keywords", ()):
            text = str(item).strip().lower()
            if text and text not in keywords:
                keywords.append(text)
    return tuple(keywords)


def cluster_pairs(cluster_names: list[str]) -> tuple[str, ...]:
    pairs: list[str] = []
    for cluster_name in cluster_names:
        definition = CONFUSION_CLUSTER_DEFINITIONS.get(cluster_name, {})
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
) -> float:
    lowered = str(text or "").lower()
    subtype_lower = str(subtype or "").strip().lower()
    bonus = 0.0
    for cluster_name in cluster_names:
        definition = CONFUSION_CLUSTER_DEFINITIONS.get(cluster_name, {})
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
) -> str:
    subtype = str(record.get("experience_type", record.get("source_subtype", ""))).strip().lower()
    text = " ".join(
        [
            str(record.get("confusion_pair", "")),
            str(record.get("perception_summary", "")),
            " ".join(str(item) for item in record.get("learning_points", [])),
        ]
    )
    if subtype in {"confusion_memory", "prototype"} and cluster_match_bonus(cluster_names=cluster_names, text=text, subtype=subtype) > 0:
        return "comparison"
    if subtype in {"rule", "rule_candidate"} and any(term in text.lower() for term in ("compare", "exclude", "confusion", "prototype")):
        return "comparison"
    return "risk"


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
