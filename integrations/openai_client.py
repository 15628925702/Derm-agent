from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from openai import APIConnectionError
    from openai import APITimeoutError
    from openai import BadRequestError
    from openai import InternalServerError
    from openai import OpenAI
    from openai import RateLimitError
except ModuleNotFoundError:  # pragma: no cover - optional dependency for offline/unit-test environments
    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    OpenAI = None  # type: ignore[assignment]

from agent.evidence_package import EvidencePackage
from agent.label_space import resolve_label_space
from agent.state import CaseInput
from agent.workflow_profiles import ensure_workflow_context, is_family_routing_case, is_full_taxonomy_case


LOGGER = logging.getLogger(__name__)
MAX_INLINE_IMAGE_EDGE = int(os.getenv("DERMAGENT_MAX_INLINE_IMAGE_EDGE", "1024") or "1024")
INLINE_IMAGE_JPEG_QUALITY = int(os.getenv("DERMAGENT_INLINE_IMAGE_JPEG_QUALITY", "80") or "80")


def _case_workflow_context(case_input: CaseInput) -> dict[str, Any]:
    return ensure_workflow_context(
        dataset_name=getattr(case_input, "dataset_name", None),
        metadata=getattr(case_input, "metadata", None),
        label_space_id=getattr(case_input, "label_space_id", None),
        workflow_context=getattr(case_input, "workflow_context", None),
    )


def _build_label_space_calibration_note(case_input: CaseInput) -> str:
    """Derive a calibration note from the label space structure.

    When benign classes outnumber malignant ones, warn Qwen not to let risk
    caution flags alone shift the diagnosis toward malignant classes.
    Works for any dataset with a registered label space — no hardcoding needed.
    """
    ls = resolve_label_space(
        label_space_id=getattr(case_input, "label_space_id", None),
        dataset_name=getattr(case_input, "dataset_name", None),
        metadata=getattr(case_input, "metadata", None),
    )
    nm = len(ls.malignant_labels)
    nb = len(ls.benign_labels)
    if nb > nm:
        benign_list = ", ".join(ls.benign_labels)
        malignant_list = ", ".join(ls.malignant_labels)
        return (
            f"Label space calibration: this label space has {nb} benign classes ({benign_list}) "
            f"and {nm} malignant classes ({malignant_list}). "
            "Benign classes outnumber malignant ones. "
            "When the override layer says risk_only (no subtype override allowed), "
            "do NOT shift the diagnosis toward malignant classes based on risk caution flags alone — "
            "stay close to the image-based baseline and note the risk concern only in follow_up_considerations."
        )
    if nm > nb:
        return (
            f"Label space calibration: this label space has {nm} malignant classes and {nb} benign classes. "
            "Malignant and pre-malignant classes are relatively common in this label space."
        )
    return ""


def _build_label_space_prompt_hint(case_input: CaseInput) -> str:
    ls = resolve_label_space(
        label_space_id=getattr(case_input, "label_space_id", None),
        dataset_name=getattr(case_input, "dataset_name", None),
        metadata=getattr(case_input, "metadata", None),
    )
    labels = [str(label).strip() for label in ls.canonical_labels if str(label).strip()]
    if not labels:
        return ""
    dataset_name = str(getattr(case_input, "dataset_name", "") or "").strip().lower()
    workflow_context = _case_workflow_context(case_input)
    workflow_profile = str(workflow_context.get("workflow_profile", "")).strip().lower()
    if dataset_name == "xiangya_sft" or workflow_profile == "eczematous_family_routing_workflow":
        return (
            "Xiangya grouped-family note: this dataset currently focuses on inflammatory eczematous-family reasoning. "
            "Prefer one grouped label among `CONTACT_DERMATITIS`, `ATOPIC_DERMATITIS`, `ECZEMA_DERMATITIS`, "
            "`PERIORAL_DERMATITIS`, `HERPETIC_ECZEMA`, `HAIR_DISORDER`, and `OTHER_INFLAMMATORY`.\n"
            "Do not collapse every erythematous rash into `CONTACT_DERMATITIS`.\n"
            "Use these distinctions:\n"
            "- `ATOPIC_DERMATITIS`: chronic or recurrent eczematous process, symmetric or widespread distribution, flexural or pediatric pattern, xerosis/lichenification, repeated flare history.\n"
            "- `ECZEMA_DERMATITIS`: eczematous inflammatory rash when the evidence supports eczema-like morphology but is not specific enough for atopic/contact subtype.\n"
            "- `CONTACT_DERMATITIS`: localized or exposure-pattern dermatitis, linear/contact-distribution clues, periocular/perioral/hairline/cosmetic/topical exposure pattern, sharper trigger-linked presentation.\n"
            "- `PERIORAL_DERMATITIS`: concentrated around the mouth/nasolabial/perioral region.\n"
            "- `HERPETIC_ECZEMA`: acute erosive/crusted painful monomorphic eruption on top of eczematous skin.\n"
            "When evidence is mixed between contact and atopic/eczema, prefer `ECZEMA_DERMATITIS` rather than defaulting to `CONTACT_DERMATITIS`."
        )
    if workflow_profile == "image_archive_full_taxonomy_lesion_workflow":
        return (
            "Image-archive full-taxonomy note: this is an archive-style dermoscopy/lesion-photo workflow with a fixed lesion label space. "
            "Do not let generic caution flags or ambiguous risk wording override the dominant morphology.\n"
            "When lesion evidence is mixed, keep benign archive alternatives such as `NV`, `BKL`, and `DF` active instead of drifting to melanoma by default.\n"
            "Use metadata like `anatom_site_general` and `age_approx` only as secondary context, not as primary override evidence."
        )
    if is_full_taxonomy_case(
        workflow_context=workflow_context,
        label_space_id=str(ls.label_space_id),
    ) and "scin" in str(ls.label_space_id).strip().lower():
        metadata = dict(getattr(case_input, "metadata", {}) or {})
        related_category = str(metadata.get("related_category", "")).strip().upper()
        category_hint = ""
        if related_category == "RASH":
            category_hint = (
                "Given SCIN metadata category `RASH`, especially consider inflammatory and infectious rash labels such as "
                "`Eczema`, `Allergic Contact Dermatitis`, `Irritant Contact Dermatitis`, `Acute dermatitis, NOS`, "
                "`Acute and chronic dermatitis`, `Herpes Zoster`, `Tinea`, `Psoriasis`, `Urticaria`, and `Drug Rash`. "
            )
        elif related_category == "ACNE":
            category_hint = (
                "Given SCIN metadata category `ACNE`, especially consider labels such as "
                "`Acne`, `Folliculitis`, `Perioral Dermatitis`, `Keratosis pilaris`, and `Rosacea`. "
            )
        elif related_category == "GROWTH_OR_MOLE":
            category_hint = (
                "Given SCIN metadata category `GROWTH_OR_MOLE`, especially consider lesion/growth labels such as "
                "`Dermatofibroma`, `Melanocytic Nevus`, `Pyogenic granuloma`, `SK/ISK`, and other focal growth labels. "
            )
        examples = ", ".join(labels[:60])
        suffix = f", and {len(labels) - 60} more labels" if len(labels) > 60 else ""
        return (
            "SCIN full-label note: return the most specific SCIN-style disease label you can justify from the image and metadata. "
            "Avoid collapsing to a broader umbrella term when a more specific SCIN label is supported. "
            "For example, prefer `Allergic Contact Dermatitis` over generic `Contact Dermatitis`, "
            "`Acute dermatitis, NOS` over vague dermatitis wording, and "
            "`Herpes Zoster` over a broad inflammatory rash term when morphology supports it. "
            f"{category_hint}"
            f"Example SCIN labels include: {examples}{suffix}."
        )
    return (
        "Dataset label-space note: prefer returning a label that fits the registered dataset label space. "
        + ", ".join(labels[:30])
        + (" ..." if len(labels) > 30 else "")
    )


def _build_sparse_lesion_prompt_hint(case_input: CaseInput) -> str:
    workflow_context = _case_workflow_context(case_input)
    if str(workflow_context.get("workflow_profile", "")).strip().lower() != "sparse_lesion_workflow":
        return ""
    label_space = resolve_label_space(
        label_space_id=getattr(case_input, "label_space_id", None),
        dataset_name=getattr(case_input, "dataset_name", None),
        metadata=getattr(case_input, "metadata", None),
    )
    if str(label_space.label_space_id).strip().lower() != "ham10000_full":
        return ""
    return (
        "HAM10000 sparse-lesion note: keep reasoning inside the HAM10000 label space. "
        "Do not drift to open-set inflammatory or infectious labels such as rosacea, erythema migrans, perioral dermatitis, eczema, or psoriasis. "
        "When morphology is ambiguous, prefer the closest HAM10000 label among MEL, BCC, NV, BKL, DF, VASC, and AKIEC. "
        "In the early differential (`ddx_candidates`), keep benign-mimic alternatives visible instead of collapsing immediately to only melanoma or BCC. "
        "For rough, keratotic, or actinic-looking facial lesions, explicitly keep AKIEC in consideration. "
        "For stuck-on or benign-keratosis-like lesions, explicitly keep BKL in consideration. "
        "For scar-like firm plaques or nodules, explicitly keep DF in consideration. "
        "For vascular-looking red-purple lesions, explicitly keep VASC in consideration."
    )


def _build_sparse_benign_mimic_guard_hint(case_input: CaseInput) -> str:
    workflow_context = _case_workflow_context(case_input)
    if str(workflow_context.get("workflow_profile", "")).strip().lower() != "sparse_lesion_workflow":
        return ""
    if not bool(workflow_context.get("benign_mimic_like_signature", False)):
        return ""
    return (
        "Sparse benign-mimic guard: this lesion shows benign-mimic-like morphology under the sparse lesion workflow. "
        "Do not default to Basal Cell Carcinoma from a central dark area, central hypopigmentation, or irregular border alone. "
        "Only favor BCC when you can point to lesion-specific BCC evidence such as pearly/translucent quality, rolled border, shiny papule/nodule morphology, arborizing vessels, or focal ulceration. "
        "If those classic BCC cues are not explicit, keep BKL, DF, VASC, or NV-style benign mimic alternatives active and avoid over-calling malignancy from generic irregularity alone."
    )


def _build_scin_routing_hint(case_input: CaseInput) -> str:
    label_space_id = str(getattr(case_input, "label_space_id", "") or "").strip().lower()
    if not is_full_taxonomy_case(workflow_context=_case_workflow_context(case_input), label_space_id=label_space_id):
        return ""

    metadata = dict(getattr(case_input, "metadata", {}) or {})
    related_category = str(metadata.get("related_category", "")).strip().upper()
    duration = str(metadata.get("condition_duration", "")).strip().upper()
    body_sites = {str(item).strip().lower() for item in metadata.get("body_sites", []) if str(item).strip()}
    textures = {str(item).strip().lower() for item in metadata.get("textures_present", []) if str(item).strip()}
    symptoms = {str(item).strip().lower() for item in metadata.get("symptoms_present", []) if str(item).strip()}

    notes: list[str] = []
    shortlist: list[str] = []

    if related_category == "RASH":
        notes.append(
            "SCIN routing note for RASH: do not default to generic contact dermatitis unless the morphology and scenario truly support it."
        )
        if "rough_or_flaky" in textures:
            shortlist.extend(["Eczema", "Psoriasis", "Seborrheic Dermatitis", "Acute dermatitis, NOS"])
            notes.append("Rough or flaky texture should increase consideration of eczema/psoriasis-like labels.")
        if "flat" in textures and "leg" in body_sites:
            shortlist.extend(["Leukocytoclastic Vasculitis", "Pigmented purpuric eruption", "Erythema ab igne"])
            notes.append("Flat red lesions on the leg should trigger vasculitic or purpuric alternatives before dermatitis defaulting.")
        if duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"} and ("itching" in symptoms or "burning" in symptoms or "pain" in symptoms):
            shortlist.extend(["Herpes Zoster", "Urticaria", "Allergic Contact Dermatitis"])
            notes.append("Very acute itchy/burning rash should explicitly consider herpes zoster or urticarial processes.")
        if "raised_or_bumpy" in textures:
            shortlist.extend(["Allergic Contact Dermatitis", "Irritant Contact Dermatitis", "Insect Bite", "Folliculitis"])
        if duration in {"ONE_TO_THREE_MONTHS", "ONE_TO_FOUR_WEEKS", "MORE_THAN_THREE_MONTHS"}:
            shortlist.extend(["Eczema", "Acute and chronic dermatitis", "Acute dermatitis, NOS"])
    elif related_category == "ACNE":
        shortlist.extend(["Acne", "Folliculitis", "Perioral Dermatitis", "Keratosis pilaris", "Rosacea"])
    elif related_category == "GROWTH_OR_MOLE":
        shortlist.extend(["Dermatofibroma", "Melanocytic Nevus", "Pyogenic granuloma", "SK/ISK"])

    if not notes and not shortlist:
        return ""

    deduped_shortlist: list[str] = []
    seen: set[str] = set()
    for label in shortlist:
        if label not in seen:
            seen.add(label)
            deduped_shortlist.append(label)

    parts = notes[:3]
    if deduped_shortlist:
        parts.append("Shortlist to discriminate carefully: " + ", ".join(deduped_shortlist[:10]) + ".")
    return " ".join(parts)


def _build_xiangya_family_routing_hint(case_input: CaseInput) -> str:
    workflow_context = _case_workflow_context(case_input)
    if str(workflow_context.get("workflow_profile", "")).strip().lower() != "eczematous_family_routing_workflow":
        return ""

    metadata = dict(getattr(case_input, "metadata", {}) or {})
    body_sites = {str(item).strip().lower() for item in metadata.get("body_sites", []) if str(item).strip()}
    related_category = str(metadata.get("related_category", "")).strip().upper()

    notes = [
        "Xiangya family-routing note: explicitly discriminate contact dermatitis vs atopic dermatitis vs non-specific eczema before committing to a final label.",
        "If the rash is diffuse, recurrent, symmetric, pediatric, xerotic, or flexural-patterned, increase `ATOPIC_DERMATITIS` / `ECZEMA_DERMATITIS` weight.",
        "If the rash is localized to a likely exposure zone such as periocular, perioral, face edge, hairline, or a linear/contact-shaped distribution, increase `CONTACT_DERMATITIS` weight.",
        "If the image shows eczematous morphology but exposure specificity is weak, prefer `ECZEMA_DERMATITIS` instead of defaulting to `CONTACT_DERMATITIS`.",
        "Negative cues matter: phrases like `no clear linear/contact distribution`, `widespread symmetric eruption`, or `not sharply demarcated` should reduce `CONTACT_DERMATITIS` confidence.",
    ]
    if {"mouth", "lip", "cheek"} & body_sites:
        notes.append("Perioral/periorificial localization should keep `PERIORAL_DERMATITIS` active.")
    if related_category == "RASH":
        notes.append("This is a rash-family case, so grouped inflammatory family reasoning should dominate over lesion-style subtype guessing.")
    return " ".join(notes)


def _xiangya_grouped_label_for_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(text).strip().lower())
    normalized = " ".join(normalized.split())
    if not normalized:
        return "OTHER_INFLAMMATORY"
    if any(token in normalized for token in ("perioral dermatitis", "口周皮炎")):
        return "PERIORAL_DERMATITIS"
    if any(token in normalized for token in ("eczema herpeticum", "herpetic eczema", "疱疹性湿疹")):
        return "HERPETIC_ECZEMA"
    if any(token in normalized for token in ("atopic dermatitis", "特应性皮炎", " ad ", "ad患儿")):
        return "ATOPIC_DERMATITIS"
    if any(token in normalized for token in ("allergic contact dermatitis", "irritant contact dermatitis", "contact dermatitis", "接触性皮炎", "隐翅虫皮炎")):
        return "CONTACT_DERMATITIS"
    if any(token in normalized for token in ("eczema", "湿疹", "自身敏感性皮炎", "传染性湿疹样皮炎")):
        return "ECZEMA_DERMATITIS"
    if any(token in normalized for token in ("alopecia", "hair loss", "脱发")):
        return "HAIR_DISORDER"
    return "OTHER_INFLAMMATORY"


def _refine_xiangya_grouped_payload(
    *,
    case_input: CaseInput,
    payload: dict[str, Any],
    evidence_package: dict[str, Any] | None = None,
    baseline_mode: bool = False,
) -> dict[str, Any]:
    workflow_context = _case_workflow_context(case_input)
    if str(workflow_context.get("workflow_profile", "")).strip().lower() != "eczematous_family_routing_workflow":
        return payload

    metadata = dict(getattr(case_input, "metadata", {}) or {})
    final_label = str(payload.get("final_diagnosis", "")).strip()
    differentials = payload.get("differential_diagnoses", [])
    if not isinstance(differentials, list):
        differentials = []
    rationale = str(payload.get("rationale", "")).strip()
    follow_up = payload.get("follow_up_considerations", [])
    follow_up_text = " ".join(str(item).strip() for item in follow_up) if isinstance(follow_up, list) else str(follow_up).strip()
    selected_evidence = list((evidence_package or {}).get("selected_evidence", []) or [])
    evidence_text = " ".join(str(item.get("summary", "")).strip() for item in selected_evidence if isinstance(item, dict))
    serialized = str((evidence_package or {}).get("serialized_evidence_text", "")).strip()
    combined = " \n".join(
        [final_label]
        + [str(item).strip() for item in differentials if str(item).strip()]
        + [rationale, follow_up_text, evidence_text, serialized]
    )
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", combined.lower())
    normalized = " ".join(normalized.split())

    scores: dict[str, float] = {
        "CONTACT_DERMATITIS": 0.0,
        "ATOPIC_DERMATITIS": 0.0,
        "ECZEMA_DERMATITIS": 0.0,
        "PERIORAL_DERMATITIS": 0.0,
        "HERPETIC_ECZEMA": 0.0,
        "HAIR_DISORDER": 0.0,
        "OTHER_INFLAMMATORY": 0.0,
    }

    def boost(bucket: str, value: float) -> None:
        scores[bucket] = scores.get(bucket, 0.0) + float(value)

    def has_any(*phrases: str) -> bool:
        return any(phrase in normalized for phrase in phrases if phrase)

    mapped_initial = _xiangya_grouped_label_for_text(final_label)
    boost(mapped_initial, 4.0 if mapped_initial != "OTHER_INFLAMMATORY" else 1.0)

    if has_any("perioral dermatitis", "口周皮炎", "perioral region", "nasolabial", "around the mouth area", "mouth area"):
        boost("PERIORAL_DERMATITIS", 3.2)
    if has_any("not concentrated around the mouth", "rules out perioral dermatitis", "rule out perioral dermatitis", "not perioral dermatitis"):
        boost("PERIORAL_DERMATITIS", -5.0)
    if has_any("eczema herpeticum", "herpetic eczema", "疱疹性湿疹", "erosive", "糜烂", "painful", "疼痛"):
        boost("HERPETIC_ECZEMA", 5.5)
    if has_any(
        "atopic dermatitis",
        "特应性皮炎",
        "recurrent",
        "chronic",
        "反复",
        "慢性",
        "lichenification",
        "苔藓样变",
        "xerosis",
        "dryness",
        "aligns with the characteristics of atopic dermatitis",
        "supports the diagnosis of atopic dermatitis",
        "flexural",
        "儿童",
        "患儿",
        "child",
        "pediatric",
    ):
        boost("ATOPIC_DERMATITIS", 5.0)
    if has_any(
        "eczema",
        "湿疹",
        "eczematous",
        "scaly",
        "scaling",
        "脱屑",
        "crusted",
        "结痂",
        "more consistent with eczema",
        "supports the diagnosis of eczema",
        "widespread eczematous",
    ):
        boost("ECZEMA_DERMATITIS", 4.2)
    if has_any(
        "contact dermatitis",
        "接触性皮炎",
        "allergic contact dermatitis",
        "irritant",
        "linear streak",
        "linear distribution",
        "linearly",
        "线状",
        "exposure",
        "allergen",
        "irritant exposure",
        "clinical diagnosis is contact dermatitis",
        "typical findings in contact dermatitis",
        "localized around the eye",
        "possible contact allergen exposure",
    ):
        boost("CONTACT_DERMATITIS", 4.0)
    if has_any("alopecia", "hair loss", "脱发"):
        boost("HAIR_DISORDER", 8.0)

    if has_any("localized", "局部", "solitary", "single lesion", "localized, cheek", "around the eye", "eye area", "hairline"):
        boost("CONTACT_DERMATITIS", 1.8)
    if has_any("generalized", "diffuse", "widespread", "symmetric", "双侧", "全身", "四肢屈侧", "肘窝", "腘窝", "inner forearm"):
        boost("ATOPIC_DERMATITIS", 2.8)
        boost("ECZEMA_DERMATITIS", 1.6)
    if has_any(
        "no clear linear or contact distribution",
        "lack of specific contact related triggers",
        "lack of specific contact related patterns",
        "there is no clear evidence of a specific contact allergen or irritant",
        "there are no clear signs of a localized or exposure patterned distribution",
        "no clear signs of a localized or exposure patterned distribution",
        "not sharply demarcated",
        "less likely to be contact dermatitis",
        "rules out contact dermatitis",
    ):
        boost("CONTACT_DERMATITIS", -4.2)
        boost("ATOPIC_DERMATITIS", 1.8)
        boost("ECZEMA_DERMATITIS", 2.2)
    if has_any(
        "widespread, symmetric distribution",
        "widespread distribution",
        "diffuse, erythematous rash",
        "more characteristic of eczema",
        "generalized eczematous process",
    ):
        boost("ATOPIC_DERMATITIS", 2.0)
        boost("ECZEMA_DERMATITIS", 2.6)

    if "unlikely_candidates=contact_dermatitis" in normalized or "unlikely_candidates= contact_dermatitis" in normalized:
        boost("CONTACT_DERMATITIS", -3.0)
    if "unlikely_candidates=eczema_dermatitis" in normalized or "unlikely_candidates= eczema_dermatitis" in normalized:
        boost("ECZEMA_DERMATITIS", -2.5)
    if "unlikely_candidates=perioral_dermatitis" in normalized or "unlikely_candidates= perioral_dermatitis" in normalized:
        boost("PERIORAL_DERMATITIS", -4.0)

    if baseline_mode:
        # For baseline keep direct prompting semantics: only normalize the model's explicit final label.
        best_label = _xiangya_grouped_label_for_text(final_label or "")
        refined = dict(payload)
        refined["raw_final_diagnosis"] = final_label or refined.get("raw_final_diagnosis", "")
        refined["final_diagnosis"] = best_label
        refined["differential_diagnoses"] = [best_label] + [
            str(item).strip() for item in differentials if str(item).strip() and str(item).strip() != best_label
        ][:4]
        return refined

    if scores["CONTACT_DERMATITIS"] > 0 and scores["ATOPIC_DERMATITIS"] == 0 and scores["ECZEMA_DERMATITIS"] == 0:
        best_label = "CONTACT_DERMATITIS"
    else:
        best_label = max(scores, key=lambda key: scores[key])

    if best_label == "CONTACT_DERMATITIS":
        if scores["ATOPIC_DERMATITIS"] >= scores["CONTACT_DERMATITIS"] - 0.8:
            best_label = "ATOPIC_DERMATITIS"
        elif scores["ECZEMA_DERMATITIS"] >= scores["CONTACT_DERMATITIS"] - 0.5:
            best_label = "ECZEMA_DERMATITIS"

    refined = dict(payload)
    refined["raw_final_diagnosis"] = final_label or refined.get("raw_final_diagnosis", "")
    refined["final_diagnosis"] = best_label
    ordered = [best_label] + [str(item).strip() for item in differentials if str(item).strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        mapped = _xiangya_grouped_label_for_text(item)
        if mapped in seen:
            continue
        seen.add(mapped)
        deduped.append(mapped)
    refined["differential_diagnoses"] = deduped[:5] or [best_label]
    return refined


def _refine_scin_full_label_payload(case_input: CaseInput, payload: dict[str, Any]) -> dict[str, Any]:
    label_space_id = str(getattr(case_input, "label_space_id", "") or "").strip().lower()
    if not is_full_taxonomy_case(workflow_context=_case_workflow_context(case_input), label_space_id=label_space_id):
        return payload

    label_space = resolve_label_space(
        label_space_id=getattr(case_input, "label_space_id", None),
        dataset_name=getattr(case_input, "dataset_name", None),
        metadata=getattr(case_input, "metadata", None),
    )
    scin_labels = {str(label).strip() for label in label_space.canonical_labels if str(label).strip()}
    final_label = str(payload.get("final_diagnosis", "")).strip()
    if not final_label:
        return payload
    if final_label in scin_labels:
        return payload

    differentials = payload.get("differential_diagnoses", [])
    if not isinstance(differentials, list):
        differentials = []
    rationale = str(payload.get("rationale", "")).strip()
    follow_up = payload.get("follow_up_considerations", [])
    if isinstance(follow_up, list):
        follow_up_text = " ".join(str(item).strip() for item in follow_up if str(item).strip())
    else:
        follow_up_text = str(follow_up).strip()

    refinement_text = " \n".join(
        [final_label]
        + [str(item).strip() for item in differentials if str(item).strip()]
        + [rationale, follow_up_text]
    )
    normalized_text = re.sub(r"[^a-z0-9]+", " ", refinement_text.strip().lower())
    normalized_text = " ".join(normalized_text.split())

    prioritized_labels = (
        "Allergic Contact Dermatitis",
        "Irritant Contact Dermatitis",
        "Acute and chronic dermatitis",
        "Acute dermatitis, NOS",
        "Acute dermatitis",
        "Eczema",
        "Herpes Zoster",
        "Leukocytoclastic Vasculitis",
        "Hemangioma",
        "Acne",
    )
    detected: list[str] = []
    for label in prioritized_labels:
        normalized_label = re.sub(r"[^a-z0-9]+", " ", label.strip().lower())
        normalized_label = " ".join(normalized_label.split())
        if normalized_label and normalized_label in normalized_text and label in scin_labels and label not in detected:
            detected.append(label)

    if not detected:
        heuristic_label = _infer_scin_specific_label_from_metadata(case_input=case_input, payload=payload)
        if heuristic_label and heuristic_label in scin_labels:
            detected.append(heuristic_label)
        else:
            return payload

    refined = dict(payload)
    refined["raw_final_diagnosis"] = final_label
    refined["final_diagnosis"] = detected[0]
    ordered_differentials = [detected[0]] + [str(item).strip() for item in differentials if str(item).strip()]
    for label in detected[1:]:
        if label not in ordered_differentials:
            ordered_differentials.append(label)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ordered_differentials:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    refined["differential_diagnoses"] = deduped[:5]
    return refined


def _infer_scin_specific_label_from_metadata(*, case_input: CaseInput, payload: dict[str, Any]) -> str:
    metadata = dict(getattr(case_input, "metadata", {}) or {})
    final_label = str(payload.get("final_diagnosis", "")).strip()
    rationale = str(payload.get("rationale", "")).strip().lower()
    related_category = str(metadata.get("related_category", "")).strip().upper()
    duration = str(metadata.get("condition_duration", "")).strip().upper()
    body_sites = {str(item).strip().lower() for item in metadata.get("body_sites", []) if str(item).strip()}
    textures = {str(item).strip().lower() for item in metadata.get("textures_present", []) if str(item).strip()}
    symptoms = {str(item).strip().lower() for item in metadata.get("symptoms_present", []) if str(item).strip()}
    dermatitis_family = {
        "Contact Dermatitis",
        "Allergic Contact Dermatitis",
        "Irritant Contact Dermatitis",
        "Acute dermatitis",
        "Acute dermatitis, NOS",
        "Acute and chronic dermatitis",
        "Eczema",
    }
    if final_label not in dermatitis_family:
        return ""

    if "rough_or_flaky" in textures and duration in {"ONE_TO_THREE_MONTHS", "ONE_TO_FOUR_WEEKS", "MORE_THAN_THREE_MONTHS"}:
        return "Eczema"
    if "flat" in textures and "leg" in body_sites:
        return "Leukocytoclastic Vasculitis"
    if duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"} and ("cluster" in rationale or "clustered" in rationale):
        return "Herpes Zoster"
    if related_category == "ACNE":
        return "Acne"
    if "allergic contact dermatitis" in rationale:
        return "Allergic Contact Dermatitis"
    if "irritant contact dermatitis" in rationale:
        return "Irritant Contact Dermatitis"
    if "acute and chronic dermatitis" in rationale:
        return "Acute and chronic dermatitis"
    if "acute dermatitis" in rationale:
        return "Acute dermatitis, NOS"
    if "itching" in symptoms and "raised_or_bumpy" in textures and duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"}:
        return "Allergic Contact Dermatitis"
    return ""


def _refine_scin_grouped_agent_payload(
    *,
    case_input: CaseInput,
    payload: dict[str, Any],
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    label_space_id = str(getattr(case_input, "label_space_id", "") or "").strip().lower()
    if not is_family_routing_case(workflow_context=_case_workflow_context(case_input), label_space_id=label_space_id):
        return payload

    metadata = dict(getattr(case_input, "metadata", {}) or {})
    related_category = str(metadata.get("related_category", "")).strip().upper()
    duration = str(metadata.get("condition_duration", "")).strip().upper()
    body_sites = {str(item).strip().lower() for item in metadata.get("body_sites", []) if str(item).strip()}
    textures = {str(item).strip().lower() for item in metadata.get("textures_present", []) if str(item).strip()}
    symptoms = {str(item).strip().lower() for item in metadata.get("symptoms_present", []) if str(item).strip()}
    final_label = str(payload.get("final_diagnosis", "")).strip()

    text_parts = [final_label, str(payload.get("rationale", "")).strip()]
    differentials = payload.get("differential_diagnoses", [])
    if isinstance(differentials, list):
        text_parts.extend(str(item).strip() for item in differentials if str(item).strip())
    for item in evidence_package.get("selected_evidence", [])[:8]:
        if isinstance(item, dict):
            text_parts.append(str(item.get("summary", "")).strip())
    text_parts.append(str(evidence_package.get("serialized_evidence_text", "")).strip())
    combined = " \n".join(text_parts)
    normalized = re.sub(r"[^a-z0-9]+", " ", combined.lower())
    normalized = " ".join(normalized.split())

    scores: dict[str, float] = {
        "DERMATITIS_ECZEMA": 0.0,
        "INFECTION_VIRAL_FUNGAL": 0.0,
        "VASCULAR_PURPURIC": 0.0,
        "ACNE_ROSACEA_FOLLICULAR": 0.0,
        "PIGMENT_KERATOSIS_NEVUS": 0.0,
        "MALIGNANT_PREMALIGNANT": 0.0,
        "URTICARIA_BITE_FOLLICULITIS": 0.0,
    }

    def boost(bucket: str, value: float) -> None:
        scores[bucket] = scores.get(bucket, 0.0) + float(value)

    # Start from the raw diagnosis family.
    normalized_final = re.sub(r"[^a-z0-9]+", " ", final_label.lower()).strip()
    if "contact dermatitis" in normalized_final or "eczema" in normalized_final or "acute dermatitis" in normalized_final:
        boost("DERMATITIS_ECZEMA", 4.0)
    if "herpes zoster" in normalized_final or "tinea" in normalized_final or "impetigo" in normalized_final:
        boost("INFECTION_VIRAL_FUNGAL", 4.0)
    if "vasculitis" in normalized_final or "hemangioma" in normalized_final or "purpura" in normalized_final:
        boost("VASCULAR_PURPURIC", 4.0)
    if "acne" in normalized_final or "folliculitis" in normalized_final or "rosacea" in normalized_final:
        boost("ACNE_ROSACEA_FOLLICULAR", 4.0)
    if "nevus" in normalized_final or "keratosis" in normalized_final or "dermatofibroma" in normalized_final:
        boost("PIGMENT_KERATOSIS_NEVUS", 4.0)

    # Evidence-text cues.
    if any(keyword in normalized for keyword in ("herpes zoster", "herpes simplex", "tinea", "impetigo", "candida", "molluscum", "cellulitis")):
        boost("INFECTION_VIRAL_FUNGAL", 5.0)
    if any(keyword in normalized for keyword in ("vasculitis", "purpura", "purpuric", "hemangioma", "petech", "ecchym", "erythema ab igne")):
        boost("VASCULAR_PURPURIC", 5.0)
    if any(keyword in normalized for keyword in ("acne", "folliculitis", "rosacea", "perioral dermatitis", "comedone")):
        boost("ACNE_ROSACEA_FOLLICULAR", 5.0)
    if any(keyword in normalized for keyword in ("allergic contact dermatitis", "irritant contact dermatitis", "acute dermatitis", "eczema", "psoriasis")):
        boost("DERMATITIS_ECZEMA", 4.5)

    # Metadata-aware routing for the hard non-dermatitis SCIN cases.
    if related_category == "RASH":
        boost("DERMATITIS_ECZEMA", 1.0)
        if "flat" in textures and "leg" in body_sites:
            boost("VASCULAR_PURPURIC", 6.0)
        if "flat" in textures and "arm" in body_sites and duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"} and "itching" not in symptoms:
            boost("VASCULAR_PURPURIC", 4.0)
        if "flat" in textures and ("leg" in body_sites or "arm" in body_sites):
            boost("VASCULAR_PURPURIC", 2.5)
        if "patchy" in normalized and "flat" in textures and ("leg" in body_sites or "arm" in body_sites):
            boost("VASCULAR_PURPURIC", 1.5)
        if "purpuric" in normalized or "petech" in normalized or "erythema ab igne" in normalized:
            boost("VASCULAR_PURPURIC", 3.0)
        if "itching" not in symptoms and "bothersome appearance" not in normalized:
            boost("VASCULAR_PURPURIC", 0.8)
        if duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"} and "clustered" in normalized and "back" in normalized:
            boost("INFECTION_VIRAL_FUNGAL", 5.0)
        if duration in {"ONE_DAY", "LESS_THAN_ONE_WEEK"} and ("burning" in symptoms or "pain" in symptoms):
            boost("INFECTION_VIRAL_FUNGAL", 3.0)
    if related_category == "ACNE":
        boost("ACNE_ROSACEA_FOLLICULAR", 4.0)
    if related_category == "GROWTH_OR_MOLE":
        boost("PIGMENT_KERATOSIS_NEVUS", 3.0)

    best_label = max(scores, key=lambda key: scores[key])
    if scores.get(best_label, 0.0) <= 0:
        return payload

    refined = dict(payload)
    refined["raw_final_diagnosis"] = final_label or refined.get("raw_final_diagnosis", "")
    refined["final_diagnosis"] = best_label
    existing = payload.get("differential_diagnoses", [])
    if isinstance(existing, list):
        refined["differential_diagnoses"] = [best_label] + [str(item).strip() for item in existing[:4] if str(item).strip()]
    else:
        refined["differential_diagnoses"] = [best_label]
    return refined


def _refine_scin_payload_for_runtime(
    *,
    case_input: CaseInput,
    payload: dict[str, Any],
    evidence_package: dict[str, Any] | None = None,
    baseline_mode: bool = False,
    allow_grouped_baseline_refinement: bool = False,
) -> dict[str, Any]:
    label_space_id = str(getattr(case_input, "label_space_id", "") or "").strip().lower()
    workflow_context = _case_workflow_context(case_input)
    if label_space_id == "xiangya_sft_grouped":
        return _refine_xiangya_grouped_payload(
            case_input=case_input,
            payload=dict(payload),
            evidence_package=evidence_package,
            baseline_mode=baseline_mode,
        )
    if not is_family_routing_case(workflow_context=workflow_context, label_space_id=label_space_id) and label_space_id != "scin_full":
        return payload

    refined = dict(payload)
    if label_space_id == "scin_full":
        return _refine_scin_full_label_payload(case_input, refined)
    if label_space_id == "scin_grouped":
        if baseline_mode:
            if allow_grouped_baseline_refinement:
                return _refine_scin_grouped_agent_payload(
                    case_input=case_input,
                    payload=refined,
                    evidence_package=evidence_package or {},
                )
            return refined
        return _refine_scin_grouped_agent_payload(
            case_input=case_input,
            payload=refined,
            evidence_package=evidence_package or {},
        )
    return refined
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_IMAGES_PER_PROMPT = 1
PROMPT_STACK_VERSION = "dermagent_prompt_stack_v1"
INITIAL_PERCEPTION_PROMPT_VERSION = "initial_perception_v1"
SKILL_PROMPT_VERSION = "skill_reasoning_v1"
FINAL_DIAGNOSIS_PROMPT_VERSION = "final_diagnosis_v1"
BASELINE_DIAGNOSIS_PROMPT_VERSION = "direct_baseline_v1"
INITIAL_PERCEPTION_MAX_TOKENS = 320
SKILL_MAX_TOKENS = 640
FINAL_DIAGNOSIS_MAX_TOKENS = 680
BASELINE_DIAGNOSIS_MAX_TOKENS = 420
SKINVL_MODEL_NAME_HINT = "skinvl"
SKINVL_ALLOWED_LABELS = (
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Actinic Keratosis",
    "Seborrheic Keratosis",
    "Malignant Melanoma",
    "Nevus",
    "Contact Dermatitis",
    "Psoriasis",
    "Seborrheic Dermatitis",
    "Atopic Dermatitis",
)

FULL_CLINICAL_PROFILE_ID = "full_clinical"
COMPACT_PROFILE_PRESETS = (
    {"profile_id": "standard", "retrieval_top_k": 4, "max_skill_count": 16, "max_skill_fields": 7, "serialized_max_length": 3600},
    {"profile_id": "tight", "retrieval_top_k": 3, "max_skill_count": 12, "max_skill_fields": 5, "serialized_max_length": 2200},
    {"profile_id": "minimal", "retrieval_top_k": 2, "max_skill_count": 8, "max_skill_fields": 4, "serialized_max_length": 1200},
    {"profile_id": "emergency", "retrieval_top_k": 1, "max_skill_count": 5, "max_skill_fields": 3, "serialized_max_length": 600},
)

GENERIC_SKILL_FIELD_PRIORITY = (
    "primary_lesion_morphology",
    "lesion_type",
    "primary_color",
    "border_clarity",
    "border_irregularity",
    "surface_texture",
    "body_location",
    "risk_level",
    "candidate_pairs",
    "unlikely_candidates",
    "contradictions",
    "missing_information",
    "whether_escalation_needed",
    "critical_supporting_evidence",
    "supporting_evidence",
    "opposing_evidence",
    "conflicting_evidence",
    "alarm_signals",
    "risk_evidence",
    "exclusion_evidence",
    "required_missing_evidence",
    "differentiation_features",
    "reasoning_gaps",
    "missing_links",
    "caution_flags",
    "reasons",
    "why_it_matters",
    "impact_on_differential",
    "counterexample_watchouts",
    "further_observation_suggestions",
    "consistency_score",
    "conflicts",
    "suspicious_points",
    "uncertainty_level",
    "uncertainty_if_missing",
    "exclusion_confidence",
    "onset_type",
    "progression_speed",
    "stability",
    "recurrence",
    "size_range",
    "elevation",
    "count",
    "color_variation",
    "pigmentation_pattern",
    "asymmetry_color",
    "symmetry",
    "localized_vs_generalized",
    "clustering_pattern",
    "color",
    "border",
    "surface",
    "size_count",
    "distribution",
    "associated_context",
    "referenced_confusion_patterns",
    "evidence_strength",
    "recommendation_type",
)

SKILL_FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "lesion_description_structuring_skill": (
        "primary_lesion_morphology",
        "color",
        "border",
        "surface",
        "size_count",
        "distribution",
        "associated_context",
    ),
    "morphology_analysis_skill": ("lesion_type", "elevation", "count", "size_range"),
    "color_pattern_analysis_skill": ("primary_color", "color_variation", "pigmentation_pattern", "asymmetry_color"),
    "border_surface_analysis_skill": ("border_clarity", "border_irregularity", "surface_texture", "scaling_presence"),
    "distribution_analysis_skill": ("body_location", "localized_vs_generalized", "clustering_pattern", "symmetry"),
    "metadata_consistency_skill": ("consistency_score", "conflicts", "suspicious_points"),
    "temporal_evolution_skill": ("onset_type", "progression_speed", "stability", "recurrence"),
    "malignancy_risk_assessment_skill": ("risk_level", "alarm_signals", "risk_evidence"),
    "differential_compare_skill": ("candidate_pairs", "supporting_evidence", "conflicting_evidence"),
    "exclusion_reasoning_skill": ("unlikely_candidates", "exclusion_evidence", "required_missing_evidence", "exclusion_confidence"),
    "information_gap_detection_skill": (
        "missing_information",
        "why_it_matters",
        "impact_on_differential",
        "uncertainty_if_missing",
    ),
    "uncertainty_assessment_skill": ("uncertainty_level", "reasons", "missing_information"),
    "contradiction_check_skill": ("contradictions", "missing_links", "reasoning_gaps"),
    "escalation_recommendation_skill": (
        "whether_escalation_needed",
        "escalation_reason",
        "suggested_next_check_type",
        "caution_flags",
    ),
    "mel_nev_specialist_skill": (
        "differentiation_features",
        "critical_supporting_evidence",
        "supporting_evidence",
        "opposing_evidence",
        "counterexample_watchouts",
        "referenced_confusion_patterns",
        "further_observation_suggestions",
    ),
    "ack_scc_specialist_skill": (
        "differentiation_features",
        "critical_supporting_evidence",
        "supporting_evidence",
        "opposing_evidence",
        "counterexample_watchouts",
        "referenced_confusion_patterns",
        "further_observation_suggestions",
    ),
}


class DermOpenAIClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.getenv("OPENAI_MODEL", "Qwen2.5-VL-7B-Instruct")
        self.timeout = timeout if timeout is not None else _read_float_env("OPENAI_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        configured_retries = (
            max_retries if max_retries is not None else _read_int_env("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )
        self.max_retries = max(0, configured_retries)
        self.max_images_per_prompt = max(
            1,
            _read_int_env("OPENAI_MAX_IMAGES_PER_PROMPT", DEFAULT_MAX_IMAGES_PER_PROMPT),
        )
        if OpenAI is None:
            raise ModuleNotFoundError(
                "openai package is not installed. Install `openai` to use DermOpenAIClient runtime calls."
            )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def _is_skinvl_model(self) -> bool:
        return SKINVL_MODEL_NAME_HINT in self.model.strip().lower()

    @staticmethod
    def _skinvl_allowed_labels_text() -> str:
        return ", ".join(SKINVL_ALLOWED_LABELS)

    def prompt_manifest(self) -> dict[str, Any]:
        return {
            "prompt_stack_version": PROMPT_STACK_VERSION,
            "initial_perception_prompt_version": INITIAL_PERCEPTION_PROMPT_VERSION,
            "skill_prompt_version": SKILL_PROMPT_VERSION,
            "final_diagnosis_prompt_version": FINAL_DIAGNOSIS_PROMPT_VERSION,
            "baseline_diagnosis_prompt_version": BASELINE_DIAGNOSIS_PROMPT_VERSION,
        }

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "configured_model_name": self.model,
            "served_model_name": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_images_per_prompt": self.max_images_per_prompt,
            "prompt_manifest": self.prompt_manifest(),
        }

    def initial_perception(self, case_input: CaseInput) -> dict[str, Any]:
        label_space_hint = _build_label_space_prompt_hint(case_input)
        label_space_line = f"{label_space_hint}\n" if label_space_hint else ""
        sparse_hint = _build_sparse_lesion_prompt_hint(case_input)
        sparse_line = f"{sparse_hint}\n" if sparse_hint else ""
        benign_mimic_guard_hint = _build_sparse_benign_mimic_guard_hint(case_input)
        benign_mimic_guard_line = f"{benign_mimic_guard_hint}\n" if benign_mimic_guard_hint else ""
        user_text = (
            "You are the initial perception stage in DermAgent.\n"
            "Return structured observation only. Do not produce a final diagnosis label.\n"
            "You must mimic the early dermatologist reasoning stage: observation, coarse description, "
            "tentative differential candidates, and uncertainty disclosure.\n"
            "Return JSON only with exactly these keys:\n"
            "- image_summary: short clinical visual summary\n"
            "- ddx_candidates: list of a few candidate diagnoses considered at the perception stage\n"
            "- uncertainty: { level: low/medium/high, reasons: [..] }\n"
            "- notes: list of short observation notes\n"
            f"{label_space_line}"
            f"{sparse_line}"
            f"{benign_mimic_guard_line}"
            f"Metadata: {case_input.clinical_metadata()}"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You extract dermatologist-style observations for downstream reasoning. "
                    "Never provide a final diagnosis."
                ),
            },
            {"role": "user", "content": self._build_case_multimodal_content(case_input, user_text)},
        ]
        payload = self._create_json_payload(
            messages=messages,
            max_tokens=INITIAL_PERCEPTION_MAX_TOKENS,
            request_name=f"initial_perception:{case_input.case_id}",
        )
        return self._normalize_initial_perception(payload)

    def run_skill_prompt(
        self,
        case_input: CaseInput,
        skill_name: str,
        prompt: str,
        output_schema: str,
    ) -> dict[str, Any]:
        system_text = (
            "You are a clinical reasoning skill inside DermAgent.\n"
            "You perform one atomic reasoning action only.\n"
            "You must not output a final diagnosis, disease classification, or treatment decision.\n"
            "You must return structured evidence only in JSON.\n"
            f"Current skill: {skill_name}\n"
            f"Required output fields:\n{output_schema}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": self._build_case_multimodal_content(case_input, prompt)},
        ]
        return self._create_json_payload(
            messages=messages,
            max_tokens=SKILL_MAX_TOKENS,
            request_name=f"skill:{skill_name}:{case_input.case_id}",
        )

    def final_diagnosis(self, case_input: CaseInput, evidence_package: EvidencePackage) -> dict[str, Any]:
        evidence_payload = self._canonicalize_evidence_package(evidence_package.to_dict())
        last_error: Exception | None = None
        request_name = f"final_diagnosis:{case_input.case_id}"
        if self._is_skinvl_model():
            profile_sequence: list[dict[str, Any]] = [{"profile_id": FULL_CLINICAL_PROFILE_ID}] + list(COMPACT_PROFILE_PRESETS)
            for profile in profile_sequence:
                prepared_evidence = self._prepare_evidence_for_profile(evidence_payload, profile)
                serialized_payload = json.dumps(prepared_evidence, ensure_ascii=False, separators=(",", ":"))
                prompt = (
                    "You are the only final diagnostic decision maker for the SkinVL evaluation line in DermAgent.\n"
                    "Return valid JSON only.\n"
                    "Do not output markdown.\n"
                    "Do not describe the whole image as the diagnosis.\n"
                    "Use the evidence package as structured support, not as an overriding instruction.\n"
                    "The field `final_diagnosis` must be exactly one label from the allowed label set.\n"
                    f"Allowed labels: {self._skinvl_allowed_labels_text()}.\n"
                    "The field `differential_diagnoses` must be a short list containing only labels from the same set.\n"
                    "The evidence package contains two layers:\n"
                    "1. `risk_layer`: malignant-risk warnings, caution flags, follow-up suggestions, and the supporting shortlist.\n"
                    "2. `diagnosis_override_layer`: whether the agent evidence is strong enough to justify changing the diagnosis direction.\n"
                    "Interpret the override layer conservatively:\n"
                    "- if `subtype_override_allowed` is true, you may output a specific diagnostic subtype.\n"
                    "- if only `malignancy_override_allowed` is true, prefer a suspicious-style subtype label rather than a fully confident subtype override.\n"
                    "- if neither override is allowed, stay close to a baseline-style diagnosis from the image and metadata, but preserve risk, caution, and follow-up context.\n"
                    "Within the override layer, separately weigh `Supporting Evidence` against `Opposing / Exclusion Evidence`.\n"
                    "Prioritize the `selected_evidence` block as the curated shortlist chosen by the evidence calibrator.\n"
                    "Use `serialized_evidence_text` as supporting narrative context when it agrees with the selected evidence.\n"
                    "Do not copy evidence lines verbatim into diagnosis fields.\n"
                    "Integrate image, metadata, and evidence, then return a structured final diagnosis result.\n"
                    "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
                    "When override is not allowed, keep the diagnosis conservative but explicitly explain the residual uncertainty and what evidence was insufficient to justify a different subtype.\n"
                    f"Metadata: {case_input.clinical_metadata()}\n"
                    f"Evidence package: {serialized_payload}"
                )
                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": (
                            "You are the final diagnosis stage for SkinVL. Preserve independent judgment while using structured supporting evidence."
                        ),
                    },
                    {"role": "user", "content": self._build_case_multimodal_content(case_input, prompt)},
                ]
                try:
                    payload = self._create_json_payload(
                        messages=messages,
                        max_tokens=FINAL_DIAGNOSIS_MAX_TOKENS,
                        request_name=request_name,
                    )
                    return _refine_scin_payload_for_runtime(
                        case_input=case_input,
                        payload=payload,
                        evidence_package=evidence_package.to_dict(),
                        baseline_mode=False,
                    )
                except BadRequestError as exc:
                    last_error = exc
                    if not self._is_context_length_error(exc):
                        raise
                    LOGGER.warning(
                        "Retrying %s with more aggressive evidence handling after context overflow on profile `%s`.",
                        request_name,
                        profile["profile_id"],
                    )
                    continue

            if last_error is not None:
                raise last_error
            raise RuntimeError(f"Failed final diagnosis for case: {case_input.case_id}")

        workflow_context = _case_workflow_context(case_input)
        workflow_cell_id = str(workflow_context.get("workflow_cell_id", "")).strip().lower()
        compact_final_output = workflow_cell_id == "medgemma__pad20__clinical_core_v2"
        final_max_tokens = 900 if compact_final_output else FINAL_DIAGNOSIS_MAX_TOKENS
        compact_final_line = (
            "Return compact JSON: final_diagnosis must be one short label; rationale must be at most two short sentences; "
            "follow_up_considerations must contain at most two short strings; do not repeat evidence text.\n"
            if compact_final_output
            else ""
        )
        profile_sequence: list[dict[str, Any]] = [{"profile_id": FULL_CLINICAL_PROFILE_ID}] + list(COMPACT_PROFILE_PRESETS)
        calibration_note = _build_label_space_calibration_note(case_input)
        calibration_line = f"{calibration_note}\n" if calibration_note else ""
        label_space_hint = _build_label_space_prompt_hint(case_input)
        label_space_line = f"{label_space_hint}\n" if label_space_hint else ""
        scin_routing_hint = _build_scin_routing_hint(case_input)
        scin_routing_line = f"{scin_routing_hint}\n" if scin_routing_hint else ""
        benign_mimic_guard_hint = _build_sparse_benign_mimic_guard_hint(case_input)
        benign_mimic_guard_line = f"{benign_mimic_guard_hint}\n" if benign_mimic_guard_hint else ""

        # Check if this is a MEL vs NV confusion case
        confusion_summary = evidence_package.confusion_cluster_summary or {}
        active_clusters = confusion_summary.get("active_clusters", [])
        is_mel_nev_confusion = any("mel_nev" in str(cluster).lower() for cluster in active_clusters)
        mel_nev_note = ""
        if is_mel_nev_confusion:
            mel_nev_note = (
                "\n"
                "ATTENTION - Melanoma vs Nevus Confusion Detected:\n"
                "This case involves MEL vs NV differential confusion.\n"
                "The mel_nev_specialist_skill has analyzed both supporting and opposing evidence.\n"
                "Carefully weigh BOTH sides:\n"
                "- Supporting evidence: features favoring melanoma (asymmetry, irregular border, color variation)\n"
                "- Opposing evidence: features favoring nevus (symmetry, regular border, uniform pigmentation)\n"
                "Do NOT diagnose Malignant Melanoma based solely on color variation without structural irregularity.\n"
                "Do NOT diagnose Nevus if strong structural chaos, ulceration, or rapid growth is present.\n"
                "Balance both sides and make the diagnosis based on which evidence is stronger.\n"
                "\n"
            )

        for profile in profile_sequence:
            prepared_evidence = self._prepare_evidence_for_profile(evidence_payload, profile)
            serialized_payload = json.dumps(prepared_evidence, ensure_ascii=False, separators=(",", ":"))
            prompt = (
                "You are the only final diagnostic decision maker in DermAgent.\n"
                "Use the evidence package as structured support, not as an overriding instruction.\n"
                f"{mel_nev_note}"
                "Treat `risk_layer.baseline_preview` as the image-and-metadata-only anchor diagnosis.\n"
                "The evidence package contains two layers:\n"
                "1. `risk_layer`: malignant-risk warnings, caution flags, follow-up suggestions, and the supporting shortlist.\n"
                "2. `diagnosis_override_layer`: whether the agent evidence is strong enough to justify changing the diagnosis direction.\n"
                "Interpret the override layer conservatively:\n"
                "- if `subtype_override_allowed` is true, you may output a specific diagnostic subtype.\n"
                "- if only `malignancy_override_allowed` is true, prefer a suspicious-style subtype label such as `Suspicious for Basal Cell Carcinoma` instead of a fully confident subtype override.\n"
                "- if neither override is allowed, stay close to a baseline-style diagnosis from the image and metadata, but preserve the risk warnings, caution, and follow-up context.\n"
                "- if `selected_evidence` is empty, or there is no subtype-specific supporting evidence, do not drift away from `baseline_preview`.\n"
                "- generic uncertainty, follow-up suggestions, or malignancy caution flags are not by themselves subtype evidence.\n"
                "- when the evidence mostly says 'need more information', keep the baseline diagnosis and express uncertainty in the rationale/follow_up_considerations instead of changing the label.\n"
                "Within the override layer, separately weigh `Supporting Evidence` against `Opposing / Exclusion Evidence`.\n"
                "Do not let generic risk or descriptive evidence count as subtype-specific support unless the override layer says subtype support is sufficient.\n"
                "Prioritize the `selected_evidence` block as the curated shortlist chosen by the evidence calibrator.\n"
                "Use `serialized_evidence_text` as supporting narrative context when it agrees with the selected evidence.\n"
                "When opposing evidence is present in the evidence package, weigh it carefully against supporting evidence.\n"
                "Integrate image, metadata, and evidence, then return a structured final diagnosis result.\n"
                "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
                f"{compact_final_line}"
                "When override is not allowed, keep the diagnosis conservative but include risk, caution, follow-up, and why the evidence was not strong enough to override.\n"
                f"{calibration_line}"
                f"{label_space_line}"
                f"{scin_routing_line}"
                f"{benign_mimic_guard_line}"
                f"Evidence package: {serialized_payload}"
            )
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are the final diagnosis stage. Preserve independent judgment while using supporting evidence."
                    ),
                },
                {"role": "user", "content": self._build_case_multimodal_content(case_input, prompt)},
            ]
            try:
                payload = self._create_json_payload(
                    messages=messages,
                    max_tokens=final_max_tokens,
                    request_name=request_name,
                )
                return _refine_scin_payload_for_runtime(
                    case_input=case_input,
                    payload=payload,
                    evidence_package=evidence_package.to_dict(),
                    baseline_mode=False,
                )
            except BadRequestError as exc:
                last_error = exc
                if not self._is_context_length_error(exc):
                    raise
                LOGGER.warning(
                    "Retrying %s with more aggressive evidence handling after context overflow on profile `%s`.",
                    request_name,
                    profile["profile_id"],
                )
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed final diagnosis for case: {case_input.case_id}")

    def baseline_diagnosis(self, case_input: CaseInput) -> dict[str, Any]:
        if self._is_skinvl_model():
            prompt = (
                "You are a dermatology diagnosis model.\n"
                "Use only the image and metadata.\n"
                "Return valid JSON only.\n"
                "Do not output markdown.\n"
                "Do not describe the whole image as the diagnosis.\n"
                "The field `final_diagnosis` must be exactly one label from the allowed label set.\n"
                f"Allowed labels: {self._skinvl_allowed_labels_text()}.\n"
                "The field `differential_diagnoses` must be a short list containing only labels from the same set.\n"
                "Schema:\n"
                "{"
                "\"final_diagnosis\":\"short disease label\","
                "\"differential_diagnoses\":[\"short disease label\"],"
                "\"rationale\":\"short explanation\","
                "\"confidence\":\"low|medium|high\","
                "\"follow_up_considerations\":[\"short follow-up item\"]"
                "}\n"
                f"Metadata: {case_input.clinical_metadata()}"
            )
        else:
            workflow_profile = str(_case_workflow_context(case_input).get("workflow_profile", "")).strip().lower()
            if workflow_profile == "eczematous_family_routing_workflow":
                label_space_hint = (
                    "Dataset label-space note: return one grouped label from `CONTACT_DERMATITIS`, `ATOPIC_DERMATITIS`, "
                    "`ECZEMA_DERMATITIS`, `PERIORAL_DERMATITIS`, `HERPETIC_ECZEMA`, `HAIR_DISORDER`, or `OTHER_INFLAMMATORY`."
                )
            else:
                label_space_hint = _build_label_space_prompt_hint(case_input)
            label_space_line = f"{label_space_hint}\n" if label_space_hint else ""
            sparse_hint = _build_sparse_lesion_prompt_hint(case_input)
            sparse_line = f"{sparse_hint}\n" if sparse_hint else ""
            benign_mimic_guard_hint = _build_sparse_benign_mimic_guard_hint(case_input)
            benign_mimic_guard_line = f"{benign_mimic_guard_hint}\n" if benign_mimic_guard_hint else ""
            scin_routing_hint = _build_scin_routing_hint(case_input)
            scin_routing_line = f"{scin_routing_hint}\n" if scin_routing_hint else ""
            prompt = (
                "You are the direct Qwen baseline diagnostic path for DermAgent evaluation.\n"
                "There is no agent evidence package in this path.\n"
                "Use only the image and metadata to produce a structured diagnosis result.\n"
                "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
                f"{label_space_line}"
                f"{sparse_line}"
                f"{benign_mimic_guard_line}"
                f"{scin_routing_line}"
                f"Metadata: {case_input.clinical_metadata()}"
            )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are the baseline final diagnosis stage. Diagnose directly from the case input only."
                ),
            },
            {"role": "user", "content": self._build_case_multimodal_content(case_input, prompt)},
        ]
        payload = self._create_json_payload(
            messages=messages,
            max_tokens=BASELINE_DIAGNOSIS_MAX_TOKENS,
            request_name=f"baseline_diagnosis:{case_input.case_id}",
        )
        return _refine_scin_payload_for_runtime(
            case_input=case_input,
            payload=payload,
            evidence_package=None,
            baseline_mode=True,
        )

    def _create_json_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        request_name: str,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
            except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                sleep_seconds = min(2**attempt, 8)
                LOGGER.warning(
                    "Retrying %s after %s on attempt %s/%s; sleeping %.1fs.",
                    request_name,
                    exc.__class__.__name__,
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to complete request: {request_name}")

    def _create_json_payload(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        request_name: str,
    ) -> dict[str, Any]:
        token_budget = max_tokens
        max_parse_attempts = 3 if request_name.startswith("skill:") else 2
        for parse_attempt in range(max_parse_attempts):
            response = self._create_json_completion(
                messages=messages,
                max_tokens=token_budget,
                request_name=request_name,
            )
            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            try:
                payload = self._parse_json_response(content)
            except json.JSONDecodeError:
                if parse_attempt >= max_parse_attempts - 1:
                    raise
                token_budget += max(160, max_tokens // 2, token_budget // 3)
                LOGGER.warning(
                    "Retrying %s after malformed JSON response; increasing max_tokens to %s.",
                    request_name,
                    token_budget,
                )
                continue

            payload = self._normalize_diagnosis_payload(payload, request_name=request_name)

            if finish_reason == "length" and parse_attempt < max_parse_attempts - 1:
                token_budget += max(160, max_tokens // 2, token_budget // 3)
                LOGGER.warning(
                    "Retrying %s because the model hit the max token budget; increasing max_tokens to %s.",
                    request_name,
                    token_budget,
                )
                continue
            return payload

        raise RuntimeError(f"Failed to parse JSON payload for request: {request_name}")

    @staticmethod
    def _build_multimodal_content(image_path: str, prompt_text: str) -> list[dict[str, Any]]:
        path = Path(image_path)
        if path.exists():
            detected_mime_type, encoded_image = DermOpenAIClient._encode_image_for_prompt(path)
            image_url = f"data:{detected_mime_type};base64,{encoded_image}"
            return [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        return [{"type": "text", "text": prompt_text}]

    def _build_case_multimodal_content(self, case_input: CaseInput, prompt_text: str) -> list[dict[str, Any]]:
        image_paths: list[str] = []
        metadata = dict(getattr(case_input, "metadata", {}) or {})
        raw_paths = metadata.get("image_paths", [])
        if isinstance(raw_paths, list):
            image_paths = [str(item).strip() for item in raw_paths if str(item).strip()]
        if not image_paths:
            image_paths = [str(case_input.image_path)]

        shot_types = metadata.get("shot_types", [])
        shot_lines: list[str] = []
        if isinstance(shot_types, list):
            for index, shot_type in enumerate(shot_types[: len(image_paths)]):
                shot_text = str(shot_type).strip()
                if shot_text:
                    shot_lines.append(f"image_{index + 1}_shot_type={shot_text}")

        max_images = max(1, int(self.max_images_per_prompt or 1))
        image_note = ""
        if len(image_paths) > max_images:
            remaining = len(image_paths) - max_images
            image_note = (
                f"\nAdditional image context: this case has {len(image_paths)} total images, "
                f"but the current runtime accepts at most {max_images} image(s) per prompt, "
                f"so {remaining} additional image(s) are summarized only through metadata."
            )
        if shot_lines:
            image_note += "\nImage view metadata: " + "; ".join(shot_lines[:3])

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text + image_note}]
        seen: set[str] = set()
        for image_path in image_paths[:max_images]:
            path = Path(image_path)
            if not path.exists():
                continue
            normalized = str(path.resolve())
            if normalized in seen:
                continue
            seen.add(normalized)
            detected_mime_type, encoded_image = self._encode_image_for_prompt(path)
            image_url = f"data:{detected_mime_type};base64,{encoded_image}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        return content

    @staticmethod
    def _encode_image_for_prompt(path: Path) -> tuple[str, str]:
        mime_type, _ = mimetypes.guess_type(path.name)
        detected_mime_type = mime_type or "image/png"
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                if max(width, height) <= MAX_INLINE_IMAGE_EDGE:
                    return detected_mime_type, base64.b64encode(path.read_bytes()).decode("utf-8")

                converted = image.convert("RGB")
                resized = converted.copy()
                resized.thumbnail((MAX_INLINE_IMAGE_EDGE, MAX_INLINE_IMAGE_EDGE), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                resized.save(
                    buffer,
                    format="JPEG",
                    quality=INLINE_IMAGE_JPEG_QUALITY,
                    optimize=True,
                )
                return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            LOGGER.warning("Falling back to raw image bytes for %s after preprocessing failure: %s", path, exc)
            return detected_mime_type, base64.b64encode(path.read_bytes()).decode("utf-8")

    @staticmethod
    def _parse_json_response(content: str | None) -> dict[str, Any]:
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            candidate = content.strip()
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if len(lines) >= 3:
                    candidate = "\n".join(lines[1:-1]).strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    pass
            repaired = DermOpenAIClient._repair_truncated_json(candidate[start:] if start >= 0 else candidate)
            if repaired:
                return json.loads(repaired)
            raise

    @staticmethod
    def _normalize_diagnosis_payload(payload: dict[str, Any], *, request_name: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if not (request_name.startswith("baseline_diagnosis:") or request_name.startswith("final_diagnosis:")):
            return payload
        if payload.get("final_diagnosis"):
            final_diagnosis = DermOpenAIClient._normalize_diagnosis_label(str(payload.get("final_diagnosis", "")).strip())
            payload["final_diagnosis"] = final_diagnosis
            differentials = payload.get("differential_diagnoses", [])
            if isinstance(differentials, list):
                normalized_differentials = [
                    DermOpenAIClient._normalize_diagnosis_label(str(item).strip())
                    for item in differentials
                    if str(item).strip()
                ]
                normalized_differentials = [item for item in normalized_differentials if item]
                payload["differential_diagnoses"] = normalized_differentials[:5] or ([final_diagnosis] if final_diagnosis else [])
            if request_name.startswith("baseline_diagnosis:") or request_name.startswith("final_diagnosis:"):
                payload = DermOpenAIClient._apply_skinvl_selector(payload)
            return payload
        raw_text = str(payload.get("raw_text", "")).strip()
        if not raw_text:
            return payload
        normalized = DermOpenAIClient._diagnosis_payload_from_raw_text(raw_text)
        if request_name.startswith("baseline_diagnosis:") or request_name.startswith("final_diagnosis:"):
            normalized = DermOpenAIClient._apply_skinvl_selector(normalized)
        return normalized

    @staticmethod
    def _diagnosis_payload_from_raw_text(raw_text: str) -> dict[str, Any]:
        text = " ".join(raw_text.strip().split())
        diagnosis = text
        match = re.search(
            r"(?:final diagnosis is|diagnosis is|most likely|likely diagnosis is|impression is)\s+(.*)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            diagnosis = match.group(1).strip()
        diagnosis = diagnosis.rstrip(". ")
        diagnosis = re.sub(r"^[Tt]he\s+", "", diagnosis).strip()
        diagnosis = DermOpenAIClient._normalize_diagnosis_label(diagnosis)
        differential: list[str] = []
        for pattern in (
            r"differential diagnoses? (?:include|are)\s*:\s*(.*)",
            r"differential diagnoses?\s*:\s*(.*)",
        ):
            diff_match = re.search(pattern, text, flags=re.IGNORECASE)
            if diff_match:
                tail = diff_match.group(1).strip().rstrip(".")
                candidates = [item.strip(" .") for item in re.split(r",|;|/| or ", tail) if item.strip(" .")]
                differential = candidates[:5]
                break
        if not differential and diagnosis:
            differential = [diagnosis]
        return {
            "final_diagnosis": diagnosis,
            "differential_diagnoses": differential,
            "rationale": raw_text,
            "confidence": "unknown",
            "follow_up_considerations": [],
        }

    @staticmethod
    def _normalize_diagnosis_label(text: str) -> str:
        lowered = text.strip().lower()
        label_map = [
            ("allergic contact dermatitis", "Allergic Contact Dermatitis"),
            ("irritant contact dermatitis", "Irritant Contact Dermatitis"),
            ("acute dermatitis, nos", "Acute dermatitis, NOS"),
            ("acute and chronic dermatitis", "Acute and chronic dermatitis"),
            ("herpes zoster", "Herpes Zoster"),
            ("leukocytoclastic vasculitis", "Leukocytoclastic Vasculitis"),
            ("erythema ab igne", "Erythema ab igne"),
            ("hemangioma", "Hemangioma"),
            ("acne", "Acne"),
            ("basal cell carcinoma", "Basal Cell Carcinoma"),
            ("bcc", "Basal Cell Carcinoma"),
            ("squamous cell carcinoma", "Squamous Cell Carcinoma"),
            ("scc", "Squamous Cell Carcinoma"),
            ("actinic keratosis", "Actinic Keratosis"),
            ("seborrheic keratosis", "Seborrheic Keratosis"),
            ("malignant melanoma", "Malignant Melanoma"),
            ("melanoma", "Malignant Melanoma"),
            ("spitz nevus", "Nevus"),
            ("nevus", "Nevus"),
            ("mole", "Nevus"),
            ("contact dermatitis", "Contact Dermatitis"),
            ("psoriasis", "Psoriasis"),
            ("seborrheic dermatitis", "Seborrheic Dermatitis"),
            ("atopic dermatitis", "Atopic Dermatitis"),
        ]
        for needle, label in sorted(label_map, key=lambda item: len(item[0]), reverse=True):
            if needle in lowered:
                return label
        return text.strip()

    @staticmethod
    def _coerce_skinvl_allowed_label(text: str) -> str:
        normalized = text.strip()
        if normalized in SKINVL_ALLOWED_LABELS:
            return normalized
        lowered = normalized.lower()
        for label in SKINVL_ALLOWED_LABELS:
            if label.lower() in lowered:
                return label
        heuristic_map = [
            ("pearly", "Basal Cell Carcinoma"),
            ("telangiect", "Basal Cell Carcinoma"),
            ("dark center", "Squamous Cell Carcinoma"),
            ("lighter periphery", "Squamous Cell Carcinoma"),
            ("crater", "Squamous Cell Carcinoma"),
            ("scaly", "Actinic Keratosis"),
            ("waxy", "Seborrheic Keratosis"),
            ("stuck-on", "Seborrheic Keratosis"),
            ("melanoma", "Malignant Melanoma"),
            ("nevus", "Nevus"),
            ("mole", "Nevus"),
            ("dermatitis", "Contact Dermatitis"),
            ("psoriasis", "Psoriasis"),
        ]
        for needle, label in heuristic_map:
            if needle in lowered:
                return label
        return normalized

    @staticmethod
    def _apply_skinvl_selector(payload: dict[str, Any]) -> dict[str, Any]:
        final_text = str(payload.get("final_diagnosis", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        differentials = payload.get("differential_diagnoses", [])
        if not isinstance(differentials, list):
            differentials = []
        candidates = [final_text] + [str(item).strip() for item in differentials if str(item).strip()]
        candidates.append(rationale)

        selected = ""
        for candidate in candidates:
            mapped = DermOpenAIClient._coerce_skinvl_allowed_label(candidate)
            if mapped in SKINVL_ALLOWED_LABELS:
                selected = mapped
                break

        if not selected:
            selected = "Basal Cell Carcinoma" if "malignan" in rationale.lower() else final_text
            selected = DermOpenAIClient._coerce_skinvl_allowed_label(selected)

        if selected not in SKINVL_ALLOWED_LABELS:
            return payload

        filtered_differentials: list[str] = []
        for item in candidates:
            mapped = DermOpenAIClient._coerce_skinvl_allowed_label(item)
            if mapped in SKINVL_ALLOWED_LABELS and mapped not in filtered_differentials:
                filtered_differentials.append(mapped)

        payload["final_diagnosis"] = selected
        payload["differential_diagnoses"] = filtered_differentials[:5] or [selected]
        return payload

    @staticmethod
    def _build_skinvl_final_evidence_summary(payload: dict[str, Any]) -> str:
        lines: list[str] = []
        selected = payload.get("selected_evidence", [])
        if isinstance(selected, list):
            for item in selected[:5]:
                if not isinstance(item, dict):
                    continue
                section = str(item.get("section", "")).strip() or "evidence"
                summary = str(item.get("summary", "")).strip()
                if summary:
                    lines.append(f"- {section}: {summary[:180]}")
        risk_flags = payload.get("risk_flags", [])
        if isinstance(risk_flags, list):
            for flag in risk_flags[:2]:
                flag_text = str(flag).strip()
                if flag_text:
                    lines.append(f"- risk: {flag_text[:180]}")
        uncertainty_summary = payload.get("uncertainty_summary", {})
        if isinstance(uncertainty_summary, dict):
            level = str(uncertainty_summary.get("uncertainty_level", "") or uncertainty_summary.get("level", "")).strip()
            if level:
                lines.append(f"- uncertainty: {level}")
        if not lines:
            lines.append("- no additional agent evidence")
        return "\n".join(lines[:8])

    @staticmethod
    def _repair_truncated_json(candidate: str) -> str | None:
        if not candidate:
            return None
        start = candidate.find("{")
        if start >= 0:
            candidate = candidate[start:]
        if not candidate.startswith("{"):
            return None

        result_chars: list[str] = []
        stack: list[str] = []
        in_string = False
        escape = False

        for char in candidate:
            result_chars.append(char)
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in {"}", "]"} and stack and stack[-1] == char:
                stack.pop()

        repaired = "".join(result_chars).rstrip()
        if in_string:
            repaired += '"'
        repaired = repaired.rstrip(", \n\t")
        while repaired.endswith(","):
            repaired = repaired[:-1].rstrip()
        while stack:
            closing = stack.pop()
            repaired = repaired.rstrip(", \n\t")
            repaired += closing
        repaired = repaired.replace(",}", "}").replace(",]", "]")
        return repaired

    @staticmethod
    def _normalize_initial_perception(payload: dict[str, Any]) -> dict[str, Any]:
        ddx_candidates = payload.get("ddx_candidates", [])
        if isinstance(ddx_candidates, str):
            ddx_candidates = [ddx_candidates]
        elif not isinstance(ddx_candidates, list):
            ddx_candidates = []
        ddx_candidates = [str(item) for item in ddx_candidates if str(item).strip()]

        uncertainty = payload.get("uncertainty", {})
        if isinstance(uncertainty, str):
            uncertainty = {"level": uncertainty, "reasons": []}
        elif not isinstance(uncertainty, dict):
            uncertainty = {}
        uncertainty = {
            "level": str(uncertainty.get("level", "unknown")).lower(),
            "reasons": [str(item) for item in uncertainty.get("reasons", [])]
            if isinstance(uncertainty.get("reasons", []), list)
            else [str(uncertainty.get("reasons"))] if uncertainty.get("reasons") else [],
        }

        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        elif not isinstance(notes, list):
            notes = []

        return {
            "image_summary": str(payload.get("image_summary", "")),
            "ddx_candidates": ddx_candidates,
            "uncertainty": uncertainty,
            "notes": [str(item) for item in notes if str(item).strip()],
        }

    @staticmethod
    def _compact_evidence_package(
        payload: dict[str, Any],
        *,
        retrieval_top_k: int = 4,
        max_skill_count: int = 16,
        max_skill_fields: int = 6,
        serialized_max_length: int = 3600,
        profile_id: str = "standard",
    ) -> dict[str, Any]:
        compact_raw = DermOpenAIClient._compact_retrieval_slice(payload.get("retrieved_raw_cases_summary", []), top_k=retrieval_top_k)
        compact_tactical = DermOpenAIClient._compact_retrieval_slice(
            payload.get("retrieved_tactical_experiences_summary", []),
            top_k=retrieval_top_k,
        )
        compact_abstract = DermOpenAIClient._compact_retrieval_slice(
            payload.get("retrieved_abstract_experiences_summary", []),
            top_k=retrieval_top_k,
        )
        if not compact_raw and not compact_tactical and not compact_abstract:
            compact_legacy = DermOpenAIClient._compact_retrieval_slice(payload.get("retrieved_experience", []), top_k=retrieval_top_k)
            compact_abstract = compact_legacy

        skill_outputs = payload.get("skill_outputs", {})
        compact_skills = {}
        preferred_skill_order = [
            "lesion_description_structuring_skill",
            "morphology_analysis_skill",
            "color_pattern_analysis_skill",
            "border_surface_analysis_skill",
            "distribution_analysis_skill",
            "metadata_consistency_skill",
            "temporal_evolution_skill",
            "differential_compare_skill",
            "exclusion_reasoning_skill",
            "information_gap_detection_skill",
            "mel_nev_specialist_skill",
            "ack_scc_specialist_skill",
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
            "contradiction_check_skill",
            "escalation_recommendation_skill",
        ]
        ordered_skill_names: list[str] = []
        for skill_name in preferred_skill_order:
            if skill_name in skill_outputs and skill_name not in ordered_skill_names:
                ordered_skill_names.append(skill_name)
        for skill_name in skill_outputs:
            if skill_name not in ordered_skill_names:
                ordered_skill_names.append(skill_name)
        for skill_name in ordered_skill_names[:max_skill_count]:
            output = skill_outputs.get(skill_name)
            if isinstance(output, dict):
                compact_skills[skill_name] = DermOpenAIClient._compact_skill_output(
                    skill_name,
                    output,
                    max_fields=max_skill_fields,
                )

        notes = payload.get("notes", [])
        if isinstance(notes, list):
            notes = [str(item).strip()[:160] for item in notes[:4] if str(item).strip()]
        else:
            notes = [str(notes).strip()[:160]] if notes else []

        planner_rationale = payload.get("planner_rationale", {})
        compact_planner_rationale = {
            "selected_skills": planner_rationale.get("selected_skills", [])[: min(max_skill_count, 10)],
            "selection_reasons": {
                skill_name: [str(item)[:180] for item in reasons[:2]]
                for skill_name, reasons in list(planner_rationale.get("selection_reasons", {}).items())[: min(max_skill_count, 6)]
            },
        }

        serialized_evidence_text = str(payload.get("serialized_evidence_text", "")).strip()
        if len(serialized_evidence_text) > serialized_max_length:
            serialized_evidence_text = DermOpenAIClient._compact_serialized_evidence_text(
                serialized_evidence_text,
                max_length=serialized_max_length,
            )
        selected_evidence = DermOpenAIClient._compact_selected_evidence(
            payload.get("selected_evidence", []),
            top_k=max(4, min(max_skill_count, 8)),
        )
        evidence_decision_policy = DermOpenAIClient._compact_evidence_decision_policy(
            payload.get("evidence_decision_policy", {}),
            top_k=max(4, min(max_skill_count, 6)),
        )

        return {
            "compression_profile": profile_id,
            "initial_perception_summary": payload.get("initial_perception_summary", payload.get("perception", {})),
            "retrieved_raw_cases_summary": compact_raw,
            "retrieved_tactical_experiences_summary": compact_tactical,
            "retrieved_abstract_experiences_summary": compact_abstract,
            "skill_outputs": compact_skills,
            "risk_flags": payload.get("risk_flags", []),
            "uncertainty_summary": payload.get("uncertainty_summary", payload.get("uncertainty", {})),
            "contradiction_summary": payload.get("contradiction_summary", {}),
            "information_gap_summary": payload.get("information_gap_summary", {}),
            "escalation_summary": payload.get("escalation_summary", {}),
            "planner_rationale": compact_planner_rationale,
            "notes": notes,
            "selected_evidence": selected_evidence,
            "evidence_decision_policy": evidence_decision_policy,
            "serialized_evidence_text": serialized_evidence_text,
        }

    @staticmethod
    def _compact_retrieval_slice(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        compact_records: list[dict[str, Any]] = []
        for record in records[:top_k]:
            compact_records.append(
                {
                    "source_id": record.get("source_id"),
                    "source_layer": record.get("source_layer"),
                    "experience_type": record.get("experience_type"),
                    "case_id": record.get("case_id"),
                    "perception_summary": str(record.get("perception_summary", ""))[:180],
                    "confusion_pair": str(record.get("confusion_pair", ""))[:80],
                    "learning_points": [str(item).strip()[:120] for item in record.get("learning_points", [])[:2] if str(item).strip()],
                }
            )
        return compact_records

    @staticmethod
    def _compact_skill_output(skill_name: str, output: dict[str, Any], *, max_fields: int = 6) -> dict[str, Any]:
        compact_output: dict[str, Any] = {}
        for field_name in DermOpenAIClient._ordered_skill_field_names(skill_name, output):
            value = output.get(field_name)
            if field_name == "referenced_experiences":
                continue
            if value in (None, "", [], {}, "unknown"):
                continue
            if isinstance(value, list):
                compact_values = [str(item).strip()[:140] for item in value[:2] if str(item).strip()]
                if compact_values:
                    compact_output[field_name] = compact_values
            else:
                compact_output[field_name] = str(value).strip()[:140]
            if len(compact_output) >= max_fields:
                break
        return compact_output

    @staticmethod
    def _ordered_skill_field_names(skill_name: str, output: dict[str, Any]) -> list[str]:
        preferred = list(SKILL_FIELD_PRIORITY.get(skill_name, ())) + list(GENERIC_SKILL_FIELD_PRIORITY)
        ordered: list[str] = []
        for field_name in preferred:
            if field_name in output and field_name not in ordered:
                ordered.append(field_name)
        for field_name in output:
            if field_name not in ordered:
                ordered.append(field_name)
        return ordered

    @staticmethod
    def _canonicalize_evidence_package(payload: dict[str, Any]) -> dict[str, Any]:
        skill_outputs = payload.get("skill_outputs", {})
        canonical_skills: dict[str, dict[str, Any]] = {}
        if isinstance(skill_outputs, dict):
            for skill_name, output in skill_outputs.items():
                if isinstance(output, dict):
                    canonical_skills[str(skill_name)] = DermOpenAIClient._full_skill_output(str(skill_name), output)
        notes = payload.get("notes", [])
        normalized_notes = []
        if isinstance(notes, list):
            normalized_notes = [str(item).strip()[:220] for item in notes[:6] if str(item).strip()]
        elif notes:
            normalized_notes = [str(notes).strip()[:220]]
        raw_cases = payload.get("retrieved_raw_cases_summary", [])
        tactical = payload.get("retrieved_tactical_experiences_summary", [])
        abstract = payload.get("retrieved_abstract_experiences_summary", [])
        risk_flags = payload.get("risk_flags", [])
        return {
            "compression_profile": FULL_CLINICAL_PROFILE_ID,
            "initial_perception_summary": payload.get("initial_perception_summary", payload.get("perception", {})),
            "retrieved_raw_cases_summary": list(raw_cases) if isinstance(raw_cases, list) else [],
            "retrieved_tactical_experiences_summary": list(tactical) if isinstance(tactical, list) else [],
            "retrieved_abstract_experiences_summary": list(abstract) if isinstance(abstract, list) else [],
            "skill_outputs": canonical_skills,
            "risk_flags": list(risk_flags) if isinstance(risk_flags, list) else [],
            "uncertainty_summary": payload.get("uncertainty_summary", payload.get("uncertainty", {})),
            "contradiction_summary": payload.get("contradiction_summary", {}),
            "information_gap_summary": payload.get("information_gap_summary", {}),
            "escalation_summary": payload.get("escalation_summary", {}),
            "planner_rationale": DermOpenAIClient._canonicalize_planner_rationale(payload.get("planner_rationale", {})),
            "notes": normalized_notes,
            "selected_evidence": DermOpenAIClient._canonicalize_selected_evidence(payload.get("selected_evidence", [])),
            "evidence_decision_policy": DermOpenAIClient._canonicalize_evidence_decision_policy(
                payload.get("evidence_decision_policy", {})
            ),
            "serialized_evidence_text": str(payload.get("serialized_evidence_text", "")).strip(),
        }

    @staticmethod
    def _canonicalize_planner_rationale(planner_rationale: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(planner_rationale, dict):
            return {}
        selection_reasons = planner_rationale.get("selection_reasons", {})
        selected_skills = planner_rationale.get("selected_skills", [])
        compact_reasons: dict[str, list[str]] = {}
        if isinstance(selection_reasons, dict):
            for skill_name, reasons in selection_reasons.items():
                if isinstance(reasons, list):
                    compact_reasons[str(skill_name)] = [str(item).strip()[:220] for item in reasons[:4] if str(item).strip()]
        return {
            "planner_type": str(planner_rationale.get("planner_type", "")).strip(),
            "planner_version": str(planner_rationale.get("planner_version", "")).strip(),
            "selected_skills": [str(item).strip() for item in selected_skills if str(item).strip()]
            if isinstance(selected_skills, list)
            else [],
            "selection_reasons": compact_reasons,
        }

    @staticmethod
    def _prepare_evidence_for_profile(payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(profile.get("profile_id", FULL_CLINICAL_PROFILE_ID))
        if profile_id == FULL_CLINICAL_PROFILE_ID:
            prepared = dict(payload)
            prepared["compression_profile"] = FULL_CLINICAL_PROFILE_ID
            return prepared
        return DermOpenAIClient._compact_evidence_package(
            payload,
            retrieval_top_k=int(profile["retrieval_top_k"]),
            max_skill_count=int(profile["max_skill_count"]),
            max_skill_fields=int(profile["max_skill_fields"]),
            serialized_max_length=int(profile["serialized_max_length"]),
            profile_id=profile_id,
        )

    @staticmethod
    def _full_skill_output(skill_name: str, output: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for field_name in DermOpenAIClient._ordered_skill_field_names(skill_name, output):
            if field_name == "referenced_experiences":
                continue
            value = output.get(field_name)
            if value in (None, "", [], {}, "unknown"):
                continue
            if isinstance(value, list):
                cleaned = [str(item).strip()[:220] for item in value[:6] if str(item).strip()]
                if cleaned:
                    normalized[field_name] = cleaned
            else:
                normalized[field_name] = str(value).strip()[:220]
        return normalized

    @staticmethod
    def _canonicalize_selected_evidence(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "source_type": str(item.get("source_type", "")).strip(),
                    "source_name": str(item.get("source_name", "")).strip()[:160],
                    "skill_name": str(item.get("skill_name", "")).strip()[:120],
                    "retrieval_type": str(item.get("retrieval_type", "")).strip()[:120],
                    "category": str(item.get("category", "")).strip()[:120],
                    "summary": str(item.get("summary", "")).strip()[:360],
                    "score": item.get("score"),
                    "rank": item.get("rank"),
                    "keep_reason": str(item.get("keep_reason", "")).strip()[:120],
                    "section": str(item.get("section", "")).strip()[:120],
                }
            )
        return normalized

    @staticmethod
    def _canonicalize_evidence_decision_policy(policy: Any) -> dict[str, Any]:
        if not isinstance(policy, dict):
            return {}
        risk_layer = dict(policy.get("risk_layer", {}))
        diagnosis_layer = dict(policy.get("diagnosis_override_layer", {}))
        return {
            "risk_layer": {
                "risk_flag": str(risk_layer.get("risk_flag", "")).strip()[:120],
                "caution_flags": [str(item).strip()[:120] for item in risk_layer.get("caution_flags", [])[:4] if str(item).strip()],
                "follow_up_suggestion": str(risk_layer.get("follow_up_suggestion", "")).strip()[:220],
                "baseline_preview": dict(risk_layer.get("baseline_preview", {})) if isinstance(risk_layer.get("baseline_preview", {}), dict) else {},
                "selected_evidence": DermOpenAIClient._canonicalize_selected_evidence(risk_layer.get("selected_evidence", [])),
            },
            "diagnosis_override_layer": {
                "override_allowed": bool(diagnosis_layer.get("override_allowed", False)),
                "malignancy_override_allowed": bool(diagnosis_layer.get("malignancy_override_allowed", False)),
                "subtype_override_allowed": bool(diagnosis_layer.get("subtype_override_allowed", False)),
                "override_mode": str(diagnosis_layer.get("override_mode", "")).strip()[:60],
                "override_reasons": [
                    str(item).strip()[:120] for item in diagnosis_layer.get("override_reasons", [])[:6] if str(item).strip()
                ],
                "why_not_confident_enough_to_override": [
                    str(item).strip()[:180]
                    for item in diagnosis_layer.get("why_not_confident_enough_to_override", [])[:6]
                    if str(item).strip()
                ],
                "supporting_score": diagnosis_layer.get("supporting_score"),
                "opposing_score": diagnosis_layer.get("opposing_score"),
                "support_margin": diagnosis_layer.get("support_margin"),
                "selected_evidence_present": bool(diagnosis_layer.get("selected_evidence_present", False)),
                "subtype_support_score": diagnosis_layer.get("subtype_support_score"),
                "subtype_support_margin": diagnosis_layer.get("subtype_support_margin"),
                "specialist_support_present": bool(diagnosis_layer.get("specialist_support_present", False)),
                "consistent_retrieval_count": diagnosis_layer.get("consistent_retrieval_count"),
                "contradiction_count": diagnosis_layer.get("contradiction_count"),
                "uncertainty_level": str(diagnosis_layer.get("uncertainty_level", "")).strip()[:60],
                "opposing_quota_satisfied": bool(diagnosis_layer.get("opposing_quota_satisfied", False)),
                "subtype_support_quota_satisfied": bool(diagnosis_layer.get("subtype_support_quota_satisfied", False)),
                "supporting_evidence": DermOpenAIClient._canonicalize_selected_evidence(
                    diagnosis_layer.get("supporting_evidence", [])
                ),
                "opposing_evidence": DermOpenAIClient._canonicalize_selected_evidence(
                    diagnosis_layer.get("opposing_evidence", [])
                ),
            },
        }

    @staticmethod
    def _compact_selected_evidence(items: Any, *, top_k: int) -> list[dict[str, Any]]:
        canonical = DermOpenAIClient._canonicalize_selected_evidence(items)
        compacted: list[dict[str, Any]] = []
        for item in canonical[:top_k]:
            compacted.append(
                {
                    "source_type": item.get("source_type", ""),
                    "source_name": str(item.get("source_name", ""))[:120],
                    "skill_name": str(item.get("skill_name", ""))[:80],
                    "retrieval_type": str(item.get("retrieval_type", ""))[:80],
                    "category": str(item.get("category", ""))[:80],
                    "summary": str(item.get("summary", ""))[:220],
                    "score": item.get("score"),
                    "rank": item.get("rank"),
                    "keep_reason": str(item.get("keep_reason", ""))[:80],
                }
            )
        return compacted

    @staticmethod
    def _compact_evidence_decision_policy(policy: Any, *, top_k: int) -> dict[str, Any]:
        canonical = DermOpenAIClient._canonicalize_evidence_decision_policy(policy)
        risk_layer = dict(canonical.get("risk_layer", {}))
        diagnosis_layer = dict(canonical.get("diagnosis_override_layer", {}))
        return {
            "risk_layer": {
                "risk_flag": risk_layer.get("risk_flag", ""),
                "caution_flags": list(risk_layer.get("caution_flags", []))[:4],
                "follow_up_suggestion": risk_layer.get("follow_up_suggestion", ""),
                "baseline_preview": dict(risk_layer.get("baseline_preview", {})) if isinstance(risk_layer.get("baseline_preview", {}), dict) else {},
                "selected_evidence": DermOpenAIClient._compact_selected_evidence(
                    risk_layer.get("selected_evidence", []),
                    top_k=top_k,
                ),
            },
            "diagnosis_override_layer": {
                "override_allowed": bool(diagnosis_layer.get("override_allowed", False)),
                "malignancy_override_allowed": bool(diagnosis_layer.get("malignancy_override_allowed", False)),
                "subtype_override_allowed": bool(diagnosis_layer.get("subtype_override_allowed", False)),
                "override_mode": diagnosis_layer.get("override_mode", ""),
                "override_reasons": list(diagnosis_layer.get("override_reasons", []))[:4],
                "why_not_confident_enough_to_override": list(
                    diagnosis_layer.get("why_not_confident_enough_to_override", [])
                )[:4],
                "supporting_score": diagnosis_layer.get("supporting_score"),
                "opposing_score": diagnosis_layer.get("opposing_score"),
                "support_margin": diagnosis_layer.get("support_margin"),
                "selected_evidence_present": bool(diagnosis_layer.get("selected_evidence_present", False)),
                "subtype_support_score": diagnosis_layer.get("subtype_support_score"),
                "subtype_support_margin": diagnosis_layer.get("subtype_support_margin"),
                "specialist_support_present": bool(diagnosis_layer.get("specialist_support_present", False)),
                "consistent_retrieval_count": diagnosis_layer.get("consistent_retrieval_count"),
                "contradiction_count": diagnosis_layer.get("contradiction_count"),
                "uncertainty_level": diagnosis_layer.get("uncertainty_level", ""),
                "opposing_quota_satisfied": bool(diagnosis_layer.get("opposing_quota_satisfied", False)),
                "subtype_support_quota_satisfied": bool(diagnosis_layer.get("subtype_support_quota_satisfied", False)),
                "supporting_evidence": DermOpenAIClient._compact_selected_evidence(
                    diagnosis_layer.get("supporting_evidence", []),
                    top_k=top_k,
                ),
                "opposing_evidence": DermOpenAIClient._compact_selected_evidence(
                    diagnosis_layer.get("opposing_evidence", []),
                    top_k=max(2, top_k // 2),
                ),
            },
        }

    @staticmethod
    def _compact_serialized_evidence_text(text: str, *, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        sections = DermOpenAIClient._split_serialized_sections(text)
        if not sections:
            return text[: max(0, max_length - 3)] + "..."

        section_priority = [
            "[Observation Evidence]",
            "[Exclusion And Comparison Evidence]",
            "[Risk Evidence]",
            "[Conflict And Uncertainty]",
            "[Planner Rationale]",
        ]
        prioritized_sections = sorted(
            sections,
            key=lambda item: section_priority.index(item[0]) if item[0] in section_priority else len(section_priority),
        )
        rendered_sections = [{"header": header, "lines": [header], "body_lines": list(body_lines)} for header, body_lines in prioritized_sections]
        merged = "\n\n".join(section["header"] for section in rendered_sections)
        if len(merged) > max_length:
            return merged[: max(0, max_length - 3)] + "..."

        remaining_budget = max_length - len(merged)
        line_index = 0
        while remaining_budget > 0:
            added_any = False
            for section in rendered_sections:
                body_lines = section["body_lines"]
                if line_index >= len(body_lines):
                    continue
                candidate = body_lines[line_index]
                line_cost = len(candidate) + 1
                if line_cost > remaining_budget:
                    clipped = candidate[: max(0, remaining_budget - 4)]
                    if clipped:
                        section["lines"].append(clipped + "...")
                        remaining_budget = 0
                        added_any = True
                    break
                section["lines"].append(candidate)
                remaining_budget -= line_cost
                added_any = True
            if not added_any:
                break
            line_index += 1

        compact_sections = ["\n".join(section["lines"]) for section in rendered_sections]
        merged = "\n\n".join(compact_sections)
        if len(merged) <= max_length:
            return merged
        return merged[: max(0, max_length - 3)] + "..."

    @staticmethod
    def _split_serialized_sections(text: str) -> list[tuple[str, list[str]]]:
        current_header = ""
        current_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.startswith("[") and line.endswith("]"):
                if current_header:
                    sections.append((current_header, current_lines))
                current_header = line
                current_lines = []
                continue
            if line:
                current_lines.append(line)
        if current_header:
            sections.append((current_header, current_lines))
        return sections

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "maximum context length" in message or "input length" in message or "context length" in message


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid float environment value for %s=%r", name, raw_value)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid integer environment value for %s=%r", name, raw_value)
        return default
