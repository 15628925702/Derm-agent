from __future__ import annotations

import csv
import json
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from project_paths import outputs_root, repo_root

AUDITABILITY_VERSION = "auditability_scoring_v1"
WANTED_CASE_FILES = (
    "case_execution_record.json",
    "evidence_package.json",
    "reflection.json",
    "state.json",
)


def auditability_root() -> Path:
    root = outputs_root() / "auditability"
    root.mkdir(parents=True, exist_ok=True)
    return root


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_rules(rules_path: Path | None = None) -> dict[str, Any]:
    path = (rules_path or (repo_root() / "configs" / "auditability_rules.json")).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower())
    return lowered.strip("_") or "auditability"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _unknown_tokens(rules: dict[str, Any]) -> set[str]:
    tokens = {_normalize_text(item) for item in list(rules.get("unknown_tokens", []))}
    tokens.update({"unknown", "", "none", "null", "n/a"})
    return tokens


def _is_meaningful(value: Any, *, rules: dict[str, Any]) -> bool:
    unknown_tokens = _unknown_tokens(rules)
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return _normalize_text(value) not in unknown_tokens
    if isinstance(value, dict):
        return any(_is_meaningful(v, rules=rules) for v in value.values())
    if isinstance(value, list):
        return any(_is_meaningful(v, rules=rules) for v in value)
    return True


def _flatten_strings(value: Any) -> list[str]:
    items: list[str] = []
    if isinstance(value, str):
        if value.strip():
            items.append(value.strip())
        return items
    if isinstance(value, dict):
        for child in value.values():
            items.extend(_flatten_strings(child))
        return items
    if isinstance(value, list):
        for child in value:
            items.extend(_flatten_strings(child))
        return items
    return items


def _keyword_hits(texts: Iterable[str], keywords: Iterable[str]) -> list[str]:
    haystack = " ".join(_normalize_text(text) for text in texts if str(text or "").strip())
    hits: list[str] = []
    for keyword in keywords:
        normalized = _normalize_text(keyword)
        if not normalized:
            continue
        if normalized in haystack:
            hits.append(keyword)
    return sorted(set(hits))


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _maybe_parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _extract_ground_truth(record: dict[str, Any]) -> str:
    ground_truth = dict(record.get("ground_truth", {}) or {})
    if ground_truth.get("canonical_label"):
        return str(ground_truth["canonical_label"]).strip()
    if ground_truth.get("raw_label"):
        return str(ground_truth["raw_label"]).strip()
    reflection_summary = dict(record.get("reflection_summary", {}) or {})
    case_outcome = dict(reflection_summary.get("case_outcome", {}) or {})
    if case_outcome.get("reference_canonical_label"):
        return str(case_outcome["reference_canonical_label"]).strip()
    if case_outcome.get("reference_label"):
        return str(case_outcome["reference_label"]).strip()
    return ""


def _extract_final_prediction(record: dict[str, Any]) -> str:
    qwen_final = dict(record.get("qwen_final", {}) or {})
    if qwen_final.get("final_diagnosis"):
        return str(qwen_final["final_diagnosis"]).strip()
    final_decision = dict(record.get("final_decision", {}) or {})
    if final_decision.get("label"):
        return str(final_decision["label"]).strip()
    if record.get("qwen_final_diagnosis"):
        return str(record["qwen_final_diagnosis"]).strip()
    reflection_summary = dict(record.get("reflection_summary", {}) or {})
    case_outcome = dict(reflection_summary.get("case_outcome", {}) or {})
    if case_outcome.get("predicted_canonical_label"):
        return str(case_outcome["predicted_canonical_label"]).strip()
    if case_outcome.get("predicted_label"):
        return str(case_outcome["predicted_label"]).strip()
    return ""


def _extract_initial_prediction(record: dict[str, Any]) -> str:
    qwen_initial = dict(record.get("qwen_initial", {}) or {})
    ddx_candidates = list(qwen_initial.get("ddx_candidates", []) or [])
    if ddx_candidates:
        return str(ddx_candidates[0]).strip()
    baseline_qwen = dict(record.get("baseline_qwen", {}) or {})
    if baseline_qwen.get("final_diagnosis"):
        return str(baseline_qwen["final_diagnosis"]).strip()
    qwen_final = dict(record.get("qwen_final", {}) or {})
    differential = list(qwen_final.get("differential_diagnoses", []) or [])
    if differential:
        return str(differential[0]).strip()
    return ""


def _extract_correct_flag(record: dict[str, Any]) -> bool | None:
    evaluation = dict(record.get("evaluation", {}) or {})
    if "correct" in evaluation:
        return bool(evaluation["correct"])
    correctness = dict(record.get("correctness", {}) or {})
    if "is_correct" in correctness:
        return bool(correctness["is_correct"])
    reflection_summary = dict(record.get("reflection_summary", {}) or {})
    case_outcome = dict(reflection_summary.get("case_outcome", {}) or {})
    if "is_correct" in case_outcome:
        return bool(case_outcome["is_correct"])
    return None


def _extract_baseline_correct_flag(record: dict[str, Any]) -> bool | None:
    evaluation = dict(record.get("evaluation", {}) or {})
    if "baseline_correct" in evaluation:
        return bool(evaluation["baseline_correct"])
    return None


def _extract_high_risk_flag(record: dict[str, Any], rules: dict[str, Any], ground_truth: str) -> bool:
    ground_truth_payload = dict(record.get("ground_truth", {}) or {})
    if isinstance(ground_truth_payload.get("malignant_flag"), bool):
        return bool(ground_truth_payload["malignant_flag"])
    normalized = _normalize_text(ground_truth)
    for token in list(rules.get("high_risk_label_keywords", [])):
        if _normalize_text(token) and _normalize_text(token) in normalized:
            return True
    return False


def _find_expected_risk_terms(ground_truth: str, rules: dict[str, Any]) -> list[str]:
    normalized = _normalize_text(ground_truth)
    for payload in dict(rules.get("risk_label_groups", {}) or {}).values():
        labels = [_normalize_text(item) for item in list(payload.get("labels", []))]
        if any(label and label in normalized for label in labels):
            return [str(item) for item in list(payload.get("expected_terms", []))]
    return []


def _canonical_label_token(value: str) -> str:
    normalized = _normalize_text(value)
    aliases = {
        "nevus": "nv",
        "nv": "nv",
        "malignant melanoma": "mel",
        "melanoma": "mel",
        "mel": "mel",
        "basal cell carcinoma": "bcc",
        "bcc": "bcc",
        "actinic keratosis": "akiec",
        "akiec": "akiec",
        "benign keratosis": "bkl",
        "bkl": "bkl",
    }
    return aliases.get(normalized, normalized)


def _summarize_payload(payload: Any, *, max_chars: int = 3000) -> str:
    if isinstance(payload, str):
        return payload[:max_chars]
    text = _json_dumps(payload)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_evidence_text(
    *,
    record: dict[str, Any],
    evidence_package: dict[str, Any],
    reflection: dict[str, Any],
    system_name: str,
) -> str:
    parts: list[str] = []
    serialized = str(evidence_package.get("serialized_evidence_text", "")).strip()
    if serialized:
        parts.append(serialized)

    physician_summary = record.get("physician_evidence_summary", {})
    if physician_summary:
        parts.append("[Physician Evidence Summary]\n" + _summarize_payload(physician_summary))

    qwen_final = dict(record.get("qwen_final", {}) or {})
    qwen_rationale = _maybe_parse_jsonish(qwen_final.get("rationale"))
    if qwen_rationale:
        parts.append("[Final Diagnosis Rationale]\n" + _summarize_payload(qwen_rationale))

    baseline_qwen = dict(record.get("baseline_qwen", {}) or {})
    baseline_rationale = _maybe_parse_jsonish(baseline_qwen.get("rationale"))
    if baseline_rationale and slugify(system_name) != "direct_baseline":
        parts.append("[Baseline Preview Rationale]\n" + _summarize_payload(baseline_rationale))

    if reflection:
        reflection_parts: list[str] = []
        if reflection.get("reasoning_summary"):
            reflection_parts.append(str(reflection["reasoning_summary"]).strip())
        case_outcome = dict(reflection.get("case_outcome", {}) or {})
        if case_outcome:
            reflection_parts.append(_summarize_payload(case_outcome, max_chars=1200))
        if reflection_parts:
            parts.append("[Reflection]\n" + "\n".join(reflection_parts))

    if not parts:
        evidence_bundle = dict(record.get("evidence_bundle", {}) or {})
        if evidence_bundle:
            parts.append("[Evidence Bundle]\n" + _summarize_payload(evidence_bundle))

    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _normalize_text(part)
        if normalized and normalized not in seen:
            unique_parts.append(part)
            seen.add(normalized)
    return "\n\n".join(unique_parts).strip()


def _scoring_text_fragments(entry: "AuditEntry") -> list[str]:
    evidence_package = dict(entry.evidence_package or {})
    texts: list[str] = []
    serialized = str(evidence_package.get("serialized_evidence_text", "")).strip()
    if serialized:
        texts.append(serialized)

    if entry.system_type == "baseline" or slugify(entry.system_name) == "direct_baseline":
        qwen_final = dict(entry.record.get("qwen_final", {}) or {})
        baseline_qwen = dict(entry.record.get("baseline_qwen", {}) or {})
        for payload in (qwen_final.get("rationale"), baseline_qwen.get("rationale")):
            parsed = _maybe_parse_jsonish(payload)
            if parsed:
                texts.append(_summarize_payload(parsed, max_chars=3000))
    else:
        physician_summary = entry.record.get("physician_evidence_summary", {})
        if physician_summary:
            texts.append(_summarize_payload(physician_summary, max_chars=3000))
        selected_evidence = evidence_package.get("selected_evidence", [])
        if selected_evidence:
            texts.extend(_flatten_strings(selected_evidence))
        for key in ("uncertainty_summary", "contradiction_summary", "information_gap_summary"):
            payload = evidence_package.get(key, {})
            if payload:
                texts.extend(_flatten_strings(payload))

    return [text for text in texts if str(text or "").strip()]


@dataclass
class AuditEntry:
    case_id: str
    dataset: str
    system_name: str
    target_id: str
    system_type: str
    source_root: str
    record_path: str
    evidence_package_path: str
    reflection_path: str
    state_path: str
    ground_truth: str
    final_prediction: str
    initial_prediction: str
    direct_baseline_prediction: str
    direct_baseline_correct: bool | None
    correct: bool | None
    is_malignant_or_high_risk: bool
    evidence_text: str
    record: dict[str, Any]
    evidence_package: dict[str, Any]
    reflection: dict[str, Any]
    state: dict[str, Any]


def _build_entry(
    *,
    source_root: Path,
    record_path: Path,
    evidence_package_path: Path | None,
    reflection_path: Path | None,
    state_path: Path | None,
    system_name: str,
    target_id: str,
    system_type: str,
    direct_baseline_prediction: str = "",
    direct_baseline_correct: bool | None = None,
    rules: dict[str, Any],
) -> AuditEntry:
    record = _safe_load_json(record_path)
    evidence_package = _safe_load_json(evidence_package_path) if evidence_package_path else {}
    reflection = _safe_load_json(reflection_path) if reflection_path else dict(record.get("reflection_summary", {}) or {})
    state = _safe_load_json(state_path) if state_path else {}
    ground_truth = _extract_ground_truth(record)
    evidence_text = _build_evidence_text(
        record=record,
        evidence_package=evidence_package or dict(record.get("evidence_bundle", {}) or {}),
        reflection=reflection,
        system_name=system_name,
    )
    return AuditEntry(
        case_id=str(record.get("case_id", "")).strip() or record_path.parent.name,
        dataset=str(record.get("dataset_name", "")).strip(),
        system_name=system_name,
        target_id=target_id,
        system_type=system_type,
        source_root=str(source_root),
        record_path=str(record_path),
        evidence_package_path=str(evidence_package_path or ""),
        reflection_path=str(reflection_path or ""),
        state_path=str(state_path or ""),
        ground_truth=ground_truth,
        final_prediction=_extract_final_prediction(record),
        initial_prediction=_extract_initial_prediction(record),
        direct_baseline_prediction=direct_baseline_prediction,
        direct_baseline_correct=direct_baseline_correct,
        correct=_extract_correct_flag(record),
        is_malignant_or_high_risk=_extract_high_risk_flag(record, rules, ground_truth),
        evidence_text=evidence_text,
        record=record,
        evidence_package=evidence_package or dict(record.get("evidence_bundle", {}) or {}),
        reflection=reflection,
        state=state,
    )


def _collect_compare_run_entries(run_root: Path, rules: dict[str, Any]) -> list[AuditEntry]:
    manifest = _safe_load_json(run_root / "evaluation_manifest.json")
    run_targets = list(manifest.get("run_targets", []) or [])
    entries: list[AuditEntry] = []
    baseline_index: dict[str, tuple[str, bool | None]] = {}

    for target in run_targets:
        target_id = str(target.get("target_id", "")).strip()
        target_root = run_root / "targets" / target_id
        record_root = target_root / "records"
        if not record_root.exists():
            continue
        if str(target.get("mode", "")).strip() == "baseline":
            for record_path in sorted(record_root.glob("*/case_execution_record.json")):
                record = _safe_load_json(record_path)
                baseline_index[str(record.get("case_id", "")).strip() or record_path.parent.name] = (
                    _extract_final_prediction(record),
                    _extract_correct_flag(record),
                )

    for target in run_targets:
        target_id = str(target.get("target_id", "")).strip()
        target_root = run_root / "targets" / target_id
        record_root = target_root / "records"
        artifact_root = target_root / "artifacts"
        if not record_root.exists():
            continue
        label = str(target.get("label", target_id)).strip() or target_id
        system_type = str(target.get("target_type", target.get("mode", "unknown"))).strip() or "unknown"
        for record_path in sorted(record_root.glob("*/case_execution_record.json")):
            case_id = record_path.parent.name
            baseline_prediction, baseline_correct = baseline_index.get(case_id, ("", None))
            evidence_package_path = artifact_root / case_id / "evidence_package.json"
            reflection_path = artifact_root / case_id / "reflection.json"
            state_path = artifact_root / case_id / "state.json"
            entry = _build_entry(
                source_root=run_root,
                record_path=record_path,
                evidence_package_path=evidence_package_path if evidence_package_path.exists() else None,
                reflection_path=reflection_path if reflection_path.exists() else None,
                state_path=state_path if state_path.exists() else None,
                system_name=label,
                target_id=target_id,
                system_type=system_type,
                direct_baseline_prediction=baseline_prediction,
                direct_baseline_correct=baseline_correct,
                rules=rules,
            )
            entries.append(entry)
    return entries


def _collect_manifest_entries(root: Path, rules: dict[str, Any]) -> list[AuditEntry]:
    manifest_path = root / "subset_manifest.json"
    manifest = _safe_load_json(manifest_path)
    entries: list[AuditEntry] = []
    for item in list(manifest.get("cases", []) or []):
        case_dir = root / str(item.get("relative_case_dir", "")).strip()
        if not case_dir.exists():
            continue
        entry = _build_entry(
            source_root=root,
            record_path=case_dir / "case_execution_record.json",
            evidence_package_path=case_dir / "evidence_package.json",
            reflection_path=case_dir / "reflection.json",
            state_path=case_dir / "state.json",
            system_name=str(item.get("system_name", manifest.get("system_name", root.name))).strip() or root.name,
            target_id=str(item.get("target_id", manifest.get("target_id", "generic_bundle"))).strip() or "generic_bundle",
            system_type=str(item.get("system_type", manifest.get("system_type", "generic_bundle"))).strip() or "generic_bundle",
            rules=rules,
        )
        entries.append(entry)
    return entries


def _collect_compare_manifest_entries(root: Path, rules: dict[str, Any]) -> list[AuditEntry]:
    manifest = _safe_load_json(root / "compare_run_manifest.json")
    entries: list[AuditEntry] = []
    for item in list(manifest.get("extracted_run_dirs", []) or []):
        run_root = root / str(item.get("relative_run_dir", "")).strip()
        if not run_root.exists():
            continue
        entries.extend(_collect_compare_run_entries(run_root, rules))
    return entries


def _collect_generic_case_tree_entries(root: Path, rules: dict[str, Any]) -> list[AuditEntry]:
    entries: list[AuditEntry] = []
    for record_path in sorted(root.rglob("case_execution_record.json")):
        case_dir = record_path.parent
        relative_parts = case_dir.relative_to(root).parts
        system_name = root.name
        target_id = "generic_bundle"
        system_type = "generic_bundle"
        if len(relative_parts) >= 3:
            system_name = relative_parts[0]
            target_id = relative_parts[0]
            system_type = relative_parts[0]
        entry = _build_entry(
            source_root=root,
            record_path=record_path,
            evidence_package_path=case_dir / "evidence_package.json",
            reflection_path=case_dir / "reflection.json",
            state_path=case_dir / "state.json",
            system_name=system_name,
            target_id=target_id,
            system_type=system_type,
            rules=rules,
        )
        entries.append(entry)
    return entries


def collect_audit_entries(input_path: Path, *, rules: dict[str, Any] | None = None) -> list[AuditEntry]:
    resolved = input_path.resolve()
    scoring_rules = rules or load_rules()
    if (resolved / "compare_run_manifest.json").exists():
        return _collect_compare_manifest_entries(resolved, scoring_rules)
    if (resolved / "evaluation_manifest.json").exists():
        return _collect_compare_run_entries(resolved, scoring_rules)
    if (resolved / "subset_manifest.json").exists():
        return _collect_manifest_entries(resolved, scoring_rules)
    return _collect_generic_case_tree_entries(resolved, scoring_rules)


def _non_unknown_fields(payload: dict[str, Any], keys: Iterable[str], rules: dict[str, Any]) -> bool:
    for key in keys:
        if _is_meaningful(payload.get(key), rules=rules):
            return True
    return False


def _detect_evidence_categories(entry: AuditEntry, rules: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    evidence_package = dict(entry.evidence_package or {})
    skill_outputs = dict(evidence_package.get("skill_outputs", {}) or {})
    morphology = dict(skill_outputs.get("morphology_analysis_skill", {}) or {})
    color = dict(skill_outputs.get("color_pattern_analysis_skill", {}) or {})
    distribution = dict(skill_outputs.get("distribution_analysis_skill", {}) or {})
    lesion = dict(skill_outputs.get("lesion_description_structuring_skill", {}) or {})
    metadata = dict(skill_outputs.get("metadata_consistency_skill", {}) or {})
    qwen_final = dict(entry.record.get("qwen_final", {}) or {})
    baseline_qwen = dict(entry.record.get("baseline_qwen", {}) or {})
    evidence_texts = _scoring_text_fragments(entry)
    keywords = dict(rules.get("evidence_category_keywords", {}) or {})

    uncertainty_summary = dict(evidence_package.get("uncertainty_summary", {}) or {})
    contradiction_summary = dict(evidence_package.get("contradiction_summary", {}) or {})
    information_gap = dict(evidence_package.get("information_gap_summary", {}) or {})
    selected_evidence = list(evidence_package.get("selected_evidence", []) or [])
    differentials = list(qwen_final.get("differential_diagnoses", []) or []) + list(
        baseline_qwen.get("differential_diagnoses", []) or []
    )

    categories = {
        "lesion_morphology": (
            _non_unknown_fields(morphology, ("lesion_type", "size_range", "elevation", "count"), rules)
            or _non_unknown_fields(lesion, ("primary_lesion_morphology", "border", "surface", "size_count"), rules)
            or bool(_keyword_hits(evidence_texts, keywords.get("lesion_morphology", [])))
        ),
        "color_or_pigmentation": (
            _non_unknown_fields(color, ("primary_color", "color_variation", "pigmentation_pattern", "asymmetry_color"), rules)
            or _non_unknown_fields(lesion, ("color",), rules)
            or bool(_keyword_hits(evidence_texts, keywords.get("color_or_pigmentation", [])))
        ),
        "anatomical_site_or_context": (
            _non_unknown_fields(distribution, ("body_location", "symmetry", "localized_vs_generalized", "clustering_pattern"), rules)
            or _non_unknown_fields(lesion, ("distribution", "associated_context"), rules)
            or bool(_keyword_hits(evidence_texts, keywords.get("anatomical_site_or_context", [])))
        ),
        "differential_diagnosis": (
            bool(differentials)
            or any(str(item.get("category", "")).strip() == "differential_support" for item in selected_evidence if isinstance(item, dict))
            or bool(_keyword_hits(evidence_texts, keywords.get("differential_diagnosis", [])))
        ),
        "malignant_risk_cues": (
            bool(evidence_package.get("risk_flags"))
            or bool(_keyword_hits(evidence_texts, keywords.get("malignant_risk_cues", [])))
        ),
        "uncertainty_or_missing_info": (
            _is_meaningful(uncertainty_summary, rules=rules)
            or _is_meaningful(contradiction_summary, rules=rules)
            or _is_meaningful(information_gap, rules=rules)
            or bool(_keyword_hits(evidence_texts, keywords.get("uncertainty_or_missing_info", [])))
        ),
    }
    all_text_fragments = evidence_texts + _flatten_strings(selected_evidence) + _flatten_strings(uncertainty_summary) + _flatten_strings(contradiction_summary)
    return categories, all_text_fragments


def _detect_contradiction_signals(
    entry: AuditEntry,
    rules: dict[str, Any],
    text_fragments: list[str],
) -> dict[str, bool]:
    evidence_package = dict(entry.evidence_package or {})
    selected_evidence = list(evidence_package.get("selected_evidence", []) or [])
    contradiction_summary = dict(evidence_package.get("contradiction_summary", {}) or {})
    uncertainty_summary = dict(evidence_package.get("uncertainty_summary", {}) or {})
    skill_outputs = dict(evidence_package.get("skill_outputs", {}) or {})
    metadata = dict(skill_outputs.get("metadata_consistency_skill", {}) or {})
    qwen_final = dict(entry.record.get("qwen_final", {}) or {})
    baseline_qwen = dict(entry.record.get("baseline_qwen", {}) or {})
    alternative_pool = list(qwen_final.get("differential_diagnoses", []) or []) + list(
        baseline_qwen.get("differential_diagnoses", []) or []
    )
    contradiction_keywords = list(rules.get("contradiction_keywords", []))
    support_text = " ".join(text_fragments) or entry.evidence_text or _json_dumps(qwen_final)
    uncertainty_keywords = list(rules.get("evidence_category_keywords", {}).get("uncertainty_or_missing_info", []))

    return {
        "support_evidence": bool(selected_evidence) or len(_normalize_text(support_text)) > 40,
        "conflicting_evidence": (
            _is_meaningful(contradiction_summary, rules=rules)
            or _is_meaningful(metadata.get("conflicts"), rules=rules)
            or _is_meaningful(metadata.get("suspicious_points"), rules=rules)
            or bool(_keyword_hits(text_fragments, contradiction_keywords))
        ),
        "uncertainty_statement": _is_meaningful(uncertainty_summary, rules=rules) or bool(
            _keyword_hits(text_fragments, uncertainty_keywords)
        ),
        "alternative_diagnosis": len([item for item in alternative_pool if str(item or "").strip()]) >= 2,
    }


def _detect_risk_cues(
    entry: AuditEntry,
    rules: dict[str, Any],
    text_fragments: list[str],
) -> tuple[list[str], list[str]]:
    expected_terms = _find_expected_risk_terms(entry.ground_truth, rules)
    keywords = expected_terms or list(rules.get("evidence_category_keywords", {}).get("malignant_risk_cues", []))
    detected_terms = _keyword_hits(text_fragments, keywords)
    risk_flags = [str(item) for item in list(entry.evidence_package.get("risk_flags", []) or [])]
    detected_terms.extend(risk_flags)
    return sorted(set(expected_terms)), sorted(set(detected_terms))


def _detect_error_signals(
    entry: AuditEntry,
    rules: dict[str, Any],
    categories: dict[str, bool],
    contradiction_signals: dict[str, bool],
    text_fragments: list[str],
) -> dict[str, bool]:
    reflection = dict(entry.reflection or {})
    case_outcome = dict(reflection.get("case_outcome", {}) or {})
    confusion_pair = str(case_outcome.get("confusion_pair", "")).strip()
    evidence_package = dict(entry.evidence_package or {})
    uncertainty_summary = dict(evidence_package.get("uncertainty_summary", {}) or {})
    information_gap = dict(evidence_package.get("information_gap_summary", {}) or {})
    skill_outputs = dict(evidence_package.get("skill_outputs", {}) or {})
    meaningful_skill_payload = any(
        _is_meaningful(payload, rules=rules)
        for payload in list(skill_outputs.values())
        if isinstance(payload, dict)
    )
    qwen_final = dict(entry.record.get("qwen_final", {}) or {})
    differential_count = len([item for item in list(qwen_final.get("differential_diagnoses", []) or []) if str(item).strip()])
    correct = bool(entry.correct) if entry.correct is not None else False
    predicted_high_risk = _extract_high_risk_flag(entry.record, rules, entry.final_prediction)
    completeness_value = sum(1 for value in categories.values() if value) / 6.0

    return {
        "error_visual_misread": (
            not correct
            and not categories["lesion_morphology"]
            and not categories["color_or_pigmentation"]
            and not categories["anatomical_site_or_context"]
        ),
        "error_confusion_pair": bool(confusion_pair)
        or (not correct and _canonical_label_token(entry.final_prediction) != _canonical_label_token(entry.ground_truth)),
        "error_risk_downgrade": entry.is_malignant_or_high_risk and not predicted_high_risk,
        "error_premature_closure": not correct and differential_count <= 1 and not contradiction_signals["uncertainty_statement"],
        "error_missing_cue": (
            _is_meaningful(uncertainty_summary.get("missing_information"), rules=rules)
            or _is_meaningful(information_gap.get("missing_information"), rules=rules)
            or contradiction_signals["conflicting_evidence"]
            or (not correct and completeness_value < 0.5 and not meaningful_skill_payload)
        ),
    }


def _case_group(entry: AuditEntry) -> str:
    if entry.direct_baseline_correct is False and entry.correct is True:
        return "baseline_error_corrected"
    if entry.direct_baseline_correct is False and entry.correct is False:
        return "baseline_error_not_corrected"
    if entry.is_malignant_or_high_risk:
        return "high_risk_case"
    if entry.correct is True:
        return "correct_case"
    if entry.correct is False:
        return "error_case"
    return "unlabeled_case"


def normalize_audit_input_row(entry: AuditEntry) -> dict[str, Any]:
    return {
        "case_id": entry.case_id,
        "dataset": entry.dataset,
        "system_name": entry.system_name,
        "target_id": entry.target_id,
        "system_type": entry.system_type,
        "ground_truth": entry.ground_truth,
        "final_prediction": entry.final_prediction,
        "initial_prediction": entry.initial_prediction,
        "direct_baseline_prediction": entry.direct_baseline_prediction,
        "correct": entry.correct,
        "direct_baseline_correct": entry.direct_baseline_correct,
        "is_malignant_or_high_risk": entry.is_malignant_or_high_risk,
        "source_root": entry.source_root,
        "record_path": entry.record_path,
        "evidence_package_path": entry.evidence_package_path,
        "reflection_path": entry.reflection_path,
        "state_path": entry.state_path,
        "evidence_text": entry.evidence_text,
    }


def score_entries(entries: list[AuditEntry], rules: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        categories, text_fragments = _detect_evidence_categories(entry, rules)
        contradiction_signals = _detect_contradiction_signals(entry, rules, text_fragments)
        expected_risk_terms, detected_risk_terms = _detect_risk_cues(entry, rules, text_fragments)
        error_signals = _detect_error_signals(entry, rules, categories, contradiction_signals, text_fragments)

        evidence_completeness = sum(1 for value in categories.values() if value) / 6.0
        contradiction_awareness = sum(1 for value in contradiction_signals.values() if value) / 4.0
        risk_cue_preservation = None
        if expected_risk_terms:
            overlap = len({item.lower() for item in detected_risk_terms} & {item.lower() for item in expected_risk_terms})
            risk_cue_preservation = overlap / max(len(expected_risk_terms), 1)
        error_localization = sum(1 for value in error_signals.values() if value) / 5.0
        metrics = [evidence_completeness, contradiction_awareness, error_localization]
        if risk_cue_preservation is not None:
            metrics.append(risk_cue_preservation)
        overall = sum(metrics) / len(metrics)

        extracted_evidence = {
            "evidence_categories": categories,
            "contradiction_signals": contradiction_signals,
            "risk_expected_terms": expected_risk_terms,
            "risk_detected_terms": detected_risk_terms,
            "error_signals": error_signals,
        }

        rows.append(
            {
                "case_id": entry.case_id,
                "dataset": entry.dataset,
                "case_group": _case_group(entry),
                "system_name": entry.system_name,
                "target_id": entry.target_id,
                "system_type": entry.system_type,
                "ground_truth": entry.ground_truth,
                "final_prediction": entry.final_prediction,
                "correct": entry.correct,
                "initial_prediction": entry.initial_prediction,
                "direct_baseline_prediction": entry.direct_baseline_prediction,
                "direct_baseline_correct": entry.direct_baseline_correct,
                "correction_success": int(entry.direct_baseline_correct is False and entry.correct is True),
                "is_malignant_or_high_risk": entry.is_malignant_or_high_risk,
                "evidence_completeness": round(evidence_completeness, 4),
                "contradiction_awareness": round(contradiction_awareness, 4),
                "risk_cue_preservation": "" if risk_cue_preservation is None else round(risk_cue_preservation, 4),
                "error_localization": round(error_localization, 4),
                "overall_auditability": round(overall, 4),
                "risk_cue_detected_count": len(detected_risk_terms),
                "risk_cue_expected_count": len(expected_risk_terms),
                "error_missing_cue": int(error_signals["error_missing_cue"]),
                "error_confusion_pair": int(error_signals["error_confusion_pair"]),
                "error_risk_downgrade": int(error_signals["error_risk_downgrade"]),
                "error_premature_closure": int(error_signals["error_premature_closure"]),
                "error_visual_misread": int(error_signals["error_visual_misread"]),
                "record_path": entry.record_path,
                "evidence_package_path": entry.evidence_package_path,
                "reflection_path": entry.reflection_path,
                "source_root": entry.source_root,
                "extracted_evidence_json": _json_dumps(extracted_evidence),
            }
        )
    return rows


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[str(row.get("system_name", ""))].append(row)

    system_summaries: list[dict[str, Any]] = []
    for system_name, system_rows in sorted(by_system.items()):
        def mean(field: str) -> float | None:
            values: list[float] = []
            for row in system_rows:
                value = row.get(field)
                if value == "" or value is None:
                    continue
                values.append(float(value))
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        system_summaries.append(
            {
                "system_name": system_name,
                "rows": len(system_rows),
                "mean_evidence_completeness": mean("evidence_completeness"),
                "mean_contradiction_awareness": mean("contradiction_awareness"),
                "mean_risk_cue_preservation": mean("risk_cue_preservation"),
                "mean_error_localization": mean("error_localization"),
                "mean_overall_auditability": mean("overall_auditability"),
                "mean_correction_success": mean("correction_success"),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[(str(row.get("dataset", "")), str(row.get("case_id", "")))].append(row)
    for (dataset, case_id), case_rows in sorted(by_case.items()):
        if len(case_rows) < 2:
            continue
        ordered = sorted(case_rows, key=lambda item: float(item.get("overall_auditability", 0.0)), reverse=True)
        best = ordered[0]
        worst = ordered[-1]
        pairwise_rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "best_system": best["system_name"],
                "best_overall_auditability": best["overall_auditability"],
                "worst_system": worst["system_name"],
                "worst_overall_auditability": worst["overall_auditability"],
                "delta": round(float(best["overall_auditability"]) - float(worst["overall_auditability"]), 4),
            }
        )

    return {
        "version": AUDITABILITY_VERSION,
        "generated_at": utc_now(),
        "num_rows": len(rows),
        "num_systems": len(by_system),
        "system_summaries": system_summaries,
        "pairwise_case_comparisons": pairwise_rows,
    }


def render_summary_markdown(summary: dict[str, Any], inputs: list[Path]) -> str:
    lines = [
        "# Evidence Auditability Summary",
        "",
        f"- Generated at: `{summary.get('generated_at', '')}`",
        f"- Total case-system rows: `{summary.get('num_rows', 0)}`",
        f"- Systems: `{summary.get('num_systems', 0)}`",
        "",
        "## Inputs",
    ]
    for item in inputs:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## System Means",
            "",
            "| system_name | rows | overall | completeness | contradiction | risk cue | error localization | correction success |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in list(summary.get("system_summaries", []) or []):
        lines.append(
            "| {system_name} | {rows} | {mean_overall_auditability} | {mean_evidence_completeness} | {mean_contradiction_awareness} | {mean_risk_cue_preservation} | {mean_error_localization} | {mean_correction_success} |".format(
                system_name=item.get("system_name", ""),
                rows=item.get("rows", 0),
                mean_overall_auditability=item.get("mean_overall_auditability", ""),
                mean_evidence_completeness=item.get("mean_evidence_completeness", ""),
                mean_contradiction_awareness=item.get("mean_contradiction_awareness", ""),
                mean_risk_cue_preservation=item.get("mean_risk_cue_preservation", ""),
                mean_error_localization=item.get("mean_error_localization", ""),
                mean_correction_success=item.get("mean_correction_success", ""),
            )
        )
    pairwise = list(summary.get("pairwise_case_comparisons", []) or [])
    if pairwise:
        display_pairwise = pairwise[:50]
        lines.extend(
            [
                "",
                "## Same-Case Winners",
                "",
                f"- Total paired comparisons: `{len(pairwise)}`",
                f"- Showing first `{len(display_pairwise)}` rows below.",
                "",
                "| dataset | case_id | best_system | best_overall | worst_system | worst_overall | delta |",
                "|---|---|---|---:|---|---:|---:|",
            ]
        )
        for item in display_pairwise:
            lines.append(
                "| {dataset} | {case_id} | {best_system} | {best_overall_auditability} | {worst_system} | {worst_overall_auditability} | {delta} |".format(
                    **item
                )
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The scorer uses structured fields first and falls back to keyword matching only when needed.",
            "- `risk_cue_preservation` is left blank on cases whose ground-truth label does not match the configured high-risk groups.",
            "- `correction_success` is only `1` when a system turns a direct-baseline error into a correct final diagnosis.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_run(
    *,
    input_paths: list[Path],
    output_root: Path,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoring_rules = rules or load_rules()
    all_entries: list[AuditEntry] = []
    for input_path in input_paths:
        all_entries.extend(collect_audit_entries(input_path, rules=scoring_rules))

    audit_input_rows = [normalize_audit_input_row(entry) for entry in all_entries]
    score_rows = score_entries(all_entries, scoring_rules)
    summary = summarize_scores(score_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "audit_inputs.jsonl", audit_input_rows)
    write_csv(output_root / "audit_scores.csv", score_rows)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    (output_root / "summary.md").write_text(render_summary_markdown(summary, input_paths), encoding="utf-8")
    if summary.get("pairwise_case_comparisons"):
        write_csv(output_root / "pairwise_case_comparisons.csv", list(summary["pairwise_case_comparisons"]))

    manifest = {
        "version": AUDITABILITY_VERSION,
        "generated_at": utc_now(),
        "input_paths": [str(path.resolve()) for path in input_paths],
        "output_root": str(output_root.resolve()),
        "num_entries": len(all_entries),
        "artifacts": {
            "audit_inputs_jsonl": str((output_root / "audit_inputs.jsonl").resolve()),
            "audit_scores_csv": str((output_root / "audit_scores.csv").resolve()),
            "summary_json": str((output_root / "summary.json").resolve()),
            "summary_md": str((output_root / "summary.md").resolve()),
        },
    }
    with (output_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def extract_tar_subset(
    *,
    tar_path: Path,
    output_root: Path,
    system_segment: str = "final-boostrap",
    dataset_segment: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected_case_members: dict[str, dict[str, tarfile.TarInfo]] = {}
    selected_case_names: list[str] = []

    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            parts = member.name.split("/")
            if len(parts) < 5:
                continue
            system_name = parts[1]
            dataset_name = parts[2]
            case_id = parts[3]
            filename = parts[4]
            if system_name != system_segment:
                continue
            if dataset_segment and dataset_name != dataset_segment:
                continue
            if filename not in WANTED_CASE_FILES:
                continue
            if case_id not in selected_case_members:
                if len(selected_case_members) >= limit:
                    break
                selected_case_members[case_id] = {}
                selected_case_names.append(case_id)
            selected_case_members[case_id][filename] = member

        extracted_cases: list[dict[str, Any]] = []
        for case_id in selected_case_names:
            case_dir = output_root / system_segment / (dataset_segment or dataset_name) / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            member_map = selected_case_members.get(case_id, {})
            for filename, member in member_map.items():
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                (case_dir / filename).write_bytes(extracted.read())

            record = _safe_load_json(case_dir / "case_execution_record.json")
            extracted_cases.append(
                {
                    "case_id": case_id,
                    "dataset": str(record.get("dataset_name", "")).strip() or (dataset_segment or ""),
                    "ground_truth": _extract_ground_truth(record),
                    "final_prediction": _extract_final_prediction(record),
                    "relative_case_dir": str(case_dir.relative_to(output_root)),
                    "system_name": system_segment,
                    "target_id": system_segment,
                    "system_type": system_segment,
                    "files": sorted(member_map.keys()),
                }
            )

    manifest = {
        "version": "auditability_subset_v1",
        "generated_at": utc_now(),
        "tar_path": str(tar_path.resolve()),
        "output_root": str(output_root.resolve()),
        "system_name": system_segment,
        "target_id": system_segment,
        "system_type": system_segment,
        "dataset_segment": dataset_segment,
        "limit": limit,
        "cases": extracted_cases,
    }
    with (output_root / "subset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def extract_compare_runs_from_tar(
    *,
    tar_path: Path,
    output_root: Path,
    root_segment: str = "final-data",
    dataset: str = "pad",
    experiment: str = "hulumed_agent",
    model: str = "hulumed",
    dataset_variant: str = "pad20",
    require_targets: tuple[str, ...] = ("direct_baseline", "full_dermagent"),
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    run_case_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: {target: set() for target in require_targets})
    run_meta: dict[str, dict[str, Any]] = {}

    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if not member.name.endswith("case_execution_record.json"):
                continue
            parts = member.name.split("/")
            if len(parts) < 8 or parts[1] != root_segment:
                continue
            if parts[2] != dataset or parts[3] != experiment:
                continue
            if parts[6] != model or parts[7] != dataset_variant:
                continue
            if "targets" not in parts:
                continue
            target_index = parts.index("targets")
            if target_index + 4 >= len(parts):
                continue
            target = parts[target_index + 1]
            lane = parts[target_index + 2]
            if target not in require_targets:
                continue
            if lane not in {"records", "artifacts"}:
                continue
            if target == "full_dermagent" and lane != "records":
                continue
            compare_index = -1
            for index, segment in enumerate(parts):
                if segment.startswith("compare_agent_vs_qwen_"):
                    compare_index = index
                    break
            if compare_index == -1:
                continue
            run_root = "/".join(parts[1 : compare_index + 1])
            case_id = parts[target_index + 3]
            run_case_map[run_root][target].add(case_id)
            if run_root not in run_meta:
                run_meta[run_root] = {
                    "root_segment": parts[1],
                    "dataset": parts[2],
                    "experiment": parts[3],
                    "machine": parts[5] if len(parts) > 5 else "",
                    "model": parts[6] if len(parts) > 6 else "",
                    "dataset_variant": parts[7] if len(parts) > 7 else "",
                    "shard": parts[9] if len(parts) > 9 and parts[8] == "compare_shards" else "",
                    "compare_run_id": parts[compare_index],
                }

    selected_runs: dict[str, dict[str, Any]] = {}
    for run_root, payload in run_case_map.items():
        paired_cases = set.intersection(*(payload[target] for target in require_targets)) if require_targets else set()
        if not paired_cases:
            continue
        meta = run_meta.get(run_root, {})
        shard = str(meta.get("shard", "")).strip() or "shard"
        compare_run_id = str(meta.get("compare_run_id", "")).strip() or "compare_run"
        relative_run_dir = f"runs/{slugify(dataset)}_{slugify(model)}_{slugify(shard)}_{slugify(compare_run_id)}"
        selected_runs[run_root] = {
            **meta,
            "relative_run_dir": relative_run_dir,
            "paired_cases": sorted(paired_cases),
            "paired_case_count": len(paired_cases),
            "per_target_case_counts": {target: len(payload[target]) for target in require_targets},
        }

    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            parts = member.name.split("/")
            if len(parts) < 2 or parts[1] != root_segment:
                continue
            member_relative = "/".join(parts[1:])
            matched_run_root = None
            for run_root in selected_runs:
                if member_relative == run_root or member_relative.startswith(run_root + "/"):
                    matched_run_root = run_root
                    break
            if not matched_run_root:
                continue

            should_extract = False
            if member_relative.endswith("evaluation_manifest.json") or member_relative.endswith("result_manifest.json"):
                should_extract = True
            elif "/targets/direct_baseline/summary.json" in member_relative or "/targets/full_dermagent/summary.json" in member_relative:
                should_extract = True
            elif member_relative.endswith("case_execution_records.jsonl"):
                should_extract = True
            elif "/targets/direct_baseline/records/" in member_relative and member_relative.endswith("/case_execution_record.json"):
                case_id = parts[-2]
                should_extract = case_id in selected_runs[matched_run_root]["paired_cases"]
            elif "/targets/full_dermagent/records/" in member_relative and member_relative.endswith("/case_execution_record.json"):
                case_id = parts[-2]
                should_extract = case_id in selected_runs[matched_run_root]["paired_cases"]
            elif "/targets/full_dermagent/artifacts/" in member_relative:
                case_id = parts[-2]
                filename = parts[-1]
                should_extract = (
                    case_id in selected_runs[matched_run_root]["paired_cases"]
                    and filename in {"case_execution_record.json", "state.json", "evidence_package.json", "reflection.json"}
                )

            if not should_extract:
                continue
            relative_run_dir = str(selected_runs[matched_run_root]["relative_run_dir"]).strip()
            suffix = member_relative[len(matched_run_root) :].lstrip("/")
            destination = output_root / relative_run_dir / suffix
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            destination.write_bytes(extracted.read())

    manifest = {
        "version": "compare_run_extract_v1",
        "generated_at": utc_now(),
        "tar_path": str(tar_path.resolve()),
        "output_root": str(output_root.resolve()),
        "filters": {
            "root_segment": root_segment,
            "dataset": dataset,
            "experiment": experiment,
            "model": model,
            "dataset_variant": dataset_variant,
            "require_targets": list(require_targets),
        },
        "aggregate": {
            "num_runs": len(selected_runs),
            "total_paired_cases": sum(int(item["paired_case_count"]) for item in selected_runs.values()),
            "max_cases_per_run": max((int(item["paired_case_count"]) for item in selected_runs.values()), default=0),
            "min_cases_per_run": min((int(item["paired_case_count"]) for item in selected_runs.values()), default=0),
        },
        "extracted_run_dirs": [
            {
                "original_run_root": run_root,
                **meta,
            }
            for run_root, meta in sorted(selected_runs.items())
        ],
    }
    with (output_root / "compare_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
