from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


CALIBRATOR_VERSION = "v1"

DEFAULT_EVIDENCE_POLICY = {
    "enable_evidence_calibrator": True,
    "calibrator_mode": "heuristic",  # off | heuristic | learned | hybrid
    "calibrator_checkpoint_path": "",
    "learned_calibration_weight": 1.2,
    "max_observation_skills": 4,
    "max_comparison_skills": 3,
    "max_risk_skills": 2,
    "max_uncertainty_skills": 3,
    "max_raw_cases": 1,
    "max_tactical": 2,
    "max_abstract": 2,
    "max_planner_reasons": 3,
    "omit_low_value_evidence": True,
    "min_effective_score": 2.4,
    "debug_output": True,
}


SKILL_SECTION_MAP = {
    "lesion_description_structuring_skill": "observation",
    "morphology_analysis_skill": "observation",
    "color_pattern_analysis_skill": "observation",
    "border_surface_analysis_skill": "observation",
    "distribution_analysis_skill": "observation",
    "temporal_evolution_skill": "observation",
    "metadata_consistency_skill": "comparison",
    "differential_compare_skill": "comparison",
    "exclusion_reasoning_skill": "comparison",
    "mel_nev_specialist_skill": "comparison",
    "ack_scc_specialist_skill": "comparison",
    "malignancy_risk_assessment_skill": "risk",
    "contradiction_check_skill": "conflict_uncertainty",
    "uncertainty_assessment_skill": "conflict_uncertainty",
    "information_gap_detection_skill": "conflict_uncertainty",
    "escalation_recommendation_skill": "conflict_uncertainty",
}

SKILL_BASE_WEIGHT = {
    "lesion_description_structuring_skill": 4.8,
    "morphology_analysis_skill": 4.1,
    "color_pattern_analysis_skill": 4.0,
    "border_surface_analysis_skill": 3.9,
    "distribution_analysis_skill": 3.6,
    "temporal_evolution_skill": 2.8,
    "metadata_consistency_skill": 3.0,
    "differential_compare_skill": 3.6,
    "exclusion_reasoning_skill": 4.3,
    "mel_nev_specialist_skill": 3.2,
    "ack_scc_specialist_skill": 3.8,
    "malignancy_risk_assessment_skill": 4.5,
    "contradiction_check_skill": 4.0,
    "uncertainty_assessment_skill": 3.8,
    "information_gap_detection_skill": 3.8,
    "escalation_recommendation_skill": 3.5,
}

EVIDENCE_STRENGTH_WEIGHT = {
    "high": 2.0,
    "medium": 1.1,
    "low": 0.2,
    "unknown": 0.0,
}

RECOMMENDATION_WEIGHT = {
    "risk_signal": 1.4,
    "comparative_support": 1.2,
    "uncertainty_signal": 1.2,
    "conflict_alert": 1.3,
    "descriptive_evidence": 0.9,
}

CRITICAL_SKILLS = {
    "lesion_description_structuring_skill",
    "morphology_analysis_skill",
    "exclusion_reasoning_skill",
    "malignancy_risk_assessment_skill",
    "uncertainty_assessment_skill",
    "contradiction_check_skill",
    "information_gap_detection_skill",
}

SECTION_PRIORITY = ("observation", "comparison", "risk", "conflict_uncertainty", "planner")


@dataclass
class CalibrationItemScore:
    item_id: str
    item_type: str
    section: str
    heuristic_score: float
    learned_score: float
    final_score: float
    selected: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceCalibrationOutput:
    calibrator_type: str
    calibrator_version: str
    section_plan: dict[str, Any] = field(default_factory=dict)
    item_scores: list[dict[str, Any]] = field(default_factory=list)
    omitted_items: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _EvidenceCalibrationMLP(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Any) -> Any:
        return self.layers(x).squeeze(-1)


class LearnedEvidenceCalibrationScorer:
    def __init__(self, checkpoint_path: str | Path) -> None:
        if torch is None or nn is None:
            raise RuntimeError("torch is unavailable for learned evidence calibration scorer.")
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.checkpoint_path = str(checkpoint_path)
        self.feature_vocab = dict(checkpoint.get("feature_vocab", {}))
        if not self.feature_vocab:
            raise ValueError(f"Invalid evidence calibrator checkpoint: {checkpoint_path}")
        input_dim = int(checkpoint.get("input_dim", len(self.feature_vocab)))
        hidden_dim = int(checkpoint.get("hidden_dim", 64))
        dropout = float(checkpoint.get("dropout", 0.1))
        self.model = _EvidenceCalibrationMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.model.load_state_dict(checkpoint.get("model_state_dict", {}))
        self.model.eval()

    def score(self, feature_map: dict[str, float]) -> float:
        if torch is None:
            return 0.0
        vector = torch.zeros((len(self.feature_vocab),), dtype=torch.float32)
        for key, value in feature_map.items():
            index = self.feature_vocab.get(key)
            if index is None:
                continue
            vector[index] = float(value)
        with torch.no_grad():
            logit = self.model(vector.unsqueeze(0))
            prob = torch.sigmoid(logit).item()
        return float(prob)


class EvidenceCalibrator:
    def __init__(self, *, policy: dict[str, Any] | None = None) -> None:
        self.policy = _normalize_evidence_policy(policy)
        self.mode = str(self.policy.get("calibrator_mode", "heuristic")).strip().lower() or "heuristic"
        self.learned_weight = float(self.policy.get("learned_calibration_weight", 1.2))
        self.learned_scorer: LearnedEvidenceCalibrationScorer | None = None
        self._load_learned_scorer_if_needed()

    def calibrate(self, calibration_input: dict[str, Any]) -> EvidenceCalibrationOutput:
        if not bool(self.policy.get("enable_evidence_calibrator", True)) or self.mode == "off":
            return EvidenceCalibrationOutput(
                calibrator_type="off",
                calibrator_version=CALIBRATOR_VERSION,
                section_plan=self._fallback_section_plan(calibration_input),
                item_scores=[],
                omitted_items=[],
                debug={"reason": "evidence calibrator disabled by policy"},
            )

        skill_outputs = dict(calibration_input.get("skill_outputs", {}))
        skill_retrieval_scores = dict(calibration_input.get("skill_retrieval_scores", {}))
        uncertainty_summary = dict(calibration_input.get("uncertainty_summary", {}))
        contradiction_summary = dict(calibration_input.get("contradiction_summary", {}))
        risk_flags = [str(item).strip() for item in calibration_input.get("risk_flags", []) if str(item).strip()]
        contradiction_count = _contradiction_count(contradiction_summary)
        uncertainty_level = str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower() or "unknown"

        item_scores: list[CalibrationItemScore] = []
        selected_skills_by_section: dict[str, list[tuple[str, float]]] = {section: [] for section in SECTION_PRIORITY}
        omitted_items: list[dict[str, Any]] = []
        for skill_name, output in skill_outputs.items():
            if not isinstance(output, dict) or not _has_meaningful_skill_output(output):
                continue
            section = SKILL_SECTION_MAP.get(skill_name, "comparison")
            heuristic = _score_skill_output(
                skill_name=skill_name,
                output=output,
                retrieval_score=float(skill_retrieval_scores.get(skill_name, 0.0) or 0.0),
                uncertainty_level=uncertainty_level,
                contradiction_count=contradiction_count,
                risk_flags=risk_flags,
            )
            learned = self._score_with_learned(
                _build_skill_feature_map(
                    skill_name=skill_name,
                    section=section,
                    output=output,
                    heuristic_score=heuristic,
                    uncertainty_level=uncertainty_level,
                    contradiction_count=contradiction_count,
                    risk_flags=risk_flags,
                )
            )
            final_score = heuristic + self.learned_weight * learned if self.mode == "hybrid" else (
                learned if self.mode == "learned" else heuristic
            )
            should_omit = (
                bool(self.policy.get("omit_low_value_evidence", True))
                and final_score < float(self.policy.get("min_effective_score", 2.4))
                and skill_name not in CRITICAL_SKILLS
            )
            if should_omit:
                omitted_items.append(
                    {
                        "item_id": skill_name,
                        "item_type": "skill_output",
                        "reason": "below_min_effective_score",
                        "score": round(final_score, 6),
                    }
                )
            else:
                selected_skills_by_section.setdefault(section, []).append((skill_name, final_score))
            item_scores.append(
                CalibrationItemScore(
                    item_id=skill_name,
                    item_type="skill_output",
                    section=section,
                    heuristic_score=round(heuristic, 6),
                    learned_score=round(learned, 6),
                    final_score=round(final_score, 6),
                    selected=not should_omit,
                    reason="kept" if not should_omit else "omitted_low_value",
                )
            )

        max_by_section = {
            "observation": int(self.policy.get("max_observation_skills", 4)),
            "comparison": int(self.policy.get("max_comparison_skills", 3)),
            "risk": int(self.policy.get("max_risk_skills", 2)),
            "conflict_uncertainty": int(self.policy.get("max_uncertainty_skills", 3)),
        }
        ordered_section_skills: dict[str, list[str]] = {}
        for section_name, rows in selected_skills_by_section.items():
            rows.sort(key=lambda item: (item[1], item[0]), reverse=True)
            if section_name in max_by_section:
                rows = rows[: max(0, max_by_section[section_name])]
            ordered_section_skills[section_name] = [skill_name for skill_name, _ in rows]

        # Keep critical evidence if present and section is empty.
        for critical in CRITICAL_SKILLS:
            if critical not in skill_outputs:
                continue
            section = SKILL_SECTION_MAP.get(critical, "comparison")
            if critical not in ordered_section_skills.get(section, []):
                if len(ordered_section_skills.get(section, [])) < max_by_section.get(section, 99):
                    ordered_section_skills.setdefault(section, []).append(critical)

        raw_records = list(calibration_input.get("retrieved_raw_cases_summary", []))
        tactical_records = list(calibration_input.get("retrieved_tactical_experiences_summary", []))
        abstract_records = list(calibration_input.get("retrieved_abstract_experiences_summary", []))
        raw_selected, raw_item_scores = self._select_retrieval_records(
            records=raw_records,
            source_layer="raw_case_memory",
            top_k=int(self.policy.get("max_raw_cases", 1)),
            uncertainty_level=uncertainty_level,
            contradiction_count=contradiction_count,
            risk_flags=risk_flags,
        )
        tactical_selected, tactical_item_scores = self._select_retrieval_records(
            records=tactical_records,
            source_layer="tactical_experience",
            top_k=int(self.policy.get("max_tactical", 2)),
            uncertainty_level=uncertainty_level,
            contradiction_count=contradiction_count,
            risk_flags=risk_flags,
        )
        abstract_selected, abstract_item_scores = self._select_retrieval_records(
            records=abstract_records,
            source_layer="abstract_experience",
            top_k=int(self.policy.get("max_abstract", 2)),
            uncertainty_level=uncertainty_level,
            contradiction_count=contradiction_count,
            risk_flags=risk_flags,
        )
        item_scores.extend(raw_item_scores)
        item_scores.extend(tactical_item_scores)
        item_scores.extend(abstract_item_scores)

        section_plan = {
            "observation": {
                "skill_names": ordered_section_skills.get("observation", []),
                "merged_groups": _observation_merged_groups(ordered_section_skills.get("observation", [])),
                "raw_case_source_ids": [item.get("source_id") for item in raw_selected if item.get("source_id")],
            },
            "comparison": {
                "skill_names": ordered_section_skills.get("comparison", []),
                "merged_groups": _comparison_merged_groups(ordered_section_skills.get("comparison", [])),
                "tactical_source_ids": [item.get("source_id") for item in tactical_selected if item.get("source_id")],
            },
            "risk": {
                "skill_names": ordered_section_skills.get("risk", []),
                "abstract_source_ids": [item.get("source_id") for item in abstract_selected if item.get("source_id")],
            },
            "conflict_uncertainty": {
                "skill_names": ordered_section_skills.get("conflict_uncertainty", []),
            },
            "planner": {
                "max_reason_skills": int(self.policy.get("max_planner_reasons", 3)),
            },
            "section_order": list(SECTION_PRIORITY),
        }
        return EvidenceCalibrationOutput(
            calibrator_type=self._effective_mode(),
            calibrator_version=CALIBRATOR_VERSION,
            section_plan=section_plan,
            item_scores=[item.to_dict() for item in sorted(item_scores, key=lambda item: item.final_score, reverse=True)],
            omitted_items=omitted_items,
            debug={
                "policy": deepcopy(self.policy),
                "learned_scorer_loaded": self.learned_scorer is not None,
                "contradiction_count": contradiction_count,
                "uncertainty_level": uncertainty_level,
                "risk_flag_count": len(risk_flags),
            },
        )

    def _select_retrieval_records(
        self,
        *,
        records: list[dict[str, Any]],
        source_layer: str,
        top_k: int,
        uncertainty_level: str,
        contradiction_count: int,
        risk_flags: list[str],
    ) -> tuple[list[dict[str, Any]], list[CalibrationItemScore]]:
        scored: list[tuple[dict[str, Any], float, float, float]] = []
        rows: list[CalibrationItemScore] = []
        for record in records:
            source_id = str(record.get("source_id", "")).strip()
            if not source_id:
                continue
            heuristic = _score_retrieval_record(
                record=record,
                source_layer=source_layer,
                uncertainty_level=uncertainty_level,
                contradiction_count=contradiction_count,
                risk_flags=risk_flags,
            )
            learned = self._score_with_learned(
                _build_retrieval_feature_map(
                    record=record,
                    source_layer=source_layer,
                    heuristic_score=heuristic,
                    uncertainty_level=uncertainty_level,
                    contradiction_count=contradiction_count,
                    risk_flags=risk_flags,
                )
            )
            final_score = heuristic + self.learned_weight * learned if self.mode == "hybrid" else (
                learned if self.mode == "learned" else heuristic
            )
            scored.append((record, final_score, heuristic, learned))
        scored.sort(
            key=lambda item: (
                item[1],
                float(item[0].get("retrieval_score", 0.0) or 0.0),
                str(item[0].get("source_id", "")),
            ),
            reverse=True,
        )
        selected_ids = {
            str(item[0].get("source_id", "")).strip()
            for item in scored[: max(0, top_k)]
            if str(item[0].get("source_id", "")).strip()
        }
        selected = [record for record, _, _, _ in scored if str(record.get("source_id", "")).strip() in selected_ids]
        for record, final_score, heuristic, learned in scored:
            source_id = str(record.get("source_id", "")).strip()
            rows.append(
                CalibrationItemScore(
                    item_id=source_id,
                    item_type="retrieval_record",
                    section=source_layer,
                    heuristic_score=round(heuristic, 6),
                    learned_score=round(learned, 6),
                    final_score=round(final_score, 6),
                    selected=source_id in selected_ids,
                    reason="selected" if source_id in selected_ids else "dropped_by_rank",
                )
            )
        return selected, rows

    def _load_learned_scorer_if_needed(self) -> None:
        if self.mode not in {"learned", "hybrid"}:
            self.learned_scorer = None
            return
        checkpoint_path = str(self.policy.get("calibrator_checkpoint_path", "")).strip()
        if not checkpoint_path:
            self.learned_scorer = None
            return
        try:
            self.learned_scorer = LearnedEvidenceCalibrationScorer(checkpoint_path=checkpoint_path)
        except Exception:
            self.learned_scorer = None

    def _score_with_learned(self, feature_map: dict[str, float]) -> float:
        if self.learned_scorer is None:
            return 0.0
        try:
            return self.learned_scorer.score(feature_map)
        except Exception:
            return 0.0

    def _effective_mode(self) -> str:
        if self.mode in {"learned", "hybrid"} and self.learned_scorer is None:
            return "heuristic_fallback"
        return self.mode

    @staticmethod
    def _fallback_section_plan(calibration_input: dict[str, Any]) -> dict[str, Any]:
        skill_outputs = dict(calibration_input.get("skill_outputs", {}))
        section_map: dict[str, list[str]] = {section: [] for section in SECTION_PRIORITY}
        for skill_name in skill_outputs.keys():
            section = SKILL_SECTION_MAP.get(skill_name, "comparison")
            section_map.setdefault(section, []).append(skill_name)
        return {
            "observation": {
                "skill_names": section_map.get("observation", []),
                "merged_groups": [],
                "raw_case_source_ids": [
                    item.get("source_id")
                    for item in calibration_input.get("retrieved_raw_cases_summary", [])[:1]
                    if item.get("source_id")
                ],
            },
            "comparison": {
                "skill_names": section_map.get("comparison", []),
                "merged_groups": [],
                "tactical_source_ids": [
                    item.get("source_id")
                    for item in calibration_input.get("retrieved_tactical_experiences_summary", [])[:2]
                    if item.get("source_id")
                ],
            },
            "risk": {
                "skill_names": section_map.get("risk", []),
                "abstract_source_ids": [
                    item.get("source_id")
                    for item in calibration_input.get("retrieved_abstract_experiences_summary", [])[:2]
                    if item.get("source_id")
                ],
            },
            "conflict_uncertainty": {
                "skill_names": section_map.get("conflict_uncertainty", []),
            },
            "planner": {"max_reason_skills": 3},
            "section_order": list(SECTION_PRIORITY),
        }


def build_default_evidence_calibrator(policy: dict[str, Any] | None = None) -> EvidenceCalibrator:
    return EvidenceCalibrator(policy=policy)


def _normalize_evidence_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_EVIDENCE_POLICY)
    merged.update(dict(policy or {}))
    mode = str(merged.get("calibrator_mode", "heuristic")).strip().lower()
    if mode not in {"off", "heuristic", "learned", "hybrid"}:
        mode = "heuristic"
    merged["calibrator_mode"] = mode
    return merged


def _score_skill_output(
    *,
    skill_name: str,
    output: dict[str, Any],
    retrieval_score: float,
    uncertainty_level: str,
    contradiction_count: int,
    risk_flags: list[str],
) -> float:
    score = float(SKILL_BASE_WEIGHT.get(skill_name, 2.2))
    score += float(EVIDENCE_STRENGTH_WEIGHT.get(str(output.get("evidence_strength", "unknown")).strip().lower(), 0.0))
    recommendation_type = str(output.get("recommendation_type", "descriptive_evidence")).strip().lower()
    score += float(RECOMMENDATION_WEIGHT.get(recommendation_type, 0.4))
    score += min(1.5, 0.08 * float(retrieval_score))
    score += min(1.2, 0.1 * _count_non_control_fields(output))
    if output.get("referenced_experiences"):
        score += 0.3
    if uncertainty_level == "high" and skill_name in {
        "uncertainty_assessment_skill",
        "information_gap_detection_skill",
        "contradiction_check_skill",
        "escalation_recommendation_skill",
    }:
        score += 1.0
    if contradiction_count > 0 and skill_name in {
        "contradiction_check_skill",
        "metadata_consistency_skill",
        "exclusion_reasoning_skill",
        "information_gap_detection_skill",
    }:
        score += 0.8
    if skill_name in {"ack_scc_specialist_skill", "mel_nev_specialist_skill", "exclusion_reasoning_skill"}:
        if _output_contains_terms(output, ("confusion", "opposing", "counterexample", "unlikely", "exclude")):
            score += 0.7
        if contradiction_count > 0 and _output_contains_terms(output, ("opposing_evidence", "exclusion_evidence", "counterexample")):
            score += 0.4
        if risk_flags and _output_contains_terms(output, ("bcc", "scc", "mel", "nev", "actinic", "seborrheic")):
            score += 0.3
    if risk_flags and skill_name in {"malignancy_risk_assessment_skill", "escalation_recommendation_skill"}:
        score += 0.9
    return float(score)


def _score_retrieval_record(
    *,
    record: dict[str, Any],
    source_layer: str,
    uncertainty_level: str,
    contradiction_count: int,
    risk_flags: list[str],
) -> float:
    layer_weight = {
        "raw_case_memory": 1.0,
        "tactical_experience": 1.8,
        "abstract_experience": 2.2,
    }.get(source_layer, 1.0)
    score = layer_weight + 0.2 * float(record.get("retrieval_score", 0.0) or 0.0)
    subtype = str(record.get("experience_type", record.get("source_subtype", ""))).strip().lower()
    if subtype == "confusion_memory":
        score += 1.0
    elif subtype in {"rule", "rule_candidate"}:
        score += 0.8
    elif subtype == "prototype":
        score += 0.6
    elif subtype == "composite_skill_seed":
        score += 0.5
    text = f"{record.get('perception_summary', '')} {' '.join(str(item) for item in record.get('learning_points', []))}".lower()
    if any(term in text for term in ("risk", "alarm", "malignan")) and risk_flags:
        score += 0.4
    if uncertainty_level in {"medium", "high"} and any(term in text for term in ("uncertainty", "missing", "gap")):
        score += 0.5
    if contradiction_count > 0 and any(term in text for term in ("conflict", "contradiction", "inconsisten")):
        score += 0.5
    if record.get("confusion_pair"):
        score += 0.3
    return float(score)


def _build_skill_feature_map(
    *,
    skill_name: str,
    section: str,
    output: dict[str, Any],
    heuristic_score: float,
    uncertainty_level: str,
    contradiction_count: int,
    risk_flags: list[str],
) -> dict[str, float]:
    feature_map: dict[str, float] = {
        "item_type::skill_output": 1.0,
        f"section::{section}": 1.0,
        f"skill::{skill_name}": 1.0,
        f"evidence_strength::{str(output.get('evidence_strength', 'unknown')).strip().lower() or 'unknown'}": 1.0,
        f"recommendation::{str(output.get('recommendation_type', 'unknown')).strip().lower() or 'unknown'}": 1.0,
        f"uncertainty::{uncertainty_level}": 1.0,
        "heuristic_score": float(heuristic_score),
        "contradiction_count": float(max(0, contradiction_count)),
        "risk_flag_count": float(len(risk_flags)),
        "field_count": float(_count_non_control_fields(output)),
    }
    if output.get("referenced_experiences"):
        feature_map["has_references"] = 1.0
    return feature_map


def _build_retrieval_feature_map(
    *,
    record: dict[str, Any],
    source_layer: str,
    heuristic_score: float,
    uncertainty_level: str,
    contradiction_count: int,
    risk_flags: list[str],
) -> dict[str, float]:
    feature_map: dict[str, float] = {
        "item_type::retrieval_record": 1.0,
        f"source_layer::{source_layer}": 1.0,
        f"source_subtype::{str(record.get('source_subtype', 'unknown')).strip().lower() or 'unknown'}": 1.0,
        f"experience_type::{str(record.get('experience_type', 'unknown')).strip().lower() or 'unknown'}": 1.0,
        f"uncertainty::{uncertainty_level}": 1.0,
        "heuristic_score": float(heuristic_score),
        "retrieval_score": float(record.get("retrieval_score", 0.0) or 0.0),
        "contradiction_count": float(max(0, contradiction_count)),
        "risk_flag_count": float(len(risk_flags)),
    }
    if record.get("confusion_pair"):
        feature_map["has_confusion_pair"] = 1.0
    return feature_map


def _count_non_control_fields(output: dict[str, Any]) -> int:
    count = 0
    for key, value in output.items():
        if key in {"referenced_experiences", "evidence_strength", "recommendation_type"}:
            continue
        if value in (None, "", [], {}, "unknown"):
            continue
        count += 1
    return count


def _output_contains_terms(output: dict[str, Any], terms: tuple[str, ...]) -> bool:
    blob = " ".join(
        f"{key} {value}"
        for key, value in output.items()
        if key not in {"referenced_experiences", "evidence_strength", "recommendation_type"}
    ).lower()
    return any(term in blob for term in terms)


def _has_meaningful_skill_output(output: dict[str, Any]) -> bool:
    return _count_non_control_fields(output) > 0 or bool(output.get("evidence_strength"))


def _contradiction_count(summary: dict[str, Any]) -> int:
    total = 0
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts", "suspicious_points"):
        total += len(summary.get(field_name, []) or [])
    return total


def _observation_merged_groups(skill_names: list[str]) -> list[dict[str, Any]]:
    names = [name for name in skill_names if name]
    if "lesion_description_structuring_skill" in names:
        members = [
            name
            for name in (
                "lesion_description_structuring_skill",
                "morphology_analysis_skill",
                "color_pattern_analysis_skill",
                "border_surface_analysis_skill",
            )
            if name in names
        ]
        if len(members) >= 2:
            return [
                {
                    "label": "structured_lesion_observation",
                    "skills": members,
                    "max_members": 3,
                }
            ]
    return []


def _comparison_merged_groups(skill_names: list[str]) -> list[dict[str, Any]]:
    names = [name for name in skill_names if name]
    if "differential_compare_skill" in names and "exclusion_reasoning_skill" in names:
        return [
            {
                "label": "differential_and_exclusion_reasoning",
                "skills": ["differential_compare_skill", "exclusion_reasoning_skill"],
                "max_members": 2,
            }
        ]
    return []
