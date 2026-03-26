from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.contamination_guard import build_split_state_version, infer_split_from_path, normalize_split_name

DEFAULT_COGNITION_PATH = Path("/root/DermAgent/state/cognition_state.json")


@dataclass
class CognitionState:
    self_capability_summary: str = (
        "DermAgent MVP provides structured clinical reasoning evidence, tracks uncertainty, "
        "records confusion patterns, and preserves Qwen as the only final diagnostic decision maker."
    )
    known_confusion_patterns: dict[str, int] = field(default_factory=dict)
    preferred_skills: list[str] = field(
        default_factory=lambda: [
            "morphology_analysis_skill",
            "color_pattern_analysis_skill",
            "border_surface_analysis_skill",
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
        ]
    )
    retrieval_preferences: dict[str, Any] = field(
        default_factory=lambda: {"top_k": 3, "prioritize_confusion_cases": True}
    )
    failure_statistics: dict[str, int] = field(
        default_factory=lambda: {
            "total_cases": 0,
            "failed_cases": 0,
            "confusion_cases": 0,
            "hard_cases": 0,
        }
    )
    skill_statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_split: str = "global"
    state_version: str = ""

    def __post_init__(self) -> None:
        self.known_confusion_patterns = dict(self.known_confusion_patterns or {})
        self.preferred_skills = list(self.preferred_skills or [])
        self.retrieval_preferences = {
            "top_k": 3,
            "prioritize_confusion_cases": True,
            **dict(self.retrieval_preferences or {}),
        }
        self.failure_statistics = {
            "total_cases": 0,
            "failed_cases": 0,
            "confusion_cases": 0,
            "hard_cases": 0,
            **dict(self.failure_statistics or {}),
        }
        normalized_skill_statistics: dict[str, dict[str, Any]] = {}
        for skill_name, payload in dict(self.skill_statistics or {}).items():
            normalized_skill_statistics[str(skill_name)] = self._normalize_skill_stats(payload)
        self.skill_statistics = normalized_skill_statistics
        self.state_split = normalize_split_name(self.state_split, default="global")
        if not str(self.state_version).strip():
            self.state_version = self._compute_state_version()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def initialize_default_state(cls, path: Path | None = None) -> "CognitionState":
        state = cls()
        state.save(path)
        return state

    @classmethod
    def load(cls, path: Path | None = None) -> "CognitionState":
        target = path or DEFAULT_COGNITION_PATH
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            return cls.initialize_default_state(target)
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and not str(payload.get("state_split", "")).strip():
            payload["state_split"] = infer_split_from_path(target, default="global")
        return cls(**payload)

    def save(self, path: Path | None = None) -> Path:
        target = path or DEFAULT_COGNITION_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        self.state_split = normalize_split_name(self.state_split, default=infer_split_from_path(target, default="global"))
        self.state_version = self._compute_state_version()
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        return target

    def update_failure_statistics(
        self,
        *,
        total_cases_increment: int = 0,
        failed_cases_increment: int = 0,
        confusion_cases_increment: int = 0,
        hard_cases_increment: int = 0,
    ) -> None:
        self.failure_statistics["total_cases"] += total_cases_increment
        self.failure_statistics["failed_cases"] += failed_cases_increment
        self.failure_statistics["confusion_cases"] += confusion_cases_increment
        self.failure_statistics["hard_cases"] += hard_cases_increment

    def update_known_confusion_patterns(self, confusion_pair: str | None) -> None:
        if not confusion_pair:
            return
        self.known_confusion_patterns[confusion_pair] = self.known_confusion_patterns.get(confusion_pair, 0) + 1

    def update_preferred_skills(self, skills: list[str]) -> None:
        if not skills:
            return
        self.preferred_skills = list(dict.fromkeys(skills))

    def update_skill_statistics(self, skill_updates: list[dict[str, Any]]) -> None:
        for update in skill_updates:
            skill_name = str(update.get("skill_name", "")).strip()
            if not skill_name:
                continue
            current = self._normalize_skill_stats(self.skill_statistics.get(skill_name, {}))
            current["call_count"] += int(update.get("call_increment", 0))
            current["selected_count"] += int(update.get("selected_increment", 0))
            current["success_count"] += int(update.get("success_increment", 0))
            current["partially_helpful_count"] += int(update.get("partially_helpful_increment", 0))
            current["harmful_count"] += int(update.get("harmful_increment", 0))
            current["failure_count"] += int(update.get("failure_increment", 0))
            current["reusable_count"] += int(update.get("reusable_increment", 0))
            current["uncertainty_reduction_count"] += int(update.get("uncertainty_reduction_increment", 0))
            current["contradiction_detection_count"] += int(update.get("contradiction_detection_increment", 0))
            current["malignant_flag_support_count"] += int(update.get("malignant_flag_support_increment", 0))
            current["evidence_strength_sum"] += float(update.get("evidence_strength_sum", 0.0))
            current["evidence_strength_observation_count"] += int(update.get("evidence_strength_count", 0))
            current["helpful_count"] = current["success_count"] + current["partially_helpful_count"]
            if current["call_count"] > 0:
                current["helpful_rate"] = current["helpful_count"] / current["call_count"]
                current["failure_rate"] = current["failure_count"] / current["call_count"]
            else:
                current["helpful_rate"] = 0.0
                current["failure_rate"] = 0.0
            if current["evidence_strength_observation_count"] > 0:
                current["average_evidence_strength"] = (
                    current["evidence_strength_sum"] / current["evidence_strength_observation_count"]
                )
            else:
                current["average_evidence_strength"] = 0.0
            self.skill_statistics[skill_name] = current

    @staticmethod
    def _normalize_skill_stats(payload: dict[str, Any] | None) -> dict[str, Any]:
        source = dict(payload or {})
        call_count = int(source.get("call_count", 0))
        selected_count = int(source.get("selected_count", source.get("call_count", 0)))
        success_count = int(source.get("success_count", 0))
        partially_helpful_count = int(source.get("partially_helpful_count", 0))
        harmful_count = int(source.get("harmful_count", source.get("failure_count", 0)))
        failure_count = int(source.get("failure_count", harmful_count))
        reusable_count = int(source.get("reusable_count", 0))
        uncertainty_reduction_count = int(source.get("uncertainty_reduction_count", 0))
        contradiction_detection_count = int(source.get("contradiction_detection_count", 0))
        malignant_flag_support_count = int(source.get("malignant_flag_support_count", 0))
        evidence_strength_sum = float(source.get("evidence_strength_sum", 0.0))
        evidence_strength_observation_count = int(source.get("evidence_strength_observation_count", 0))
        average_evidence_strength = float(source.get("average_evidence_strength", 0.0))
        helpful_count = int(source.get("helpful_count", success_count + partially_helpful_count))
        helpful_rate = float(source.get("helpful_rate", 0.0))
        failure_rate = float(source.get("failure_rate", 0.0))
        if call_count > 0:
            helpful_rate = helpful_count / call_count
            failure_rate = failure_count / call_count
        else:
            helpful_rate = 0.0
            failure_rate = 0.0
        if evidence_strength_observation_count > 0:
            average_evidence_strength = evidence_strength_sum / evidence_strength_observation_count
        else:
            average_evidence_strength = 0.0
        return {
            "call_count": call_count,
            "selected_count": selected_count,
            "success_count": success_count,
            "partially_helpful_count": partially_helpful_count,
            "helpful_count": helpful_count,
            "harmful_count": harmful_count,
            "failure_count": failure_count,
            "reusable_count": reusable_count,
            "uncertainty_reduction_count": uncertainty_reduction_count,
            "contradiction_detection_count": contradiction_detection_count,
            "malignant_flag_support_count": malignant_flag_support_count,
            "evidence_strength_sum": evidence_strength_sum,
            "evidence_strength_observation_count": evidence_strength_observation_count,
            "average_evidence_strength": average_evidence_strength,
            "helpful_rate": helpful_rate,
            "failure_rate": failure_rate,
        }

    def _compute_state_version(self) -> str:
        payload = {
            "self_capability_summary": self.self_capability_summary,
            "known_confusion_patterns": self.known_confusion_patterns,
            "preferred_skills": self.preferred_skills,
            "retrieval_preferences": self.retrieval_preferences,
            "failure_statistics": self.failure_statistics,
            "skill_statistics": self.skill_statistics,
        }
        return build_split_state_version(
            component_id="cognition_state",
            split_name=self.state_split,
            payload=payload,
        )
