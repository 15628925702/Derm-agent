from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.contamination_guard import build_split_state_version, infer_split_from_path, normalize_split_name

POLICY_STATE_ROOT = Path("/root/DermAgent/state/policy")
POLICY_VERSIONS_DIR = POLICY_STATE_ROOT / "versions"
POLICY_EVALUATIONS_DIR = POLICY_STATE_ROOT / "evaluations"
CURRENT_STABLE_POLICY_PATH = POLICY_STATE_ROOT / "current_stable_policy.json"
POLICY_MANIFEST_PATH = POLICY_STATE_ROOT / "manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PolicyVersion:
    policy_id: str
    version: int = 1
    status: str = "candidate"
    created_at: str = field(default_factory=utc_now)
    parent_policy_id: str = ""
    description: str = ""
    change_summary: dict[str, Any] = field(default_factory=dict)
    planner_policy: dict[str, Any] = field(default_factory=dict)
    retrieval_policy: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    skill_config_refs: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    evaluation_gate: dict[str, Any] = field(default_factory=dict)
    last_evaluation: dict[str, Any] = field(default_factory=dict)
    rollback_history: list[dict[str, Any]] = field(default_factory=list)
    state_split: str = "global"
    state_version: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_policy_payload() -> dict[str, Any]:
    return {
        "policy_id": "stable_default_policy",
        "version": 1,
        "status": "stable",
        "created_at": utc_now(),
        "parent_policy_id": "",
        "description": "Default conservative stable planner policy.",
        "change_summary": {
            "planner": ["Baseline rule-based planner with structured evidence support."],
            "skill_overrides": [],
            "retrieval": ["Use cognition-guided retrieval with conservative top-k defaults."],
            "cognition_defaults": [],
        },
        "planner_policy": {
            "controller_family": "heuristic",
            "controller_checkpoint_path": "",
            "learned_controller_weight": 4.0,
            "learned_controller_select_threshold": 0.4,
            "learned_controller_top_k": 0,
            "learned_controller_force_top_k": 0,
            "score_threshold_default": 4,
            "retrieval_score_cap": 3,
            "foundational_bonus": 6,
            "preferred_skill_bonus": 3,
            "helpful_rate_bonus": 2,
            "failure_rate_penalty": 2,
            "signal_match_bonus": 2,
            "known_confusion_bonus": 2,
            "experience_compare_bonus": 2,
            "experience_risk_bonus": 2,
            "experience_gap_bonus": 2,
            "experience_conflict_bonus": 2,
            "experience_escalation_bonus": 2,
            "enable_signals": {
                "experience_compare_pattern": True,
                "experience_risk_pattern": True,
                "experience_gap_pattern": True,
                "experience_conflict_pattern": True,
                "experience_escalation_pattern": True,
            },
            "force_select_skills": [],
            "force_disable_skills": [],
            "skill_overrides": {},
        },
        "retrieval_policy": {
            "top_k_experience": 3,
            "top_k_tactical": 3,
            "top_k_abstract": 3,
            "top_k_skill_candidates": 14,
            "prioritize_confusion_cases": True,
            "enable_learned_retrieval_reranker": False,
            "retrieval_reranker_checkpoint_path": "",
            "retrieval_reranker_blend_weight": 2.5,
            "retrieval_reranker_fail_open": True,
        },
        "evidence_policy": {
            "enable_evidence_calibrator": True,
            "calibrator_mode": "heuristic",
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
        },
        "skill_config_refs": [],
        "evidence_refs": {},
        "evaluation_gate": {
            "minimum_cases": 10,
            "max_top1_drop": 0.0,
            "max_topk_drop": 0.02,
            "max_error_rate_increase": 0.02,
            "max_key_confusion_drop": 0.0,
            "require_non_decreasing_malignant_recall": True,
        },
        "last_evaluation": {},
        "rollback_history": [],
        "state_split": "global",
        "state_version": "",
        "source_path": str(CURRENT_STABLE_POLICY_PATH),
    }


def normalize_policy_payload(payload: dict[str, Any] | None, *, source_path: str = "") -> dict[str, Any]:
    source = deepcopy(payload or {})
    merged = default_policy_payload()
    merged.update(
        {
            key: value
            for key, value in source.items()
            if key not in {"planner_policy", "retrieval_policy", "evidence_policy", "evaluation_gate"}
        }
    )
    merged["planner_policy"] = {
        **default_policy_payload()["planner_policy"],
        **dict(source.get("planner_policy", {}) or {}),
    }
    merged["retrieval_policy"] = {
        **default_policy_payload()["retrieval_policy"],
        **dict(source.get("retrieval_policy", {}) or {}),
    }
    merged["evidence_policy"] = {
        **default_policy_payload()["evidence_policy"],
        **dict(source.get("evidence_policy", {}) or {}),
    }
    merged["evaluation_gate"] = {
        **default_policy_payload()["evaluation_gate"],
        **dict(source.get("evaluation_gate", {}) or {}),
    }
    split_name = source.get("state_split")
    if not str(split_name or "").strip():
        split_name = infer_split_from_path(source_path or source.get("source_path"), default="global")
    merged["state_split"] = normalize_split_name(str(split_name), default="global")
    merged["state_version"] = str(source.get("state_version", "")).strip() or _compute_policy_state_version(merged)
    merged["source_path"] = source_path or str(source.get("source_path", "")).strip()
    return merged


def normalize_policy(payload: dict[str, Any] | None, *, source_path: str = "") -> PolicyVersion:
    return PolicyVersion(**normalize_policy_payload(payload, source_path=source_path))


def ensure_policy_store() -> dict[str, Path]:
    POLICY_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    stable_payload = None
    if not CURRENT_STABLE_POLICY_PATH.exists():
        stable = normalize_policy(default_policy_payload(), source_path=str(CURRENT_STABLE_POLICY_PATH))
        stable_payload = stable.to_dict()
        CURRENT_STABLE_POLICY_PATH.write_text(json.dumps(stable_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if stable_payload is None:
        stable_payload = json.loads(CURRENT_STABLE_POLICY_PATH.read_text(encoding="utf-8"))
    if not POLICY_MANIFEST_PATH.exists():
        manifest = {
            "stable_policy_id": stable_payload.get("policy_id"),
            "stable_policy_path": str(CURRENT_STABLE_POLICY_PATH),
            "policies": [
                {
                    "policy_id": stable_payload.get("policy_id"),
                    "version": stable_payload.get("version"),
                    "status": stable_payload.get("status"),
                    "created_at": stable_payload.get("created_at"),
                    "source_path": str(CURRENT_STABLE_POLICY_PATH),
                    "description": stable_payload.get("description"),
                }
            ],
            "evaluations": [],
            "updated_at": utc_now(),
        }
        POLICY_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "root": POLICY_STATE_ROOT,
        "versions_dir": POLICY_VERSIONS_DIR,
        "evaluations_dir": POLICY_EVALUATIONS_DIR,
        "stable_path": CURRENT_STABLE_POLICY_PATH,
        "manifest_path": POLICY_MANIFEST_PATH,
    }


def load_policy(path: str | Path | None = None) -> PolicyVersion:
    ensure_policy_store()
    target = Path(path) if path else CURRENT_STABLE_POLICY_PATH
    if not target.exists():
        raise FileNotFoundError(f"Policy config not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return normalize_policy(payload, source_path=str(target))


def load_stable_policy(path: str | Path | None = None) -> PolicyVersion:
    return load_policy(path or CURRENT_STABLE_POLICY_PATH)


def save_policy(policy: PolicyVersion | dict[str, Any], path: str | Path, *, register: bool = True) -> Path:
    POLICY_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    raw_payload = policy.to_dict() if isinstance(policy, PolicyVersion) else normalize_policy(policy).to_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_policy_payload(raw_payload, source_path=str(target))
    payload["source_path"] = str(target)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if register:
        register_policy(normalize_policy(payload, source_path=str(target)))
    return target


def register_policy(policy: PolicyVersion | dict[str, Any]) -> dict[str, Any]:
    ensure_policy_store()
    payload = policy.to_dict() if isinstance(policy, PolicyVersion) else normalize_policy(policy).to_dict()
    manifest = _load_manifest()
    policies = [item for item in manifest.get("policies", []) if item.get("policy_id") != payload.get("policy_id")]
    policies.append(
        {
            "policy_id": payload.get("policy_id"),
            "version": payload.get("version"),
            "status": payload.get("status"),
            "created_at": payload.get("created_at"),
            "source_path": payload.get("source_path"),
            "description": payload.get("description"),
        }
    )
    manifest["policies"] = sorted(policies, key=lambda item: (str(item.get("created_at", "")), str(item.get("policy_id", ""))))
    if payload.get("status") == "stable":
        manifest["stable_policy_id"] = payload.get("policy_id")
        manifest["stable_policy_path"] = payload.get("source_path") or str(CURRENT_STABLE_POLICY_PATH)
    manifest["updated_at"] = utc_now()
    _save_manifest(manifest)
    return manifest


def save_evaluation_record(record: dict[str, Any], *, policy_id: str) -> Path:
    ensure_policy_store()
    target = POLICY_EVALUATIONS_DIR / f"{policy_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = _load_manifest()
    evaluations = list(manifest.get("evaluations", []))
    evaluations.append(
        {
            "policy_id": policy_id,
            "path": str(target),
            "created_at": record.get("evaluated_at", utc_now()),
            "decision": record.get("gate_decision", {}).get("decision"),
        }
    )
    manifest["evaluations"] = evaluations[-100:]
    manifest["updated_at"] = utc_now()
    _save_manifest(manifest)
    return target


def promote_candidate_to_stable(candidate: PolicyVersion | dict[str, Any], *, evaluation_record: dict[str, Any]) -> Path:
    normalized = candidate if isinstance(candidate, PolicyVersion) else normalize_policy(candidate)
    normalized.status = "stable"
    normalized.last_evaluation = deepcopy(evaluation_record)
    version_path = POLICY_VERSIONS_DIR / f"{normalized.policy_id}.json"
    save_policy(normalized, version_path, register=True)
    save_policy(normalized, CURRENT_STABLE_POLICY_PATH, register=True)
    return version_path


def reject_or_rollback_candidate(
    candidate: PolicyVersion | dict[str, Any],
    *,
    evaluation_record: dict[str, Any],
    rollback_to_policy_id: str,
    reason: str,
) -> Path:
    normalized = candidate if isinstance(candidate, PolicyVersion) else normalize_policy(candidate)
    normalized.status = "rolled_back" if evaluation_record.get("gate_decision", {}).get("rollback_required") else "rejected"
    normalized.last_evaluation = deepcopy(evaluation_record)
    normalized.rollback_history.append(
        {
            "rolled_back_at": utc_now(),
            "rollback_to_policy_id": rollback_to_policy_id,
            "reason": reason,
            "evaluation_path": evaluation_record.get("evaluation_record_path", ""),
        }
    )
    version_path = POLICY_VERSIONS_DIR / f"{normalized.policy_id}.json"
    save_policy(normalized, version_path, register=True)
    return version_path


def snapshot_policy(policy: PolicyVersion | dict[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return load_stable_policy().to_dict()
    if isinstance(policy, PolicyVersion):
        return policy.to_dict()
    return normalize_policy(policy).to_dict()


def _load_manifest() -> dict[str, Any]:
    if not POLICY_MANIFEST_PATH.exists():
        ensure_policy_store()
    return json.loads(POLICY_MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(payload: dict[str, Any]) -> None:
    POLICY_MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_policy_state_version(payload: dict[str, Any]) -> str:
    stable_payload = {
        "policy_id": payload.get("policy_id"),
        "version": payload.get("version"),
        "planner_policy": payload.get("planner_policy", {}),
        "retrieval_policy": payload.get("retrieval_policy", {}),
        "evidence_policy": payload.get("evidence_policy", {}),
        "evaluation_gate": payload.get("evaluation_gate", {}),
    }
    return build_split_state_version(
        component_id="policy_config",
        split_name=normalize_split_name(str(payload.get("state_split", "global")), default="global"),
        payload=stable_payload,
    )
