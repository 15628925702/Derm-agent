from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.experience_schema import dedupe_strings, stable_hash
from memory.experience_store import ExperienceStore
from project_paths import outputs_root, repo_root


DEFAULT_BATCH_CRITIQUE_PATH = outputs_root() / "batch_reflection" / "batch_critique.json"
DEFAULT_REFINEMENT_CANDIDATES_PATH = outputs_root() / "skill_refinement_candidates" / "skill_refinement_candidates.jsonl"
DEFAULT_PROPOSALS_DIR = repo_root() / "proposals" / "composite_skills"
DEFAULT_MIN_SUPPORTING_CASES = 2


@dataclass
class CompositeSkillProposal:
    proposal_id: str
    source_seed_ids: list[str]
    trigger_pattern: dict[str, Any]
    intended_scope: dict[str, Any]
    proposed_workflow_text: str
    skill_sequence: list[str]
    supporting_cases: list[str]
    counter_cases: list[str]
    expected_benefit: dict[str, Any]
    risk_notes: list[str]
    review_status: str = "pending_review"
    review_checklist: list[str] = field(default_factory=list)
    proposal_artifacts: dict[str, Any] = field(default_factory=dict)
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_composite_skill_proposals(
    *,
    batch_critique_path: str | Path = DEFAULT_BATCH_CRITIQUE_PATH,
    refinement_candidates_path: str | Path = DEFAULT_REFINEMENT_CANDIDATES_PATH,
    experience_root: str | Path | None = None,
    min_supporting_cases: int = DEFAULT_MIN_SUPPORTING_CASES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_critique = _load_json(Path(batch_critique_path), {})
    refinement_candidates = _load_jsonl(Path(refinement_candidates_path))
    store = ExperienceStore(Path(experience_root)) if experience_root else ExperienceStore()
    abstract_experiences = store.load_abstract_experiences()

    if not batch_critique:
        return [], _summary([], batch_critique_path, refinement_candidates_path, min_supporting_cases)

    proposals: list[dict[str, Any]] = []
    abstract_seeds = [
        record for record in abstract_experiences
        if str(record.get("type", "")).strip() == "composite_skill_seed"
    ]
    by_signature = _index_abstract_seeds(abstract_seeds)
    refinement_by_skill = _refinement_candidates_by_skill(refinement_candidates)
    failure_clusters = list(batch_critique.get("failure_clusters", []))
    success_clusters = list(batch_critique.get("success_clusters", []))
    rule_candidates = list(batch_critique.get("rule_candidates", []))

    for seed_candidate in batch_critique.get("composite_skill_seed_candidates", []):
        skill_sequence = [str(item).strip() for item in seed_candidate.get("skill_sequence", []) if str(item).strip()]
        supporting_cases = dedupe_strings(seed_candidate.get("supporting_cases", []))
        if len(skill_sequence) < 3 or len(supporting_cases) < min_supporting_cases:
            continue

        signature = _seed_signature(
            trigger_pattern=seed_candidate.get("trigger_pattern", {}),
            skill_sequence=skill_sequence,
        )
        related_abstract = dedupe_records(
            by_signature.get(signature, []) + by_signature.get(_sequence_signature(skill_sequence), [])
        )
        source_seed_ids = dedupe_strings(
            [seed_candidate.get("seed_id", "")]
            + [record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id", "") for record in related_abstract]
        )
        counter_cases = _counter_cases_for_sequence(skill_sequence, failure_clusters, success_clusters, related_abstract)
        risk_notes = _risk_notes_for_sequence(skill_sequence, refinement_by_skill, failure_clusters)
        intended_scope = _intended_scope(seed_candidate, success_clusters, failure_clusters, counter_cases)
        suggested_skill_name = _suggested_skill_name(seed_candidate, skill_sequence)
        suggested_skill_id = _suggested_skill_id(seed_candidate, skill_sequence)
        workflow_text = _build_proposed_workflow_text(
            trigger_pattern=seed_candidate.get("trigger_pattern", {}),
            skill_sequence=skill_sequence,
            intended_scope=intended_scope,
        )
        expected_benefit = _expected_benefit(seed_candidate, related_abstract, rule_candidates)
        evidence_refs = _evidence_refs(seed_candidate, related_abstract, refinement_by_skill, batch_critique)

        proposal = CompositeSkillProposal(
            proposal_id=f"proposal_{stable_hash({'signature': signature, 'supporting_cases': supporting_cases})}",
            source_seed_ids=source_seed_ids,
            trigger_pattern=dict(seed_candidate.get("trigger_pattern", {})),
            intended_scope=intended_scope,
            proposed_workflow_text=workflow_text,
            skill_sequence=skill_sequence,
            supporting_cases=supporting_cases,
            counter_cases=counter_cases,
            expected_benefit=expected_benefit,
            risk_notes=risk_notes,
            review_checklist=[
                "Confirm the trigger pattern is stable across supporting cases rather than overfit to one image style.",
                "Review whether any component skill should stay independent instead of being fused into a composite workflow.",
                "Check that counter-cases do not expose harmful shortcut behavior or hidden diagnostic leakage.",
                "Ensure the proposal improves evidence organization and CoT explainability rather than replacing final diagnosis.",
                "If approved, materialize as a new skill spec and registry entry manually; do not auto-enable.",
            ],
            proposal_artifacts={
                "suggested_skill_id": suggested_skill_id,
                "suggested_skill_name": suggested_skill_name,
                "target_proposal_dir": str(DEFAULT_PROPOSALS_DIR / f"{suggested_skill_name}.json"),
                "future_skill_type": "composite_workflow_skill",
                "manual_integration_targets": {
                    "design_skill_spec_path": str(repo_root() / "design" / "skill_specs" / "composite" / f"{suggested_skill_name}.md"),
                    "python_skill_path": str(repo_root() / "skills" / f"{suggested_skill_name.replace('_skill', '')}.py"),
                    "registry_update_required": True,
                },
            },
            evidence_refs=evidence_refs,
        ).to_dict()
        proposals.append(proposal)

    proposals.sort(key=lambda item: (len(item.get("supporting_cases", [])), -len(item.get("counter_cases", []))), reverse=True)
    return proposals, _summary(proposals, batch_critique_path, refinement_candidates_path, min_supporting_cases)


def save_composite_skill_proposals(
    proposals: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_PROPOSALS_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    proposals_path = root / "composite_skill_proposals.jsonl"
    with proposals_path.open("w", encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps(proposal, ensure_ascii=False) + "\n")

    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id", "")).strip()
        if not proposal_id:
            continue
        target = root / f"{proposal_id}.json"
        target.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "proposals_path": str(proposals_path),
        "summary_path": str(summary_path),
    }


def _summary(
    proposals: list[dict[str, Any]],
    batch_critique_path: str | Path,
    refinement_candidates_path: str | Path,
    min_supporting_cases: int,
) -> dict[str, Any]:
    return {
        "proposal_count": len(proposals),
        "min_supporting_cases": min_supporting_cases,
        "batch_critique_path": str(batch_critique_path),
        "refinement_candidates_path": str(refinement_candidates_path),
        "proposal_ids": [proposal.get("proposal_id") for proposal in proposals],
        "supporting_case_count_distribution": dict(Counter(len(proposal.get("supporting_cases", [])) for proposal in proposals)),
    }


def _index_abstract_seeds(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        seed_payload = record.get("composite_skill_seed", {})
        trigger_pattern = seed_payload.get("trigger_pattern") or record.get("pattern_summary", {}).get("trigger_pattern", {})
        skill_sequence = seed_payload.get("skill_sequence") or record.get("pattern_summary", {}).get("seed_skills", [])
        signature = _seed_signature(trigger_pattern=trigger_pattern, skill_sequence=skill_sequence)
        if signature:
            grouped[signature].append(record)
        sequence_signature = _sequence_signature(skill_sequence)
        if sequence_signature:
            grouped[sequence_signature].append(record)
    return grouped


def _seed_signature(*, trigger_pattern: dict[str, Any], skill_sequence: list[str]) -> str:
    normalized_trigger = _normalize_trigger_pattern(trigger_pattern)
    return stable_hash({"trigger_pattern": normalized_trigger, "skill_sequence": list(skill_sequence)})


def _sequence_signature(skill_sequence: list[str]) -> str:
    cleaned = [str(item).strip() for item in skill_sequence if str(item).strip()]
    if not cleaned:
        return ""
    return f"sequence::{stable_hash(cleaned)}"


def _normalize_trigger_pattern(trigger_pattern: Any) -> dict[str, Any]:
    if isinstance(trigger_pattern, dict):
        return {
            "decision_pattern": str(trigger_pattern.get("decision_pattern", "")).strip(),
            "uncertainty_level": str(trigger_pattern.get("uncertainty_level", "")).strip(),
            "confusion_pair": str(trigger_pattern.get("confusion_pair", "")).strip(),
            "region": str(trigger_pattern.get("region", "")).strip(),
        }
    if isinstance(trigger_pattern, list):
        return {
            "decision_pattern": "",
            "uncertainty_level": "",
            "confusion_pair": "",
            "region": "",
            "supporting_pattern_preview": "|".join(str(item).strip() for item in trigger_pattern[:3] if str(item).strip()),
        }
    if trigger_pattern is None:
        return {
            "decision_pattern": "",
            "uncertainty_level": "",
            "confusion_pair": "",
            "region": "",
        }
    normalized_trigger = {
        "decision_pattern": str(trigger_pattern).strip(),
        "uncertainty_level": "",
        "confusion_pair": "",
        "region": "",
    }
    return normalized_trigger


def _refinement_candidates_by_skill(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        skill_name = str(record.get("target_skill_name", "")).strip()
        if skill_name:
            grouped[skill_name].append(record)
    return grouped


def _counter_cases_for_sequence(
    skill_sequence: list[str],
    failure_clusters: list[dict[str, Any]],
    success_clusters: list[dict[str, Any]],
    related_abstract: list[dict[str, Any]],
) -> list[str]:
    counters: list[str] = []
    skill_set = set(skill_sequence)
    for cluster in failure_clusters:
        cluster_sequence = set(str(item).strip() for item in cluster.get("common_skill_sequence", []) if str(item).strip())
        if skill_set.intersection(cluster_sequence):
            counters.extend(str(item).strip() for item in cluster.get("case_ids", []) if str(item).strip())
    for record in related_abstract:
        counters.extend(str(item).strip() for item in record.get("counter_cases", []) if str(item).strip())
    success_cases = {
        str(item).strip()
        for cluster in success_clusters
        for item in cluster.get("case_ids", [])
        if str(item).strip()
    }
    return [case_id for case_id in dedupe_strings(counters) if case_id not in success_cases][:10]


def _risk_notes_for_sequence(
    skill_sequence: list[str],
    refinement_by_skill: dict[str, list[dict[str, Any]]],
    failure_clusters: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    for skill_name in skill_sequence:
        for candidate in refinement_by_skill.get(skill_name, []):
            update_type = str(candidate.get("proposed_update_type", "")).strip()
            reason = str(candidate.get("proposed_change_summary", {}).get("reason", "")).strip()
            if update_type or reason:
                notes.append(f"{skill_name}: review {update_type or 'refinement'} because {reason or 'existing failure evidence is present'}.")
    for cluster in failure_clusters[:5]:
        failure_type = str(cluster.get("failure_type", "")).strip()
        cluster_sequence = set(str(item).strip() for item in cluster.get("common_skill_sequence", []) if str(item).strip())
        if cluster_sequence.intersection(skill_sequence):
            notes.append(
                f"Related failure cluster `{failure_type}` overlaps this sequence; inspect counter-cases before approval."
            )
    notes.append("Composite workflow must remain evidence-organizing only and cannot replace Qwen as final diagnostician.")
    return dedupe_strings(notes)[:12]


def _intended_scope(
    seed_candidate: dict[str, Any],
    success_clusters: list[dict[str, Any]],
    failure_clusters: list[dict[str, Any]],
    counter_cases: list[str],
) -> dict[str, Any]:
    trigger_pattern = dict(seed_candidate.get("trigger_pattern", {}))
    decision_pattern = str(trigger_pattern.get("decision_pattern", "")).strip() or "structured_reasoning_path"
    applicable_when = [
        f"decision_pattern:{decision_pattern}",
    ]
    if trigger_pattern.get("uncertainty_level"):
        applicable_when.append(f"uncertainty:{trigger_pattern.get('uncertainty_level')}")
    if trigger_pattern.get("confusion_pair"):
        applicable_when.append(f"confusion_pair:{trigger_pattern.get('confusion_pair')}")
    if trigger_pattern.get("region"):
        applicable_when.append(f"region:{trigger_pattern.get('region')}")
    for cluster in success_clusters[:3]:
        applicable_when.extend(str(item).strip() for item in cluster.get("representative_signals", [])[:3] if str(item).strip())

    do_not_use_when = []
    if counter_cases:
        do_not_use_when.append("similar trigger pattern appears in known counter-cases")
    for cluster in failure_clusters[:3]:
        failure_type = str(cluster.get("failure_type", "")).strip()
        if failure_type:
            do_not_use_when.append(f"unresolved {failure_type} pattern remains dominant")

    return {
        "use_case": "Bundle a repeated doctor-style skill sequence into a reviewable composite workflow proposal for planning and evidence organization.",
        "applicable_when": dedupe_strings(applicable_when),
        "do_not_use_when": dedupe_strings(do_not_use_when),
    }


def _build_proposed_workflow_text(
    *,
    trigger_pattern: dict[str, Any],
    skill_sequence: list[str],
    intended_scope: dict[str, Any],
) -> str:
    decision_pattern = str(trigger_pattern.get("decision_pattern", "")).strip() or "structured reasoning"
    lines = [
        f"Use this composite workflow when the case matches `{decision_pattern}` and the planner needs a repeatable evidence-building path.",
        "Start by establishing the structured lesion description and morphology-driven observation layer before narrowing reasoning.",
        "Then run the comparison and risk-oriented skills in sequence so negative evidence, conflict signals, and uncertainty stay explicit.",
        "Finish by packaging the outputs into a coherent evidence bundle for Qwen, without letting the composite workflow make the final diagnosis.",
        f"Component sequence: {', '.join(skill_sequence)}.",
        f"Intended scope markers: {', '.join(intended_scope.get('applicable_when', [])[:6])}.",
    ]
    return "\n".join(lines)


def _expected_benefit(
    seed_candidate: dict[str, Any],
    related_abstract: list[dict[str, Any]],
    rule_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    benefits = {
        "planner_benefit": [
            "Reduce repeated planner work for known stable multi-skill sequences.",
            "Make candidate skill selection more consistent under repeated trigger patterns.",
        ],
        "cot_explainability_benefit": [
            "Preserve a doctor-style ordered workflow instead of scattered skill outputs.",
            "Keep negative evidence, uncertainty, and escalation framing visible in a single reusable path.",
        ],
        "evidence_bundle_benefit": [
            "Improve evidence serialization consistency for final Qwen reasoning.",
            "Create a more replayable and auditable intermediate evidence package.",
        ],
    }
    if related_abstract:
        benefits["planner_benefit"].append(
            f"Leverage {len(related_abstract)} related abstract experiences already linked to this sequence."
        )
    if rule_candidates:
        benefits["evidence_bundle_benefit"].append(
            "Align with existing rule candidates distilled from repeated tactical condition->action traces."
        )
    if seed_candidate.get("success_count"):
        benefits["cot_explainability_benefit"].append(
            f"Backed by {seed_candidate.get('success_count')} repeated successful supporting cases."
        )
    for key, values in benefits.items():
        benefits[key] = dedupe_strings(values)
    return benefits


def _evidence_refs(
    seed_candidate: dict[str, Any],
    related_abstract: list[dict[str, Any]],
    refinement_by_skill: dict[str, list[dict[str, Any]]],
    batch_critique: dict[str, Any],
) -> dict[str, Any]:
    skill_sequence = [str(item).strip() for item in seed_candidate.get("skill_sequence", []) if str(item).strip()]
    matched_refinement_ids = []
    for skill_name in skill_sequence:
        matched_refinement_ids.extend(
            str(item.get("candidate_id", "")).strip()
            for item in refinement_by_skill.get(skill_name, [])
            if str(item.get("candidate_id", "")).strip()
        )
    return {
        "batch_critique_sections": [
            "success_clusters",
            "composite_skill_seed_candidates",
            "rule_candidates",
            "refinement_inputs.success_patterns_for_policy_reuse",
        ],
        "batch_id": batch_critique.get("batch_id"),
        "abstract_experience_ids": [
            str(record.get("abs_id", "")).strip()
            for record in related_abstract
            if str(record.get("abs_id", "")).strip()
        ],
        "refinement_candidate_ids": dedupe_strings(matched_refinement_ids),
    }


def _suggested_skill_name(seed_candidate: dict[str, Any], skill_sequence: list[str]) -> str:
    trigger_pattern = dict(seed_candidate.get("trigger_pattern", {}))
    decision_pattern = str(trigger_pattern.get("decision_pattern", "")).strip().lower().replace(" ", "_")
    if decision_pattern and decision_pattern != "batch_success_sequence":
        return f"{decision_pattern}_composite_skill"
    return f"{skill_sequence[0]}_{skill_sequence[-1]}_composite_skill"


def _suggested_skill_id(seed_candidate: dict[str, Any], skill_sequence: list[str]) -> str:
    return f"skill.{_suggested_skill_name(seed_candidate, skill_sequence).replace('_skill', '')}.proposal.v1"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("abs_id") or record.get("seed_id") or stable_hash(record)).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result
