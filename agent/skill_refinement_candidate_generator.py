from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.hard_case_miner import load_execution_records
from cognition.cognition_state import CognitionState
from memory.experience_schema import stable_hash
from memory.experience_store import ExperienceStore
from skills.registry import build_default_registry


DEFAULT_HARD_CASES_PATH = Path("/root/DermAgent/outputs/hard_case_mining/hard_cases.jsonl")
DEFAULT_SKILL_HELPFULNESS_DIR = Path("/root/DermAgent/outputs/skill_helpfulness")
DEFAULT_OUTPUT_DIR = Path("/root/DermAgent/outputs/skill_refinement_candidates")
CONTRADICTION_FOCUSED_SKILLS = {
    "metadata_consistency_skill",
    "contradiction_check_skill",
    "differential_compare_skill",
    "uncertainty_assessment_skill",
    "information_gap_detection_skill",
    "exclusion_reasoning_skill",
    "malignancy_risk_assessment_skill",
}


@dataclass
class SkillRefinementCandidate:
    candidate_id: str
    target_skill_id: str
    target_skill_name: str
    trigger_pattern: dict[str, Any]
    failure_pattern: dict[str, Any]
    supporting_cases: list[str]
    counter_cases: list[str]
    proposed_update_type: str
    proposed_change_summary: dict[str, Any]
    confidence: str
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    source_evidence_refs: dict[str, Any] = field(default_factory=dict)
    integration_hints: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_skill_refinement_candidates(
    *,
    records_root: str | Path,
    hard_cases_path: str | Path = DEFAULT_HARD_CASES_PATH,
    skill_helpfulness_dir: str | Path = DEFAULT_SKILL_HELPFULNESS_DIR,
    experience_root: str | Path | None = None,
    cognition_path: str | Path | None = None,
    dataset_name: str | None = None,
    skill_name: str | None = None,
    proposed_update_type: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution_records = load_execution_records(records_root)
    hard_cases = _load_jsonl(Path(hard_cases_path))
    helpfulness_reports = _load_jsonl(Path(skill_helpfulness_dir) / "skill_helpfulness_reports.jsonl")
    helpfulness_summary = _load_json(Path(skill_helpfulness_dir) / "skill_helpfulness_summary.json", {})
    experience_store = ExperienceStore(Path(experience_root)) if experience_root else ExperienceStore()
    abstract_experiences = experience_store.load_abstract_experiences()
    cognition = CognitionState.load(Path(cognition_path)) if cognition_path else CognitionState.load()

    registry = build_default_registry()
    skill_metadata_map: dict[str, dict[str, Any]] = {}
    for registered_skill in registry._skills.values():
        skill_object = registered_skill.get_skill_object()
        module = __import__(registered_skill.__class__.__module__, fromlist=[registered_skill.__class__.__name__])
        module_path = str(Path(getattr(module, "__file__", "")).resolve()) if getattr(module, "__file__", None) else ""
        skill_metadata_map[registered_skill.name] = {
            "skill_name": skill_object.name,
            "skill_id": skill_object.skill_id,
            "skill_type": skill_object.skill_type,
            "source": module_path or skill_object.source,
            "workflow_text": skill_object.workflow_text,
        }
    skill_spec_paths = _discover_skill_spec_paths()

    record_by_case = {str(record.get("case_id", "")).strip(): record for record in execution_records if record.get("case_id")}
    helpfulness_by_skill = {
        str(report.get("skill_name", "")).strip(): report
        for report in helpfulness_reports
        if str(report.get("skill_name", "")).strip()
    }
    hard_cases_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hard_case_pattern_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missed_trigger_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for hard_case in hard_cases:
        if dataset_name and str(hard_case.get("dataset_name", "")).strip() != dataset_name:
            continue
        involved_skills = [str(item).strip() for item in hard_case.get("involved_skills", []) if str(item).strip()]
        for involved_skill in involved_skills:
            hard_cases_by_skill[involved_skill].append(hard_case)
            hard_case_pattern_counts[involved_skill][str(hard_case.get("failure_type", "unknown"))] += 1

    for hard_case in hard_cases:
        if dataset_name and str(hard_case.get("dataset_name", "")).strip() != dataset_name:
            continue
        case_id = str(hard_case.get("case_id", "")).strip()
        record = record_by_case.get(case_id, {})
        hard_case_scenarios = _derive_case_scenarios(record, hard_case)
        involved_skills = {str(item).strip() for item in hard_case.get("involved_skills", []) if str(item).strip()}
        for report in helpfulness_reports:
            candidate_skill = str(report.get("skill_name", "")).strip()
            if not candidate_skill or candidate_skill in involved_skills:
                continue
            scenario_names = {str(item.get("name", "")).strip() for item in report.get("common_applicable_scenarios", [])}
            overlap = sorted(item for item in hard_case_scenarios if item in scenario_names)
            if overlap:
                for item in overlap:
                    missed_trigger_counts[candidate_skill][item] += 1

    candidates: list[dict[str, Any]] = []
    all_skill_names = sorted(
        {
            *skill_metadata_map.keys(),
            *hard_cases_by_skill.keys(),
            *helpfulness_by_skill.keys(),
            *missed_trigger_counts.keys(),
        }
    )
    for current_skill_name in all_skill_names:
        if skill_name and current_skill_name != skill_name:
            continue
        metadata = skill_metadata_map.get(
            current_skill_name,
            {
                "skill_name": current_skill_name,
                "skill_id": current_skill_name,
                "skill_type": "unknown",
                "source": "",
                "workflow_text": "",
            },
        )
        hard_case_records = hard_cases_by_skill.get(current_skill_name, [])
        helpfulness_report = helpfulness_by_skill.get(current_skill_name, {})
        related_abstract = _find_related_abstract_experiences(current_skill_name, hard_case_records, abstract_experiences)
        candidate_specs = _propose_candidates_for_skill(
            skill_name=current_skill_name,
            metadata=metadata,
            hard_case_records=hard_case_records,
            helpfulness_report=helpfulness_report,
            record_by_case=record_by_case,
            related_abstract_experiences=related_abstract,
            cognition=cognition,
            missed_trigger_counter=missed_trigger_counts.get(current_skill_name, Counter()),
            skill_spec_path=skill_spec_paths.get(current_skill_name, ""),
        )
        for spec in candidate_specs:
            if proposed_update_type and spec["proposed_update_type"] != proposed_update_type:
                continue
            candidates.append(spec)

    candidates.sort(
        key=lambda item: (
            _confidence_rank(item.get("confidence")),
            len(item.get("supporting_cases", [])),
            str(item.get("target_skill_name", "")),
        ),
        reverse=True,
    )
    summary = _build_candidate_summary(
        execution_records=execution_records,
        hard_cases=hard_cases,
        candidates=candidates,
        helpfulness_summary=helpfulness_summary,
        filters={
            "dataset_name": dataset_name,
            "skill_name": skill_name,
            "proposed_update_type": proposed_update_type,
        },
    )
    return candidates, summary


def save_skill_refinement_candidates(
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    candidates_path = root / "skill_refinement_candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    summary_path = root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "candidates_path": str(candidates_path),
        "summary_path": str(summary_path),
    }


def _propose_candidates_for_skill(
    *,
    skill_name: str,
    metadata: dict[str, Any],
    hard_case_records: list[dict[str, Any]],
    helpfulness_report: dict[str, Any],
    record_by_case: dict[str, dict[str, Any]],
    related_abstract_experiences: list[dict[str, Any]],
    cognition: CognitionState,
    missed_trigger_counter: Counter[str],
    skill_spec_path: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not helpfulness_report and not hard_case_records and not missed_trigger_counter:
        return candidates

    failure_types = Counter(str(item.get("failure_type", "unknown")) for item in hard_case_records)
    contradiction_heavy_count = int(failure_types.get("contradiction_heavy_case", 0))
    uncertainty_heavy_count = int(failure_types.get("high_uncertainty_case", 0))
    confusion_heavy_count = int(failure_types.get("confusion_driven_case", 0))
    hard_case_count = len(hard_case_records)
    harmful_rate = float(helpfulness_report.get("harmful_rate", 0.0))
    harmful_count = int(helpfulness_report.get("harmful_count", 0))
    average_evidence_strength = float(helpfulness_report.get("average_evidence_strength", 0.0))
    common_failure_modes = [str(item.get("name", "")).strip() for item in helpfulness_report.get("common_failure_modes", [])]
    contradiction_detection_count = int(helpfulness_report.get("contradiction_detection_count", 0))
    uncertainty_reduction_count = int(helpfulness_report.get("uncertainty_reduction_count", 0))

    if hard_case_count > 0:
        dominant_failure_type = failure_types.most_common(1)[0][0]
        if (
            harmful_count > 0
            or harmful_rate >= 0.25
            or any(mode in common_failure_modes for mode in ("final_reasoning_failed", "active_confusion_unresolved", "uncertainty_remained_high"))
            or (contradiction_heavy_count > 0 and skill_name in {"contradiction_check_skill", "metadata_consistency_skill"})
        ):
            candidates.append(
                _build_candidate(
                    target_skill_name=skill_name,
                    target_skill_id=str(metadata.get("skill_id", skill_name)),
                    trigger_pattern=_build_trigger_pattern(
                        skill_name=skill_name,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                    ),
                    failure_pattern=_build_failure_pattern(
                        dominant_failure_type=dominant_failure_type,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        failure_modes=common_failure_modes,
                    ),
                    supporting_cases=_supporting_cases(hard_case_records),
                    counter_cases=_counter_cases(skill_name, dominant_failure_type, record_by_case, hard_case_records),
                    proposed_update_type="workflow_update",
                    proposed_change_summary={
                        "reason": "Skill is repeatedly present in difficult executions but the current reasoning workflow is not robust enough for the observed failure pattern.",
                        "focus": [
                            "tighten step ordering under hard-case conditions",
                            "make negative evidence and conflict reconciliation explicit",
                            "improve downstream handoff for Qwen final integration",
                        ],
                    },
                    confidence=_confidence_from_evidence(
                        hard_case_count=hard_case_count,
                        harmful_count=harmful_count,
                        related_abstract_count=len(related_abstract_experiences),
                    ),
                    evidence_summary=_build_evidence_summary(
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        related_abstract_experiences=related_abstract_experiences,
                        cognition=cognition,
                    ),
                    source_evidence_refs=_build_source_refs(hard_case_records, related_abstract_experiences),
                    integration_hints=_integration_hints(
                        skill_name=skill_name,
                        metadata=metadata,
                        skill_spec_path=skill_spec_path,
                        proposed_update_type="workflow_update",
                    ),
                )
            )

        if contradiction_detection_count > 0 or (contradiction_heavy_count > 0 and skill_name in CONTRADICTION_FOCUSED_SKILLS):
            candidates.append(
                _build_candidate(
                    target_skill_name=skill_name,
                    target_skill_id=str(metadata.get("skill_id", skill_name)),
                    trigger_pattern=_build_trigger_pattern(
                        skill_name=skill_name,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                    ),
                    failure_pattern=_build_failure_pattern(
                        dominant_failure_type=dominant_failure_type,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        failure_modes=common_failure_modes,
                    ),
                    supporting_cases=_supporting_cases(hard_case_records),
                    counter_cases=_counter_cases(skill_name, dominant_failure_type, record_by_case, hard_case_records),
                    proposed_update_type="watchout_update",
                    proposed_change_summary={
                        "reason": "Hard cases show repeated contradiction-heavy evidence or unresolved conflicts that should be surfaced more explicitly as doctor-style watch-outs.",
                        "focus": [
                            "add high-risk misread scenarios to Watch Out For",
                            "separate contradiction from ambiguity",
                            "state which negative evidence should block overconfident narrowing",
                        ],
                    },
                    confidence=_confidence_from_evidence(
                        hard_case_count=contradiction_heavy_count or hard_case_count,
                        harmful_count=0,
                        related_abstract_count=len(related_abstract_experiences),
                    ),
                    evidence_summary=_build_evidence_summary(
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        related_abstract_experiences=related_abstract_experiences,
                        cognition=cognition,
                    ),
                    source_evidence_refs=_build_source_refs(hard_case_records, related_abstract_experiences),
                    integration_hints=_integration_hints(
                        skill_name=skill_name,
                        metadata=metadata,
                        skill_spec_path=skill_spec_path,
                        proposed_update_type="watchout_update",
                    ),
                )
            )

        if average_evidence_strength < 0.45 or any(mode in common_failure_modes for mode in ("no_structured_output", "weak_evidence_strength")):
            candidates.append(
                _build_candidate(
                    target_skill_name=skill_name,
                    target_skill_id=str(metadata.get("skill_id", skill_name)),
                    trigger_pattern=_build_trigger_pattern(
                        skill_name=skill_name,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                    ),
                    failure_pattern=_build_failure_pattern(
                        dominant_failure_type=dominant_failure_type,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        failure_modes=common_failure_modes,
                    ),
                    supporting_cases=_supporting_cases(hard_case_records),
                    counter_cases=_counter_cases(skill_name, dominant_failure_type, record_by_case, hard_case_records),
                    proposed_update_type="output_schema_update",
                    proposed_change_summary={
                        "reason": "The skill output does not appear expressive enough for difficult cases.",
                        "focus": [
                            "add fields that preserve negative evidence or missing evidence",
                            "make downstream aggregation easier to audit",
                            "separate supportive evidence from exclusion evidence",
                        ],
                    },
                    confidence=_confidence_from_evidence(
                        hard_case_count=hard_case_count,
                        harmful_count=harmful_count,
                        related_abstract_count=0,
                    ),
                    evidence_summary=_build_evidence_summary(
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        related_abstract_experiences=related_abstract_experiences,
                        cognition=cognition,
                    ),
                    source_evidence_refs=_build_source_refs(hard_case_records, related_abstract_experiences),
                    integration_hints=_integration_hints(
                        skill_name=skill_name,
                        metadata=metadata,
                        skill_spec_path=skill_spec_path,
                        proposed_update_type="output_schema_update",
                    ),
                )
            )

        if confusion_heavy_count > 0 and str(metadata.get("skill_type", "")).strip() == "specialist":
            candidates.append(
                _build_candidate(
                    target_skill_name=skill_name,
                    target_skill_id=str(metadata.get("skill_id", skill_name)),
                    trigger_pattern=_build_trigger_pattern(
                        skill_name=skill_name,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                    ),
                    failure_pattern=_build_failure_pattern(
                        dominant_failure_type=dominant_failure_type,
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        failure_modes=common_failure_modes,
                    ),
                    supporting_cases=_supporting_cases(hard_case_records),
                    counter_cases=_counter_cases(skill_name, dominant_failure_type, record_by_case, hard_case_records),
                    proposed_update_type="split_skill",
                    proposed_change_summary={
                        "reason": "Specialist scope may be too broad for the observed confusion-driven cases.",
                        "focus": [
                            "check whether one specialist skill is covering more than one confusion family",
                            "consider separating distinct confusion workflows",
                        ],
                    },
                    confidence=_confidence_from_evidence(
                        hard_case_count=confusion_heavy_count,
                        harmful_count=harmful_count,
                        related_abstract_count=len(related_abstract_experiences),
                    ),
                    evidence_summary=_build_evidence_summary(
                        hard_case_records=hard_case_records,
                        helpfulness_report=helpfulness_report,
                        related_abstract_experiences=related_abstract_experiences,
                        cognition=cognition,
                    ),
                    source_evidence_refs=_build_source_refs(hard_case_records, related_abstract_experiences),
                    integration_hints=_integration_hints(
                        skill_name=skill_name,
                        metadata=metadata,
                        skill_spec_path=skill_spec_path,
                        proposed_update_type="split_skill",
                    ),
                )
            )

    if missed_trigger_counter:
        candidates.append(
            _build_candidate(
                target_skill_name=skill_name,
                target_skill_id=str(metadata.get("skill_id", skill_name)),
                trigger_pattern={
                    "missed_trigger_scenarios": [
                        {"name": name, "count": count} for name, count in missed_trigger_counter.most_common(5)
                    ],
                    "current_common_applicable_scenarios": helpfulness_report.get("common_applicable_scenarios", [])[:5],
                },
                failure_pattern={
                    "pattern_type": "missed_trigger_opportunity",
                    "observed_issue": "The skill is not being selected in some hard cases whose trigger patterns overlap with its known applicable scenarios.",
                    "related_failure_modes": ["skill_not_selected_under_matching_pattern"],
                },
                supporting_cases=[],
                counter_cases=_counter_cases(skill_name, "missed_trigger_opportunity", record_by_case, hard_case_records),
                proposed_update_type="trigger_update",
                proposed_change_summary={
                    "reason": "Hard-case scenarios overlap with this skill's known application profile, but the skill was not selected.",
                    "focus": [
                        "expand or refine trigger metadata",
                        "add planner-visible trigger phrases linked to hard-case scenarios",
                    ],
                },
                confidence="low",
                evidence_summary={
                    "missed_trigger_count": sum(missed_trigger_counter.values()),
                    "missed_trigger_scenarios": [
                        {"name": name, "count": count} for name, count in missed_trigger_counter.most_common(5)
                    ],
                    "harmful_rate": harmful_rate,
                },
                source_evidence_refs={
                    "hard_case_ids": [],
                    "abstract_experience_ids": [],
                },
                integration_hints=_integration_hints(
                    skill_name=skill_name,
                    metadata=metadata,
                    skill_spec_path=skill_spec_path,
                    proposed_update_type="trigger_update",
                ),
            )
        )

    merge_candidate = _build_merge_candidate(
        skill_name=skill_name,
        metadata=metadata,
        hard_case_records=hard_case_records,
        related_abstract_experiences=related_abstract_experiences,
        record_by_case=record_by_case,
        skill_spec_path=skill_spec_path,
    )
    if merge_candidate is not None:
        candidates.append(merge_candidate)

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        deduped[candidate["candidate_id"]] = candidate
    return list(deduped.values())


def _build_merge_candidate(
    *,
    skill_name: str,
    metadata: dict[str, Any],
    hard_case_records: list[dict[str, Any]],
    related_abstract_experiences: list[dict[str, Any]],
    record_by_case: dict[str, dict[str, Any]],
    skill_spec_path: str,
) -> dict[str, Any] | None:
    composite_records = [
        record
        for record in related_abstract_experiences
        if str(record.get("type", "")).strip() == "composite_skill_seed"
    ]
    if not composite_records:
        return None
    seed = composite_records[0]
    composite_payload = seed.get("composite_skill_seed", {})
    skill_sequence = [str(item).strip() for item in composite_payload.get("skill_sequence", []) if str(item).strip()]
    if skill_name not in skill_sequence or len(skill_sequence) < 2:
        return None
    neighboring_skills = [item for item in skill_sequence if item != skill_name]
    if not neighboring_skills:
        return None
    target_neighbor = neighboring_skills[0]
    supporting_cases = _supporting_cases(hard_case_records)
    return _build_candidate(
        target_skill_name=skill_name,
        target_skill_id=str(metadata.get("skill_id", skill_name)),
        trigger_pattern={
            "composite_seed_id": composite_payload.get("seed_id") or seed.get("seed_id"),
            "skill_sequence": skill_sequence,
            "trigger_pattern": composite_payload.get("trigger_pattern", seed.get("pattern_summary", {}).get("trigger_pattern", {})),
        },
        failure_pattern={
            "pattern_type": "composite_sequence_overlap",
            "observed_issue": "A repeated multi-skill sequence already exists as a composite seed and may justify a future merged workflow boundary.",
            "related_failure_modes": [],
        },
        supporting_cases=supporting_cases,
        counter_cases=_counter_cases(skill_name, "composite_sequence_overlap", record_by_case, hard_case_records),
        proposed_update_type="merge_skill",
        proposed_change_summary={
            "reason": f"{skill_name} frequently travels with `{target_neighbor}` inside an existing composite skill seed.",
            "focus": [
                "review whether these skills should share a tighter composite workflow boundary",
                "do not auto-merge yet; use this as a design-review seed only",
            ],
            "candidate_partner_skill": target_neighbor,
        },
        confidence="low" if len(supporting_cases) < 2 else "medium",
        evidence_summary={
            "composite_seed_id": composite_payload.get("seed_id") or seed.get("seed_id"),
            "supporting_case_count": len(supporting_cases),
            "skill_sequence": skill_sequence,
        },
        source_evidence_refs={
            "hard_case_ids": [record.get("hard_case_id") for record in hard_case_records if record.get("hard_case_id")],
            "abstract_experience_ids": [seed.get("abs_id")] if seed.get("abs_id") else [],
        },
        integration_hints=_integration_hints(
            skill_name=skill_name,
            metadata=metadata,
            skill_spec_path=skill_spec_path,
            proposed_update_type="merge_skill",
        ),
    )


def _build_candidate(
    *,
    target_skill_name: str,
    target_skill_id: str,
    trigger_pattern: dict[str, Any],
    failure_pattern: dict[str, Any],
    supporting_cases: list[str],
    counter_cases: list[str],
    proposed_update_type: str,
    proposed_change_summary: dict[str, Any],
    confidence: str,
    evidence_summary: dict[str, Any],
    source_evidence_refs: dict[str, Any],
    integration_hints: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "target_skill_id": target_skill_id,
        "proposed_update_type": proposed_update_type,
        "trigger_pattern": trigger_pattern,
        "failure_pattern": failure_pattern,
        "supporting_cases": supporting_cases,
    }
    return SkillRefinementCandidate(
        candidate_id=f"src_{stable_hash(payload)}",
        target_skill_id=target_skill_id,
        target_skill_name=target_skill_name,
        trigger_pattern=trigger_pattern,
        failure_pattern=failure_pattern,
        supporting_cases=supporting_cases,
        counter_cases=counter_cases,
        proposed_update_type=proposed_update_type,
        proposed_change_summary=proposed_change_summary,
        confidence=confidence,
        evidence_summary=evidence_summary,
        source_evidence_refs=source_evidence_refs,
        integration_hints=integration_hints,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ).to_dict()


def _build_trigger_pattern(
    *,
    skill_name: str,
    hard_case_records: list[dict[str, Any]],
    helpfulness_report: dict[str, Any],
) -> dict[str, Any]:
    datasets = Counter(str(item.get("dataset_name", "unknown_dataset")) for item in hard_case_records)
    labels = Counter(str(item.get("ground_truth", {}).get("canonical_label", "unknown")) for item in hard_case_records)
    confusion_tags = Counter(
        tag
        for record in hard_case_records
        for tag in record.get("confusion_tags", [])
        if str(tag).strip()
    )
    return {
        "skill_name": skill_name,
        "common_hard_case_datasets": [{"name": name, "count": count} for name, count in datasets.most_common(3)],
        "common_hard_case_labels": [{"name": name, "count": count} for name, count in labels.most_common(3)],
        "common_confusion_tags": [{"name": name, "count": count} for name, count in confusion_tags.most_common(3)],
        "common_applicable_scenarios": helpfulness_report.get("common_applicable_scenarios", [])[:5],
    }


def _build_failure_pattern(
    *,
    dominant_failure_type: str,
    hard_case_records: list[dict[str, Any]],
    helpfulness_report: dict[str, Any],
    failure_modes: list[str],
) -> dict[str, Any]:
    contradiction_counts = [
        int(record.get("evidence_snapshot", {}).get("contradiction_count", 0))
        for record in hard_case_records
    ]
    uncertainty_levels = Counter(
        str(record.get("agent_result", {}).get("uncertainty_level", "unknown")).strip().lower()
        for record in hard_case_records
    )
    return {
        "pattern_type": dominant_failure_type,
        "common_failure_modes": failure_modes[:5],
        "hard_case_failure_types": [
            {"name": name, "count": count}
            for name, count in Counter(str(item.get("failure_type", "unknown")) for item in hard_case_records).most_common(5)
        ],
        "contradiction_count_summary": {
            "max": max(contradiction_counts) if contradiction_counts else 0,
            "mean": round(sum(contradiction_counts) / len(contradiction_counts), 3) if contradiction_counts else 0.0,
        },
        "uncertainty_levels": [{"name": name, "count": count} for name, count in uncertainty_levels.most_common(3)],
        "helpfulness_stats": {
            "harmful_count": int(helpfulness_report.get("harmful_count", 0)),
            "harmful_rate": float(helpfulness_report.get("harmful_rate", 0.0)),
            "average_evidence_strength": float(helpfulness_report.get("average_evidence_strength", 0.0)),
        },
    }


def _build_evidence_summary(
    *,
    hard_case_records: list[dict[str, Any]],
    helpfulness_report: dict[str, Any],
    related_abstract_experiences: list[dict[str, Any]],
    cognition: CognitionState,
) -> dict[str, Any]:
    return {
        "hard_case_count": len(hard_case_records),
        "hard_case_ids": [record.get("hard_case_id") for record in hard_case_records if record.get("hard_case_id")],
        "helpfulness_report": deepcopy(helpfulness_report),
        "related_abstract_experience_count": len(related_abstract_experiences),
        "related_abstract_experience_types": dict(
            Counter(str(record.get("type", "unknown")) for record in related_abstract_experiences)
        ),
        "cognition_skill_statistics": cognition.skill_statistics.get(
            str(helpfulness_report.get("skill_name", "")),
            {},
        ),
    }


def _build_source_refs(hard_case_records: list[dict[str, Any]], related_abstract_experiences: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hard_case_ids": [record.get("hard_case_id") for record in hard_case_records if record.get("hard_case_id")],
        "case_ids": [record.get("case_id") for record in hard_case_records if record.get("case_id")],
        "execution_record_paths": [record.get("source_record_path") for record in hard_case_records if record.get("source_record_path")],
        "abstract_experience_ids": [record.get("abs_id") for record in related_abstract_experiences if record.get("abs_id")],
        "composite_seed_ids": [
            record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id")
            for record in related_abstract_experiences
            if record.get("seed_id") or record.get("composite_skill_seed", {}).get("seed_id")
        ],
    }


def _integration_hints(
    *,
    skill_name: str,
    metadata: dict[str, Any],
    skill_spec_path: str,
    proposed_update_type: str,
) -> dict[str, Any]:
    section_map = {
        "workflow_update": ["Strategy Overview", "Workflow"],
        "trigger_update": ["When to Use", "triggers"],
        "watchout_update": ["Watch Out For"],
        "output_schema_update": ["Output Contract"],
        "split_skill": ["When to Use", "Workflow", "Output Contract"],
        "merge_skill": ["Workflow", "composite workflow interface"],
    }
    return {
        "target_sections": section_map.get(proposed_update_type, []),
        "skill_type": metadata.get("skill_type", "unknown"),
        "python_source_path": str(metadata.get("source", "")),
        "skill_spec_path": skill_spec_path,
        "notes": (
            "This candidate is for manual or semi-automatic refinement only. "
            "It should inform updates to skill text, registry triggers, or schema, not direct auto-editing."
        ),
        "skill_name": skill_name,
    }


def _find_related_abstract_experiences(
    skill_name: str,
    hard_case_records: list[dict[str, Any]],
    abstract_experiences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_ids = {str(record.get("case_id", "")).strip() for record in hard_case_records if record.get("case_id")}
    related: list[dict[str, Any]] = []
    for record in abstract_experiences:
        record_type = str(record.get("type", "")).strip()
        supporting_cases = {str(item).strip() for item in record.get("supporting_cases", []) if str(item).strip()}
        skill_sequence = {
            str(item).strip()
            for item in record.get("composite_skill_seed", {}).get("skill_sequence", [])
            if str(item).strip()
        }
        action_hint = str(record.get("derived_rule", {}).get("action_hint", "")).strip().lower()
        if case_ids and supporting_cases.intersection(case_ids) and record_type in {"confusion_memory", "rule", "composite_skill_seed"}:
            related.append(record)
            continue
        if skill_name in skill_sequence:
            related.append(record)
            continue
        if skill_name.lower() in action_hint:
            related.append(record)
    return related


def _supporting_cases(hard_case_records: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(record.get("case_id", "")).strip() for record in hard_case_records if record.get("case_id")))


def _counter_cases(
    skill_name: str,
    failure_type: str,
    record_by_case: dict[str, dict[str, Any]],
    hard_case_records: list[dict[str, Any]],
) -> list[str]:
    hard_case_ids = {str(item.get("case_id", "")).strip() for item in hard_case_records if item.get("case_id")}
    results: list[str] = []
    for case_id, record in record_by_case.items():
        if case_id in hard_case_ids:
            continue
        selected_skills = {
            str(item).strip()
            for item in (
                record.get("selected_skills")
                or record.get("planner_decision", {}).get("selected_skills", [])
                or record.get("planner_decision", {}).get("ordering", [])
            )
            if str(item).strip()
        }
        if skill_name not in selected_skills:
            continue
        evaluation = record.get("evaluation", {})
        if evaluation.get("correct") is True:
            results.append(case_id)
    return results[:5]


def _derive_case_scenarios(record: dict[str, Any], hard_case: dict[str, Any]) -> set[str]:
    scenarios: set[str] = set()
    qwen_initial = record.get("qwen_initial", {})
    clinical_metadata = record.get("input_summary", {}).get("clinical_metadata", {})
    case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    for item in qwen_initial.get("ddx_candidates", [])[:2]:
        if str(item).strip():
            scenarios.add(f"ddx:{'|'.join(str(x).strip() for x in qwen_initial.get('ddx_candidates', [])[:2] if str(x).strip())}")
            break
    region = str(clinical_metadata.get("region", "")).strip()
    if region:
        scenarios.add(f"region:{region.lower()}")
    for risk_flag in case_outcome.get("risk_flags", [])[:2]:
        scenarios.add(f"risk:{risk_flag}")
    uncertainty_level = str(case_outcome.get("uncertainty_level", hard_case.get("agent_result", {}).get("uncertainty_level", "unknown"))).strip().lower()
    if uncertainty_level and uncertainty_level != "unknown":
        scenarios.add(f"uncertainty:{uncertainty_level}")
    for tag in hard_case.get("confusion_tags", [])[:2]:
        scenarios.add(f"confusion:{tag}")
    return scenarios


def _confidence_from_evidence(*, hard_case_count: int, harmful_count: int, related_abstract_count: int) -> str:
    score = 0
    if hard_case_count >= 3:
        score += 2
    elif hard_case_count >= 1:
        score += 1
    if harmful_count >= 2:
        score += 2
    elif harmful_count >= 1:
        score += 1
    if related_abstract_count >= 2:
        score += 1
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _confidence_rank(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value).strip().lower(), 0)


def _discover_skill_spec_paths() -> dict[str, str]:
    result: dict[str, str] = {}
    root = Path("/root/DermAgent/design/skill_specs")
    if not root.exists():
        return result
    for path in root.rglob("*.md"):
        result[path.stem] = str(path)
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _build_candidate_summary(
    *,
    execution_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    helpfulness_summary: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any]:
    by_update_type = Counter(str(item.get("proposed_update_type", "unknown")) for item in candidates)
    by_skill = Counter(str(item.get("target_skill_name", "unknown")) for item in candidates)
    by_confidence = Counter(str(item.get("confidence", "unknown")) for item in candidates)
    return {
        "execution_record_count": len(execution_records),
        "hard_case_count": len(hard_cases),
        "candidate_count": len(candidates),
        "filters": deepcopy(filters),
        "counts": {
            "by_update_type": dict(by_update_type),
            "by_skill": dict(by_skill),
            "by_confidence": dict(by_confidence),
        },
        "helpfulness_context": deepcopy(helpfulness_summary.get("totals", {})) if isinstance(helpfulness_summary, dict) else {},
        "top_candidates": [
            {
                "candidate_id": item.get("candidate_id"),
                "target_skill_name": item.get("target_skill_name"),
                "proposed_update_type": item.get("proposed_update_type"),
                "confidence": item.get("confidence"),
                "supporting_case_count": len(item.get("supporting_cases", [])),
            }
            for item in candidates[:10]
        ],
    }
