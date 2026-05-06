from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.hard_case_miner import build_hard_case_candidate, load_execution_records
from agent.skill_fitness_snapshot import DEFAULT_SNAPSHOT_DIR, save_skill_fitness_snapshot
from cognition.cognition_state import CognitionState
from memory.experience_schema import dedupe_strings, stable_hash
from project_paths import outputs_root


DEFAULT_OUTPUT_DIR = outputs_root() / "batch_reflection"
DEFAULT_HARD_CASES_PATH = outputs_root() / "hard_case_mining" / "hard_cases.jsonl"
DEFAULT_REFINEMENT_CANDIDATES_PATH = outputs_root() / "skill_refinement_candidates" / "skill_refinement_candidates.jsonl"
DEFAULT_MIN_SUPPORT = 2


@dataclass
class BatchCritique:
    batch_id: str
    created_at: str
    records_root: str
    source_summary: dict[str, Any]
    success_clusters: list[dict[str, Any]] = field(default_factory=list)
    failure_clusters: list[dict[str, Any]] = field(default_factory=list)
    skill_sequence_analysis: dict[str, Any] = field(default_factory=dict)
    experience_helpfulness_analysis: dict[str, Any] = field(default_factory=dict)
    emergent_confusion_memories: list[dict[str, Any]] = field(default_factory=list)
    rule_candidates: list[dict[str, Any]] = field(default_factory=list)
    composite_skill_seed_candidates: list[dict[str, Any]] = field(default_factory=list)
    refinement_inputs: dict[str, Any] = field(default_factory=dict)
    dependency_links: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_batch_reflection(
    *,
    records_root: str | Path,
    hard_cases_path: str | Path = DEFAULT_HARD_CASES_PATH,
    refinement_candidates_path: str | Path = DEFAULT_REFINEMENT_CANDIDATES_PATH,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> BatchCritique:
    records = load_execution_records(records_root)
    hard_cases = _load_jsonl(Path(hard_cases_path))
    if not hard_cases:
        hard_cases = [build_hard_case_candidate(record) for record in records]
    refinement_candidates = _load_jsonl(Path(refinement_candidates_path))

    success_records = [record for record in records if _is_success(record)]
    failure_records = [record for record in records if _is_failure(record)]

    batch_id = f"batch_{stable_hash({'records_root': str(records_root), 'records': [record.get('case_id') for record in records]})}"
    critique = BatchCritique(
        batch_id=batch_id,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        records_root=str(records_root),
        source_summary=_source_summary(records, hard_cases, success_records, failure_records),
        success_clusters=_build_success_clusters(success_records, min_support=min_support),
        failure_clusters=_build_failure_clusters(failure_records, hard_cases, min_support=min_support),
        skill_sequence_analysis=_build_skill_sequence_analysis(records, min_support=min_support),
        experience_helpfulness_analysis=_build_experience_helpfulness_analysis(records),
        emergent_confusion_memories=_build_emergent_confusion_memories(records, hard_cases, min_support=min_support),
        rule_candidates=_build_rule_candidates(records, min_support=min_support),
        composite_skill_seed_candidates=_build_composite_skill_seed_candidates(success_records, min_support=min_support),
        refinement_inputs=_build_refinement_inputs(records, hard_cases, refinement_candidates, min_support=min_support),
        dependency_links={
            "experience_consolidation": {
                "consumes": ["emergent_confusion_memories", "rule_candidates", "composite_skill_seed_candidates"],
                "rationale": "Batch reflection proposes cross-rollout patterns before periodic abstract experience consolidation.",
            },
            "skill_refinement": {
                "consumes": ["failure_clusters", "skill_sequence_analysis.success_vs_failure_deltas", "refinement_inputs.failure_patterns_for_skill_refinement"],
                "rationale": "Batch reflection aggregates repeated failure evidence into structured refinement-ready signals.",
            },
        },
    )
    return critique


def save_batch_reflection_outputs(
    critique: BatchCritique | dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cognition: CognitionState | None = None,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> dict[str, str]:
    payload = critique.to_dict() if isinstance(critique, BatchCritique) else deepcopy(critique)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    critique_path = root / "batch_critique.json"
    critique_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "batch_id": payload.get("batch_id"),
        "created_at": payload.get("created_at"),
        "source_summary": payload.get("source_summary", {}),
        "success_cluster_count": len(payload.get("success_clusters", [])),
        "failure_cluster_count": len(payload.get("failure_clusters", [])),
        "emergent_confusion_memory_count": len(payload.get("emergent_confusion_memories", [])),
        "rule_candidate_count": len(payload.get("rule_candidates", [])),
        "composite_skill_seed_candidate_count": len(payload.get("composite_skill_seed_candidates", [])),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "critique_path": str(critique_path),
        "summary_path": str(summary_path),
    }

    if cognition is not None:
        snapshot_path = save_skill_fitness_snapshot(cognition.to_dict(), output_dir=snapshot_dir)
        result["skill_fitness_snapshot_path"] = str(snapshot_path)

    return result


def _source_summary(
    records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    success_records: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "execution_record_count": len(records),
        "hard_case_count": len(hard_cases),
        "success_case_count": len(success_records),
        "failure_case_count": len(failure_records),
        "dataset_counts": dict(Counter(str(record.get("dataset_name", "unknown_dataset")) for record in records)),
        "policy_counts": dict(
            Counter(
                str(
                    record.get("policy_snapshot", {}).get("policy_id")
                    or record.get("planner_decision", {}).get("policy_id")
                    or "unknown_policy"
                )
                for record in records
            )
        ),
    }


def _build_success_clusters(records: list[dict[str, Any]], *, min_support: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sequence = _selected_skills(record)
        if not sequence:
            continue
        key = "|".join(sequence)
        grouped[key].append(record)

    clusters: list[dict[str, Any]] = []
    for key, bundle in grouped.items():
        if len(bundle) < min_support:
            continue
        case_ids = [_case_id(record) for record in bundle]
        clusters.append(
            {
                "cluster_id": f"succ_{stable_hash({'sequence': key, 'cases': case_ids})}",
                "pattern_type": "skill_sequence",
                "case_ids": case_ids,
                "support_count": len(bundle),
                "common_skill_sequence": list(key.split("|")),
                "common_retrieved_experiences": _top_experiences(bundle, top_k=5),
                "common_confusion_tags": _top_confusion_tags(bundle, top_k=5),
                "representative_signals": _representative_signals(bundle),
            }
        )
    clusters.sort(key=lambda item: (int(item.get("support_count", 0)), len(item.get("common_skill_sequence", []))), reverse=True)
    return clusters


def _build_failure_clusters(
    failure_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    *,
    min_support: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hard_case_by_case = {str(item.get("case_id", "")).strip(): item for item in hard_cases if item.get("case_id")}
    for record in failure_records:
        hard_case = hard_case_by_case.get(_case_id(record))
        failure_type = str((hard_case or {}).get("failure_type") or _case_failure_type(record) or "unknown_failure")
        confusion_tag = ",".join(_confusion_tags(record)[:1]) or "no_confusion"
        key = f"{failure_type}|{confusion_tag}"
        grouped[key].append(record)

    clusters: list[dict[str, Any]] = []
    for key, bundle in grouped.items():
        if len(bundle) < min_support:
            continue
        failure_type, _, confusion_tag = key.partition("|")
        case_ids = [_case_id(record) for record in bundle]
        hard_bundle = [hard_case_by_case.get(case_id, {}) for case_id in case_ids]
        clusters.append(
            {
                "cluster_id": f"fail_{stable_hash({'key': key, 'cases': case_ids})}",
                "failure_type": failure_type,
                "case_ids": case_ids,
                "support_count": len(bundle),
                "common_skill_sequence": _most_common_sequence(bundle),
                "common_failure_patterns": _top_failure_patterns(hard_bundle),
                "common_confusion_tags": dedupe_strings([confusion_tag] + _top_confusion_tags(bundle, top_k=5)),
                "common_retrieved_experiences": _top_experiences(bundle, top_k=5),
                "representative_signals": _representative_signals(bundle),
            }
        )
    clusters.sort(key=lambda item: int(item.get("support_count", 0)), reverse=True)
    return clusters


def _build_skill_sequence_analysis(records: list[dict[str, Any]], *, min_support: int) -> dict[str, Any]:
    success_counter: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    cases_by_sequence: dict[str, list[str]] = defaultdict(list)

    for record in records:
        sequence = _selected_skills(record)
        if not sequence:
            continue
        key = "|".join(sequence)
        cases_by_sequence[key].append(_case_id(record))
        if _is_success(record):
            success_counter[key] += 1
        elif _is_failure(record):
            failure_counter[key] += 1

    successful_sequences = [
        {
            "sequence_id": f"seq_{stable_hash(key)}",
            "skill_sequence": key.split("|"),
            "support_count": count,
            "case_ids": cases_by_sequence.get(key, []),
        }
        for key, count in success_counter.most_common()
        if count >= min_support
    ]
    failure_sequences = [
        {
            "sequence_id": f"seq_{stable_hash(key)}",
            "skill_sequence": key.split("|"),
            "support_count": count,
            "case_ids": cases_by_sequence.get(key, []),
        }
        for key, count in failure_counter.most_common()
        if count >= min_support
    ]

    deltas: list[dict[str, Any]] = []
    for key in sorted(set(success_counter) | set(failure_counter)):
        success_count = int(success_counter.get(key, 0))
        failure_count = int(failure_counter.get(key, 0))
        total = success_count + failure_count
        if total < min_support:
            continue
        deltas.append(
            {
                "sequence_id": f"seq_{stable_hash(key)}",
                "skill_sequence": key.split("|"),
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": round(success_count / total, 4) if total else None,
                "failure_rate": round(failure_count / total, 4) if total else None,
                "delta_success_minus_failure": success_count - failure_count,
            }
        )
    deltas.sort(key=lambda item: (item.get("delta_success_minus_failure", 0), item.get("success_count", 0)), reverse=True)
    return {
        "successful_sequences": successful_sequences[:20],
        "failure_sequences": failure_sequences[:20],
        "success_vs_failure_deltas": deltas[:30],
    }


def _build_experience_helpfulness_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    score_by_experience: dict[str, dict[str, Any]] = {}
    for record in records:
        case_weight = 1 if _is_success(record) else -1 if _is_failure(record) else 0
        for experience in _retrieved_experiences(record):
            source_id = str(experience.get("source_id", "")).strip()
            if not source_id:
                continue
            item = score_by_experience.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "source_layer": str(experience.get("source_layer", "")).strip(),
                    "experience_type": str(experience.get("experience_type", "")).strip(),
                    "success_count": 0,
                    "failure_count": 0,
                    "net_helpfulness": 0,
                    "case_ids": [],
                },
            )
            if case_weight > 0:
                item["success_count"] += 1
            elif case_weight < 0:
                item["failure_count"] += 1
            item["net_helpfulness"] += case_weight
            item["case_ids"].append(_case_id(record))

    ranked = sorted(
        score_by_experience.values(),
        key=lambda item: (int(item.get("net_helpfulness", 0)), int(item.get("success_count", 0)), -int(item.get("failure_count", 0))),
        reverse=True,
    )
    grouped = {"raw_case_memory": [], "tactical_experience": [], "abstract_experience": []}
    for item in ranked:
        layer = str(item.get("source_layer", "")).strip()
        if layer in grouped:
            grouped[layer].append(item)

    return {
        "most_helpful_raw_cases": grouped["raw_case_memory"][:10],
        "most_helpful_tactical_experiences": grouped["tactical_experience"][:10],
        "most_helpful_abstract_experiences": grouped["abstract_experience"][:10],
        "experience_type_summary": {
            "by_source_layer": dict(Counter(str(item.get("source_layer", "")).strip() for item in ranked)),
            "by_experience_type": dict(Counter(str(item.get("experience_type", "")).strip() for item in ranked)),
        },
    }


def _build_emergent_confusion_memories(
    records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    *,
    min_support: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"supporting": set(), "counter": set()})
    for record in records:
        tags = _confusion_tags(record)
        if not tags:
            continue
        for tag in tags:
            if _is_failure(record) or _has_confusion_outcome(record):
                grouped[tag]["supporting"].add(_case_id(record))
            elif _is_success(record):
                grouped[tag]["counter"].add(_case_id(record))
    for hard_case in hard_cases:
        for tag in [str(item).strip().lower() for item in hard_case.get("confusion_tags", []) if str(item).strip()]:
            grouped[tag]["supporting"].add(str(hard_case.get("case_id", "")).strip())

    results: list[dict[str, Any]] = []
    for confusion_pair, bundle in grouped.items():
        support_cases = sorted(case_id for case_id in bundle["supporting"] if case_id)
        counter_cases = sorted(case_id for case_id in bundle["counter"] if case_id and case_id not in bundle["supporting"])
        if len(support_cases) < min_support:
            continue
        results.append(
            {
                "confusion_pair": confusion_pair,
                "supporting_cases": support_cases,
                "counter_cases": counter_cases[:10],
                "evidence_strength": _evidence_strength_label(len(support_cases), len(counter_cases)),
                "reason_to_create": "Repeated confusion-linked failures or ambiguity suggest a new confusion_memory candidate.",
            }
        )
    results.sort(key=lambda item: len(item.get("supporting_cases", [])), reverse=True)
    return results


def _build_rule_candidates(records: list[dict[str, Any]], *, min_support: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"support": set(), "counter": set(), "records": []})
    for record in records:
        for tactical in _tactical_experiences(record):
            condition = dict(tactical.get("condition", {}) or {})
            action = dict(tactical.get("action", {}) or {})
            trigger_type = str(condition.get("trigger_type", "")).strip() or "unknown_trigger"
            decision_pattern = str(action.get("decision_pattern", "")).strip() or "unknown_decision_pattern"
            key = f"{trigger_type}|{decision_pattern}"
            grouped[key]["records"].append(tactical)
            case_id = str(tactical.get("case_id", "")).strip() or _case_id(record)
            if _is_success(record):
                grouped[key]["support"].add(case_id)
            elif _is_failure(record):
                grouped[key]["counter"].add(case_id)

    candidates: list[dict[str, Any]] = []
    for key, bundle in grouped.items():
        support_cases = sorted(case_id for case_id in bundle["support"] if case_id)
        counter_cases = sorted(case_id for case_id in bundle["counter"] if case_id and case_id not in bundle["support"])
        if len(support_cases) < min_support:
            continue
        trigger_type, _, decision_pattern = key.partition("|")
        top_skills = Counter(
            skill_name
            for record in bundle["records"]
            for skill_name in record.get("action", {}).get("skills_used", [])
            if str(skill_name).strip()
        )
        candidates.append(
            {
                "candidate_id": f"batch_rule_{stable_hash({'key': key, 'cases': support_cases})}",
                "pattern_summary": {
                    "trigger_type": trigger_type,
                    "decision_pattern": decision_pattern,
                    "top_skills": [{"name": name, "count": count} for name, count in top_skills.most_common(5)],
                },
                "supporting_cases": support_cases,
                "counter_cases": counter_cases[:10],
                "derived_rule": {
                    "rule_text": "Repeated tactical condition->action traces suggest a reusable rule candidate for planning and reflection.",
                    "action_hint": decision_pattern,
                },
                "evidence_strength": _evidence_strength_label(len(support_cases), len(counter_cases)),
            }
        )
    candidates.sort(key=lambda item: len(item.get("supporting_cases", [])), reverse=True)
    return candidates


def _build_composite_skill_seed_candidates(records: list[dict[str, Any]], *, min_support: int) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sequence = _selected_skills(record)
        if len(sequence) < 2:
            continue
        key = "|".join(sequence)
        grouped[key].add(_case_id(record))

    results: list[dict[str, Any]] = []
    for key, support_cases in grouped.items():
        if len(support_cases) < min_support:
            continue
        sequence = key.split("|")
        results.append(
            {
                "seed_id": f"batch_seed_{stable_hash({'sequence': sequence, 'cases': sorted(support_cases)})}",
                "trigger_pattern": {
                    "decision_pattern": "batch_success_sequence",
                    "case_outcome": "success",
                },
                "skill_sequence": sequence,
                "supporting_cases": sorted(support_cases),
                "success_count": len(support_cases),
                "notes": [
                    "Repeated successful cross-rollout sequence detected.",
                    "Suitable as upstream input for composite skill seed consolidation.",
                ],
            }
        )
    results.sort(key=lambda item: int(item.get("success_count", 0)), reverse=True)
    return results


def _build_refinement_inputs(
    records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    refinement_candidates: list[dict[str, Any]],
    *,
    min_support: int,
) -> dict[str, Any]:
    failure_pattern_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    for hard_case in hard_cases:
        failure_type = str(hard_case.get("failure_type", "unknown_failure")).strip()
        failure_pattern_counter[failure_type] += 1
        for skill_name in [str(item).strip() for item in hard_case.get("involved_skills", []) if str(item).strip()]:
            skill_counter[skill_name] += 1

    success_sequences = _build_composite_skill_seed_candidates([record for record in records if _is_success(record)], min_support=min_support)
    consolidation_ready_signals = []
    for section_name, payload in (
        ("emergent_confusion_memories", _build_emergent_confusion_memories(records, hard_cases, min_support=min_support)),
        ("rule_candidates", _build_rule_candidates(records, min_support=min_support)),
        ("composite_skill_seed_candidates", success_sequences),
    ):
        if payload:
            consolidation_ready_signals.append({"section": section_name, "count": len(payload)})

    mapped_refinement = [
        {
            "target_skill_name": str(item.get("target_skill_name", "")).strip(),
            "proposed_update_type": str(item.get("proposed_update_type", "")).strip(),
            "confidence": str(item.get("confidence", "")).strip(),
            "supporting_case_count": len(item.get("supporting_cases", [])),
        }
        for item in refinement_candidates
        if str(item.get("target_skill_name", "")).strip()
    ]

    failure_patterns_for_skill_refinement = [
        {
            "failure_type": name,
            "count": count,
            "most_involved_skills": [
                {"name": skill_name, "count": skill_count}
                for skill_name, skill_count in skill_counter.most_common(5)
            ],
        }
        for name, count in failure_pattern_counter.most_common()
        if count >= min_support
    ]

    return {
        "failure_patterns_for_skill_refinement": failure_patterns_for_skill_refinement,
        "success_patterns_for_policy_reuse": success_sequences[:10],
        "consolidation_ready_signals": consolidation_ready_signals,
        "existing_refinement_candidates_summary": mapped_refinement[:20],
    }


def _retrieved_experiences(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("retrieval_bundle", {}).get("after_skills", {}).get("aggregator_summary", []))


def _tactical_experiences(record: dict[str, Any]) -> list[dict[str, Any]]:
    writeback_bundle = record.get("writeback_ops", {}).get("planned_writeback_bundle", {})
    return [item for item in writeback_bundle.get("tactical_experiences", []) if isinstance(item, dict)]


def _selected_skills(record: dict[str, Any]) -> list[str]:
    planner = record.get("planner_decision", {})
    ordering = [str(item).strip() for item in planner.get("ordering", []) if str(item).strip()]
    if ordering:
        return ordering
    return [str(item).strip() for item in record.get("selected_skills", []) if str(item).strip()]


def _case_id(record: dict[str, Any]) -> str:
    return str(record.get("case_id", "")).strip()


def _is_success(record: dict[str, Any]) -> bool:
    evaluation = record.get("evaluation", {})
    if evaluation.get("correct") is True:
        return True
    status = str(record.get("reflection_summary", {}).get("case_outcome", {}).get("status", "")).strip().lower()
    return status == "success"


def _is_failure(record: dict[str, Any]) -> bool:
    evaluation = record.get("evaluation", {})
    if evaluation.get("correct") is False:
        return True
    status = str(record.get("reflection_summary", {}).get("case_outcome", {}).get("status", "")).strip().lower()
    return status == "failure"


def _confusion_tags(record: dict[str, Any]) -> list[str]:
    tags = []
    pair = str(record.get("reflection_summary", {}).get("case_outcome", {}).get("confusion_pair", "")).strip()
    if pair:
        tags.append(pair.lower())
    query_pair = str(record.get("skill_retrieval", {}).get("query_summary", {}).get("confusion_pair", "")).strip()
    if query_pair:
        tags.append(query_pair.lower())
    return dedupe_strings(tags)


def _has_confusion_outcome(record: dict[str, Any]) -> bool:
    return bool(_confusion_tags(record))


def _case_failure_type(record: dict[str, Any]) -> str:
    case_outcome = record.get("reflection_summary", {}).get("case_outcome", {})
    failure_type = str(case_outcome.get("failure_type", "")).strip()
    if failure_type:
        return failure_type
    uncertainty_level = str(record.get("evidence_bundle", {}).get("uncertainty_summary", {}).get("uncertainty_level", "")).lower()
    if uncertainty_level == "high":
        return "high_uncertainty_case"
    contradiction_count = _contradiction_count(record)
    if contradiction_count >= 3:
        return "contradiction_heavy_case"
    return "unknown_failure"


def _contradiction_count(record: dict[str, Any]) -> int:
    summary = record.get("evidence_bundle", {}).get("contradiction_summary", {})
    total = 0
    for field_name in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts"):
        total += len(summary.get(field_name, []) or [])
    return total


def _top_experiences(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    counter: dict[tuple[str, str, str], int] = Counter()
    for record in records:
        seen: set[tuple[str, str, str]] = set()
        for experience in _retrieved_experiences(record):
            key = (
                str(experience.get("source_id", "")).strip(),
                str(experience.get("source_layer", "")).strip(),
                str(experience.get("experience_type", "")).strip(),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            counter[key] += 1
    return [
        {"source_id": source_id, "source_layer": layer, "experience_type": exp_type, "count": count}
        for (source_id, layer, exp_type), count in counter.most_common(top_k)
    ]


def _top_confusion_tags(records: list[dict[str, Any]], *, top_k: int) -> list[str]:
    counter = Counter(tag for record in records for tag in _confusion_tags(record))
    return [name for name, _ in counter.most_common(top_k)]


def _representative_signals(records: list[dict[str, Any]]) -> list[str]:
    counter = Counter()
    for record in records:
        counter.update(_case_signal_tokens(record))
    return [name for name, _ in counter.most_common(8)]


def _case_signal_tokens(record: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    qwen_initial = record.get("qwen_initial", {})
    ddx = [str(item).strip() for item in qwen_initial.get("ddx_candidates", []) if str(item).strip()]
    if ddx:
        tokens.append(f"ddx:{'|'.join(ddx[:2])}")
    metadata = record.get("input_summary", {}).get("clinical_metadata", {})
    region = str(metadata.get("region", "")).strip().lower()
    if region:
        tokens.append(f"region:{region}")
    uncertainty = str(record.get("evidence_bundle", {}).get("uncertainty_summary", {}).get("uncertainty_level", "")).strip().lower()
    if uncertainty:
        tokens.append(f"uncertainty:{uncertainty}")
    for flag in record.get("reflection_summary", {}).get("case_outcome", {}).get("risk_flags", [])[:3]:
        text = str(flag).strip()
        if text:
            tokens.append(f"risk:{text}")
    return dedupe_strings(tokens)


def _most_common_sequence(records: list[dict[str, Any]]) -> list[str]:
    counter = Counter("|".join(_selected_skills(record)) for record in records if _selected_skills(record))
    if not counter:
        return []
    return counter.most_common(1)[0][0].split("|")


def _top_failure_patterns(hard_cases: list[dict[str, Any]]) -> list[str]:
    counter = Counter(
        reason
        for hard_case in hard_cases
        if isinstance(hard_case, dict)
        for reason in hard_case.get("hardness_reason", [])
        if str(reason).strip()
    )
    return [name for name, _ in counter.most_common(6)]


def _evidence_strength_label(support_count: int, counter_count: int) -> str:
    if support_count >= 5 and counter_count <= 1:
        return "high"
    if support_count >= 3:
        return "medium"
    return "low"


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
