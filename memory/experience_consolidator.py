from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.label_space import canonicalize_label
from memory.experience_schema import AbstractExperience, CompositeSkillSeed, dedupe_strings, stable_hash
from memory.experience_store import ExperienceStore


CONSOLIDATION_VERSION = "consolidated_v1"
DEFAULT_MIN_SUPPORTING_CASES = 2
DEFAULT_OUTPUT_DIR = Path("/root/DermAgent/outputs/experience_consolidation")
DEFAULT_HARD_CASES_PATH = Path("/root/DermAgent/outputs/hard_case_mining/hard_cases.jsonl")


@dataclass
class ConsolidationResult:
    consolidated_records: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def consolidate_experiences(
    *,
    store: ExperienceStore,
    hard_cases_path: str | Path = DEFAULT_HARD_CASES_PATH,
    min_supporting_cases: int = DEFAULT_MIN_SUPPORTING_CASES,
    write_to_store: bool = True,
) -> ConsolidationResult:
    raw_records = store.load_raw_case_memories()
    tactical_records = store.load_tactical_experiences()
    abstract_records = store.load_abstract_experiences()
    hard_cases = _load_jsonl(Path(hard_cases_path))

    generated: list[AbstractExperience] = []
    generated.extend(
        _consolidate_prototypes(
            raw_records=raw_records,
            hard_cases=hard_cases,
            min_supporting_cases=min_supporting_cases,
        )
    )
    generated.extend(
        _consolidate_confusion_memories(
            raw_records=raw_records,
            tactical_records=tactical_records,
            hard_cases=hard_cases,
            min_supporting_cases=min_supporting_cases,
        )
    )
    generated.extend(
        _consolidate_rule_candidates(
            tactical_records=tactical_records,
            raw_records=raw_records,
            hard_cases=hard_cases,
            min_supporting_cases=min_supporting_cases,
        )
    )
    generated.extend(
        _consolidate_composite_skill_seeds(
            tactical_records=tactical_records,
            abstract_records=abstract_records,
            hard_cases=hard_cases,
            min_supporting_cases=min_supporting_cases,
        )
    )

    consolidated_dicts = [record.to_dict() for record in generated]
    persisted_records = store.upsert_abstract_experiences(consolidated_dicts) if write_to_store and consolidated_dicts else consolidated_dicts
    if write_to_store:
        store.refresh_metadata()

    summary = _build_consolidation_summary(
        raw_records=raw_records,
        tactical_records=tactical_records,
        abstract_records=abstract_records,
        hard_cases=hard_cases,
        consolidated_records=persisted_records,
        min_supporting_cases=min_supporting_cases,
        write_to_store=write_to_store,
    )
    return ConsolidationResult(
        consolidated_records=persisted_records,
        summary=summary,
    )


def save_consolidation_outputs(
    result: ConsolidationResult,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "consolidated_abstract_experiences.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in result.consolidated_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)
    return {
        "records_path": str(records_path),
        "summary_path": str(summary_path),
    }


def _consolidate_prototypes(
    *,
    raw_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    min_supporting_cases: int,
) -> list[AbstractExperience]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        correctness = record.get("correctness", {})
        if correctness.get("is_correct") is not True:
            continue
        label = canonicalize_label(
            record.get("true_label") or record.get("final_decision", {}).get("canonical_label"),
            dataset_name=record.get("dataset_name"),
            label_space_id=record.get("label_space_id"),
            metadata=record.get("metadata", {}),
        )
        if not label:
            continue
        grouped[label].append(record)

    consolidated: list[AbstractExperience] = []
    for label, records in grouped.items():
        distinct_cases = _distinct_case_ids(records)
        if len(distinct_cases) < min_supporting_cases:
            continue
        counter_cases = _prototype_counter_cases(label=label, raw_records=raw_records, hard_cases=hard_cases)
        supporting_patterns = _summaries_from_raw(records)
        region_counter = Counter(_region_from_raw(record) for record in records if _region_from_raw(record))
        concept = str(label).lower()
        abs_id = f"abs_proto_cons_{stable_hash({'type': 'prototype', 'concept': concept})}"
        consolidated.append(
            AbstractExperience(
                abs_id=abs_id,
                type="prototype",
                concept=concept,
                pattern_summary={
                    "supporting_pattern": supporting_patterns[:5],
                    "common_regions": [{"name": name, "count": count} for name, count in region_counter.most_common(3)],
                    "prototype_kind": "label_stable_pattern",
                },
                supporting_cases=distinct_cases,
                counter_cases=counter_cases,
                derived_rule={
                    "rule_text": "Use this as a reference prototype for supportive comparison, not as an automatic label shortcut.",
                    "action_hint": "retrieve_as_reference_pattern",
                },
                version=CONSOLIDATION_VERSION,
                provenance=_provenance(
                    source="periodic_consolidation",
                    source_case_ids=distinct_cases,
                    source_exp_ids=[],
                    source_abs_ids=[],
                    source_hard_case_ids=_hard_case_ids_for_cases(hard_cases, distinct_cases),
                    evidence_thresholds={"min_supporting_cases": min_supporting_cases},
                    input_counts={"raw_records": len(records)},
                    generator="experience_consolidator_v1",
                ),
            )
        )
    return consolidated


def _consolidate_confusion_memories(
    *,
    raw_records: list[dict[str, Any]],
    tactical_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    min_supporting_cases: int,
) -> list[AbstractExperience]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"raw": [], "tactical": [], "hard": []})
    for record in raw_records:
        confusion_pair = _raw_confusion_pair(record)
        if confusion_pair:
            grouped[confusion_pair]["raw"].append(record)
    for record in tactical_records:
        confusion_pair = str(record.get("condition", {}).get("confusion_pair", "")).strip()
        if confusion_pair:
            grouped[confusion_pair]["tactical"].append(record)
    for record in hard_cases:
        for confusion_tag in [str(item).strip() for item in record.get("confusion_tags", []) if str(item).strip()]:
            grouped[confusion_tag]["hard"].append(record)

    consolidated: list[AbstractExperience] = []
    for confusion_pair, bundle in grouped.items():
        supporting_cases = _distinct_case_ids(bundle["raw"] + bundle["tactical"] + bundle["hard"])
        if len(supporting_cases) < min_supporting_cases:
            continue
        concept = confusion_pair.lower().replace(" ", "_")
        predicted_label, _, true_label = confusion_pair.partition("->")
        counter_cases = _confusion_counter_cases(
            predicted_label=canonicalize_label(predicted_label),
            true_label=canonicalize_label(true_label),
            raw_records=raw_records,
        )
        supporting_pattern = dedupe_strings(
            _summaries_from_raw(bundle["raw"])
            + _summaries_from_tactical(bundle["tactical"])
        )
        abs_id = f"abs_confusion_cons_{stable_hash({'type': 'confusion_memory', 'concept': concept})}"
        consolidated.append(
            AbstractExperience(
                abs_id=abs_id,
                type="confusion_memory",
                concept=concept,
                pattern_summary={
                    "confusion_pair": confusion_pair,
                    "supporting_pattern": supporting_pattern[:5],
                    "failure_pattern": _hard_case_failure_types(bundle["hard"]),
                },
                supporting_cases=supporting_cases,
                counter_cases=counter_cases,
                derived_rule={
                    "rule_text": "When this confusion pair appears repeatedly, force differential openness and explicit negative-evidence review before final Qwen integration.",
                    "action_hint": "retrieve_confusion_memory",
                },
                version=CONSOLIDATION_VERSION,
                provenance=_provenance(
                    source="periodic_consolidation",
                    source_case_ids=supporting_cases,
                    source_exp_ids=[record.get("exp_id") for record in bundle["tactical"] if record.get("exp_id")],
                    source_abs_ids=[],
                    source_hard_case_ids=[record.get("hard_case_id") for record in bundle["hard"] if record.get("hard_case_id")],
                    evidence_thresholds={"min_supporting_cases": min_supporting_cases},
                    input_counts={
                        "raw_records": len(bundle["raw"]),
                        "tactical_records": len(bundle["tactical"]),
                        "hard_cases": len(bundle["hard"]),
                    },
                    generator="experience_consolidator_v1",
                ),
            )
        )
    return consolidated


def _consolidate_rule_candidates(
    *,
    tactical_records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    min_supporting_cases: int,
) -> list[AbstractExperience]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"support": [], "counter": []})
    for record in tactical_records:
        key = _rule_candidate_key(record)
        if not key:
            continue
        outcome_result = str(record.get("outcome", {}).get("result", "")).strip().lower()
        case_status = str(record.get("outcome", {}).get("case_status", "")).strip().lower()
        if outcome_result in {"success", "partially_helpful"} or case_status in {"success", "partially_helpful"}:
            grouped[key]["support"].append(record)
        else:
            grouped[key]["counter"].append(record)

    consolidated: list[AbstractExperience] = []
    for key, bundle in grouped.items():
        supporting_cases = _distinct_case_ids(bundle["support"])
        if len(supporting_cases) < min_supporting_cases:
            continue
        trigger_type, decision_pattern = key.split("|", 1)
        counter_cases = _distinct_case_ids(bundle["counter"]) + _rule_counter_cases_from_hard_cases(
            support_records=bundle["support"],
            hard_cases=hard_cases,
        )
        counter_cases = list(dict.fromkeys(counter_cases))[:10]
        skill_counter = Counter(
            skill_name
            for record in bundle["support"]
            for skill_name in record.get("action", {}).get("skills_used", [])
            if str(skill_name).strip()
        )
        concept = decision_pattern or trigger_type
        abs_id = f"abs_rule_candidate_{stable_hash({'type': 'rule_candidate', 'concept': concept, 'trigger_type': trigger_type})}"
        consolidated.append(
            AbstractExperience(
                abs_id=abs_id,
                type="rule_candidate",
                concept=concept,
                pattern_summary={
                    "trigger_type": trigger_type,
                    "decision_pattern": decision_pattern,
                    "supporting_pattern": _summaries_from_tactical(bundle["support"])[:5],
                    "top_skills": [{"name": name, "count": count} for name, count in skill_counter.most_common(5)],
                },
                supporting_cases=supporting_cases,
                counter_cases=counter_cases,
                derived_rule={
                    "rule_text": (
                        "This is a conservative rule candidate distilled from repeated tactical condition->action traces. "
                        "It should guide planning and auditing, not replace diagnosis."
                    ),
                    "action_hint": decision_pattern,
                },
                version=CONSOLIDATION_VERSION,
                provenance=_provenance(
                    source="periodic_consolidation",
                    source_case_ids=supporting_cases,
                    source_exp_ids=[record.get("exp_id") for record in bundle["support"] if record.get("exp_id")],
                    source_abs_ids=[],
                    source_hard_case_ids=_hard_case_ids_for_cases(hard_cases, counter_cases),
                    evidence_thresholds={"min_supporting_cases": min_supporting_cases},
                    input_counts={
                        "support_tactical_records": len(bundle["support"]),
                        "counter_tactical_records": len(bundle["counter"]),
                    },
                    generator="experience_consolidator_v1",
                ),
            )
        )
    return consolidated


def _consolidate_composite_skill_seeds(
    *,
    tactical_records: list[dict[str, Any]],
    abstract_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    min_supporting_cases: int,
) -> list[AbstractExperience]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"support_cases": set(), "seed_records": [], "tactical_records": []})
    for record in abstract_records:
        if str(record.get("type", "")).strip() != "composite_skill_seed":
            continue
        composite = record.get("composite_skill_seed", {})
        skill_sequence = [str(item).strip() for item in composite.get("skill_sequence", []) if str(item).strip()]
        decision_pattern = str(composite.get("trigger_pattern", {}).get("decision_pattern", "")).strip()
        key = "|".join([decision_pattern, ",".join(skill_sequence)])
        if not key.strip("|,"):
            continue
        grouped[key]["seed_records"].append(record)
        grouped[key]["support_cases"].update(str(item).strip() for item in composite.get("supporting_cases", []) if str(item).strip())

    for record in tactical_records:
        skills_used = [str(item).strip() for item in record.get("action", {}).get("skills_used", []) if str(item).strip()]
        if len(skills_used) < 2:
            continue
        decision_pattern = str(record.get("action", {}).get("decision_pattern", "")).strip()
        key = "|".join([decision_pattern, ",".join(skills_used)])
        grouped[key]["tactical_records"].append(record)
        if record.get("case_id"):
            grouped[key]["support_cases"].add(str(record.get("case_id")).strip())

    consolidated: list[AbstractExperience] = []
    for key, bundle in grouped.items():
        supporting_cases = sorted(case_id for case_id in bundle["support_cases"] if case_id)
        if len(supporting_cases) < min_supporting_cases:
            continue
        decision_pattern, _, sequence_text = key.partition("|")
        skill_sequence = [item for item in sequence_text.split(",") if item]
        if len(skill_sequence) < 2:
            continue
        counter_cases = _composite_counter_cases(
            decision_pattern=decision_pattern,
            skill_sequence=skill_sequence,
            hard_cases=hard_cases,
        )
        trigger_pattern = {
            "decision_pattern": decision_pattern,
            "seed_skills": skill_sequence,
        }
        seed_id = f"seed_consolidated_{stable_hash({'decision_pattern': decision_pattern, 'skill_sequence': skill_sequence})}"
        composite_seed = CompositeSkillSeed(
            seed_id=seed_id,
            trigger_pattern=trigger_pattern,
            skill_sequence=skill_sequence,
            supporting_cases=supporting_cases,
            success_count=len(supporting_cases),
            notes=[
                "Consolidated from repeated composite-skill patterns across distinct cases.",
                "This seed is intended for future workflow promotion, not automatic execution.",
            ],
            promotion_interface={
                "builder": "promote_composite_skill_seed",
                "output_contract_source": "component_skill_union",
                "trigger_pattern_source": "composite_skill_seed.trigger_pattern",
            },
        )
        abs_id = f"abs_composite_cons_{stable_hash({'type': 'composite_skill_seed', 'seed_id': seed_id})}"
        consolidated.append(
            AbstractExperience(
                abs_id=abs_id,
                type="composite_skill_seed",
                concept=decision_pattern or "_".join(skill_sequence),
                pattern_summary={
                    "trigger_pattern": trigger_pattern,
                    "seed_skills": skill_sequence,
                    "supporting_pattern": [decision_pattern] if decision_pattern else skill_sequence,
                },
                supporting_cases=supporting_cases,
                counter_cases=counter_cases,
                derived_rule={
                    "rule_text": "Repeated multi-skill sequences can be stored as composite seeds for future workflow promotion.",
                    "action_hint": decision_pattern or "composite_sequence",
                    "promotion_interface": deepcopy(composite_seed.promotion_interface),
                },
                version=CONSOLIDATION_VERSION,
                seed_id=seed_id,
                composite_skill_seed=composite_seed.to_dict(),
                provenance=_provenance(
                    source="periodic_consolidation",
                    source_case_ids=supporting_cases,
                    source_exp_ids=[record.get("exp_id") for record in bundle["tactical_records"] if record.get("exp_id")],
                    source_abs_ids=[record.get("abs_id") for record in bundle["seed_records"] if record.get("abs_id")],
                    source_hard_case_ids=_hard_case_ids_for_cases(hard_cases, counter_cases),
                    evidence_thresholds={"min_supporting_cases": min_supporting_cases},
                    input_counts={
                        "source_seed_records": len(bundle["seed_records"]),
                        "source_tactical_records": len(bundle["tactical_records"]),
                    },
                    generator="experience_consolidator_v1",
                ),
            )
        )
    return consolidated


def _build_consolidation_summary(
    *,
    raw_records: list[dict[str, Any]],
    tactical_records: list[dict[str, Any]],
    abstract_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
    consolidated_records: list[dict[str, Any]],
    min_supporting_cases: int,
    write_to_store: bool,
) -> dict[str, Any]:
    by_type = Counter(str(record.get("type", "unknown")) for record in consolidated_records)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "consolidation_version": CONSOLIDATION_VERSION,
        "write_to_store": write_to_store,
        "thresholds": {
            "min_supporting_cases": min_supporting_cases,
        },
        "input_counts": {
            "raw_case_memory": len(raw_records),
            "tactical_experience": len(tactical_records),
            "abstract_experience_existing": len(abstract_records),
            "hard_cases": len(hard_cases),
        },
        "output_counts": {
            "consolidated_records": len(consolidated_records),
            "by_type": dict(by_type),
        },
    }


def _rule_candidate_key(record: dict[str, Any]) -> str:
    condition = record.get("condition", {})
    action = record.get("action", {})
    trigger_type = str(condition.get("trigger_type", "")).strip()
    decision_pattern = str(action.get("decision_pattern", "")).strip()
    if not trigger_type and not decision_pattern:
        return ""
    return f"{trigger_type}|{decision_pattern}"


def _prototype_counter_cases(
    *,
    label: str,
    raw_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
) -> list[str]:
    counter_cases: list[str] = []
    for record in raw_records:
        case_id = str(record.get("case_id", "")).strip()
        if not case_id:
            continue
        true_label = canonicalize_label(
            record.get("true_label"),
            dataset_name=record.get("dataset_name"),
            label_space_id=record.get("label_space_id"),
            metadata=record.get("metadata", {}),
        )
        correctness = record.get("correctness", {})
        if true_label == label and correctness.get("is_correct") is not True:
            counter_cases.append(case_id)
    for record in hard_cases:
        case_id = str(record.get("case_id", "")).strip()
        if canonicalize_label(
            record.get("ground_truth", {}).get("canonical_label"),
            dataset_name=record.get("dataset_name"),
        ) == label and case_id:
            counter_cases.append(case_id)
    return list(dict.fromkeys(counter_cases))[:10]


def _confusion_counter_cases(
    *,
    predicted_label: str | None,
    true_label: str | None,
    raw_records: list[dict[str, Any]],
) -> list[str]:
    counter_cases: list[str] = []
    for record in raw_records:
        case_id = str(record.get("case_id", "")).strip()
        if not case_id:
            continue
        correctness = record.get("correctness", {})
        if correctness.get("is_correct") is not True:
            continue
        raw_true = canonicalize_label(
            record.get("true_label"),
            dataset_name=record.get("dataset_name"),
            label_space_id=record.get("label_space_id"),
            metadata=record.get("metadata", {}),
        )
        final_label = canonicalize_label(
            record.get("final_decision", {}).get("canonical_label"),
            dataset_name=record.get("dataset_name"),
            label_space_id=record.get("label_space_id"),
            metadata=record.get("metadata", {}),
        )
        if raw_true in {predicted_label, true_label} or final_label in {predicted_label, true_label}:
            counter_cases.append(case_id)
    return list(dict.fromkeys(counter_cases))[:10]


def _rule_counter_cases_from_hard_cases(
    *,
    support_records: list[dict[str, Any]],
    hard_cases: list[dict[str, Any]],
) -> list[str]:
    support_skill_set = {
        skill_name
        for record in support_records
        for skill_name in record.get("action", {}).get("skills_used", [])
        if str(skill_name).strip()
    }
    results: list[str] = []
    for hard_case in hard_cases:
        involved_skills = {str(item).strip() for item in hard_case.get("involved_skills", []) if str(item).strip()}
        if support_skill_set and support_skill_set.intersection(involved_skills):
            case_id = str(hard_case.get("case_id", "")).strip()
            if case_id:
                results.append(case_id)
    return list(dict.fromkeys(results))[:10]


def _composite_counter_cases(
    *,
    decision_pattern: str,
    skill_sequence: list[str],
    hard_cases: list[dict[str, Any]],
) -> list[str]:
    sequence_set = set(skill_sequence)
    results: list[str] = []
    for hard_case in hard_cases:
        involved_skills = {str(item).strip() for item in hard_case.get("involved_skills", []) if str(item).strip()}
        if decision_pattern and decision_pattern in json.dumps(hard_case, ensure_ascii=False):
            case_id = str(hard_case.get("case_id", "")).strip()
            if case_id:
                results.append(case_id)
            continue
        if sequence_set and sequence_set.issubset(involved_skills):
            case_id = str(hard_case.get("case_id", "")).strip()
            if case_id:
                results.append(case_id)
    return list(dict.fromkeys(results))[:10]


def _raw_confusion_pair(record: dict[str, Any]) -> str:
    true_label = canonicalize_label(
        record.get("true_label"),
        dataset_name=record.get("dataset_name"),
        label_space_id=record.get("label_space_id"),
        metadata=record.get("metadata", {}),
    )
    predicted_label = canonicalize_label(
        record.get("qwen_output", {}).get("final_diagnosis") or record.get("final_decision", {}).get("label"),
        dataset_name=record.get("dataset_name"),
        label_space_id=record.get("label_space_id"),
        metadata=record.get("metadata", {}),
    )
    correctness = record.get("correctness", {})
    if correctness.get("is_correct") is False and predicted_label and true_label:
        return f"{predicted_label}->{true_label}"
    return ""


def _summaries_from_raw(records: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for record in records:
        perception = record.get("agent_output", {}).get("perception", {})
        image_summary = str(perception.get("image_summary", "")).strip()
        if image_summary:
            summaries.append(image_summary)
            continue
        qwen_rationale = str(record.get("qwen_output", {}).get("rationale", "")).strip()
        if qwen_rationale:
            summaries.append(qwen_rationale[:160])
    return dedupe_strings(summaries)


def _summaries_from_tactical(records: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for record in records:
        condition = record.get("condition", {})
        observed_state = condition.get("observed_state", {})
        perception_summary = str(observed_state.get("perception_summary", "")).strip()
        if perception_summary:
            summaries.append(perception_summary)
        decision_pattern = str(record.get("action", {}).get("decision_pattern", "")).strip()
        if decision_pattern:
            summaries.append(f"decision_pattern={decision_pattern}")
    return dedupe_strings(summaries)


def _region_from_raw(record: dict[str, Any]) -> str:
    return str(record.get("metadata", {}).get("region", "")).strip()


def _distinct_case_ids(records: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(record.get("case_id", "")).strip()
            for record in records
            if str(record.get("case_id", "")).strip()
        )
    )


def _hard_case_failure_types(records: list[dict[str, Any]]) -> list[str]:
    return [
        f"{name}:{count}"
        for name, count in Counter(str(record.get("failure_type", "unknown")) for record in records).most_common(5)
    ]


def _hard_case_ids_for_cases(hard_cases: list[dict[str, Any]], case_ids: list[str]) -> list[str]:
    case_id_set = {str(item).strip() for item in case_ids if str(item).strip()}
    return [
        str(record.get("hard_case_id", "")).strip()
        for record in hard_cases
        if str(record.get("case_id", "")).strip() in case_id_set and str(record.get("hard_case_id", "")).strip()
    ]


def _provenance(
    *,
    source: str,
    source_case_ids: list[str],
    source_exp_ids: list[str],
    source_abs_ids: list[str],
    source_hard_case_ids: list[str],
    evidence_thresholds: dict[str, Any],
    input_counts: dict[str, Any],
    generator: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "generator": generator,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_case_ids": list(dict.fromkeys(source_case_ids)),
        "source_exp_ids": list(dict.fromkeys(item for item in source_exp_ids if item)),
        "source_abs_ids": list(dict.fromkeys(item for item in source_abs_ids if item)),
        "source_hard_case_ids": list(dict.fromkeys(item for item in source_hard_case_ids if item)),
        "evidence_thresholds": deepcopy(evidence_thresholds),
        "input_counts": deepcopy(input_counts),
    }


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
