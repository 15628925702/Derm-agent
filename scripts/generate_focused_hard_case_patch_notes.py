from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("/root/DermAgent/outputs/smoke_cycles/medium_signal_no_ablation_v1")
DEFAULT_HARD_CASE_SUMMARY = DEFAULT_RUN_ROOT / "hard_case_mining" / "summary.json"
DEFAULT_BATCH_CRITIQUE = DEFAULT_RUN_ROOT / "batch_reflection" / "batch_critique.json"
DEFAULT_REFINEMENT_DIR = DEFAULT_RUN_ROOT / "skill_refinement_candidates"


ROOT_CAUSE_MAP: dict[str, list[str]] = {
    "scc_bcc_family": ["specialist_insufficient", "abstract_retrieval_insufficient", "evidence_ordering_insufficient"],
    "ak_bcc_family": ["exclusion_reasoning_insufficient", "evidence_ordering_insufficient"],
    "sek_bcc_family": ["exclusion_reasoning_insufficient", "abstract_retrieval_insufficient"],
    "bcc_benign_mimic_family": ["specialist_trigger_insufficient", "benign_mimic_routing_insufficient", "abstract_retrieval_insufficient"],
    "inflammatory_ack_family": ["metadata_consistency_handling_insufficient", "exclusion_reasoning_insufficient"],
    "mel_nev_family": ["specialist_trigger_insufficient", "abstract_retrieval_insufficient"],
}


PATCH_PLAN: list[dict[str, Any]] = [
    {
        "patch_id": "step5_patch_keratin_specialist_workflow",
        "target_case_type": "scc_bcc_family",
        "update_types": ["specialist_workflow_text_refinement", "trigger_refinement"],
        "files": ["/root/DermAgent/skills/ack_scc_specialist.py"],
    },
    {
        "patch_id": "step5_patch_specialist_trigger_routing",
        "target_case_type": "scc_bcc_family",
        "update_types": ["trigger_refinement", "planner_routing_refinement"],
        "files": ["/root/DermAgent/agent/planner.py", "/root/DermAgent/agent/skill_retriever.py"],
    },
    {
        "patch_id": "step5_patch_exclusion_rules",
        "target_case_type": "ak_bcc_family",
        "update_types": ["exclusion_rule_enhancement"],
        "files": ["/root/DermAgent/skills/exclusion_reasoning.py"],
    },
    {
        "patch_id": "step5_patch_bcc_benign_mimic_routing",
        "target_case_type": "bcc_benign_mimic_family",
        "update_types": ["specialist_trigger_refinement", "workflow_routing_refinement"],
        "files": ["/root/DermAgent/agent/skill_retriever.py", "/root/DermAgent/agent/planner.py", "/root/DermAgent/skills/benign_mimic_specialist.py"],
    },
    {
        "patch_id": "step5_patch_metadata_inflammatory_guard",
        "target_case_type": "inflammatory_ack_family",
        "update_types": ["metadata_consistency_handling_refinement"],
        "files": ["/root/DermAgent/skills/metadata_consistency.py"],
    },
    {
        "patch_id": "step5_patch_abstract_retrieval_preference",
        "target_case_type": "scc_bcc_family",
        "update_types": ["retrieval_preference_refinement"],
        "files": [
            "/root/DermAgent/memory/experience_retriever.py",
            "/root/DermAgent/skills/base.py",
            "/root/DermAgent/skills/mel_nev_specialist.py",
        ],
    },
    {
        "patch_id": "step5_patch_evidence_ordering",
        "target_case_type": "ak_bcc_family",
        "update_types": ["evidence_ordering_refinement"],
        "files": ["/root/DermAgent/agent/evidence_calibrator.py"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewed focused hard-case patch notes for Step 5.")
    parser.add_argument("--hard-case-summary", type=Path, default=DEFAULT_HARD_CASE_SUMMARY)
    parser.add_argument("--batch-critique", type=Path, default=DEFAULT_BATCH_CRITIQUE)
    parser.add_argument("--refinement-dir", type=Path, default=DEFAULT_REFINEMENT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_family(tag: str) -> str:
    text = str(tag).strip().lower()
    if "bcc_benign_mimic" in text:
        return "bcc_benign_mimic_family"
    if ("scc" in text or "squamous" in text) and "bcc" in text:
        return "scc_bcc_family"
    if ("ack" in text or "actinic" in text) and "bcc" in text:
        return "ak_bcc_family"
    if ("seborrheic" in text or "sek" in text) and "bcc" in text:
        return "sek_bcc_family"
    if ("lichen" in text or "psoriasis" in text or "dermatitis" in text or "eczema" in text) and ("ack" in text or "actinic" in text):
        return "inflammatory_ack_family"
    if ("mel" in text or "melanoma" in text) and ("nev" in text or "naevus" in text or "mole" in text):
        return "mel_nev_family"
    return "other"


def _collect_top_families(hard_case_summary: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    by_confusion = dict(hard_case_summary.get("counts", {}).get("by_confusion_tag", {}) or {})
    family_counts: Counter[str] = Counter()
    family_examples: dict[str, list[str]] = {}
    for tag, count in by_confusion.items():
        family = _cluster_family(tag)
        if family == "other":
            continue
        family_counts[family] += int(count)
        family_examples.setdefault(family, [])
        if len(family_examples[family]) < 4:
            family_examples[family].append(str(tag))
    rows: list[dict[str, Any]] = []
    for family, count in family_counts.most_common(max(3, top_k)):
        rows.append(
            {
                "cluster_family": family,
                "support_count": int(count),
                "example_tags": family_examples.get(family, []),
                "root_causes": list(ROOT_CAUSE_MAP.get(family, ["unknown"])),
            }
        )
    return rows[:top_k]


def _build_patch_candidates(top_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    support_lookup = {item.get("cluster_family"): int(item.get("support_count", 0)) for item in top_clusters}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    candidates: list[dict[str, Any]] = []
    for item in PATCH_PLAN:
        family = str(item.get("target_case_type", ""))
        candidates.append(
            {
                "candidate_id": item["patch_id"],
                "target_case_type": family,
                "root_causes": list(ROOT_CAUSE_MAP.get(family, [])),
                "support_count": int(support_lookup.get(family, 0)),
                "proposed_update_type": "focused_patch",
                "proposed_change_summary": {
                    "update_types": list(item.get("update_types", [])),
                    "files": list(item.get("files", [])),
                    "goal": "Conservative patch for medium-run hard clusters without adding new skills.",
                },
                "status": "implemented",
                "created_at": now,
            }
        )
    return candidates


def main() -> int:
    args = parse_args()
    hard_case_summary = _load_json(args.hard_case_summary)
    batch_critique = _load_json(args.batch_critique)
    args.refinement_dir.mkdir(parents=True, exist_ok=True)

    top_clusters = _collect_top_families(hard_case_summary, top_k=max(3, int(args.top_k)))
    patch_candidates = _build_patch_candidates(top_clusters)
    payload = {
        "note_id": "focused_hard_case_patch_step5",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_files": {
            "hard_case_summary": str(args.hard_case_summary),
            "batch_critique": str(args.batch_critique),
        },
        "source_counts": {
            "hard_case_count": int(hard_case_summary.get("hard_case_count", 0) or 0),
            "failure_cluster_count": int(batch_critique.get("source_summary", {}).get("failure_case_count", 0) or 0),
        },
        "top_confusion_clusters": top_clusters,
        "patch_plan": patch_candidates,
        "notes": [
            "This is a conservative focused patch record for pre-medium-run repair.",
            "No new skill family is introduced; only workflow/trigger/retrieval/order refinements are applied.",
        ],
    }

    notes_path = args.refinement_dir / "reviewed_patch_notes_step5.json"
    notes_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates_path = args.refinement_dir / "focused_patch_candidates_step5.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in patch_candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "notes_path": str(notes_path),
                "candidates_path": str(candidates_path),
                "top_cluster_count": len(top_clusters),
                "patch_candidate_count": len(patch_candidates),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
