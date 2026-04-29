from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.confusion_clusters import get_confusion_cluster_definitions
from cognition.cognition_state import CognitionState
from memory.experience_schema import stable_hash

DEFAULT_PROPOSALS_DIR = Path("/root/DermAgent/proposals/confusion_triggered_skills")
DEFAULT_THRESHOLD = 3


def _slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _covered_pairs(dataset_name: str | None = None) -> set[str]:
    clusters = get_confusion_cluster_definitions(dataset_name)
    covered: set[str] = set()
    for definition in clusters.values():
        for pair in definition.get("pairs", ()):
            covered.add(str(pair).strip().lower())
    return covered


def _build_proposal(pair: str, count: int) -> dict[str, Any]:
    parts = pair.split("->")
    predicted = parts[0].strip() if parts else pair
    reference = parts[1].strip() if len(parts) > 1 else ""

    pair_slug = _slugify(pair.replace("->", "_vs_"))
    skill_name = f"{pair_slug}_specialist_skill"
    skill_id = f"skill.{pair_slug}_specialist.v1"
    proposal_id = stable_hash({"pair": pair, "type": "confusion_triggered"})

    workflow_text = (
        f"Focus narrowly on the {predicted.upper()}-versus-{reference.upper() if reference else 'unknown'} confusion pair. "
        f"Inspect the lesion for features that would help an expert distinguish between these two diagnoses. "
        f"Surface explicit supporting and opposing evidence for each candidate. "
        f"Do not make a final diagnosis — provide structured differentiation evidence only."
    )

    return {
        "proposal_id": proposal_id,
        "_source_type": "confusion_triggered",
        "confusion_pair": pair,
        "confusion_count": count,
        "source_seed_ids": [],
        "trigger_pattern": {
            "condition": f"Trigger when both {predicted} and {reference} are active DDx candidates, or when confusion memory indicates recurrent {predicted}↔{reference} ambiguity.",
            "decision_pattern": f"confusion:{pair}",
            "confusion_pair": pair,
        },
        "intended_scope": {
            "use_case": f"Specialist differentiation for the {predicted}↔{reference} confusion pair.",
            "applicable_when": f"Both {predicted} and {reference} appear in the differential diagnosis.",
            "do_not_use_when": f"Neither {predicted} nor {reference} is in the active differential.",
        },
        "proposed_workflow_text": workflow_text,
        "skill_sequence": [],
        "supporting_cases": [],
        "counter_cases": [],
        "expected_benefit": {
            "planner": f"Reduces {predicted}↔{reference} confusion by surfacing targeted differentiation evidence.",
            "evidence_bundle": "Provides explicit supporting/opposing evidence for each candidate.",
        },
        "risk_notes": [
            f"Auto-generated from confusion threshold ({count} occurrences). Review clinical accuracy before deploying.",
            "Do not make a final diagnosis in this skill.",
        ],
        "review_status": "pending_review",
        "review_checklist": [
            f"Verify that {predicted} and {reference} are clinically meaningful confusion pair.",
            "Check that workflow_text is clinically accurate.",
            "Confirm trigger condition is not too broad.",
            "Ensure skill does not output a final diagnosis.",
        ],
        "proposal_artifacts": {
            "suggested_skill_name": skill_name,
            "suggested_skill_id": skill_id,
            "description": f"Specialist skill for {predicted}↔{reference} differentiation, auto-generated from confusion threshold.",
            "integration_targets": ["planner", "skill_retriever"],
        },
        "evidence_refs": {
            "confusion_pair": pair,
            "confusion_count": count,
            "source": "cognition_state.known_confusion_patterns",
        },
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def generate_confusion_triggered_proposals(
    cognition: CognitionState,
    threshold: int = DEFAULT_THRESHOLD,
    dataset_name: str | None = None,
    output_dir: Path = DEFAULT_PROPOSALS_DIR,
) -> list[dict[str, Any]]:
    covered = _covered_pairs(dataset_name)
    proposals = []

    for pair, count in cognition.known_confusion_patterns.items():
        if count < threshold:
            continue
        normalized = str(pair).strip().lower()
        if normalized in covered:
            continue
        proposals.append(_build_proposal(pair, count))

    if proposals:
        output_dir.mkdir(parents=True, exist_ok=True)
        for proposal in proposals:
            pair_slug = _slugify(proposal["confusion_pair"].replace("->", "_vs_"))
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = output_dir / f"{pair_slug}_{timestamp}.json"
            if not out_path.exists():
                out_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    return proposals
