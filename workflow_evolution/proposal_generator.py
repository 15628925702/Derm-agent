from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.label_space import canonicalize_label
from project_paths import repo_root


DEFAULT_OUTPUT_DIR = repo_root() / "proposals" / "workflow_evolution"

CORE_SKILLS = [
    "morphology_analysis_skill",
    "color_pattern_analysis_skill",
    "border_surface_analysis_skill",
    "distribution_analysis_skill",
    "lesion_description_structuring_skill",
    "malignancy_risk_assessment_skill",
    "differential_compare_skill",
]

HEAVY_REVIEW_SKILLS = [
    "metadata_consistency_skill",
    "uncertainty_assessment_skill",
    "information_gap_detection_skill",
    "contradiction_check_skill",
    "escalation_recommendation_skill",
    "exclusion_reasoning_skill",
]


def generate_workflow_evolution_proposal(
    *,
    report_paths: list[Path],
    model_name: str,
    dataset_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    doctor_experience_path: Path | None = None,
    base_workflow_cell_id: str = "",
) -> dict[str, Any]:
    cases = _load_cases(report_paths)
    metrics = _summarize_cases(cases, dataset_name=dataset_name)
    doctor_experience = _load_doctor_experience(doctor_experience_path)
    proposal_id = _proposal_id(
        model_name=model_name,
        dataset_name=dataset_name,
        metrics=metrics,
        doctor_experience=doctor_experience,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cell_id = base_workflow_cell_id or f"{_slug(model_name)}__{_slug(dataset_name)}__evolved_candidate_v1"

    proposal = {
        "schema_version": "workflow_evolution_candidate_v1",
        "proposal_type": "workflow_evolution_candidate",
        "proposal_id": proposal_id,
        "created_at": timestamp,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "review_status": "pending_review",
        "default_enabled": False,
        "activation": {
            "enabled": False,
            "requires_env": "DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1",
            "approved_dir": "state/workflow_evolution/approved",
            "one_key_enable_command": (
                f"python workflow_evolution/apply_proposal.py --proposal "
                f"proposals/workflow_evolution/{proposal_id}/workflow_evolution_proposal.json --approve --reviewer <name>"
            ),
        },
        "source_evidence": metrics,
        "doctor_experience": doctor_experience,
        "proposed_workflow_cell": _propose_workflow_cell(
            model_name=model_name,
            dataset_name=dataset_name,
            workflow_cell_id=cell_id,
            metrics=metrics,
        ),
        "proposed_conservative_fusion_adjustments": _propose_fusion_adjustments(metrics),
        "proposed_skill_adjustments": _propose_skill_adjustments(metrics),
        "proposed_runtime_refinements": _propose_runtime_refinements(metrics, doctor_experience),
        "safety_gates": [
            "Never activate directly from test-split results without human review.",
            "Run frozen compare on the same case batch before accepting a proposal.",
            "Require agent top1 > direct baseline top1 and record topk, malignant recall, helped/hurt/unchanged.",
            "If malignant recall drops, mark the proposal as risk_accepted or reject it.",
            "Keep generated skill/workflow candidates outside runtime until explicitly approved.",
        ],
    }

    save_workflow_evolution_proposal(proposal, output_dir=output_dir)
    return proposal


def save_workflow_evolution_proposal(proposal: dict[str, Any], *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    proposal_id = str(proposal.get("proposal_id", "")).strip() or "workflow_evolution_candidate"
    target_dir = output_dir / proposal_id
    target_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = target_dir / "workflow_evolution_proposal.json"
    readme_path = target_dir / "README.md"
    proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(_proposal_readme(proposal), encoding="utf-8")
    return {"proposal_path": str(proposal_path), "readme_path": str(readme_path)}


def _load_cases(report_paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in report_paths:
        payload = _read_json(path)
        for case in payload.get("cases", []) if isinstance(payload.get("cases", []), list) else []:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id", "")).strip()
            dedupe_key = case_id or f"{path}:{len(cases)}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cases.append(case)
    return cases


def _summarize_cases(cases: list[dict[str, Any]], *, dataset_name: str) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "baseline_to_truth": Counter(),
        "agent_to_truth": Counter(),
        "workflow_profile": Counter(),
        "workflow_cell": Counter(),
        "label_space": Counter(),
        "selected_skill": Counter(),
        "fusion_reason": Counter(),
        "helpful_agent_labels": Counter(),
        "hurt_agent_labels": Counter(),
    }
    helped: list[str] = []
    hurt: list[str] = []
    unchanged: list[str] = []
    malformed = Counter()
    baseline_top1 = agent_top1 = baseline_topk = agent_topk = 0
    malignant_n = malignant_baseline = malignant_agent = 0
    evaluated_n = 0

    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        gt = str(case.get("ground_truth", {}).get("canonical_label", "")).strip()
        if not gt:
            continue
        evaluated_n += 1
        evaluation = dict(case.get("evaluation", {}) or {})
        baseline_correct = bool(evaluation.get("baseline_correct", False))
        agent_correct = bool(evaluation.get("correct", False))
        baseline_top1 += int(baseline_correct)
        agent_top1 += int(agent_correct)
        baseline_topk += int(bool(evaluation.get("baseline_topk_hit", False)))
        agent_topk += int(bool(evaluation.get("topk_hit", False)))
        if bool(case.get("ground_truth", {}).get("malignant_flag", False)):
            malignant_n += 1
            malignant_baseline += int(bool(evaluation.get("baseline_malignant_recall_hit", False)))
            malignant_agent += int(bool(evaluation.get("malignant_recall_hit", False)))

        baseline_label = _canonical(case.get("baseline_qwen", {}).get("final_diagnosis"), dataset_name)
        agent_label = _canonical(case.get("qwen_final", {}).get("final_diagnosis"), dataset_name)
        if not str(case.get("qwen_final", {}).get("final_diagnosis", "")).strip():
            malformed["empty_final"] += 1
        if baseline_label != gt:
            counters["baseline_to_truth"][f"{baseline_label}->{gt}"] += 1
        if agent_label != gt:
            counters["agent_to_truth"][f"{agent_label}->{gt}"] += 1
        if agent_correct and not baseline_correct:
            helped.append(case_id)
            counters["helpful_agent_labels"][agent_label] += 1
        elif baseline_correct and not agent_correct:
            hurt.append(case_id)
            counters["hurt_agent_labels"][agent_label] += 1
        else:
            unchanged.append(case_id)

        workflow_context = _workflow_context(case)
        counters["workflow_profile"][str(workflow_context.get("workflow_profile", "unknown"))] += 1
        counters["workflow_cell"][str(workflow_context.get("workflow_cell_id", "unknown"))] += 1
        counters["label_space"][
            str(case.get("input_summary", {}).get("label_space_id") or workflow_context.get("label_space_id") or "unknown")
        ] += 1
        for skill in case.get("selected_skills", []) or []:
            counters["selected_skill"][str(skill)] += 1
        for reason in case.get("qwen_final", {}).get("fusion_decision", {}).get("reasons", []) or []:
            counters["fusion_reason"][str(reason)] += 1

    return {
        "case_count": len(cases),
        "evaluated_case_count": evaluated_n,
        "baseline_top1": baseline_top1,
        "agent_top1": agent_top1,
        "baseline_topk": baseline_topk,
        "agent_topk": agent_topk,
        "malignant_recall": {"baseline": malignant_baseline, "agent": malignant_agent, "denominator": malignant_n},
        "helped_case_ids": helped,
        "hurt_case_ids": hurt,
        "unchanged_count": len(unchanged),
        "malformed_timeout_empty": dict(malformed),
        "top_confusions": {
            name: counter.most_common(12)
            for name, counter in counters.items()
            if name in {"baseline_to_truth", "agent_to_truth"}
        },
        "distributions": {
            name: dict(counter)
            for name, counter in counters.items()
            if name not in {"baseline_to_truth", "agent_to_truth"}
        },
    }


def _propose_workflow_cell(*, model_name: str, dataset_name: str, workflow_cell_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    dataset_key = str(dataset_name).strip().lower()
    cell: dict[str, Any] = {
        "workflow_cell_id": workflow_cell_id,
        "inherit_dataset_workflow": True,
        "force_conservative_fusion": True,
        "disable_legacy_final_path": True,
        "workflow_capabilities": ["baseline_anchored_final", "graded_conservative_fusion"],
    }
    if dataset_key in {"scin", "sd198"}:
        cell["label_space_id"] = f"{dataset_key}_grouped"
        cell["environment"] = {f"DERMAGENT_{dataset_key.upper()}_LABEL_SPACE_ID": f"{dataset_key}_grouped"}
        cell["workflow_profile"] = "coarse_taxonomy_workflow"
        cell["workflow_capabilities"] = ["coarse_taxonomy_reasoning", "grouped_label_reasoning", "graded_conservative_fusion"]
    if dataset_key == "isic2019":
        cell["label_space_id"] = "isic2019_full"
        cell["workflow_profile"] = f"{_slug(model_name)}_isic2019_archive_guard_workflow"
        cell["workflow_capabilities"] = ["baseline_anchored_final", "melanocytic_guard_reasoning"]
    if metrics.get("malformed_timeout_empty"):
        cell["fallback_on_malformed_final"] = True
    if _hurt_count(metrics) > 0:
        cell["force_disable_skills"] = HEAVY_REVIEW_SKILLS
    return cell


def _propose_fusion_adjustments(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = []
    for pair, count in metrics.get("top_confusions", {}).get("baseline_to_truth", [])[:8]:
        proposals.append(
            {
                "confusion_pair": pair,
                "observed_count": count,
                "suggested_change": "consider_narrow_override_or_refinement",
                "guardrails": {
                    "selected_evidence_present": True,
                    "uncertainty_level": ["low", "medium"],
                    "minimum_support_margin": 44.0,
                    "minimum_subtype_support_margin": 7.0,
                    "block_if_malignant_recall_drops": True,
                },
            }
        )
    if _hurt_count(metrics) > 0:
        proposals.append(
            {
                "confusion_pair": "baseline_correct_agent_wrong",
                "observed_count": _hurt_count(metrics),
                "suggested_change": "tighten_conservative_fusion_or_anchor_baseline",
                "guardrails": {"do_not_allow_agent_final_passthrough": True},
            }
        )
    return proposals


def _propose_skill_adjustments(metrics: dict[str, Any]) -> dict[str, Any]:
    selected = Counter(metrics.get("distributions", {}).get("selected_skill", {}))
    core_selected = [skill for skill, _ in selected.most_common() if skill in CORE_SKILLS]
    return {
        "allowed_skills_candidate": core_selected or CORE_SKILLS,
        "force_disable_skills_candidate": HEAVY_REVIEW_SKILLS if _hurt_count(metrics) > 0 else [],
        "doctor_review_note": "A physician may add real clinical heuristics here; generated skills remain pending until reviewed.",
    }


def _propose_runtime_refinements(metrics: dict[str, Any], doctor_experience: dict[str, Any]) -> list[dict[str, Any]]:
    refinements = []
    for pair, count in metrics.get("top_confusions", {}).get("baseline_to_truth", [])[:6]:
        predicted, _, truth = pair.partition("->")
        refinements.append(
            {
                "trigger": {
                    "baseline_family": predicted,
                    "candidate_truth_family": truth,
                    "minimum_observed_count": count,
                },
                "action": "generate_candidate_family_refinement_rule",
                "source": "frozen_compare_error_pattern",
                "status": "pending_review",
            }
        )
    if doctor_experience.get("rules"):
        refinements.append(
            {
                "trigger": {"doctor_experience_rules": len(doctor_experience["rules"])},
                "action": "convert_reviewed_physician_rule_to_skill_or_fusion_guard",
                "source": "physician_experience_intake",
                "status": "pending_review",
            }
        )
    return refinements


def _load_doctor_experience(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "input_path": "",
            "rules": [],
            "note": "No physician experience file supplied. Use workflow_evolution/doctor_experience_template.md.",
        }
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    rules = [
        line.strip("- ").strip()
        for line in text.splitlines()
        if line.strip().startswith("-") and len(line.strip("- ").strip()) > 8
    ]
    return {"input_path": str(path), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "", "rules": rules}


def _proposal_readme(proposal: dict[str, Any]) -> str:
    metrics = proposal.get("source_evidence", {})
    return (
        f"# Workflow Evolution Proposal `{proposal.get('proposal_id')}`\n\n"
        "This is a disabled-by-default candidate. It does not affect runtime until reviewed, approved, and enabled with "
        "`DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1`.\n\n"
        f"- model: `{proposal.get('model_name')}`\n"
        f"- dataset: `{proposal.get('dataset_name')}`\n"
        f"- top1: `{metrics.get('baseline_top1')} -> {metrics.get('agent_top1')}`\n"
        f"- topk: `{metrics.get('baseline_topk')} -> {metrics.get('agent_topk')}`\n"
        f"- helped/hurt: `{len(metrics.get('helped_case_ids', []))}/{len(metrics.get('hurt_case_ids', []))}`\n\n"
        "To approve after review:\n\n"
        "```bash\n"
        f"{proposal.get('activation', {}).get('one_key_enable_command')}\n"
        "```\n"
    )


def _workflow_context(case: dict[str, Any]) -> dict[str, Any]:
    input_summary = dict(case.get("input_summary", {}) or {})
    metadata = dict(input_summary.get("clinical_metadata", {}) or {})
    return dict(metadata.get("workflow_context") or input_summary.get("workflow_context") or {})


def _canonical(label: Any, dataset_name: str) -> str:
    canonical = canonicalize_label(str(label or ""), dataset_name=dataset_name)
    return str(canonical or label or "UNKNOWN").strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _proposal_id(*, model_name: str, dataset_name: str, metrics: dict[str, Any], doctor_experience: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"model": model_name, "dataset": dataset_name, "metrics": metrics, "doctor": doctor_experience.get("sha256", "")},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{_slug(model_name)}__{_slug(dataset_name)}__workflow_evolution__{digest}"


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _hurt_count(metrics: dict[str, Any]) -> int:
    return len(metrics.get("hurt_case_ids", []) or [])
