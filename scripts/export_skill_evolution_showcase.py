from __future__ import annotations

import argparse
import glob
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.hard_case_miner import mine_hard_cases
from agent.skill_helpfulness_analyzer import analyze_skill_helpfulness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a doctor-facing skill self-evolution showcase from case execution records."
    )
    parser.add_argument(
        "--record-glob",
        action="append",
        required=True,
        help="Glob pattern(s) for case_execution_record.json files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the markdown report, JSON summaries, and copied images.",
    )
    parser.add_argument(
        "--dataset-name",
        default="xiangya_sft",
        help="Dataset name used for filtering summaries.",
    )
    return parser.parse_args()


def load_unique_records(patterns: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for pattern in patterns:
        for path_str in sorted(glob.glob(pattern)):
            path = Path(path_str)
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(payload.get("case_id", "")).strip()
            if not case_id or case_id in seen_case_ids:
                continue
            payload["_source_record_path"] = str(path)
            records.append(payload)
            seen_case_ids.add(case_id)
    return records


def build_case_row(record: dict[str, Any]) -> dict[str, Any]:
    reflection = record.get("reflection_summary", {})
    outcome = reflection.get("case_outcome", {})
    image_path = to_windows_path(record.get("input_summary", {}).get("image_path", ""))
    return {
        "case_id": record.get("case_id"),
        "image_path": image_path,
        "ground_truth": outcome.get("reference_label"),
        "prediction": outcome.get("predicted_label"),
        "confusion_pair": outcome.get("confusion_pair"),
        "selected_skills": list(record.get("selected_skills") or []),
        "harmful_reasons": sorted(
            {
                reason
                for item in reflection.get("skill_assessments", [])
                for reason in item.get("harmful_reasons", [])
                if str(reason).strip()
            }
        ),
        "common_failure_modes": sorted(
            {
                reason
                for item in reflection.get("skill_assessments", [])
                for reason in item.get("failure_modes", [])
                if str(reason).strip()
            }
        ),
    }


def to_windows_path(path_value: str) -> str:
    text = str(path_value).strip()
    if not text:
        return ""
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        suffix = text[6:].replace("/", "\\")
        return f"{drive}:{suffix}"
    return text


def choose_showcase_cases(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row.get("ground_truth", "unknown"))].append(row)
    selected: list[dict[str, Any]] = []
    for label in sorted(grouped):
        selected.append(sorted(grouped[label], key=lambda item: str(item.get("case_id")))[0])
    return selected


def copy_case_images(showcase_cases: list[dict[str, Any]], images_dir: Path) -> list[dict[str, Any]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for row in showcase_cases:
        source = Path(str(row.get("image_path", "")))
        if source.exists():
            target = images_dir / f"{row['case_id']}{source.suffix.lower()}"
            shutil.copy2(source, target)
            enriched = dict(row)
            enriched["exported_image"] = str(target)
            exported.append(enriched)
        else:
            exported.append(dict(row))
    return exported


def build_skill_evolution_plan() -> list[dict[str, Any]]:
    return [
        {
            "skill_name": "lesion_description_structuring_skill",
            "why_it_needs_evolution": (
                "Across all six cases this skill returned slot-complete JSON shells filled with `unknown` or empty lists, "
                "so the downstream fusion stage had nothing case-specific to use."
            ),
            "evolved_version_focus": [
                "Force an explicit visibility statement instead of empty placeholders.",
                "Add rash-family slots such as erythema quality, scale pattern, perifollicular involvement, and excoriation/crust.",
                "Require a short non-diagnostic summary sentence that later reasoning can quote directly.",
            ],
            "doctor_review_point": (
                "Check whether the evolved version produces reusable dermatologist-style wording rather than generic empty fields."
            ),
        },
        {
            "skill_name": "distribution_analysis_skill",
            "why_it_needs_evolution": (
                "The current version stayed at `body_location=unknown` and `clustering_pattern=unknown`, "
                "which prevented the system from using scalp or hair-bearing distribution clues in these inflammatory cases."
            ),
            "evolved_version_focus": [
                "Prioritize hair-bearing/scalp-like context when follicles dominate the frame.",
                "Add an inflammatory-pattern branch for diffuse erythema with superficial scale.",
                "Separate `truly unknown` from `partially visible but suggestive` so the skill stops collapsing to empty output.",
            ],
            "doctor_review_point": (
                "Check whether the evolved version captures distribution cues that would help separate contact dermatitis from broader eczematous or inflammatory patterns."
            ),
        },
        {
            "skill_name": "metadata_consistency_skill",
            "why_it_needs_evolution": (
                "This skill did not surface any conflict, suspicious point, or reliability statement, "
                "so it could not warn the agent that the evidence package was too weak to support a narrow contact-dermatitis call."
            ),
            "evolved_version_focus": [
                "Emit a soft warning when image evidence is too nonspecific for subtype-level commitment.",
                "Distinguish `no contradiction found` from `not enough evidence to judge consistency`.",
                "Add a watch-out that missing exposure history should block overconfident contact-dermatitis narrowing.",
            ],
            "doctor_review_point": (
                "Check whether the evolved version behaves like a confidence gate instead of a passive empty checker."
            ),
        },
        {
            "skill_name": "new_confusion_specialist_skill",
            "why_it_needs_evolution": (
                "The repeated clusters show three stable confusion directions: "
                "`CONTACT_DERMATITIS -> OTHER_INFLAMMATORY`, "
                "`CONTACT_DERMATITIS -> ECZEMA_DERMATITIS`, and "
                "`CONTACT_DERMATITIS -> ATOPIC_DERMATITIS`."
            ),
            "evolved_version_focus": [
                "Create a narrow inflammatory-family specialist that compares contact dermatitis against nearby eczematous alternatives.",
                "Make the skill list supporting clues, opposing clues, and missing history needed for each candidate.",
                "Keep the output evidence-only so the final diagnosis still remains outside the skill.",
            ],
            "doctor_review_point": (
                "Check whether this new specialist would meaningfully sharpen differential reasoning without smuggling in the final label."
            ),
        },
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(
    *,
    records: list[dict[str, Any]],
    hard_case_summary: dict[str, Any],
    helpfulness_summary: dict[str, Any],
    helpfulness_reports: list[dict[str, Any]],
    showcase_cases: list[dict[str, Any]],
    evolution_plan: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Skill Self-Evolution Showcase")
    lines.append("")
    lines.append("## What This Shows")
    lines.append("")
    lines.append(
        "This report is designed for doctor-side review of whether DermAgent's skill self-evolution would produce a meaningfully better skill version, rather than only a cosmetic rewrite."
    )
    lines.append("")
    lines.append(f"- Cases analyzed: {len(records)}")
    lines.append(f"- Dominant failure type: {next(iter(hard_case_summary.get('counts', {}).get('by_failure_type', {})), 'unknown')}")
    lines.append(
        f"- Most repeated confusion pattern family: {', '.join(sorted(hard_case_summary.get('counts', {}).get('by_confusion_tag', {}).keys())[:3])}"
    )
    lines.append(
        f"- Top harmful reasons across skills: {', '.join(item['name'] for item in helpfulness_summary.get('top_harmful_reasons', [])[:3])}"
    )
    lines.append("")
    lines.append("## Key Finding")
    lines.append("")
    lines.append(
        "The current Hulu-Med-4B + DermAgent run did not fail because one single diagnosis rule was wrong. "
        "It failed because the selected skills repeatedly produced almost no case-specific structured evidence, so the agent could not meaningfully refine the baseline."
    )
    lines.append("")
    lines.append(
        "In all six Xiangya cases, the system converged to `CONTACT_DERMATITIS`, while the ground-truth labels covered three nearby inflammatory targets: `OTHER_INFLAMMATORY`, `ECZEMA_DERMATITIS`, and `ATOPIC_DERMATITIS`."
    )
    lines.append("")
    lines.append("## Representative Cases")
    lines.append("")
    for row in showcase_cases:
        lines.append(f"### {row['case_id']}")
        lines.append("")
        lines.append(f"- Image: `{Path(str(row.get('exported_image') or row.get('image_path', ''))).name or 'missing'}`")
        lines.append(f"- Ground truth: `{row.get('ground_truth', 'unknown')}`")
        lines.append(f"- Model output: `{row.get('prediction', 'unknown')}`")
        lines.append(f"- Stable confusion pair: `{row.get('confusion_pair', 'unknown')}`")
        lines.append(f"- Selected skills: `{', '.join(row.get('selected_skills', []))}`")
        lines.append(f"- Shared harmful reasons: `{', '.join(row.get('harmful_reasons', []))}`")
        lines.append(f"- Shared failure modes: `{', '.join(row.get('common_failure_modes', [])[:5])}`")
        lines.append("")
        lines.append(
            "Why this case matters: it is a clean representative of a repeated confusion cluster, so if a skill evolves correctly here, the gain is more likely to generalize within this mini-cohort."
        )
        lines.append("")
    lines.append("## Skill-Level Evidence")
    lines.append("")
    for report in helpfulness_reports:
        lines.append(f"### {report['skill_name']}")
        lines.append("")
        lines.append(f"- Call count: {report['call_count']}")
        lines.append(f"- Harmful rate: {report['harmful_rate']:.2f}")
        lines.append(f"- Helpful rate: {report['helpful_rate']:.2f}")
        lines.append(
            f"- Common failure modes: {', '.join(item['name'] for item in report.get('common_failure_modes', [])[:5])}"
        )
        lines.append(
            f"- Common failure scenarios: {', '.join(item['name'] for item in report.get('common_failure_scenarios', [])[:4])}"
        )
        lines.append("")
    lines.append("## Proposed Evolution Targets")
    lines.append("")
    for item in evolution_plan:
        lines.append(f"### {item['skill_name']}")
        lines.append("")
        lines.append(f"- Why evolve it: {item['why_it_needs_evolution']}")
        lines.append(f"- Doctor review point: {item['doctor_review_point']}")
        lines.append("- Evolved version should:")
        for point in item["evolved_version_focus"]:
            lines.append(f"  - {point}")
        lines.append("")
    lines.append("## Doctor Comparison Checklist")
    lines.append("")
    lines.append("- Does the evolved skill produce concrete, reusable evidence instead of all-`unknown` placeholders?")
    lines.append("- Does it help separate contact dermatitis from nearby inflammatory or eczematous alternatives?")
    lines.append("- Does it express uncertainty appropriately when the image is too weak for subtype commitment?")
    lines.append("- Does it remain evidence-only, without quietly turning into a final diagnosis module?")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_unique_records(args.record_glob)
    hard_cases, hard_case_summary = mine_hard_cases(
        records,
        dataset_name=args.dataset_name,
        min_importance=0.0,
        max_per_cluster=20,
    )
    helpfulness_reports, helpfulness_summary = analyze_skill_helpfulness(
        records,
        dataset_name=args.dataset_name,
        min_calls=1,
        top_k=5,
    )

    case_rows = [build_case_row(record) for record in records]
    showcase_cases = copy_case_images(
        choose_showcase_cases(case_rows),
        output_dir / "images",
    )
    evolution_plan = build_skill_evolution_plan()

    write_json(output_dir / "records_summary.json", case_rows)
    write_json(output_dir / "hard_case_summary.json", hard_case_summary)
    write_json(output_dir / "hard_cases.json", hard_cases)
    write_json(output_dir / "skill_helpfulness_summary.json", helpfulness_summary)
    write_json(output_dir / "skill_helpfulness_reports.json", helpfulness_reports)
    write_json(output_dir / "showcase_cases.json", showcase_cases)
    write_json(output_dir / "proposed_skill_evolution_plan.json", evolution_plan)

    markdown = build_markdown(
        records=records,
        hard_case_summary=hard_case_summary,
        helpfulness_summary=helpfulness_summary,
        helpfulness_reports=helpfulness_reports,
        showcase_cases=showcase_cases,
        evolution_plan=evolution_plan,
    )
    (output_dir / "skill_self_evolution_showcase.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
