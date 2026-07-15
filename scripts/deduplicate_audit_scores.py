from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.auditability_scoring import render_summary_markdown, summarize_scores, write_csv

NUMERIC_FIELDS = {
    "evidence_completeness",
    "contradiction_awareness",
    "risk_cue_preservation",
    "error_localization",
    "overall_auditability",
    "correction_success",
    "risk_cue_detected_count",
    "risk_cue_expected_count",
    "error_missing_cue",
    "error_confusion_pair",
    "error_risk_downgrade",
    "error_premature_closure",
    "error_visual_misread",
}

BOOLEAN_TEXT_FIELDS = {
    "correct",
    "direct_baseline_correct",
    "is_malignant_or_high_risk",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _truthy(value: str) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _majority(values: list[str]) -> str:
    cleaned = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    winner, winner_count = counts.most_common(1)[0]
    tied = sorted([item for item, count in counts.items() if count == winner_count])
    if len(tied) == 1:
        return winner
    return " | ".join(tied)


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("case_id", "")), str(row.get("system_name", "")))
        grouped[key].append(row)

    deduped_rows: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_extra_rows = 0

    for (_, _, _), group in grouped.items():
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_extra_rows += len(group) - 1
        base = dict(group[0])
        for field in NUMERIC_FIELDS:
            values = []
            for item in group:
                raw = str(item.get(field, "")).strip()
                if not raw:
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
            if not values:
                base[field] = ""
            else:
                base[field] = round(_mean(values), 4)

        for field in BOOLEAN_TEXT_FIELDS:
            bool_values = [value for value in (_truthy(item.get(field, "")) for item in group) if value is not None]
            if not bool_values:
                base[field] = ""
            else:
                mean_value = _mean([1.0 if value else 0.0 for value in bool_values]) or 0.0
                base[field] = "True" if mean_value >= 0.5 else "False"

        for field in ("final_prediction", "initial_prediction", "direct_baseline_prediction", "ground_truth", "case_group"):
            base[field] = _majority([item.get(field, "") for item in group])

        base["source_run_count"] = len(group)
        base["source_root"] = " || ".join(sorted({str(item.get("source_root", "")).strip() for item in group if str(item.get("source_root", "")).strip()}))
        base["record_path"] = " || ".join(sorted({str(item.get("record_path", "")).strip() for item in group if str(item.get("record_path", "")).strip()}))
        base["evidence_package_path"] = " || ".join(sorted({str(item.get("evidence_package_path", "")).strip() for item in group if str(item.get("evidence_package_path", "")).strip()}))
        base["reflection_path"] = " || ".join(sorted({str(item.get("reflection_path", "")).strip() for item in group if str(item.get("reflection_path", "")).strip()}))
        base["extracted_evidence_json"] = json.dumps(
            {
                "deduplicated": len(group) > 1,
                "source_run_count": len(group),
                "unique_final_predictions": sorted({str(item.get("final_prediction", "")).strip() for item in group if str(item.get("final_prediction", "")).strip()}),
                "unique_overall_scores": sorted({str(item.get("overall_auditability", "")).strip() for item in group if str(item.get("overall_auditability", "")).strip()}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        deduped_rows.append(base)

    metadata = {
        "input_row_count": len(rows),
        "deduplicated_row_count": len(deduped_rows),
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows_removed": duplicate_extra_rows,
    }
    return deduped_rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate audit score rows by dataset + case_id + system_name and rebuild the summary.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(input_csv)
    deduped_rows, metadata = deduplicate_rows(rows)
    summary = summarize_scores(deduped_rows)
    summary["deduplication"] = metadata

    write_csv(output_root / "audit_scores_deduplicated.csv", deduped_rows)
    with (output_root / "summary_deduplicated.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    (output_root / "summary_deduplicated.md").write_text(
        render_summary_markdown(summary, [input_csv]),
        encoding="utf-8",
    )
    with (output_root / "dedup_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_csv": str(input_csv),
                "output_root": str(output_root),
                "deduplication": metadata,
                "artifacts": {
                    "audit_scores_deduplicated_csv": str((output_root / "audit_scores_deduplicated.csv").resolve()),
                    "summary_deduplicated_json": str((output_root / "summary_deduplicated.json").resolve()),
                    "summary_deduplicated_md": str((output_root / "summary_deduplicated.md").resolve()),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps({"deduplication": metadata, "output_root": str(output_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
