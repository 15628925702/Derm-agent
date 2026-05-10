from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export doctor-facing physician_evidence_summary fields per case.")
    parser.add_argument("--compare-report", type=Path, action="append", default=[])
    parser.add_argument("--compare-report-glob", type=str, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(text: str) -> str:
    keep = []
    for ch in text.strip():
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "unknown_case"


def summary_to_markdown(case: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        f"# {case.get('case_id', 'unknown_case')}",
        "",
        f"- Dataset: `{case.get('dataset_name', '')}`",
        f"- Ground truth: `{(case.get('ground_truth') or {}).get('canonical_label', '')}`",
        f"- Agent final: `{(case.get('qwen_final') or {}).get('final_diagnosis', '')}`",
        "",
    ]
    for key, value in summary.items():
        if key in {"status", "schema", "schema_version"}:
            continue
        title = key.replace("_", " ").title()
        lines.append(f"## {title}")
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item if not isinstance(item, (dict, list)) else json.dumps(item, ensure_ascii=False)}")
        elif isinstance(value, dict):
            lines.append("```json")
            lines.append(json.dumps(value, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    reports = list(args.compare_report)
    for pattern in args.compare_report_glob:
        reports.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    reports = sorted(set(path for path in reports if path.is_file()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    missing = 0
    failed = 0
    for report in reports:
        payload = read_json(report)
        run_config = payload.get("run_config", {}) or {}
        model = safe_name(str(run_config.get("agent_model") or run_config.get("model") or "model"))
        dataset = safe_name(str(run_config.get("inferred_dataset_name") or run_config.get("dataset_name") or "dataset"))
        target_dir = args.output_dir / model / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        for case in payload.get("cases", []) or []:
            case_id = safe_name(str(case.get("case_id", "unknown_case")))
            summary = case.get("physician_evidence_summary")
            if not isinstance(summary, dict) or not summary:
                missing += 1
                continue
            if str(summary.get("status", "")).lower() in {"failed", "empty_model_output"}:
                failed += 1
            json_path = target_dir / f"{case_id}.json"
            md_path = target_dir / f"{case_id}.md"
            json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            md_path.write_text(summary_to_markdown(case, summary), encoding="utf-8")
            exported.append(
                {
                    "case_id": case.get("case_id", ""),
                    "model": model,
                    "dataset": dataset,
                    "json_path": str(json_path),
                    "markdown_path": str(md_path),
                    "status": summary.get("status", "ok"),
                    "source_report": str(report),
                }
            )

    manifest = {
        "output_dir": str(args.output_dir),
        "report_count": len(reports),
        "exported_count": len(exported),
        "missing_count": missing,
        "failed_count": failed,
        "exports": exported,
    }
    manifest_path = args.output_dir / "doctor_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
