from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.evidence_package import EvidencePackage
from agent.state import CaseInput
from integrations.openai_client import DermOpenAIClient
from scripts.export_doctor_evidence_packages import safe_name, summary_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate doctor evidence packages from saved compare reports.")
    parser.add_argument("--compare-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--detail", type=str, choices=("brief", "detailed"), default="detailed")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def _evidence_package_from_record(case: dict[str, Any]) -> EvidencePackage:
    payload = dict(case.get("evidence_bundle", {}) or {})
    allowed = set(EvidencePackage.__dataclass_fields__.keys())
    return EvidencePackage(**{key: value for key, value in payload.items() if key in allowed})


def _case_input_from_record(case: dict[str, Any]) -> CaseInput:
    input_summary = dict(case.get("input_summary", {}) or {})
    ground_truth = dict(case.get("ground_truth", {}) or {})
    return CaseInput(
        case_id=str(case.get("case_id", "")).strip(),
        image_path=str(input_summary.get("image_path", "")).strip(),
        metadata=dict(input_summary.get("clinical_metadata", {}) or {}),
        label=str(ground_truth.get("raw_label", "") or ground_truth.get("canonical_label", "")).strip() or None,
        reference_label=str(ground_truth.get("raw_label", "") or ground_truth.get("canonical_label", "")).strip() or None,
        dataset_name=str(case.get("dataset_name", "")).strip() or None,
        label_space_id=str(input_summary.get("label_space_id", "")).strip() or None,
        source_metadata_path=str(input_summary.get("metadata_path", "")).strip() or None,
        workflow_context=dict((case.get("evidence_bundle", {}) or {}).get("evidence_decision_policy", {}).get("diagnosis_override_layer", {}).get("workflow_context", {}) or {}),
    )


def main() -> int:
    args = parse_args()
    payload = json.loads(args.compare_report.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []) or [])
    start = max(0, int(args.case_offset))
    end = len(cases) if int(args.limit) <= 0 else min(len(cases), start + int(args.limit))
    selected = cases[start:end]
    client = DermOpenAIClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports: list[dict[str, Any]] = []
    for local_index, case in enumerate(selected, start=start):
        case_input = _case_input_from_record(case)
        evidence_package = _evidence_package_from_record(case)
        try:
            summary = client.physician_evidence_summary(
                case_input,
                evidence_package,
                detail_level=args.detail,
            )
            if not isinstance(summary, dict):
                summary = {"status": "malformed", "raw_summary_type": type(summary).__name__}
        except Exception as exc:  # pragma: no cover - long-run defensive path
            summary = {
                "summary_version": "physician_evidence_summary_v2_detailed",
                "case_id": case_input.case_id,
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "detail_level": args.detail,
                "intended_use": "doctor_support_only_not_final_diagnosis",
            }
        summary.setdefault("case_id", case_input.case_id)
        summary.setdefault("status", "ok")
        summary.setdefault("detail_level", args.detail)
        summary.setdefault("intended_use", "doctor_support_only_not_final_diagnosis")
        case_name = safe_name(case_input.case_id)
        json_path = args.output_dir / f"{case_name}.json"
        md_path = args.output_dir / f"{case_name}.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(summary_to_markdown(case, summary), encoding="utf-8")
        exports.append(
            {
                "case_index_in_report": local_index,
                "case_id": case_input.case_id,
                "status": summary.get("status", "ok"),
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            }
        )
    manifest = {
        "compare_report": str(args.compare_report),
        "case_offset": start,
        "limit": len(selected),
        "output_dir": str(args.output_dir),
        "exports": exports,
    }
    (args.output_dir / "doctor_evidence_posthoc_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
