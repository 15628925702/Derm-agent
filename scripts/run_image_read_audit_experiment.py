from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.image_read_audit import build_agent_vs_text_only_summary, build_text_only_case_input
from agent.run_agent import run_agent
from dataio.case_loader import load_case_by_index
from integrations.openai_client import DermOpenAIClient


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "image_read_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small DermAgent image-reading audit experiment.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--limit", type=int, default=3, help="Number of cases to audit.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start from this case index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON reports.")
    parser.add_argument("--data-split", type=str, default="test", choices=("train", "val", "test"), help="Logical split label.")
    parser.add_argument("--run-mode", type=str, default="image_read_audit", help="Run mode label stored in execution metadata.")
    parser.add_argument("--client-base-url", type=str, default=None, help="Optional OpenAI-compatible base URL override.")
    parser.add_argument("--client-api-key", type=str, default=None, help="Optional OpenAI-compatible API key override.")
    parser.add_argument("--client-model", type=str, default=None, help="Optional model-name override.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Optional client timeout in seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Optional client retry count.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = DermOpenAIClient(
        base_url=args.client_base_url,
        api_key=args.client_api_key,
        model=args.client_model,
        timeout=args.client_timeout,
        max_retries=args.client_max_retries,
    )

    case_rows: list[dict[str, object]] = []
    for case_index in range(args.case_offset, args.case_offset + args.limit):
        case_input = load_case_by_index(case_index, args.data_root)
        print(
            f"[audit {case_index - args.case_offset + 1}/{args.limit}] case_id={case_input.case_id}",
            flush=True,
        )
        primary_state, _ = run_agent(
            case_input,
            client=client,
            output_dir=args.output_dir / "artifacts" / "primary",
            enable_writeback=False,
            run_mode=args.run_mode,
            data_split=args.data_split,
            execution_overrides={"enable_image_read_audit": True},
        )
        text_only_case = build_text_only_case_input(case_input)
        text_only_state, _ = run_agent(
            text_only_case,
            client=client,
            output_dir=args.output_dir / "artifacts" / "text_only",
            enable_writeback=False,
            run_mode=f"{args.run_mode}_text_only",
            data_split=args.data_split,
        )
        case_rows.append(
            {
                "case_index": case_index,
                "case_id": case_input.case_id,
                "image_path": case_input.image_path,
                "reference_label": case_input.reference_label,
                "in_run_audit": dict(primary_state.image_read_audit),
                "full_agent_text_only_compare": build_agent_vs_text_only_summary(
                    primary_output=primary_state.final_diagnosis,
                    text_only_output=text_only_state.final_diagnosis,
                    ground_truth_label=case_input.reference_label or case_input.label,
                ),
            }
        )

    summary = summarize_rows(case_rows)
    payload = {
        "experiment": {
            "experiment_name": "image_read_audit_small",
            "data_root": str(args.data_root),
            "limit": args.limit,
            "case_offset": args.case_offset,
            "data_split": args.data_split,
            "run_mode": args.run_mode,
            "client_manifest": client.runtime_manifest(),
        },
        "summary": summary,
        "cases": case_rows,
    }
    output_path = args.output_dir / "image_read_audit_small_experiment.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    in_run_result_changed = 0
    in_run_quality_changed = 0
    full_agent_result_changed = 0
    full_agent_quality_changed = 0
    for row in rows:
        in_run_audit = dict(row.get("in_run_audit", {}))
        result_impact = dict(in_run_audit.get("result_impact", {}))
        if bool(result_impact.get("result_changed")):
            in_run_result_changed += 1
        if str(result_impact.get("result_impact_verdict", "")) == "image_changes_outcome_quality":
            in_run_quality_changed += 1

        full_compare = dict(row.get("full_agent_text_only_compare", {}))
        if bool(full_compare.get("result_changed")):
            full_agent_result_changed += 1
        if str(full_compare.get("result_impact_verdict", "")) == "image_changes_outcome_quality":
            full_agent_quality_changed += 1

    return {
        "case_count": total,
        "in_run_result_changed_rate": safe_rate(in_run_result_changed, total),
        "in_run_quality_changed_rate": safe_rate(in_run_quality_changed, total),
        "full_agent_text_only_result_changed_rate": safe_rate(full_agent_result_changed, total),
        "full_agent_text_only_quality_changed_rate": safe_rate(full_agent_quality_changed, total),
        "counts": {
            "in_run_result_changed": in_run_result_changed,
            "in_run_quality_changed": in_run_quality_changed,
            "full_agent_result_changed": full_agent_result_changed,
            "full_agent_quality_changed": full_agent_quality_changed,
        },
    }


def safe_rate(numerator: int, denominator: int) -> dict[str, object]:
    if denominator <= 0:
        return {"count": numerator, "total": denominator, "rate": None}
    return {"count": numerator, "total": denominator, "rate": round(numerator / denominator, 4)}


if __name__ == "__main__":
    raise SystemExit(main())
