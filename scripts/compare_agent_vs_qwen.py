from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation_protocol import DEFAULT_EVAL_OUTPUT_ROOT, EvaluationTargetSpec, run_evaluation_suite
from agent.model_workflow_router import (
    dataset_environment_overrides_for_model_dataset,
    execution_overrides_for_run_agent,
    get_model_workflow_overrides,
    merge_model_workflow_policy_overrides,
)
from agent.paper_exports import DEFAULT_CASE_LEVEL_EXPORT_ROOT, export_compare_case_data
from agent.policy_config import load_policy, load_stable_policy
from dataio.case_loader import resolve_registered_dataset_loader
from integrations.openai_client import DermOpenAIClient
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"


def infer_dataset_name_from_data_root(data_root: Path) -> str:
    spec = resolve_registered_dataset_loader(data_root)
    if spec is not None:
        return spec.dataset_name
    parts = {part.strip().lower() for part in data_root.resolve().parts}
    if "pad_ufes_20" in parts:
        return "pad20"
    if "isic2019" in parts:
        return "isic2019"
    if "scin" in parts:
        return "scin"
    if "sd198" in parts or "sd-198" in parts:
        return "sd198"
    if "ham10000" in parts:
        return "ham10000"
    if "sft数据" in parts or "xiangya_sft" in parts:
        return "xiangya_sft"
    return ""


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _redact_execution_overrides(overrides: dict[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in dict(overrides or {}).items():
        lowered = str(key).lower()
        if "api_key" in lowered or "token" in lowered or "secret" in lowered:
            redacted[key] = "<redacted>" if str(value or "").strip() else ""
        else:
            redacted[key] = value
    return redacted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a direct model baseline against full DermAgent under frozen evaluation mode.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--limit", type=int, default=10, help="Number of cases to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for traceability; current case selection remains deterministic.")
    parser.add_argument("--case-offset", type=int, default=0, help="Start from this case index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON reports.")
    parser.add_argument("--policy-config", type=Path, default=None, help="Optional policy config JSON. Defaults to the current stable policy.")
    parser.add_argument("--policy-label", type=str, default="", help="Optional human-readable label for this policy run.")
    parser.add_argument("--agent-base-url", type=str, default=None, help="Optional OpenAI-compatible base URL for the DermAgent path.")
    parser.add_argument("--agent-api-key", type=str, default=None, help="Optional API key for the DermAgent path.")
    parser.add_argument("--agent-model", type=str, default=None, help="Optional model name for the DermAgent path.")
    parser.add_argument("--agent-label", type=str, default="Full DermAgent", help="Human-readable label for the DermAgent target.")
    parser.add_argument(
        "--agent-description",
        type=str,
        default="Full DermAgent with frozen experience/cognition/policy state and writeback disabled.",
        help="Description for the DermAgent target.",
    )
    parser.add_argument("--baseline-base-url", type=str, default=None, help="Optional OpenAI-compatible base URL for the direct baseline.")
    parser.add_argument("--baseline-api-key", type=str, default=None, help="Optional API key for the direct baseline.")
    parser.add_argument("--baseline-model", type=str, default=None, help="Optional model name for the direct baseline.")
    parser.add_argument("--baseline-label", type=str, default="Direct Baseline", help="Human-readable label for the baseline target.")
    parser.add_argument(
        "--baseline-description",
        type=str,
        default="Direct model baseline with no agent evidence package.",
        help="Description for the baseline target.",
    )
    parser.add_argument("--data-split", type=str, default="test", choices=("val", "test"), help="Evaluation split label for contamination guard and manifests.")
    parser.add_argument("--split-json", type=Path, default=None, help="Optional fixed split JSON. If omitted, the built-in deterministic split is used.")
    parser.add_argument("--non-strict-frozen-eval", action="store_true", help="Allow snapshotting from resolved split state paths even if some files are missing.")
    parser.add_argument("--client-timeout", type=float, default=None, help="Override per-request timeout in seconds.")
    parser.add_argument("--client-max-retries", type=int, default=None, help="Override automatic retries for transient local inference failures.")
    parser.add_argument("--phase", type=str, default="full", help="Compatibility label for legacy run scripts; recorded but does not alter target selection.")
    parser.add_argument(
        "--disable-model-workflow-routing",
        action="store_true",
        help="Do not apply model workflow overlays; keep dataset workflow routing only.",
    )
    parser.add_argument(
        "--enable-physician-evidence-summary",
        action="store_true",
        help="Emit an optional doctor-facing evidence package for each DermAgent case via the separate Qwen summary service.",
    )
    parser.add_argument(
        "--physician-evidence-summary-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for the separate Qwen physician-summary service. Defaults to http://127.0.0.1:8200/v1.",
    )
    parser.add_argument(
        "--physician-evidence-summary-api-key",
        type=str,
        default=None,
        help="API key for the separate Qwen physician-summary service. Defaults to EMPTY via runtime config.",
    )
    parser.add_argument(
        "--physician-evidence-summary-model",
        type=str,
        default=None,
        help="Model name served by the separate Qwen physician-summary service. Defaults to Qwen2.5-VL-7B-Instruct.",
    )
    parser.add_argument(
        "--physician-evidence-summary-detail",
        type=str,
        default=None,
        choices=("brief", "detailed"),
        help="Physician evidence package detail level. `brief` is compact; `detailed` preserves more intermediate evidence.",
    )
    parser.add_argument(
        "--physician-evidence-summary-timeout",
        type=float,
        default=None,
        help="Optional per-request timeout in seconds for the separate Qwen physician-summary service.",
    )
    parser.add_argument(
        "--physician-evidence-summary-max-retries",
        type=int,
        default=None,
        help="Optional retry count for the separate Qwen physician-summary service.",
    )
    parser.add_argument(
        "--export-paper-case-data",
        action="store_true",
        help="Also export this compare run as paper-ready workflow-free case-level CSV/JSON/XLSX rows.",
    )
    parser.add_argument(
        "--paper-case-data-dir",
        type=Path,
        default=DEFAULT_CASE_LEVEL_EXPORT_ROOT / "compare_runs",
        help="Directory for --export-paper-case-data outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = DermOpenAIClient(
        base_url=args.agent_base_url,
        api_key=args.agent_api_key,
        model=args.agent_model,
        timeout=args.client_timeout,
        max_retries=args.client_max_retries,
    )
    baseline_client = DermOpenAIClient(
        base_url=args.baseline_base_url or args.agent_base_url,
        api_key=args.baseline_api_key or args.agent_api_key,
        model=args.baseline_model or args.agent_model,
        timeout=args.client_timeout,
        max_retries=args.client_max_retries,
    )
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()
    agent_model_name = args.agent_model or client.model
    inferred_dataset_name = infer_dataset_name_from_data_root(args.data_root)
    disable_model_workflow_routing = bool(args.disable_model_workflow_routing) or str(
        os.getenv("DERMAGENT_DISABLE_MODEL_WORKFLOW_ROUTING", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not disable_model_workflow_routing:
        dataset_env_overrides = dataset_environment_overrides_for_model_dataset(
            agent_model_name,
            inferred_dataset_name,
        )
        for key, value in dataset_env_overrides.items():
            os.environ.setdefault(key, value)
        if dataset_env_overrides:
            print(
                json.dumps(
                    {
                        "model_dataset_workflow_environment": {
                            "model": agent_model_name,
                            "dataset": inferred_dataset_name,
                            "environment": dataset_env_overrides,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if disable_model_workflow_routing:
        model_workflow_overrides = {}
        agent_execution_overrides = {}
    else:
        model_workflow_overrides = get_model_workflow_overrides(
            agent_model_name,
            dataset_name=inferred_dataset_name,
            base_workflow_context=None,
        )
        agent_execution_overrides = execution_overrides_for_run_agent(model_workflow_overrides)
        if model_workflow_overrides:
            policy = merge_model_workflow_policy_overrides(policy, model_workflow_overrides)
    if args.enable_physician_evidence_summary or _env_flag("DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY"):
        agent_execution_overrides["enable_physician_evidence_summary"] = True
        if args.physician_evidence_summary_base_url:
            agent_execution_overrides["physician_evidence_summary_base_url"] = args.physician_evidence_summary_base_url
        if args.physician_evidence_summary_api_key:
            agent_execution_overrides["physician_evidence_summary_api_key"] = args.physician_evidence_summary_api_key
        if args.physician_evidence_summary_model:
            agent_execution_overrides["physician_evidence_summary_model"] = args.physician_evidence_summary_model
        if args.physician_evidence_summary_detail:
            agent_execution_overrides["physician_evidence_summary_detail"] = args.physician_evidence_summary_detail
        if args.physician_evidence_summary_timeout is not None:
            agent_execution_overrides["physician_evidence_summary_timeout"] = args.physician_evidence_summary_timeout
        if args.physician_evidence_summary_max_retries is not None:
            agent_execution_overrides["physician_evidence_summary_max_retries"] = args.physician_evidence_summary_max_retries
    if model_workflow_overrides:
        print(
            json.dumps(
                {
                    "model_workflow_routing": {
                        "model": agent_model_name,
                        "model_workflow_profile": model_workflow_overrides.get("workflow_profile", ""),
                        "workflow_cell_id": model_workflow_overrides.get("workflow_cell_id", ""),
                        "execution_overrides": _redact_execution_overrides(agent_execution_overrides),
                        "skip_specialist_skills": bool(model_workflow_overrides.get("skip_specialist_skills", False)),
                        "skip_experience_retrieval": bool(model_workflow_overrides.get("skip_experience_retrieval", False)),
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    agent_target_execution_overrides = (
        {
            "model_name": agent_model_name,
            **model_workflow_overrides,
            **agent_execution_overrides,
        }
        if not disable_model_workflow_routing
        else {}
    )
    target_specs = [
        EvaluationTargetSpec(
            target_id="direct_baseline",
            label=args.baseline_label,
            target_type="baseline",
            mode="baseline",
            description=args.baseline_description,
        ),
        EvaluationTargetSpec(
            target_id="full_dermagent",
            label=args.agent_label,
            target_type="full_agent",
            mode="agent",
            description=args.agent_description,
            execution_overrides=agent_target_execution_overrides,
            notes=[
                "Model workflow routing is layered on top of dataset workflow routing."
            ] if model_workflow_overrides else [],
        ),
    ]

    suite = run_evaluation_suite(
        output_root=args.output_dir,
        data_root=args.data_root,
        client=client,
        baseline_client=baseline_client,
        policy_config=policy,
        target_specs=target_specs,
        limit=args.limit,
        case_offset=args.case_offset,
        seed=args.seed,
        suite_label="compare_agent_vs_qwen",
        data_split=args.data_split,
        split_json=args.split_json,
        strict_frozen_eval=not args.non_strict_frozen_eval,
    )

    result_manifest = suite["result_manifest"]
    target_results = {item["target"]["target_id"]: item for item in result_manifest.get("target_results", [])}
    baseline_summary = target_results.get("direct_baseline", {}).get("summary", {})
    agent_summary = target_results.get("full_dermagent", {}).get("summary", {})
    full_agent_jsonl = Path(target_results.get("full_dermagent", {}).get("artifacts", {}).get("records_jsonl_path", ""))
    agent_cases = []
    if full_agent_jsonl.exists():
        with full_agent_jsonl.open("r", encoding="utf-8") as handle:
            agent_cases = [json.loads(line) for line in handle if line.strip()]

    report = {
        "run_config": {
            "data_root": str(args.data_root),
            "limit": args.limit,
            "seed": args.seed,
            "case_offset": args.case_offset,
            "base_url": client.base_url,
            "model": client.model,
            "agent_base_url": client.base_url,
            "agent_model": client.model,
            "baseline_base_url": baseline_client.base_url,
            "baseline_model": baseline_client.model,
            "policy_id": policy.get("policy_id"),
            "policy_source_path": policy.get("source_path"),
            "policy_label": args.policy_label,
            "data_split": args.data_split,
            "split_json": str(args.split_json) if args.split_json else "",
            "phase": args.phase,
            "strict_frozen_eval": not args.non_strict_frozen_eval,
            "evaluation_protocol_version": result_manifest.get("protocol_version"),
            "agent_model_name": agent_model_name,
            "inferred_dataset_name": inferred_dataset_name,
            "disable_model_workflow_routing": disable_model_workflow_routing,
            "model_workflow_overrides": model_workflow_overrides,
            "agent_execution_overrides": _redact_execution_overrides(agent_execution_overrides),
        },
        "summary": {
            "baseline": baseline_summary,
            "agent": agent_summary,
            "agent_vs_baseline": result_manifest.get("comparisons", {}).get("full_dermagent", {}).get("vs_baseline", {}),
        },
        "artifacts": {
            "evaluation_manifest_path": suite["evaluation_manifest_path"],
            "result_manifest_path": suite["result_manifest_path"],
            "run_root": suite["run_root"],
        },
        "cases": agent_cases,
    }
    output_path = args.output_dir / f"compare_agent_vs_qwen_{suite['eval_id'].split('_')[-1]}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paper_case_export_manifest = None
    if args.export_paper_case_data or _env_flag("DERMAGENT_EXPORT_PAPER_CASE_DATA"):
        export_stem = f"case_level_compare_export_{suite['eval_id'].split('_')[-1]}"
        paper_case_export_manifest = export_compare_case_data(
            compare_report_paths=[output_path],
            output_dir=args.paper_case_data_dir,
            export_stem=export_stem,
        )
        report["artifacts"]["paper_case_export_manifest_path"] = paper_case_export_manifest.get("manifest_path", "")
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": report["summary"],
                "report_path": str(output_path),
                "evaluation_manifest_path": suite["evaluation_manifest_path"],
                "result_manifest_path": suite["result_manifest_path"],
                "paper_case_export_manifest_path": (paper_case_export_manifest or {}).get("manifest_path", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
