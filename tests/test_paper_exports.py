from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.paper_exports import FIG_FILE_ORDER, TABLE_FILE_ORDER, export_paper_fig_data, export_paper_tables


def test_export_paper_tables_and_fig_data(tmp_path) -> None:
    outputs_root = tmp_path / "outputs"
    _write_minimal_artifacts(outputs_root)

    tables_manifest = export_paper_tables(
        outputs_root=outputs_root,
        output_dir=outputs_root / "paper_exports" / "tables",
    )
    assert Path(tables_manifest["manifest_path"]).exists()
    assert len(tables_manifest["files"]) == len(TABLE_FILE_ORDER)
    for file_row in tables_manifest["files"]:
        assert Path(file_row["json_path"]).exists()
        assert Path(file_row["csv_path"]).exists()

    fig_manifest = export_paper_fig_data(
        outputs_root=outputs_root,
        output_dir=outputs_root / "paper_exports" / "fig_data",
    )
    assert Path(fig_manifest["manifest_path"]).exists()
    assert len(fig_manifest["files"]) == len(FIG_FILE_ORDER)
    for file_row in fig_manifest["files"]:
        assert Path(file_row["json_path"]).exists()
        assert Path(file_row["csv_path"]).exists()

    main_rows = json.loads(Path(outputs_root / "paper_exports" / "tables" / "main_comparison_table.json").read_text(encoding="utf-8"))
    assert any(row.get("target_id") == "full_dermagent" for row in main_rows)

    skill_rows = json.loads(Path(outputs_root / "paper_exports" / "tables" / "skill_helpfulness_summary_table.json").read_text(encoding="utf-8"))
    assert any(row.get("skill_name") == "morphology_analysis_skill" for row in skill_rows)


def _write_minimal_artifacts(outputs_root: Path) -> None:
    eval_root = outputs_root / "evaluation_protocol_step12" / "eval_brief_20260326T000000Z"
    eval_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        eval_root / "result_manifest.json",
        {
            "eval_id": "eval_brief_20260326T000000Z",
            "suite_label": "eval_brief",
            "frozen_evaluation_mode": True,
            "comparisons": {
                "direct_baseline": {"vs_baseline": {"top1_delta": 0.0, "topk_delta": 0.0, "malignant_recall_delta": 0.0, "error_rate_delta": 0.0, "key_confusion_subset_deltas": {}}},
                "full_dermagent": {
                    "vs_baseline": {
                        "top1_delta": 0.1,
                        "topk_delta": 0.05,
                        "malignant_recall_delta": 0.2,
                        "error_rate_delta": -0.1,
                        "key_confusion_subset_deltas": {
                            "melanoma->nev": {"top1_delta": 0.3, "error_rate_delta": -0.2}
                        },
                    }
                },
            },
            "target_results": [
                {
                    "target": {
                        "target_id": "direct_baseline",
                        "label": "Direct Baseline",
                        "mode": "baseline",
                        "target_type": "baseline",
                        "description": "baseline",
                        "experience_variant": "full",
                        "cognition_variant": "frozen",
                    },
                    "summary": {
                        "num_cases": 2,
                        "num_with_ground_truth": 2,
                        "top1": {"rate": 0.5},
                        "topk": {"rate": 1.0},
                        "malignant_recall": {"hits": 1, "total": 2, "rate": 0.5},
                        "error_rate": {"rate": 0.5},
                        "key_confusion_subsets": {},
                    },
                    "artifacts": {"records_jsonl_path": str(eval_root / "targets" / "direct_baseline" / "records" / "case_execution_records.jsonl")},
                },
                {
                    "target": {
                        "target_id": "full_dermagent",
                        "label": "Full DermAgent",
                        "mode": "agent",
                        "target_type": "full_agent",
                        "description": "full",
                        "experience_variant": "full",
                        "cognition_variant": "frozen",
                    },
                    "summary": {
                        "num_cases": 2,
                        "num_with_ground_truth": 2,
                        "top1": {"rate": 0.6},
                        "topk": {"rate": 1.0},
                        "malignant_recall": {"hits": 2, "total": 2, "rate": 1.0},
                        "error_rate": {"rate": 0.4},
                        "key_confusion_subsets": {
                            "melanoma->nev": {
                                "num_cases": 1,
                                "num_with_ground_truth": 1,
                                "top1": {"rate": 1.0},
                                "topk": {"rate": 1.0},
                                "malignant_recall": {"rate": 1.0},
                                "error_rate": {"rate": 0.0},
                            }
                        },
                    },
                    "artifacts": {"records_jsonl_path": str(eval_root / "targets" / "full_dermagent" / "records" / "case_execution_records.jsonl")},
                },
            ],
        },
    )

    ablation_root = outputs_root / "ablations_smoke" / "ablations_20260326T000001Z"
    ablation_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        ablation_root / "ablation_matrix_summary.json",
        {
            "eval_id": "ablations_20260326T000001Z",
            "rows": [
                {
                    "target_id": "direct_qwen_baseline",
                    "label": "Direct Qwen Baseline",
                    "top1": 0.5,
                    "topk": 1.0,
                    "malignant_recall": 0.5,
                    "error_rate": 0.5,
                    "top1_delta_vs_baseline": 0.0,
                    "top1_delta_vs_full_anchor": 0.0,
                    "topk_delta_vs_baseline": 0.0,
                    "malignant_recall_delta_vs_baseline": 0.0,
                    "error_rate_delta_vs_baseline": 0.0,
                    "controller_family": "",
                    "experience_variant": "full",
                    "cognition_variant": "frozen",
                    "experiment_meaning": "baseline",
                },
                {
                    "target_id": "full_with_rule_controller",
                    "label": "Full With Rule Controller",
                    "top1": 0.6,
                    "topk": 1.0,
                    "malignant_recall": 1.0,
                    "error_rate": 0.4,
                    "top1_delta_vs_baseline": 0.1,
                    "top1_delta_vs_full_anchor": 0.0,
                    "topk_delta_vs_baseline": 0.0,
                    "malignant_recall_delta_vs_baseline": 0.5,
                    "error_rate_delta_vs_baseline": -0.1,
                    "controller_family": "heuristic",
                    "experience_variant": "full",
                    "cognition_variant": "frozen",
                    "experiment_meaning": "rule",
                },
            ],
        },
    )

    (outputs_root / "hard_case_mining").mkdir(parents=True, exist_ok=True)
    _write_json(
        outputs_root / "hard_case_mining" / "summary.json",
        {
            "clusters": [
                {
                    "cluster_key": "contradiction_heavy_case|melanoma->nev|MEL",
                    "count": 2,
                    "average_importance": 7.0,
                    "example_case_ids": ["C1", "C2"],
                    "failure_type": "contradiction_heavy_case",
                }
            ]
        },
    )
    (outputs_root / "hard_case_mining" / "hard_cases.jsonl").write_text(
        json.dumps(
            {
                "cluster_key": "contradiction_heavy_case|melanoma->nev|MEL",
                "ground_truth": {"canonical_label": "MEL"},
                "confusion_tags": ["melanoma->nev"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (outputs_root / "skill_helpfulness").mkdir(parents=True, exist_ok=True)
    (outputs_root / "skill_helpfulness" / "skill_helpfulness_reports.jsonl").write_text(
        json.dumps(
            {
                "skill_name": "morphology_analysis_skill",
                "call_count": 10,
                "selected_count": 10,
                "helpful_count": 8,
                "partially_helpful_count": 1,
                "harmful_count": 1,
                "average_evidence_strength": 0.7,
                "helpful_rate": 0.8,
                "harmful_rate": 0.1,
                "uncertainty_reduction_count": 3,
                "contradiction_detection_count": 2,
                "malignant_flag_support_count": 4,
                "common_applicable_scenarios": [{"name": "ddx:melanoma|nev", "count": 4}],
                "common_failure_modes": ["overweighting_single_feature"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    train_run_root = outputs_root / "train_runs" / "unit_run"
    train_run_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        train_run_root / "train_run_manifest.json",
        {
            "run_id": "unit_run",
            "stages": {
                "0": {"manifest_path": str(train_run_root / "stage0_bootstrap_collect_data" / "manifest.json")},
                "1": {"manifest_path": str(train_run_root / "stage1_supervised_controller_training" / "manifest.json")},
                "2": {"manifest_path": str(train_run_root / "stage2_retrieval_scorer_training" / "manifest.json")},
                "3": {"manifest_path": str(train_run_root / "stage3_policy_candidate_evaluation" / "manifest.json")},
                "4": {"manifest_path": str(train_run_root / "stage4_stable_checkpoint_export" / "manifest.json")},
            },
        },
    )
    _write_json(
        train_run_root / "stage0_bootstrap_collect_data" / "manifest.json",
        {
            "stage_name": "bootstrap_collect_data",
            "status": "completed",
            "started_at": "2026-03-26T00:00:00Z",
            "finished_at": "2026-03-26T00:01:00Z",
            "stats": {"controller_example_count": 20, "records_selected": 20},
            "outputs": {"data_version": "bootstrap_data_unit"},
            "checkpoint": {"component": "bootstrap_data_snapshot", "version": "v1"},
        },
    )
    _write_json(
        train_run_root / "stage1_supervised_controller_training" / "manifest.json",
        {
            "stage_name": "supervised_controller_training",
            "status": "completed",
            "started_at": "2026-03-26T00:01:00Z",
            "finished_at": "2026-03-26T00:02:00Z",
            "metrics": {"val": {"micro_f1": 0.8, "topk_hit_rate": 0.9}},
            "checkpoint": {"component": "controller_planner_scorer", "version": "v1"},
        },
    )
    _write_json(
        train_run_root / "stage2_retrieval_scorer_training" / "manifest.json",
        {
            "stage_name": "retrieval_scorer_training",
            "status": "completed",
            "started_at": "2026-03-26T00:02:00Z",
            "finished_at": "2026-03-26T00:03:00Z",
            "metrics": {"val": {"pointwise": {"binary_f1": 0.85}, "grouped": {"top1_hit_rate": 0.75}}},
            "checkpoint": {"component": "retrieval_reranker", "version": "v1"},
        },
    )
    _write_json(
        train_run_root / "stage3_policy_candidate_evaluation" / "manifest.json",
        {
            "stage_name": "policy_candidate_evaluation",
            "status": "completed",
            "started_at": "2026-03-26T00:03:00Z",
            "finished_at": "2026-03-26T00:04:00Z",
            "inputs": {"data_split": "val"},
            "outputs": {
                "gate_decision": {"decision": "promote"},
                "evaluation_record_path": str(train_run_root / "stage3_policy_candidate_evaluation" / "evaluation" / "policy_eval.json"),
            },
            "checkpoint": {"component": "policy_candidate", "version": "v1"},
        },
    )
    _write_json(
        train_run_root / "stage3_policy_candidate_evaluation" / "evaluation" / "policy_eval.json",
        {
            "candidate_summary": {
                "top1": {"rate": 0.7},
                "malignant_recall": {"rate": 0.9},
            }
        },
    )
    _write_json(
        train_run_root / "stage4_stable_checkpoint_export" / "manifest.json",
        {
            "stage_name": "stable_checkpoint_export",
            "status": "completed",
            "started_at": "2026-03-26T00:04:00Z",
            "finished_at": "2026-03-26T00:05:00Z",
            "outputs": {"components": [{"component_id": "controller_planner_scorer"}], "export_tier": "run_candidate_export"},
            "checkpoint": {"component": "stable_checkpoint_bundle", "version": "v1"},
        },
    )

    record_path = eval_root / "targets" / "full_dermagent" / "records" / "case_execution_records.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "case_id": "C1",
                "evaluation": {"baseline_correct": True, "correct": False, "malignant_recall_hit": False},
                "ground_truth": {"malignant_flag": True},
                "evidence_bundle": {"contradiction_summary": {"contradiction_count": 3}, "uncertainty_summary": {"uncertainty_level": "high"}},
                "reflection_summary": {"case_outcome": {"confusion_pair": "melanoma->nev"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
