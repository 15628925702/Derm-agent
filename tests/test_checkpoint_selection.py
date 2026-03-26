from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.checkpoint_selection import (
    build_selection_report,
    discover_checkpoint_candidates,
    save_selection_report,
)


def test_checkpoint_selection_prefers_stable_paper_eligible_candidate(tmp_path) -> None:
    train_runs_root = tmp_path / "train_runs"
    checkpoints_root = tmp_path / "checkpoints"
    output_root = tmp_path / "selection"

    _build_run(
        train_runs_root,
        run_id="run_stable",
        top1=0.82,
        topk=0.91,
        malignant=1.0,
        error_rate=0.18,
        num_with_truth=12,
        gate_decision={"decision": "promote", "passed": True, "rollback_required": False, "reasons": ["ok"], "summary_delta": {}},
        include_eval=True,
    )
    _build_run(
        train_runs_root,
        run_id="run_validation_only",
        top1=0.88,
        topk=0.93,
        malignant=0.5,
        error_rate=0.12,
        num_with_truth=6,
        gate_decision={
            "decision": "insufficient_data",
            "passed": False,
            "rollback_required": False,
            "reasons": ["below minimum cases"],
            "summary_delta": {},
        },
        include_eval=True,
    )
    _build_run(
        train_runs_root,
        run_id="run_exploratory",
        top1=0.0,
        topk=0.0,
        malignant=0.0,
        error_rate=1.0,
        num_with_truth=0,
        gate_decision={},
        include_eval=False,
    )

    candidates = discover_checkpoint_candidates(train_runs_root)
    assert len(candidates) == 3

    report = build_selection_report(
        candidates,
        train_runs_root=train_runs_root,
        checkpoints_root=checkpoints_root,
    )
    assert report["selected"]["stable_paper_checkpoint"]["candidate_id"] == "policy_run_stable"
    assert report["selected"]["best_validation_checkpoint"]["candidate_id"] == "policy_run_stable"
    assert report["selected"]["exploratory_checkpoints"][0]["candidate_id"] == "run_exploratory"

    saved = save_selection_report(
        report,
        output_dir=output_root,
        checkpoint_export_root=checkpoints_root,
    )
    report_path = Path(saved["report_path"])
    assert report_path.exists()
    stable_export = Path(saved["exports"]["stable_paper_export"]["manifest_path"])
    best_validation_export = Path(saved["exports"]["best_validation_export"]["manifest_path"])
    assert stable_export.exists()
    assert best_validation_export.exists()


def _build_run(
    train_runs_root: Path,
    *,
    run_id: str,
    top1: float,
    topk: float,
    malignant: float,
    error_rate: float,
    num_with_truth: int,
    gate_decision: dict[str, object],
    include_eval: bool,
) -> None:
    run_root = train_runs_root / run_id
    stage1_dir = run_root / "stage1_supervised_controller_training"
    stage2_dir = run_root / "stage2_retrieval_scorer_training"
    stage3_dir = run_root / "stage3_policy_candidate_evaluation"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    if include_eval:
        (stage3_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    controller_ckpt = stage1_dir / "controller.pt"
    retrieval_ckpt = stage2_dir / "retrieval.pt"
    controller_ckpt.write_bytes(b"controller")
    retrieval_ckpt.write_bytes(b"retrieval")

    controller_metrics_path = stage1_dir / "controller_metrics.json"
    controller_metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "val": {
                        "micro_f1": 0.78,
                        "topk_hit_rate": 0.92,
                        "exact_match_ratio": 0.25,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    retrieval_metrics_path = stage2_dir / "retrieval_metrics.json"
    retrieval_metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "val": {
                        "pointwise": {"binary_f1": 0.84},
                        "grouped": {"top1_hit_rate": 0.81},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    stage1_manifest_path = stage1_dir / "manifest.json"
    stage1_manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "checkpoint_path": str(controller_ckpt),
                    "metrics_path": str(controller_metrics_path),
                }
            }
        ),
        encoding="utf-8",
    )
    stage2_manifest_path = stage2_dir / "manifest.json"
    stage2_manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "checkpoint_path": str(retrieval_ckpt),
                    "metrics_path": str(retrieval_metrics_path),
                }
            }
        ),
        encoding="utf-8",
    )

    stage3_manifest_path = ""
    if include_eval:
        policy_path = stage3_dir / "candidate_policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "policy_id": f"policy_{run_id}",
                    "evaluation_gate": {"minimum_cases": 10},
                }
            ),
            encoding="utf-8",
        )
        eval_path = stage3_dir / "evaluation" / f"policy_eval_{run_id}.json"
        eval_path.write_text(
            json.dumps(
                {
                    "evaluated_at": "2026-03-26T12:00:00Z",
                    "candidate_policy": {
                        "policy_id": f"policy_{run_id}",
                        "evaluation_gate": {"minimum_cases": 10},
                    },
                    "stable_summary": {
                        "top1": {"rate": 0.8},
                        "topk": {"rate": 0.9},
                        "malignant_recall": {"rate": 1.0},
                        "error_rate": {"rate": 0.2},
                        "key_confusion_subsets": {},
                    },
                    "candidate_summary": {
                        "num_with_ground_truth": num_with_truth,
                        "top1": {"rate": top1},
                        "topk": {"rate": topk},
                        "malignant_recall": {"rate": malignant},
                        "error_rate": {"rate": error_rate},
                        "key_confusion_subsets": {
                            "melanoma->nev": {
                                "num_cases": 3,
                                "num_with_ground_truth": 3,
                                "top1": {"rate": top1},
                                "topk": {"rate": topk},
                                "malignant_recall": {"rate": malignant},
                                "error_rate": {"rate": error_rate},
                            }
                        },
                    },
                    "gate_decision": gate_decision,
                }
            ),
            encoding="utf-8",
        )
        stage3_manifest_path = stage3_dir / "manifest.json"
        Path(stage3_manifest_path).write_text(
            json.dumps(
                {
                    "inputs": {"candidate_policy_path": str(policy_path)},
                    "outputs": {"evaluation_record_path": str(eval_path)},
                }
            ),
            encoding="utf-8",
        )

    run_manifest = {
        "run_id": run_id,
        "created_at": "2026-03-26T11:00:00Z",
        "finished_at": "2026-03-26T11:30:00Z",
        "status": "completed",
        "paths": {"run_root": str(run_root)},
        "stages": {
            "1": {"manifest_path": str(stage1_manifest_path)},
            "2": {"manifest_path": str(stage2_manifest_path)},
        },
    }
    if include_eval:
        run_manifest["stages"]["3"] = {"manifest_path": str(stage3_manifest_path)}
    (run_root / "train_run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
