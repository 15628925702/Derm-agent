from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_learned_components import (
    PIPELINE_SCHEMA_VERSION,
    build_stage3_candidate_policy,
    export_stable_checkpoints,
    main,
    parse_stage_selection,
)


def test_parse_stage_selection_supports_all_and_csv() -> None:
    assert parse_stage_selection("all") == [0, 1, 2, 3, 4]
    assert parse_stage_selection("0,2,4") == [0, 2, 4]


def test_stage0_bootstrap_generates_manifest_and_snapshot(tmp_path) -> None:
    records_root = tmp_path / "records"
    records_root.mkdir(parents=True)
    jsonl_path = records_root / "case_execution_records.jsonl"
    rows = [
        {
            "case_id": "CASE_A",
            "dataset_name": "toyset",
            "planner_decision": {"selected_skills": ["morphology_analysis_skill"]},
            "evaluation": {"correct": True},
        },
        {
            "case_id": "CASE_B",
            "dataset_name": "toyset",
            "planner_decision": {"selected_skills": ["color_pattern_analysis_skill"]},
            "evaluation": {"correct": False},
        },
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_root = tmp_path / "train_runs"
    run_id = "unit_stage0_run"
    exit_code = main(
        [
            "--stages",
            "0",
            "--run-id",
            run_id,
            "--records-root",
            str(records_root),
            "--output-dir",
            str(out_root),
            "--dataset-filter",
            "toyset",
            "--limit",
            "1",
        ]
    )
    assert exit_code == 0

    run_manifest_path = out_root / run_id / "train_run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["schema_version"] == PIPELINE_SCHEMA_VERSION
    assert run_manifest["status"] == "completed"
    stage0_manifest_path = Path(run_manifest["stages"]["0"]["manifest_path"])
    stage0_manifest = json.loads(stage0_manifest_path.read_text(encoding="utf-8"))
    assert stage0_manifest["stats"]["records_selected"] == 1
    assert stage0_manifest["stats"]["controller_example_count"] == 1
    assert Path(stage0_manifest["outputs"]["records_snapshot_path"]).exists()
    assert Path(stage0_manifest["outputs"]["controller_examples_path"]).exists()


def test_build_stage3_candidate_policy_wires_checkpoints() -> None:
    stable_policy = {
        "policy_id": "stable_policy_x",
        "version": 2,
        "planner_policy": {"controller_family": "heuristic", "controller_checkpoint_path": ""},
        "retrieval_policy": {"enable_learned_retrieval_reranker": False, "retrieval_reranker_checkpoint_path": ""},
        "evidence_policy": {"enable_evidence_calibrator": True, "calibrator_mode": "heuristic", "calibrator_checkpoint_path": ""},
        "change_summary": {},
    }
    payload = build_stage3_candidate_policy(
        stable_policy=stable_policy,
        run_id="run_123",
        controller_checkpoint_path="/tmp/controller.pt",
        retrieval_checkpoint_path="/tmp/retrieval.pt",
        evidence_calibrator_checkpoint_path="/tmp/evidence.pt",
    )
    assert payload["policy_id"] == "candidate_stage4_step6_run_123"
    assert payload["version"] == 3
    assert payload["planner_policy"]["controller_family"] == "learned_supervised"
    assert payload["planner_policy"]["controller_checkpoint_path"] == "/tmp/controller.pt"
    assert payload["retrieval_policy"]["enable_learned_retrieval_reranker"] is True
    assert payload["retrieval_policy"]["retrieval_reranker_checkpoint_path"] == "/tmp/retrieval.pt"
    assert payload["evidence_policy"]["calibrator_mode"] == "hybrid"
    assert payload["evidence_policy"]["calibrator_checkpoint_path"] == "/tmp/evidence.pt"


def test_export_stable_checkpoints_writes_bundle(tmp_path) -> None:
    stage4_dir = tmp_path / "stage4"
    export_root = tmp_path / "checkpoints"
    controller_ckpt = tmp_path / "controller.pt"
    retrieval_ckpt = tmp_path / "retrieval.pt"
    evidence_ckpt = tmp_path / "evidence.pt"
    policy_path = tmp_path / "candidate_policy.json"
    controller_ckpt.write_bytes(b"controller")
    retrieval_ckpt.write_bytes(b"retrieval")
    evidence_ckpt.write_bytes(b"evidence")
    policy_path.write_text("{}", encoding="utf-8")

    bundle = export_stable_checkpoints(
        run_id="run_abc",
        stage4_dir=stage4_dir,
        checkpoint_out_dir=export_root,
        controller_checkpoint_path=str(controller_ckpt),
        retrieval_checkpoint_path=str(retrieval_ckpt),
        evidence_calibrator_checkpoint_path=str(evidence_ckpt),
        candidate_policy_path=str(policy_path),
        dry_run=False,
    )
    assert len(bundle["components"]) == 3
    assert Path(bundle["global_export_dir"]).exists()
    assert Path(stage4_dir / "stable_checkpoint_bundle_manifest.json").exists()
