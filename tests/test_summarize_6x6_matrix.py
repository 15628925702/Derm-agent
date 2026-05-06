from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_6x6_matrix import collect_matrix_rows


def test_collect_matrix_rows_reports_metrics_and_failure_rows(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    report_dir = root / "skinvl" / "pad20" / "compare"
    report_dir.mkdir(parents=True)
    (root / "summary.tsv").write_text(
        "\t".join(
            [
                "model",
                "dataset",
                "status",
                "bootstrap_status",
                "compare_status",
                "start_time",
                "end_time",
                "duration_sec",
                "log_file",
            ]
        )
        + "\n"
        + "skinvl\tpad20\tOK\tOK\tOK\t\t\t\tlogs/skinvl_pad20.log\n"
        + "qwen\tsd198\tFAILED\tFAILED\tSKIPPED\t\t\t\tlogs/qwen_sd198.log\n",
        encoding="utf-8",
    )
    (report_dir / "compare_agent_vs_qwen_20260505T000000Z.json").write_text(
        json.dumps(
            {
                "summary": {
                    "baseline": {
                        "top1": {"hits": 1, "total": 2, "rate": 0.5},
                        "topk": {"hits": 1, "total": 2, "rate": 0.5},
                        "malignant_recall": {"hits": 1, "total": 1, "rate": 1.0},
                    },
                    "agent": {
                        "top1": {"hits": 0, "total": 2, "rate": 0.0},
                        "topk": {"hits": 0, "total": 2, "rate": 0.0},
                        "malignant_recall": {"hits": 0, "total": 1, "rate": 0.0},
                    },
                },
                "cases": [
                    {
                        "qwen_initial": {"image_summary": "", "ddx_candidates": []},
                        "qwen_final": {"final_diagnosis": "source_id raw_case_memory retrieval_score"},
                        "evaluation": {"agent_vs_baseline_delta": {"correct_delta": -1}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = collect_matrix_rows([root])
    by_cell = {(row["model"], row["dataset"]): row for row in rows}

    assert by_cell[("skinvl", "pad20")]["top1_delta"] == "-0.500000"
    assert by_cell[("skinvl", "pad20")]["malformed_or_empty_final_cases"] == 1
    assert by_cell[("skinvl", "pad20")]["empty_initial_perception_cases"] == 1
    assert by_cell[("qwen", "sd198")]["status"] == "FAILED"
    assert by_cell[("qwen", "sd198")]["report_path"] == ""
