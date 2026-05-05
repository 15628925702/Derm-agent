from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.xiangya_sft_loader import (
    load_xiangya_sft_case_input_by_index,
    load_xiangya_sft_case_inputs,
    load_xiangya_sft_record_by_index,
)


def test_load_xiangya_sft_records_and_case_inputs(tmp_path: Path) -> None:
    data_root = tmp_path / "sft_data"
    case_dir = data_root / "skin_merge" / "0001"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "contact_dermatitis_view_1.jpg").write_bytes(b"img1")
    (case_dir / "contact_dermatitis_view_2.jpg").write_bytes(b"img2")

    row = {
        "id": "case0001",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>please analyze",
            },
            {
                "role": "assistant",
                "content": "\u8bca\u65ad\u7ed3\u679c\u4e3a\u63a5\u89e6\u6027\u76ae\u708e\u3002\u5efa\u8bae\u907f\u514d\u53ef\u7591\u81f4\u654f\u539f\u3002",
            },
        ],
        "images": [
            "/workspace/xiangya/xiangya/skin_merge/0001/contact_dermatitis_view_1.jpg",
            "/workspace/xiangya/xiangya/skin_merge/0001/contact_dermatitis_view_2.jpg",
        ],
    }
    jsonl_path = data_root / "skin_xiangya.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    record = load_xiangya_sft_record_by_index(0, data_root=data_root)
    cases = load_xiangya_sft_case_inputs(data_root=data_root)
    case = load_xiangya_sft_case_input_by_index(0, data_root=data_root)

    assert record.case_id == "case0001"
    assert record.label == "CONTACT_DERMATITIS"
    assert Path(record.image_path).exists()
    assert record.metadata["image_count"] == 2
    assert len(record.metadata["image_paths"]) == 2
    assert record.metadata["related_category"] == "RASH"

    assert len(cases) == 1
    assert case.dataset_name == "xiangya_sft"
    assert case.label_space_id == "xiangya_sft_grouped"
    assert case.label == "CONTACT_DERMATITIS"
    assert "image_paths" not in case.clinical_metadata()
    assert case.workflow_context is not None
    assert case.workflow_context["workflow_profile"] == "eczematous_family_routing_workflow"


def test_xiangya_sft_loader_can_fallback_to_full_text_label_inference(tmp_path: Path) -> None:
    data_root = tmp_path / "sft_data"
    case_dir = data_root / "skin_merge" / "0002"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "ad_case_view_1.jpg").write_bytes(b"img1")

    row = {
        "id": "case0002",
        "messages": [
            {
                "role": "user",
                "content": "<image>please analyze",
            },
            {
                "role": "assistant",
                "content": "\u75c5\u5386\u8bb0\u5f55\u4e3b\u8bc9\u201cAD\u60a3\u513f\u590d\u8bca\u201d\uff0c\u76ae\u635f\u5206\u5e03\u548c\u5f62\u6001\u7b26\u5408\u7279\u5e94\u6027\u76ae\u708e\u8868\u73b0\u3002",
            },
        ],
        "images": [
            "/workspace/xiangya/xiangya/skin_merge/0002/ad_case_view_1.jpg",
        ],
    }
    jsonl_path = data_root / "skin_xiangya.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    case = load_xiangya_sft_case_input_by_index(0, data_root=data_root)

    assert case.label == "ATOPIC_DERMATITIS"
    assert case.workflow_context is not None
    assert case.workflow_context["workflow_profile"] == "eczematous_family_routing_workflow"
