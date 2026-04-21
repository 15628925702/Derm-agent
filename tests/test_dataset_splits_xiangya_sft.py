from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import XIANGYA_SFT_SPLIT_ID, build_fixed_split_payload


def test_build_fixed_split_payload_for_xiangya_sft_uses_case_ids_and_indices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    dataset_root = data_root / "sft数据"
    image_root = dataset_root / "skin_merge"
    image_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    labels = (
        ["接触性皮炎"] * 4
        + ["特应性皮炎（AD）"] * 4
        + ["口周皮炎"] * 2
        + ["疱疹性湿疹"] * 2
    )
    for index, label in enumerate(labels, start=1):
        case_dir = image_root / f"{index:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"case_{index:04d}.jpg"
        (case_dir / image_name).write_bytes(b"img")
        rows.append(
            {
                "id": f"case{index:04d}",
                "messages": [
                    {"role": "user", "content": "<image>"},
                    {"role": "assistant", "content": f"诊断结果为{label}。"},
                ],
                "images": [f"/workspace/xiangya/xiangya/skin_merge/{index:04d}/{image_name}"],
            }
        )

    jsonl_path = dataset_root / "skin_xiangya.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = build_fixed_split_payload(split_id=XIANGYA_SFT_SPLIT_ID, data_root=data_root)

    assert payload["dataset_name"] == "xiangya_sft"
    assert payload["train"][0].startswith("case")
    assert len(payload["train_case_indices"]) == len(payload["train"])
    assert len(payload["val_case_indices"]) == len(payload["val"])
    assert len(payload["test_case_indices"]) == len(payload["test"])
    assert set(payload["train"]).isdisjoint(payload["val"])
    assert set(payload["train"]).isdisjoint(payload["test"])
    assert set(payload["val"]).isdisjoint(payload["test"])
    assert "CONTACT_DERMATITIS" in payload["notes"] or payload["strategy"] == "stratified_by_xiangya_grouped_label"
