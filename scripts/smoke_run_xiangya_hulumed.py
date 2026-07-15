from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_first_case_id(jsonl_path: Path) -> str:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            case_id = str(row.get("id", "")).strip()
            if case_id:
                return case_id
    raise ValueError(f"No valid case id found in {jsonl_path}")


def ensure_minimal_state(repo_root: Path) -> None:
    split_root = repo_root / "state" / "split_states" / "test"
    (split_root / "experience").mkdir(parents=True, exist_ok=True)
    cognition_path = split_root / "cognition_state.json"
    if not cognition_path.exists():
        cognition_path.write_text(
            json.dumps({"version": 1, "items": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_split_json(data_root: Path, output_path: Path, case_id: str) -> None:
    payload = {
        "split_id": "xiangya_hulumed_smoke",
        "train": [],
        "val": [],
        "test": [case_id],
        "train_case_indices": [],
        "val_case_indices": [],
        "test_case_indices": [0],
        "data_root": str(data_root),
        "metadata_path": str(data_root / "skin_xiangya.jsonl"),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 1-case Xiangya + Hulu-Med smoke compare.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8013/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Hulu-Med-4B")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or (repo_root / "outputs" / "xiangya_hulumed_smoke")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = data_root / "skin_xiangya.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing Xiangya metadata: {jsonl_path}")

    case_ids: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            case_id = str(row.get("id", "")).strip()
            if case_id:
                case_ids.append(case_id)
    if not case_ids:
        raise ValueError(f"No valid case ids found in {jsonl_path}")
    selected_case_ids = case_ids[: max(1, int(args.limit))]
    split_json = output_dir / "xiangya_hulumed_smoke_split.json"
    payload = {
        "split_id": "xiangya_hulumed_smoke",
        "train": [],
        "val": [],
        "test": selected_case_ids,
        "train_case_indices": [],
        "val_case_indices": [],
        "test_case_indices": list(range(len(selected_case_ids))),
        "data_root": str(data_root),
        "metadata_path": str(data_root / "skin_xiangya.jsonl"),
    }
    split_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_minimal_state(repo_root)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "compare_agent_vs_qwen.py"),
        "--data-root",
        str(data_root),
        "--limit",
        str(len(selected_case_ids)),
        "--split-json",
        str(split_json),
        "--non-strict-frozen-eval",
        "--agent-base-url",
        args.base_url,
        "--baseline-base-url",
        args.base_url,
        "--agent-api-key",
        args.api_key,
        "--baseline-api-key",
        args.api_key,
        "--agent-model",
        args.model,
        "--baseline-model",
        args.model,
        "--output-dir",
        str(output_dir),
        "--client-timeout",
        "60",
        "--client-max-retries",
        "0",
    ]
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
