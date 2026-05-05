from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import load_policy, load_stable_policy
from dataio.ham10000_aligned_loader import DEFAULT_HAM10000_ROOT, load_ham10000_aligned_records
from integrations.openai_client import DermOpenAIClient


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "ham10000_aligned_subset" / "manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible HAM10000 aligned-subset manifests.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HAM10000_ROOT)
    parser.add_argument("--mode", choices=("smoke", "medium"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=None, help="Optional override for samples per class.")
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()
    client = DermOpenAIClient()

    per_class = args.per_class if args.per_class is not None else (5 if args.mode == "smoke" else 20)
    records = load_ham10000_aligned_records(data_root=args.data_root, limit=None, shuffle=False)
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.aligned_label].append(record.case_id)

    import random

    rng = random.Random(args.seed)
    selected_case_ids: list[str] = []
    class_counts: dict[str, int] = {}
    for aligned_label in ("MEL", "BCC", "NEV"):
        case_ids = list(grouped.get(aligned_label, []))
        rng.shuffle(case_ids)
        chosen = case_ids[: min(per_class, len(case_ids))]
        selected_case_ids.extend(chosen)
        class_counts[aligned_label] = len(chosen)

    manifest = {
        "dataset": "HAM10000",
        "task": "aligned_subset_3class",
        "included_labels": ["mel", "bcc", "nv"],
        "mapped_labels": ["MEL", "BCC", "NEV"],
        "selected_case_ids": selected_case_ids,
        "class_counts": class_counts,
        "seed": args.seed,
        "mode": args.mode,
        "limit_per_class": per_class,
        "total_case_count": len(selected_case_ids),
        "model_name": client.model,
        "policy_id": policy.get("policy_id"),
        "policy_source_path": policy.get("source_path"),
        "commit_hash": git_commit_hash(PROJECT_ROOT),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = args.output_dir / f"ham10000_aligned_subset_manifest_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(out_path), "total_case_count": len(selected_case_ids)}, ensure_ascii=False, indent=2))
    return 0


def git_commit_hash(root: Path) -> str:
    try:
        output = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        return output
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
