from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import load_policy, load_stable_policy
from dataio.ham10000_aligned_loader import DEFAULT_HAM10000_ROOT, discover_ham10000_aligned_assets, load_ham10000_aligned_records
from integrations.openai_client import DermOpenAIClient


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "ham10000_aligned_subset" / "manifests"
ALLOWED_LABELS = ("MEL", "BCC", "NEV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build custom-count HAM10000 aligned-subset manifests.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HAM10000_ROOT)
    parser.add_argument("--class-counts", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="custom")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()
    client = DermOpenAIClient()
    class_counts = parse_class_counts(args.class_counts)

    assets = discover_ham10000_aligned_assets(args.data_root)
    records = load_ham10000_aligned_records(data_root=args.data_root, limit=None, shuffle=False)
    grouped: dict[str, list[str]] = {label: [] for label in ALLOWED_LABELS}
    for record in records:
        if record.aligned_label in grouped:
            grouped[record.aligned_label].append(record.case_id)

    rng = random.Random(args.seed)
    selected_case_ids: list[str] = []
    actual_counts: dict[str, int] = {}
    for label in ALLOWED_LABELS:
        case_ids = list(grouped.get(label, []))
        rng.shuffle(case_ids)
        target_count = class_counts.get(label, 0)
        chosen = case_ids[: min(target_count, len(case_ids))]
        selected_case_ids.extend(chosen)
        actual_counts[label] = len(chosen)

    manifest = {
        "dataset": "HAM10000",
        "task": "aligned_subset_3class",
        "selection_mode": "custom_counts",
        "included_labels": ["mel", "bcc", "nv"],
        "mapped_labels": ["MEL", "BCC", "NEV"],
        "requested_class_counts": class_counts,
        "class_counts": actual_counts,
        "available_class_counts": assets.aligned_label_counts,
        "selected_case_ids": selected_case_ids,
        "seed": args.seed,
        "mode": args.tag,
        "total_case_count": len(selected_case_ids),
        "model_name": client.model,
        "policy_id": policy.get("policy_id"),
        "policy_source_path": policy.get("source_path"),
        "commit_hash": git_commit_hash(PROJECT_ROOT),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": str(args.data_root),
    }
    out_path = args.output_dir / f"ham10000_aligned_subset_manifest_{args.tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(out_path), "total_case_count": len(selected_case_ids), "class_counts": actual_counts}, ensure_ascii=False, indent=2))
    return 0


def parse_class_counts(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class-count chunk: {item}")
        label, value = item.split("=", 1)
        normalized_label = str(label).strip().upper()
        if normalized_label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported aligned label in class counts: {normalized_label}")
        parsed_value = int(value)
        if parsed_value < 0:
            raise ValueError(f"Class count must be non-negative: {item}")
        counts[normalized_label] = parsed_value
    for label in ALLOWED_LABELS:
        counts.setdefault(label, 0)
    if sum(counts.values()) <= 0:
        raise ValueError("At least one requested class count must be > 0.")
    return counts


def git_commit_hash(root: Path) -> str:
    try:
        output = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        return output
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
