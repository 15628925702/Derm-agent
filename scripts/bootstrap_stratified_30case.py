#!/usr/bin/env python3
"""
Stratified bootstrap sampling for 30-case experiments.
Ensures label balance across all datasets.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dataset_splits import build_fixed_split_payload


def stratified_sample(
    split_payload: dict[str, Any],
    split_name: str,
    n_samples: int,
    seed: int = 42,
) -> list[int]:
    """
    Stratified sampling from a split, ensuring label balance.
    Returns list of case indices.
    """
    case_indices_key = f"{split_name}_case_indices"
    case_ids_key = split_name

    if case_indices_key in split_payload:
        all_indices = split_payload[case_indices_key]
        all_ids = split_payload[case_ids_key]
    else:
        # Fallback to range-based
        range_key = f"{split_name}_range"
        start, end = split_payload[range_key]
        all_indices = list(range(start, end + 1))
        all_ids = split_payload[case_ids_key]

    # Group by label (need to load metadata to get labels)
    dataset_name = split_payload["dataset_name"]
    strategy = split_payload.get("strategy", "")

    # For stratified splits, labels are already balanced
    # For contiguous splits, we need to group by label
    if "stratified" in strategy or "balanced" in strategy:
        # Already balanced, just random sample
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(all_indices, min(n_samples, len(all_indices))))
        return selected_indices

    # For contiguous splits, load metadata and group by label
    metadata_path = Path(split_payload["metadata_path"])
    if not metadata_path.exists():
        # Fallback to simple random sampling
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(all_indices, min(n_samples, len(all_indices))))
        return selected_indices

    # Load metadata and group by label
    import csv
    label_groups: dict[str, list[int]] = defaultdict(list)

    if dataset_name == "scin":
        from dataio.scin_loader import load_scin_rows

        rows = load_scin_rows(metadata_path.parent)
    else:
        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Determine label field
    label_field = None
    if "dx" in rows[0]:
        label_field = "dx"
    elif "original_label" in rows[0]:
        label_field = "original_label"
    elif "label" in rows[0]:
        label_field = "label"

    if label_field:
        for idx in all_indices:
            if idx < len(rows):
                label = str(rows[idx].get(label_field, "unknown")).strip().lower()
                if not label:
                    continue
                label_groups[label].append(idx)
    else:
        # No label field, fallback to random
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(all_indices, min(n_samples, len(all_indices))))
        return selected_indices

    if not label_groups:
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(all_indices, min(n_samples, len(all_indices))))
        return selected_indices

    # Stratified sampling: sample proportionally from each label
    rng = random.Random(seed)
    selected_indices = []

    labels = sorted(label_groups.keys())
    total_available = sum(len(label_groups[label]) for label in labels)

    for label in labels:
        indices = label_groups[label]
        # Proportional allocation
        n_from_label = max(1, int(n_samples * len(indices) / total_available))
        n_from_label = min(n_from_label, len(indices))

        sampled = rng.sample(indices, n_from_label)
        selected_indices.extend(sampled)

    # If we haven't reached n_samples, sample more from largest groups
    if len(selected_indices) < n_samples:
        remaining = n_samples - len(selected_indices)
        available = [idx for idx in all_indices if idx not in selected_indices]
        if available:
            extra = rng.sample(available, min(remaining, len(available)))
            selected_indices.extend(extra)

    # If we oversampled, trim
    if len(selected_indices) > n_samples:
        selected_indices = rng.sample(selected_indices, n_samples)

    return sorted(selected_indices)


def main():
    parser = argparse.ArgumentParser(description="Generate stratified bootstrap sample indices")
    parser.add_argument("--split-id", type=str, required=True, help="Split ID from dataset_splits.py")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--split-name", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=30, help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, help="Output JSON file (optional)")

    args = parser.parse_args()

    # Build split payload
    split_payload = build_fixed_split_payload(args.split_id, data_root=args.data_root)

    # Stratified sample
    selected_indices = stratified_sample(
        split_payload,
        args.split_name,
        args.n_samples,
        args.seed,
    )

    result = {
        "split_id": args.split_id,
        "dataset_name": split_payload["dataset_name"],
        "split_name": args.split_name,
        "n_samples": len(selected_indices),
        "seed": args.seed,
        "selected_indices": selected_indices,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[ok] Wrote {len(selected_indices)} stratified indices to {args.output}")
    else:
        # Print to stdout
        for idx in selected_indices:
            print(idx)

    return 0


if __name__ == "__main__":
    sys.exit(main())
