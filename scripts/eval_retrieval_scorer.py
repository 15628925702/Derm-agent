from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.hard_case_miner import load_execution_records
from agent.retrieval_scorer import (
    OBJECT_TYPES,
    RetrievalRerankerMLP,
    build_retrieval_scorer_samples,
    flatten_retrieval_training_sample_features,
    grouped_top1_hit_rate,
    infer_probabilities,
    pointwise_metrics,
    vectorize_feature_maps,
)


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval scorer checkpoint on execution-record-derived samples.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--dataset-filter", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--prediction-threshold", type=float, default=None)
    parser.add_argument("--positive-label-threshold", type=float, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--predictions-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    records = load_execution_records(args.records_root)
    if args.dataset_filter.strip():
        records = [record for record in records if str(record.get("dataset_name", "")).strip() == args.dataset_filter.strip()]
    if not records:
        raise ValueError(f"No execution records loaded from {args.records_root}")

    object_types = tuple(str(item).strip() for item in checkpoint.get("object_types", []) if str(item).strip()) or OBJECT_TYPES
    priors = {
        "object_helpful_rate": dict(checkpoint.get("object_priors", {}).get("object_helpful_rate", {})),
        "object_harmful_rate": dict(checkpoint.get("object_priors", {}).get("object_harmful_rate", {})),
        "skill_helpful_rate": dict(checkpoint.get("skill_priors", {}).get("skill_helpful_rate", {})),
        "skill_failure_rate": dict(checkpoint.get("skill_priors", {}).get("skill_failure_rate", {})),
    }
    samples = build_retrieval_scorer_samples(records, object_types=object_types, priors=priors)
    if not samples:
        raise ValueError("No evaluation samples built from records.")

    split_case_ids = dict(checkpoint.get("split_case_ids", {}))
    eval_samples = _select_split_samples(samples=samples, split=args.split, split_case_ids=split_case_ids)
    feature_vocab = dict(checkpoint.get("feature_vocab", {}))
    if not feature_vocab:
        raise ValueError("Checkpoint missing feature_vocab.")
    x = vectorize_feature_maps([flatten_retrieval_training_sample_features(sample) for sample in eval_samples], feature_vocab)
    y = torch.tensor([float(sample.get("label", 0.0) or 0.0) for sample in eval_samples], dtype=torch.float32)

    model = RetrievalRerankerMLP(
        input_dim=int(checkpoint.get("input_dim", len(feature_vocab))),
        hidden_dim=int(checkpoint.get("hidden_dim", 96)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", {}))
    model.eval()
    probs = infer_probabilities(model, x)

    prediction_threshold = float(
        args.prediction_threshold
        if args.prediction_threshold is not None
        else checkpoint.get("prediction_threshold", 0.5)
    )
    positive_label_threshold = float(
        args.positive_label_threshold
        if args.positive_label_threshold is not None
        else checkpoint.get("positive_label_threshold", 0.65)
    )
    report = {
        "checkpoint_path": str(args.checkpoint_path),
        "records_root": str(args.records_root),
        "dataset_filter": args.dataset_filter.strip(),
        "split": args.split,
        "num_records": len(records),
        "num_samples": len(eval_samples),
        "object_type_counts": _object_type_counts(eval_samples),
        "metrics": {
            "pointwise": pointwise_metrics(
                labels=y,
                probs=probs,
                positive_label_threshold=positive_label_threshold,
                prediction_threshold=prediction_threshold,
            ),
            "grouped": grouped_top1_hit_rate(
                samples=eval_samples,
                probs=probs,
                positive_label_threshold=positive_label_threshold,
            ),
        },
    }

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.predictions_path:
        args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_path.open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(eval_samples):
                probability = float(probs[index].item()) if index < probs.shape[0] else 0.0
                handle.write(
                    json.dumps(
                        {
                            "sample_id": sample.get("sample_id"),
                            "case_id": sample.get("case_id"),
                            "dataset_name": sample.get("dataset_name"),
                            "object_type": sample.get("object_type"),
                            "object_id": sample.get("object_id"),
                            "label": float(sample.get("label", 0.0) or 0.0),
                            "probability": round(probability, 6),
                            "base_retrieval_score": float(sample.get("retrieval_object", {}).get("base_retrieval_score", 0.0) or 0.0),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _select_split_samples(
    *,
    samples: list[dict[str, Any]],
    split: str,
    split_case_ids: dict[str, Any],
) -> list[dict[str, Any]]:
    if split == "all":
        return samples
    target_ids = {str(item).strip() for item in split_case_ids.get(split, []) if str(item).strip()}
    if not target_ids:
        return samples if split == "train" else []
    return [sample for sample in samples if str(sample.get("case_id", "")).strip() in target_ids]


def _object_type_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.get("object_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
