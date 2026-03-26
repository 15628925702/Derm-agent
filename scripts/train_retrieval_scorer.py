from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.hard_case_miner import load_execution_records
from agent.retrieval_scorer import (
    FEATURE_SCHEMA_VERSION,
    OBJECT_TYPES,
    RetrievalRerankerMLP,
    build_feature_vocab,
    build_retrieval_scorer_samples,
    collect_retrieval_priors,
    flatten_retrieval_training_sample_features,
    grouped_top1_hit_rate,
    infer_probabilities,
    pointwise_metrics,
    select_samples_by_case_ids,
    split_case_ids,
    train_step_with_weights,
    vectorize_feature_maps,
)


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "state" / "trainable_components" / "retrieval_reranker" / "candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight retrieval scorer/reranker for tactical/abstract/skill candidates.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-filter", type=str, default="")
    parser.add_argument("--object-types", type=str, default=",".join(OBJECT_TYPES))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--positive-label-threshold", type=float, default=0.65)
    parser.add_argument("--min-feature-count", type=int, default=1)
    parser.add_argument("--blend-weight", type=float, default=2.5)
    parser.add_argument("--checkpoint-name", type=str, default="")
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional JSON path with train/val/test case_id lists (or train_case_ids/val_case_ids/test_case_ids).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_execution_records(args.records_root)
    if args.dataset_filter.strip():
        records = [record for record in records if str(record.get("dataset_name", "")).strip() == args.dataset_filter.strip()]
    if not records:
        raise ValueError(f"No execution records found from {args.records_root} (dataset filter={args.dataset_filter!r}).")

    object_types = tuple(
        item.strip()
        for item in args.object_types.split(",")
        if item.strip() in OBJECT_TYPES
    )
    if not object_types:
        raise ValueError(f"Invalid --object-types. Allowed: {OBJECT_TYPES}")

    case_split = (
        split_case_ids_from_json(records=records, split_json_path=args.split_json)
        if args.split_json
        else split_case_ids(
            records,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
        )
    )
    train_records = [record for record in records if str(record.get("case_id", "")) in set(case_split["train"])]
    priors = collect_retrieval_priors(train_records)
    all_samples = build_retrieval_scorer_samples(records, object_types=object_types, priors=priors)
    if len(all_samples) < 8:
        raise ValueError("Not enough retrieval scorer samples (<8). Run more cases first.")

    train_samples = select_samples_by_case_ids(all_samples, case_split["train"])
    val_samples = select_samples_by_case_ids(all_samples, case_split["val"])
    test_samples = select_samples_by_case_ids(all_samples, case_split["test"])
    if not train_samples:
        raise ValueError("Training split has zero samples.")

    train_feature_maps = [flatten_retrieval_training_sample_features(sample) for sample in train_samples]
    feature_vocab = build_feature_vocab(train_feature_maps, min_feature_count=max(1, int(args.min_feature_count)))
    if not feature_vocab:
        raise ValueError("Feature vocab is empty; cannot train retrieval scorer.")

    X_train, y_train, w_train = _vectorize_samples(train_samples, feature_vocab)
    X_val, y_val, _ = _vectorize_samples(val_samples, feature_vocab)
    X_test, y_test, _ = _vectorize_samples(test_samples, feature_vocab)
    model = RetrievalRerankerMLP(
        input_dim=X_train.shape[1],
        hidden_dim=max(8, int(args.hidden_dim)),
        dropout=max(0.0, min(0.8, float(args.dropout))),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, w_train),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
    )

    epoch_history: list[dict[str, Any]] = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y, batch_w in train_loader:
            losses.append(
                train_step_with_weights(
                    model=model,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    batch_w=batch_w,
                    optimizer=optimizer,
                )
            )

        train_probs = infer_probabilities(model, X_train)
        val_probs = infer_probabilities(model, X_val)
        train_metrics = _evaluate_split(
            samples=train_samples,
            labels=y_train,
            probs=train_probs,
            positive_label_threshold=float(args.positive_label_threshold),
            prediction_threshold=float(args.prediction_threshold),
        )
        val_metrics = _evaluate_split(
            samples=val_samples,
            labels=y_val,
            probs=val_probs,
            positive_label_threshold=float(args.positive_label_threshold),
            prediction_threshold=float(args.prediction_threshold),
        )
        val_key = float((val_metrics.get("pointwise", {}) or {}).get("binary_f1") or 0.0)
        if val_key >= best_val:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": round(sum(losses) / len(losses), 6) if losses else 0.0,
                "train_binary_f1": (train_metrics.get("pointwise", {}) or {}).get("binary_f1"),
                "val_binary_f1": (val_metrics.get("pointwise", {}) or {}).get("binary_f1"),
            }
        )

    model.load_state_dict(best_state)
    train_probs = infer_probabilities(model, X_train)
    val_probs = infer_probabilities(model, X_val)
    test_probs = infer_probabilities(model, X_test)
    metrics = {
        "train": _evaluate_split(
            samples=train_samples,
            labels=y_train,
            probs=train_probs,
            positive_label_threshold=float(args.positive_label_threshold),
            prediction_threshold=float(args.prediction_threshold),
        ),
        "val": _evaluate_split(
            samples=val_samples,
            labels=y_val,
            probs=val_probs,
            positive_label_threshold=float(args.positive_label_threshold),
            prediction_threshold=float(args.prediction_threshold),
        ),
        "test": _evaluate_split(
            samples=test_samples,
            labels=y_test,
            probs=test_probs,
            positive_label_threshold=float(args.positive_label_threshold),
            prediction_threshold=float(args.prediction_threshold),
        ),
    }

    checkpoint = {
        "model_type": "torch_mlp_pointwise_retrieval_scorer",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "object_types": list(object_types),
        "feature_vocab": feature_vocab,
        "input_dim": X_train.shape[1],
        "hidden_dim": max(8, int(args.hidden_dim)),
        "dropout": max(0.0, min(0.8, float(args.dropout))),
        "prediction_threshold": float(args.prediction_threshold),
        "positive_label_threshold": float(args.positive_label_threshold),
        "blend_weight": float(args.blend_weight),
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "object_priors": {
            "object_helpful_rate": dict(priors.get("object_helpful_rate", {})),
            "object_harmful_rate": dict(priors.get("object_harmful_rate", {})),
        },
        "skill_priors": {
            "skill_helpful_rate": dict(priors.get("skill_helpful_rate", {})),
            "skill_failure_rate": dict(priors.get("skill_failure_rate", {})),
        },
        "training_data": {
            "records_root": str(args.records_root),
            "dataset_filter": args.dataset_filter.strip(),
            "record_count": len(records),
            "sample_count": len(all_samples),
            "sample_count_by_split": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "sample_count_by_object_type": _count_by_object_type(all_samples),
        },
        "split_case_ids": case_split,
        "metrics": metrics,
        "epoch_history": epoch_history,
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "min_feature_count": int(args.min_feature_count),
            "split_json_path": str(args.split_json) if args.split_json else "",
        },
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = args.checkpoint_name.strip() or f"retrieval_reranker_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt"
    checkpoint_path = args.output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "num_records": len(records),
        "num_samples": len(all_samples),
        "num_features": len(feature_vocab),
        "object_types": list(object_types),
        "metrics": metrics,
    }
    report_path = args.output_dir / f"{checkpoint_path.stem}_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _vectorize_samples(samples: list[dict[str, Any]], feature_vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        return (
            torch.zeros((0, len(feature_vocab)), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )
    feature_maps = [flatten_retrieval_training_sample_features(sample) for sample in samples]
    x = vectorize_feature_maps(feature_maps, feature_vocab)
    y = torch.tensor([float(sample.get("label", 0.0) or 0.0) for sample in samples], dtype=torch.float32)
    w = torch.tensor([float(sample.get("weight", 1.0) or 1.0) for sample in samples], dtype=torch.float32)
    return x, y, w


def _evaluate_split(
    *,
    samples: list[dict[str, Any]],
    labels: torch.Tensor,
    probs: torch.Tensor,
    positive_label_threshold: float,
    prediction_threshold: float,
) -> dict[str, Any]:
    return {
        "pointwise": pointwise_metrics(
            labels=labels,
            probs=probs,
            positive_label_threshold=positive_label_threshold,
            prediction_threshold=prediction_threshold,
        ),
        "grouped": grouped_top1_hit_rate(
            samples=samples,
            probs=probs,
            positive_label_threshold=positive_label_threshold,
        ),
        "sample_count": len(samples),
    }


def _count_by_object_type(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.get("object_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def split_case_ids_from_json(
    *,
    records: list[dict[str, Any]],
    split_json_path: Path,
) -> dict[str, list[str]]:
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    train_ids = _extract_split_ids(payload, "train")
    if not train_ids:
        raise ValueError(f"Split JSON must provide non-empty train ids: {split_json_path}")
    known_case_ids = {
        str(record.get("case_id", "")).strip()
        for record in records
        if str(record.get("case_id", "")).strip()
    }
    return {
        "train": sorted(case_id for case_id in train_ids if case_id in known_case_ids),
        "val": sorted(case_id for case_id in _extract_split_ids(payload, "val") if case_id in known_case_ids),
        "test": sorted(case_id for case_id in _extract_split_ids(payload, "test") if case_id in known_case_ids),
    }


def _extract_split_ids(payload: dict[str, Any], split: str) -> set[str]:
    values = payload.get(split)
    if values is None:
        values = payload.get(f"{split}_case_ids", [])
    if not isinstance(values, list):
        return set()
    return {str(item).strip() for item in values if str(item).strip()}


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    raise SystemExit(main())
