from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.supervised_controller import (
    ControllerSelectionPolicy,
    FEATURE_SCHEMA_VERSION,
    ControllerMLP,
    build_feature_vocab,
    flatten_training_example_features,
    is_finite_tensor,
    multilabel_metrics,
    vectorize_feature_maps,
)


DEFAULT_EXAMPLES_PATH = PROJECT_ROOT / "outputs" / "controller_training_data" / "controller_training_examples.jsonl"
FALLBACK_EXAMPLES_PATH = PROJECT_ROOT / "outputs" / "controller_training_data_step7" / "controller_training_examples.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "state" / "trainable_components" / "controller_planner_scorer" / "candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised learnable controller with sparse helpfulness-driven skill selection.")
    parser.add_argument("--examples-path", type=Path, default=DEFAULT_EXAMPLES_PATH, help="Path to controller training examples JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save checkpoints and reports.")
    parser.add_argument("--dataset-filter", type=str, default="", help="Optional dataset filter.")
    parser.add_argument(
        "--label-source",
        type=str,
        default="sparse_helpfulness",
        choices=("selected", "helpful", "selected_plus_helpful", "sparse_helpfulness"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-select", type=int, default=4)
    parser.add_argument("--max-select", type=int, default=8)
    parser.add_argument("--top-k-buffer", type=int, default=1)
    parser.add_argument("--harmful-negative-weight", type=float, default=2.0)
    parser.add_argument("--weak-positive-weight", type=float, default=0.75)
    parser.add_argument("--min-label-frequency", type=int, default=1)
    parser.add_argument("--checkpoint-name", type=str, default="")
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional JSON path specifying case_id splits. Keys: train/val/test or train_case_ids/val_case_ids/test_case_ids.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples_path = resolve_examples_path(args.examples_path)
    examples = load_examples(examples_path, dataset_filter=args.dataset_filter.strip())
    if not examples:
        raise ValueError(f"No controller examples found from {examples_path} with dataset filter `{args.dataset_filter}`.")

    feature_maps = [flatten_training_example_features(example) for example in examples]
    label_vocab = build_label_vocab(examples, label_source=args.label_source, min_frequency=args.min_label_frequency)
    if not label_vocab:
        raise ValueError("No label vocabulary available after filtering. Adjust `--label-source` or `--min-label-frequency`.")

    y = build_target_score_matrix(examples, label_vocab, label_source=args.label_source)
    harmful_mask = build_harmful_mask_matrix(examples, label_vocab)
    candidate_mask = build_candidate_mask_matrix(examples, label_vocab)
    target_k = build_target_k_list(examples)
    feature_vocab = build_feature_vocab(feature_maps, min_feature_count=1)
    X = vectorize_feature_maps(feature_maps, feature_vocab)
    if X.shape[0] != y.shape[0]:
        raise RuntimeError("Feature/label size mismatch.")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to train/evaluate supervised controller.")

    split_indices = (
        build_split_indices_from_case_ids(
            examples=examples,
            split_json_path=args.split_json,
        )
        if args.split_json
        else build_split_indices(
            num_samples=X.shape[0],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    )
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    if not train_idx:
        raise ValueError("Training split is empty. Increase dataset size or adjust split ratios.")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx] if val_idx else torch.zeros((0, X.shape[1]), dtype=torch.float32)
    y_val = y[val_idx] if val_idx else torch.zeros((0, y.shape[1]), dtype=torch.float32)
    X_test = X[test_idx] if test_idx else torch.zeros((0, X.shape[1]), dtype=torch.float32)
    y_test = y[test_idx] if test_idx else torch.zeros((0, y.shape[1]), dtype=torch.float32)
    harmful_train = harmful_mask[train_idx]
    harmful_val = harmful_mask[val_idx] if val_idx else torch.zeros((0, y.shape[1]), dtype=torch.float32)
    harmful_test = harmful_mask[test_idx] if test_idx else torch.zeros((0, y.shape[1]), dtype=torch.float32)
    candidate_train = candidate_mask[train_idx]
    candidate_val = candidate_mask[val_idx] if val_idx else torch.zeros((0, y.shape[1]), dtype=torch.float32)
    candidate_test = candidate_mask[test_idx] if test_idx else torch.zeros((0, y.shape[1]), dtype=torch.float32)
    target_k_train = [target_k[index] for index in train_idx]
    target_k_val = [target_k[index] for index in val_idx]
    target_k_test = [target_k[index] for index in test_idx]

    model = ControllerMLP(
        input_dim=X.shape[1],
        hidden_dim=max(8, int(args.hidden_dim)),
        output_dim=y.shape[1],
        dropout=max(0.0, min(0.8, float(args.dropout))),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, harmful_train, candidate_train),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
    )
    best_state = model.state_dict()
    best_val_f1 = -1.0
    epoch_history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch_x, batch_y, batch_harmful, batch_candidate in train_loader:
            logits = model(batch_x)
            loss = compute_sparse_controller_loss(
                logits=logits,
                target_scores=batch_y,
                harmful_mask=batch_harmful,
                candidate_mask=batch_candidate,
                harmful_negative_weight=float(args.harmful_negative_weight),
                weak_positive_weight=float(args.weak_positive_weight),
            )
            if not is_finite_tensor(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch={epoch}.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        train_metrics = evaluate_model(
            model,
            X_train,
            y_train,
            harmful_mask=harmful_train,
            candidate_mask=candidate_train,
            threshold=float(args.threshold),
            top_k=int(args.top_k),
            target_k=target_k_train,
            min_select=int(args.min_select),
            max_select=int(args.max_select),
            top_k_buffer=int(args.top_k_buffer),
        )
        val_metrics = evaluate_model(
            model,
            X_val,
            y_val,
            harmful_mask=harmful_val,
            candidate_mask=candidate_val,
            threshold=float(args.threshold),
            top_k=int(args.top_k),
            target_k=target_k_val,
            min_select=int(args.min_select),
            max_select=int(args.max_select),
            top_k_buffer=int(args.top_k_buffer),
        )
        train_loss_mean = train_loss / train_batches if train_batches else 0.0
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss_mean, 6),
                "train_micro_f1": train_metrics.get("micro_f1"),
                "val_micro_f1": val_metrics.get("micro_f1"),
            }
        )
        current_val_f1 = float(val_metrics.get("micro_f1") or 0.0)
        if current_val_f1 >= best_val_f1:
            best_val_f1 = current_val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = {
        "train": evaluate_model(
            model,
            X_train,
            y_train,
            harmful_mask=harmful_train,
            candidate_mask=candidate_train,
            threshold=float(args.threshold),
            top_k=int(args.top_k),
            target_k=target_k_train,
            min_select=int(args.min_select),
            max_select=int(args.max_select),
            top_k_buffer=int(args.top_k_buffer),
        ),
        "val": evaluate_model(
            model,
            X_val,
            y_val,
            harmful_mask=harmful_val,
            candidate_mask=candidate_val,
            threshold=float(args.threshold),
            top_k=int(args.top_k),
            target_k=target_k_val,
            min_select=int(args.min_select),
            max_select=int(args.max_select),
            top_k_buffer=int(args.top_k_buffer),
        ),
        "test": evaluate_model(
            model,
            X_test,
            y_test,
            harmful_mask=harmful_test,
            candidate_mask=candidate_test,
            threshold=float(args.threshold),
            top_k=int(args.top_k),
            target_k=target_k_test,
            min_select=int(args.min_select),
            max_select=int(args.max_select),
            top_k_buffer=int(args.top_k_buffer),
        ),
    }

    checkpoint_payload = {
        "model_type": "torch_mlp_multilabel",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_vocab": feature_vocab,
        "label_list": label_vocab,
        "input_dim": X.shape[1],
        "output_dim": y.shape[1],
        "hidden_dim": max(8, int(args.hidden_dim)),
        "dropout": max(0.0, min(0.8, float(args.dropout))),
        "threshold": float(args.threshold),
        "top_k": int(args.top_k),
        "selection_policy": ControllerSelectionPolicy(
            threshold=float(args.threshold),
            target_top_k=int(args.top_k),
            min_select=int(args.min_select),
            max_select=int(args.max_select),
            top_k_buffer=int(args.top_k_buffer),
        ).to_dict(),
        "label_source": args.label_source,
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "train_config": {
            "examples_path": str(examples_path),
            "dataset_filter": args.dataset_filter.strip(),
            "split_json_path": str(args.split_json) if args.split_json else "",
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "min_label_frequency": args.min_label_frequency,
            "min_select": args.min_select,
            "max_select": args.max_select,
            "top_k_buffer": args.top_k_buffer,
            "harmful_negative_weight": args.harmful_negative_weight,
            "weak_positive_weight": args.weak_positive_weight,
        },
        "split_indices": split_indices,
        "case_ids": [str(example.get("case_id", "")) for example in examples],
        "target_k": target_k,
        "metrics": metrics,
        "epoch_history": epoch_history,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_name = args.checkpoint_name.strip() or f"controller_mlp_{timestamp}.pt"
    checkpoint_path = args.output_dir / checkpoint_name
    torch.save(checkpoint_payload, checkpoint_path)

    report_path = args.output_dir / f"{checkpoint_path.stem}_metrics.json"
    report = {
        "checkpoint_path": str(checkpoint_path),
        "report_path": str(report_path),
        "num_examples": len(examples),
        "num_labels": len(label_vocab),
        "num_features": len(feature_vocab),
        "label_source": args.label_source,
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def resolve_examples_path(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_EXAMPLES_PATH and FALLBACK_EXAMPLES_PATH.exists():
        return FALLBACK_EXAMPLES_PATH
    raise FileNotFoundError(f"Controller training examples not found: {path}")


def load_examples(path: Path, *, dataset_filter: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if dataset_filter and str(payload.get("dataset_name", "")).strip() != dataset_filter:
                continue
            rows.append(payload)
    return rows


def build_split_indices_from_case_ids(
    *,
    examples: list[dict[str, Any]],
    split_json_path: Path,
) -> dict[str, list[int]]:
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    train_ids = _extract_split_ids(payload, "train")
    val_ids = _extract_split_ids(payload, "val")
    test_ids = _extract_split_ids(payload, "test")
    if not train_ids:
        raise ValueError(f"Split JSON must provide non-empty train ids: {split_json_path}")
    indices_by_split = {"train": [], "val": [], "test": []}
    used_indices: set[int] = set()
    for index, example in enumerate(examples):
        case_id = str(example.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id in train_ids:
            indices_by_split["train"].append(index)
            used_indices.add(index)
        elif case_id in val_ids:
            indices_by_split["val"].append(index)
            used_indices.add(index)
        elif case_id in test_ids:
            indices_by_split["test"].append(index)
            used_indices.add(index)
    if not indices_by_split["train"]:
        raise ValueError(f"No training examples matched split JSON case ids: {split_json_path}")
    # Keep unmatched examples out of training to preserve split contract.
    return indices_by_split


def _extract_split_ids(payload: dict[str, Any], split: str) -> set[str]:
    values = payload.get(split)
    if values is None:
        values = payload.get(f"{split}_case_ids", [])
    if not isinstance(values, list):
        return set()
    return {str(item).strip() for item in values if str(item).strip()}


def extract_labels(example: dict[str, Any], label_source: str) -> list[str]:
    selected = [str(item).strip() for item in example.get("selected_skills", []) if str(item).strip()]
    helpful = [str(item).strip() for item in example.get("outcome", {}).get("helpful_skills", []) if str(item).strip()]
    if label_source == "selected":
        return selected
    if label_source == "helpful":
        return helpful
    if label_source == "selected_plus_helpful":
        return list(dict.fromkeys(selected + helpful))
    if label_source == "sparse_helpfulness":
        outcome = dict(example.get("outcome", {}))
        primary = [str(item).strip() for item in outcome.get("primary_positive_skills", []) if str(item).strip()]
        weak = [str(item).strip() for item in outcome.get("weak_positive_skills", []) if str(item).strip()]
        return list(dict.fromkeys(primary + weak))
    return selected


def build_label_vocab(
    examples: list[dict[str, Any]],
    *,
    label_source: str,
    min_frequency: int,
) -> list[str]:
    counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    for example in examples:
        for label in extract_labels(example, label_source):
            counts[label] = counts.get(label, 0) + 1
        for skill_name in example.get("available_skill_candidates", []):
            normalized = str(skill_name).strip()
            if normalized:
                candidate_counts[normalized] = candidate_counts.get(normalized, 0) + 1
    labels = sorted(label for label, count in counts.items() if count >= max(1, int(min_frequency)))
    if not labels:
        labels = sorted(candidate_counts.keys())
    return labels


def build_label_matrix(label_lists: list[list[str]], label_vocab: list[str]) -> torch.Tensor:
    label_to_index = {label: idx for idx, label in enumerate(label_vocab)}
    rows = torch.zeros((len(label_lists), len(label_vocab)), dtype=torch.float32)
    for row_index, labels in enumerate(label_lists):
        for label in labels:
            column_index = label_to_index.get(label)
            if column_index is None:
                continue
            rows[row_index, column_index] = 1.0
    return rows


def build_target_score_matrix(
    examples: list[dict[str, Any]],
    label_vocab: list[str],
    *,
    label_source: str,
) -> torch.Tensor:
    if label_source != "sparse_helpfulness":
        return build_label_matrix([extract_labels(example, label_source) for example in examples], label_vocab)
    label_to_index = {label: idx for idx, label in enumerate(label_vocab)}
    rows = torch.zeros((len(examples), len(label_vocab)), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        score_map = dict(dict(example.get("outcome", {})).get("target_skill_scores", {}))
        for skill_name, score in score_map.items():
            column_index = label_to_index.get(str(skill_name).strip())
            if column_index is None:
                continue
            rows[row_index, column_index] = float(score or 0.0)
    return rows


def build_harmful_mask_matrix(examples: list[dict[str, Any]], label_vocab: list[str]) -> torch.Tensor:
    label_to_index = {label: idx for idx, label in enumerate(label_vocab)}
    rows = torch.zeros((len(examples), len(label_vocab)), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        for skill_name in dict(example.get("outcome", {})).get("explicit_negative_skills", []):
            normalized = str(skill_name).strip()
            column_index = label_to_index.get(normalized)
            if column_index is not None:
                rows[row_index, column_index] = 1.0
    return rows


def build_candidate_mask_matrix(examples: list[dict[str, Any]], label_vocab: list[str]) -> torch.Tensor:
    label_to_index = {label: idx for idx, label in enumerate(label_vocab)}
    rows = torch.zeros((len(examples), len(label_vocab)), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        for skill_name in example.get("available_skill_candidates", []):
            normalized = str(skill_name).strip()
            column_index = label_to_index.get(normalized)
            if column_index is not None:
                rows[row_index, column_index] = 1.0
    return rows


def build_target_k_list(examples: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for example in examples:
        available_count = len(example.get("available_skill_candidates", []))
        raw = int(dict(example.get("outcome", {})).get("target_k", 0) or 0)
        values.append(max(1, min(max(available_count, 1), raw if raw > 0 else 1)))
    return values


def build_split_indices(
    *,
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if num_samples < 3:
        return {"train": indices, "val": [], "test": []}

    train_count = int(round(num_samples * (train_ratio / total_ratio)))
    val_count = int(round(num_samples * (val_ratio / total_ratio)))
    train_count = max(1, min(train_count, num_samples - 2))
    val_count = max(1, min(val_count, num_samples - train_count - 1))
    test_count = num_samples - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def evaluate_model(
    model: ControllerMLP,
    X: torch.Tensor,
    y_true: torch.Tensor,
    *,
    harmful_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    threshold: float,
    top_k: int,
    target_k: list[int],
    min_select: int,
    max_select: int,
    top_k_buffer: int,
) -> dict[str, float | None]:
    if X.shape[0] == 0:
        return multilabel_metrics(
            torch.zeros((0, y_true.shape[1] if y_true.ndim == 2 else 0), dtype=torch.float32),
            torch.zeros((0, y_true.shape[1] if y_true.ndim == 2 else 0), dtype=torch.float32),
            threshold=threshold,
            top_k=top_k,
            target_k=[],
            y_harmful=torch.zeros((0, y_true.shape[1] if y_true.ndim == 2 else 0), dtype=torch.float32),
            min_select=min_select,
            max_select=max_select,
            top_k_buffer=top_k_buffer,
        )
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_prob = torch.sigmoid(logits)
        if candidate_mask.shape == y_prob.shape:
            y_prob = y_prob * candidate_mask
    return multilabel_metrics(
        (y_true >= 0.5).float(),
        y_prob,
        threshold=threshold,
        top_k=top_k,
        target_k=target_k,
        y_harmful=harmful_mask,
        min_select=min_select,
        max_select=max_select,
        top_k_buffer=top_k_buffer,
    )


def compute_sparse_controller_loss(
    *,
    logits: torch.Tensor,
    target_scores: torch.Tensor,
    harmful_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    harmful_negative_weight: float,
    weak_positive_weight: float,
) -> torch.Tensor:
    binary_targets = (target_scores > 0).float()
    per_entry_loss = nn.functional.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")
    weights = torch.ones_like(per_entry_loss)
    weights = torch.where((target_scores > 0) & (target_scores < 1.0), torch.full_like(weights, float(weak_positive_weight)), weights)
    weights = torch.where(target_scores >= 1.0, torch.full_like(weights, 1.25), weights)
    weights = torch.where(harmful_mask > 0, torch.full_like(weights, float(harmful_negative_weight)), weights)
    masked_weights = weights * torch.clamp(candidate_mask, min=0.0, max=1.0)
    denom = torch.clamp(masked_weights.sum(), min=1.0)
    return (per_entry_loss * masked_weights).sum() / denom


if __name__ == "__main__":
    raise SystemExit(main())
