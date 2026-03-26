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

from agent.supervised_controller import ControllerMLP, flatten_training_example_features, multilabel_metrics, vectorize_feature_maps
from scripts.train_controller import (
    DEFAULT_EXAMPLES_PATH,
    FALLBACK_EXAMPLES_PATH,
    build_label_matrix,
    extract_labels,
    load_examples,
    resolve_examples_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained supervised controller checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to controller checkpoint .pt file.")
    parser.add_argument("--examples-path", type=Path, default=DEFAULT_EXAMPLES_PATH, help="Path to controller training examples JSONL.")
    parser.add_argument("--dataset-filter", type=str, default="", help="Optional dataset filter.")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--threshold", type=float, default=None, help="Override prediction threshold.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k hit metric k.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional JSON path to save evaluation report.")
    parser.add_argument("--predictions-path", type=Path, default=None, help="Optional JSONL path to save per-case predictions.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    examples_path = resolve_examples_path(args.examples_path) if args.examples_path == DEFAULT_EXAMPLES_PATH else args.examples_path
    if not examples_path.exists() and args.examples_path == DEFAULT_EXAMPLES_PATH and FALLBACK_EXAMPLES_PATH.exists():
        examples_path = FALLBACK_EXAMPLES_PATH
    examples = load_examples(examples_path, dataset_filter=args.dataset_filter.strip())
    if not examples:
        raise ValueError(f"No examples loaded from {examples_path}.")

    feature_vocab = dict(checkpoint.get("feature_vocab", {}))
    label_list = [str(item).strip() for item in checkpoint.get("label_list", []) if str(item).strip()]
    if not feature_vocab or not label_list:
        raise ValueError("Checkpoint missing feature_vocab or label_list.")

    label_source = str(checkpoint.get("label_source", "selected"))
    feature_maps = [flatten_training_example_features(example) for example in examples]
    X = vectorize_feature_maps(feature_maps, feature_vocab)
    y_true = build_label_matrix([extract_labels(example, label_source) for example in examples], label_list)

    split_indices = _resolve_split_indices(
        split=args.split,
        checkpoint=checkpoint,
        num_samples=len(examples),
    )
    X_eval = X[split_indices] if split_indices else torch.zeros((0, X.shape[1]), dtype=torch.float32)
    y_eval = y_true[split_indices] if split_indices else torch.zeros((0, y_true.shape[1]), dtype=torch.float32)
    examples_eval = [examples[index] for index in split_indices] if split_indices else []

    model = ControllerMLP(
        input_dim=int(checkpoint.get("input_dim", len(feature_vocab))),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        output_dim=int(checkpoint.get("output_dim", len(label_list))),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", {}))
    model.eval()
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.4))
    top_k = int(args.top_k if args.top_k is not None else checkpoint.get("top_k", 5))

    with torch.no_grad():
        logits = model(X_eval) if X_eval.shape[0] > 0 else torch.zeros((0, len(label_list)), dtype=torch.float32)
        probs = torch.sigmoid(logits)
    metrics = multilabel_metrics(y_eval, probs, threshold=threshold, top_k=top_k)

    prediction_rows = _build_prediction_rows(
        examples=examples_eval,
        probabilities=probs,
        label_list=label_list,
        threshold=threshold,
    )

    report = {
        "checkpoint_path": str(args.checkpoint_path),
        "examples_path": str(examples_path),
        "split": args.split,
        "num_examples_total": len(examples),
        "num_examples_eval": len(examples_eval),
        "threshold": threshold,
        "top_k": top_k,
        "label_source": label_source,
        "metrics": metrics,
    }
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.predictions_path:
        args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _resolve_split_indices(
    *,
    split: str,
    checkpoint: dict[str, Any],
    num_samples: int,
) -> list[int]:
    if split == "all":
        return list(range(num_samples))
    split_indices = checkpoint.get("split_indices", {})
    indices = [int(item) for item in split_indices.get(split, []) if isinstance(item, int)]
    indices = [item for item in indices if 0 <= item < num_samples]
    if indices:
        return indices
    if split == "train":
        return list(range(num_samples))
    return []


def _build_prediction_rows(
    *,
    examples: list[dict[str, Any]],
    probabilities: torch.Tensor,
    label_list: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if probabilities.shape[0] != len(examples):
        return rows
    for index, example in enumerate(examples):
        prob_vector = probabilities[index].tolist()
        score_map = {skill_name: float(prob) for skill_name, prob in zip(label_list, prob_vector)}
        ranked = [item[0] for item in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]
        selected = [skill_name for skill_name in ranked if score_map.get(skill_name, 0.0) >= threshold]
        rows.append(
            {
                "case_id": str(example.get("case_id", "")),
                "dataset_name": str(example.get("dataset_name", "")),
                "selected_skills_pred": selected,
                "ranked_skills_pred": ranked,
                "skill_probabilities": {key: round(value, 6) for key, value in score_map.items()},
                "selected_skills_true": [str(item).strip() for item in example.get("selected_skills", []) if str(item).strip()],
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
