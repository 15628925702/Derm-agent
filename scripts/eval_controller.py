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

from agent.supervised_controller import ControllerMLP, ControllerSelectionPolicy, flatten_training_example_features, multilabel_metrics, select_skills_with_policy, vectorize_feature_maps
from scripts.train_controller import (
    DEFAULT_EXAMPLES_PATH,
    FALLBACK_EXAMPLES_PATH,
    build_candidate_mask_matrix,
    build_harmful_mask_matrix,
    build_label_matrix,
    build_target_k_list,
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
    harmful_mask = build_harmful_mask_matrix(examples, label_list)
    candidate_mask = build_candidate_mask_matrix(examples, label_list)
    target_k_values = build_target_k_list(examples)

    split_indices = _resolve_split_indices(
        split=args.split,
        checkpoint=checkpoint,
        num_samples=len(examples),
    )
    X_eval = X[split_indices] if split_indices else torch.zeros((0, X.shape[1]), dtype=torch.float32)
    y_eval = y_true[split_indices] if split_indices else torch.zeros((0, y_true.shape[1]), dtype=torch.float32)
    harmful_eval = harmful_mask[split_indices] if split_indices else torch.zeros((0, y_true.shape[1]), dtype=torch.float32)
    candidate_eval = candidate_mask[split_indices] if split_indices else torch.zeros((0, y_true.shape[1]), dtype=torch.float32)
    target_k_eval = [target_k_values[index] for index in split_indices] if split_indices else []
    examples_eval = [examples[index] for index in split_indices] if split_indices else []

    model = ControllerMLP(
        input_dim=int(checkpoint.get("input_dim", len(feature_vocab))),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        output_dim=int(checkpoint.get("output_dim", len(label_list))),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", {}))
    model.eval()
    selection_policy_payload = dict(checkpoint.get("selection_policy", {}))
    threshold = float(args.threshold if args.threshold is not None else selection_policy_payload.get("threshold", checkpoint.get("threshold", 0.5)))
    top_k = int(args.top_k if args.top_k is not None else selection_policy_payload.get("target_top_k", checkpoint.get("top_k", 5)))
    selection_policy = ControllerSelectionPolicy(
        threshold=threshold,
        target_top_k=max(1, top_k),
        min_select=max(1, int(selection_policy_payload.get("min_select", 4) or 1)),
        max_select=max(1, int(selection_policy_payload.get("max_select", 8) or 1)),
        top_k_buffer=max(0, int(selection_policy_payload.get("top_k_buffer", 1) or 0)),
    )

    with torch.no_grad():
        logits = model(X_eval) if X_eval.shape[0] > 0 else torch.zeros((0, len(label_list)), dtype=torch.float32)
        probs = torch.sigmoid(logits)
        if candidate_eval.shape == probs.shape:
            probs = probs * candidate_eval
    metrics = multilabel_metrics(
        y_eval,
        probs,
        threshold=threshold,
        top_k=top_k,
        target_k=target_k_eval,
        y_harmful=harmful_eval,
        min_select=selection_policy.min_select,
        max_select=selection_policy.max_select,
        top_k_buffer=selection_policy.top_k_buffer,
    )

    prediction_rows = _build_prediction_rows(
        examples=examples_eval,
        probabilities=probs,
        label_list=label_list,
        selection_policy=selection_policy,
        target_k_eval=target_k_eval,
    )

    report = {
        "checkpoint_path": str(args.checkpoint_path),
        "examples_path": str(examples_path),
        "split": args.split,
        "num_examples_total": len(examples),
        "num_examples_eval": len(examples_eval),
        "threshold": threshold,
        "top_k": top_k,
        "selection_policy": selection_policy.to_dict(),
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
    selection_policy: ControllerSelectionPolicy,
    target_k_eval: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if probabilities.shape[0] != len(examples):
        return rows
    for index, example in enumerate(examples):
        prob_vector = probabilities[index].tolist()
        score_map = {skill_name: float(prob) for skill_name, prob in zip(label_list, prob_vector)}
        ranked = [item[0] for item in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]
        local_policy = ControllerSelectionPolicy(
            threshold=selection_policy.threshold,
            target_top_k=max(1, target_k_eval[index] if index < len(target_k_eval) else selection_policy.target_top_k),
            min_select=selection_policy.min_select,
            max_select=selection_policy.max_select,
            top_k_buffer=selection_policy.top_k_buffer,
        )
        selected = select_skills_with_policy(ranked_skills=ranked, score_map=score_map, policy=local_policy)
        harmful = {str(item).strip() for item in dict(example.get("outcome", {})).get("explicit_negative_skills", []) if str(item).strip()}
        rows.append(
            {
                "case_id": str(example.get("case_id", "")),
                "dataset_name": str(example.get("dataset_name", "")),
                "selected_skills_pred": selected,
                "ranked_skills_pred": ranked,
                "predicted_skill_count": len(selected),
                "target_k": int(target_k_eval[index] if index < len(target_k_eval) else selection_policy.target_top_k),
                "harmful_selected_overlap": [skill_name for skill_name in selected if skill_name in harmful],
                "skill_probabilities": {key: round(value, 6) for key, value in score_map.items()},
                "selected_skills_true": [str(item).strip() for item in example.get("selected_skills", []) if str(item).strip()],
                "primary_positive_skills_true": [str(item).strip() for item in dict(example.get("outcome", {})).get("primary_positive_skills", []) if str(item).strip()],
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
