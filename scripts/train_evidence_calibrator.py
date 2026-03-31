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

from agent.evidence_calibrator import (
    CALIBRATOR_VERSION,
    _EvidenceCalibrationMLP,
    _build_retrieval_feature_map,
    _build_skill_feature_map,
    _score_retrieval_record,
    _score_skill_output,
)
from agent.hard_case_miner import load_execution_records


DEFAULT_RECORDS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "state" / "trainable_components" / "evidence_calibrator" / "candidates"
FEATURE_SCHEMA_VERSION = "evidence_calibrator_features_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight learned evidence calibrator.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-filter", type=str, default="")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-feature-count", type=int, default=1)
    parser.add_argument("--checkpoint-name", type=str, default="")
    parser.add_argument("--split-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_execution_records(args.records_root)
    if args.dataset_filter.strip():
        records = [record for record in records if str(record.get("dataset_name", "")).strip() == args.dataset_filter.strip()]
    if not records:
        raise ValueError(f"No execution records found from {args.records_root} (dataset filter={args.dataset_filter!r}).")

    samples = build_evidence_training_samples(records)
    if len(samples) < 16:
        raise ValueError("Not enough evidence calibrator samples (<16). Run more cases first.")

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
    train_samples = select_samples_by_case_ids(samples, case_split["train"])
    val_samples = select_samples_by_case_ids(samples, case_split["val"])
    test_samples = select_samples_by_case_ids(samples, case_split["test"])
    if not train_samples:
        raise ValueError("Training split has zero evidence calibrator samples.")

    train_feature_maps = [sample["feature_map"] for sample in train_samples]
    feature_vocab = build_feature_vocab(train_feature_maps, min_feature_count=max(1, int(args.min_feature_count)))
    if not feature_vocab:
        raise ValueError("Feature vocab is empty; cannot train evidence calibrator.")

    X_train, y_train, w_train = vectorize_samples(train_samples, feature_vocab)
    X_val, y_val, w_val = vectorize_samples(val_samples, feature_vocab)
    X_test, y_test, w_test = vectorize_samples(test_samples, feature_vocab)

    model = _EvidenceCalibrationMLP(
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

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_f1 = -1.0
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y, batch_w in train_loader:
            logits = model(batch_x)
            loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y, reduction="none")
            loss = (loss_raw * batch_w).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_metrics = evaluate_split(model, X_train, y_train, w_train)
        val_metrics = evaluate_split(model, X_val, y_val, w_val)
        val_f1 = float(val_metrics.get("binary_f1", 0.0) or 0.0)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": round(sum(losses) / len(losses), 6) if losses else 0.0,
                "train_binary_f1": train_metrics.get("binary_f1"),
                "val_binary_f1": val_metrics.get("binary_f1"),
            }
        )

    model.load_state_dict(best_state)
    metrics = {
        "train": evaluate_split(model, X_train, y_train, w_train),
        "val": evaluate_split(model, X_val, y_val, w_val),
        "test": evaluate_split(model, X_test, y_test, w_test),
    }

    checkpoint = {
        "model_type": "torch_mlp_pointwise_evidence_calibrator",
        "calibrator_version": CALIBRATOR_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_vocab": feature_vocab,
        "input_dim": X_train.shape[1],
        "hidden_dim": max(8, int(args.hidden_dim)),
        "dropout": max(0.0, min(0.8, float(args.dropout))),
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "training_data": {
            "records_root": str(args.records_root),
            "dataset_filter": args.dataset_filter.strip(),
            "record_count": len(records),
            "sample_count": len(samples),
            "sample_count_by_split": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "sample_count_by_kind": count_by_kind(samples),
        },
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
    checkpoint_name = args.checkpoint_name.strip() or f"evidence_calibrator_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt"
    checkpoint_path = args.output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "num_records": len(records),
        "num_samples": len(samples),
        "num_features": len(feature_vocab),
        "metrics": metrics,
    }
    report_path = args.output_dir / f"{checkpoint_path.stem}_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_evidence_training_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        case_id = str(record.get("case_id", "")).strip()
        evaluation = dict(record.get("evaluation", {}))
        final_correct = bool(evaluation.get("correct"))
        reflection = dict(record.get("reflection_summary", {}))
        skill_assessments = {
            str(item.get("skill_name", "")).strip(): str(item.get("impact", "")).strip().lower()
            for item in reflection.get("skill_assessments", [])
            if isinstance(item, dict) and str(item.get("skill_name", "")).strip()
        }
        evidence_bundle = dict(record.get("evidence_bundle", {}))
        uncertainty_summary = dict(evidence_bundle.get("uncertainty_summary", {}))
        contradiction_summary = dict(evidence_bundle.get("contradiction_summary", {}))
        risk_flags = [str(item).strip() for item in evidence_bundle.get("risk_flags", []) if str(item).strip()]
        skill_retrieval_scores = dict(record.get("skill_retrieval", {}).get("retrieval_scores", {}))
        confusion_clusters = list(evidence_bundle.get("confusion_cluster_summary", {}).get("active_clusters", []))

        for skill_name, output in dict(record.get("skill_outputs", {})).items():
            if not isinstance(output, dict):
                continue
            heuristic = _score_skill_output(
                skill_name=skill_name,
                output=output,
                retrieval_score=float(skill_retrieval_scores.get(skill_name, 0.0) or 0.0),
                uncertainty_level=str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower() or "unknown",
                contradiction_count=contradiction_count(contradiction_summary),
                risk_flags=risk_flags,
                confusion_clusters=confusion_clusters,
                policy={},
            )
            feature_map = _build_skill_feature_map(
                skill_name=skill_name,
                section=infer_skill_section(skill_name),
                output=output,
                heuristic_score=heuristic,
                uncertainty_level=str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower() or "unknown",
                contradiction_count=contradiction_count(contradiction_summary),
                risk_flags=risk_flags,
            )
            label = label_from_skill_impact(skill_assessments.get(skill_name, ""), final_correct=final_correct)
            samples.append(
                {
                    "case_id": case_id,
                    "item_kind": "skill_output",
                    "item_id": skill_name,
                    "feature_map": feature_map,
                    "label": label,
                    "sample_weight": 1.2 if label > 0.5 else 1.0,
                }
            )

        for source_layer, records_key in (
            ("raw_case_memory", "retrieved_raw_cases_summary"),
            ("tactical_experience", "retrieved_tactical_experiences_summary"),
            ("abstract_experience", "retrieved_abstract_experiences_summary"),
        ):
            for retrieved in evidence_bundle.get(records_key, []):
                if not isinstance(retrieved, dict):
                    continue
                heuristic = _score_retrieval_record(
                    record=retrieved,
                    source_layer=source_layer,
                    uncertainty_level=str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower() or "unknown",
                    contradiction_count=contradiction_count(contradiction_summary),
                    risk_flags=risk_flags,
                    confusion_clusters=confusion_clusters,
                )
                feature_map = _build_retrieval_feature_map(
                    record=retrieved,
                    source_layer=source_layer,
                    heuristic_score=heuristic,
                    uncertainty_level=str(uncertainty_summary.get("uncertainty_level", "unknown")).strip().lower() or "unknown",
                    contradiction_count=contradiction_count(contradiction_summary),
                    risk_flags=risk_flags,
                )
                source_id = str(retrieved.get("source_id", "")).strip()
                if not source_id:
                    continue
                samples.append(
                    {
                        "case_id": case_id,
                        "item_kind": "retrieval_record",
                        "item_id": source_id,
                        "feature_map": feature_map,
                        "label": 1.0 if final_correct else 0.35,
                        "sample_weight": 0.8 if final_correct else 0.6,
                    }
                )
    return samples


def label_from_skill_impact(impact: str, *, final_correct: bool) -> float:
    if impact == "helpful":
        return 1.0
    if impact == "partially_helpful":
        return 0.75
    if impact == "harmful":
        return 0.0
    return 0.6 if final_correct else 0.25


def infer_skill_section(skill_name: str) -> str:
    if skill_name in {
        "lesion_description_structuring_skill",
        "morphology_analysis_skill",
        "color_pattern_analysis_skill",
        "border_surface_analysis_skill",
        "distribution_analysis_skill",
        "temporal_evolution_skill",
    }:
        return "observation"
    if skill_name in {
        "metadata_consistency_skill",
        "differential_compare_skill",
        "exclusion_reasoning_skill",
        "mel_nev_specialist_skill",
        "ack_scc_specialist_skill",
    }:
        return "comparison"
    if skill_name == "malignancy_risk_assessment_skill":
        return "risk"
    return "conflict_uncertainty"


def contradiction_count(summary: dict[str, Any]) -> int:
    total = 0
    for key in ("contradictions", "missing_links", "reasoning_gaps", "metadata_conflicts", "suspicious_points"):
        value = summary.get(key, [])
        if isinstance(value, list):
            total += len(value)
    return total


def split_case_ids(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    case_ids = sorted({str(record.get("case_id", "")).strip() for record in records if str(record.get("case_id", "")).strip()})
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n = len(case_ids)
    train_n = max(1, int(n * train_ratio))
    val_n = int(n * val_ratio)
    if train_n + val_n >= n:
        val_n = max(0, n - train_n - 1)
    test_start = train_n + val_n
    return {
        "train": case_ids[:train_n],
        "val": case_ids[train_n:test_start],
        "test": case_ids[test_start:],
    }


def split_case_ids_from_json(*, records: list[dict[str, Any]], split_json_path: Path) -> dict[str, list[str]]:
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    available = {str(record.get("case_id", "")).strip() for record in records if str(record.get("case_id", "")).strip()}
    result: dict[str, list[str]] = {}
    for short_key, long_key in (("train", "train_case_ids"), ("val", "val_case_ids"), ("test", "test_case_ids")):
        values = payload.get(short_key, payload.get(long_key, []))
        result[short_key] = [str(item).strip() for item in values if str(item).strip() in available]
    return result


def select_samples_by_case_ids(samples: list[dict[str, Any]], case_ids: list[str]) -> list[dict[str, Any]]:
    allowed = set(case_ids)
    return [sample for sample in samples if str(sample.get("case_id", "")) in allowed]


def build_feature_vocab(feature_maps: list[dict[str, float]], *, min_feature_count: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature_map in feature_maps:
        for key, value in feature_map.items():
            if float(value) == 0.0:
                continue
            counts[key] = counts.get(key, 0) + 1
    selected = sorted(key for key, count in counts.items() if count >= max(1, int(min_feature_count)))
    return {key: idx for idx, key in enumerate(selected)}


def vectorize_samples(samples: list[dict[str, Any]], feature_vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        return (
            torch.zeros((0, len(feature_vocab)), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )
    x = torch.zeros((len(samples), len(feature_vocab)), dtype=torch.float32)
    for row_idx, sample in enumerate(samples):
        for key, value in sample["feature_map"].items():
            index = feature_vocab.get(key)
            if index is not None:
                x[row_idx, index] = float(value)
    y = torch.tensor([float(sample.get("label", 0.0) or 0.0) for sample in samples], dtype=torch.float32)
    w = torch.tensor([float(sample.get("sample_weight", 1.0) or 1.0) for sample in samples], dtype=torch.float32)
    return x, y, w


def evaluate_split(model: _EvidenceCalibrationMLP, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> dict[str, float]:
    if x.shape[0] == 0:
        return {"binary_f1": 0.0, "weighted_accuracy": 0.0}
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
    preds = (probs >= 0.5).float()
    targets = (y >= 0.5).float()
    tp = float(((preds == 1.0) & (targets == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (targets == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (targets == 1.0)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((((preds == targets).float() * w).sum() / torch.clamp(w.sum(), min=1.0)).item())
    return {
        "binary_f1": round(f1, 6),
        "weighted_accuracy": round(accuracy, 6),
    }


def count_by_kind(samples: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for sample in samples:
        kind = str(sample.get("item_kind", "unknown"))
        result[kind] = result.get(kind, 0) + 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
