from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cognition.cognition_state import CognitionState


FEATURE_SCHEMA_VERSION = "controller_features_v1"

METADATA_FIELDS = (
    "region",
    "age",
    "diameter_1",
    "diameter_2",
    "grew",
    "changed",
    "bleed",
    "itch",
    "hurt",
    "elevation",
)


def flatten_training_example_features(example: dict[str, Any]) -> dict[str, float]:
    state_features = dict(example.get("state_features", {}))
    feature_map: dict[str, float] = {}
    initial_ddx = [str(item).strip().lower() for item in state_features.get("initial_ddx", []) if str(item).strip()]
    for item in initial_ddx:
        feature_map[f"ddx::{item}"] = 1.0
    feature_map["ddx_count"] = float(len(initial_ddx))

    uncertainty = dict(state_features.get("uncertainty", {}))
    initial_level = str(uncertainty.get("initial_level", "unknown")).strip().lower() or "unknown"
    feature_map[f"uncertainty_initial::{initial_level}"] = 1.0

    metadata_summary = dict(state_features.get("metadata_summary", {}))
    for field_name in METADATA_FIELDS:
        value = str(metadata_summary.get(field_name, "")).strip().lower()
        if not value:
            continue
        if _is_number(value):
            feature_map[f"metadata_num::{field_name}"] = float(value)
        else:
            feature_map[f"metadata::{field_name}::{value}"] = 1.0

    confusion_tags = [str(item).strip().lower() for item in state_features.get("confusion_tags", []) if str(item).strip()]
    for tag in confusion_tags:
        feature_map[f"confusion::{tag}"] = 1.0

    retrieval_summary = dict(state_features.get("retrieval_summary", {}))
    experience_layers = dict(retrieval_summary.get("experience_layers", {}))
    feature_map["retrieval_raw_case_count"] = float(experience_layers.get("raw_case_count", 0) or 0)
    feature_map["retrieval_tactical_count"] = float(experience_layers.get("tactical_count", 0) or 0)
    feature_map["retrieval_abstract_count"] = float(experience_layers.get("abstract_count", 0) or 0)

    skill_retrieval = dict(retrieval_summary.get("skill_retrieval", {}))
    retrieval_scores = dict(skill_retrieval.get("retrieval_scores", {}))
    candidate_skill_names = [
        str(item).strip()
        for item in (
            skill_retrieval.get("candidate_skill_names")
            or example.get("available_skill_candidates", [])
        )
        if str(item).strip()
    ]
    for skill_name in candidate_skill_names:
        feature_map[f"candidate::{skill_name}"] = 1.0
    for skill_name, score in retrieval_scores.items():
        normalized_name = str(skill_name).strip()
        if not normalized_name:
            continue
        feature_map[f"retrieval_score::{normalized_name}"] = float(score or 0.0)
    return feature_map


def build_runtime_controller_feature_map(
    *,
    perception: dict[str, Any],
    metadata: dict[str, Any],
    skill_retrieval_bundle: dict[str, Any],
    retrieved_experience_bundle: dict[str, Any],
    available_skill_names: list[str],
    cognition: CognitionState,
) -> dict[str, float]:
    feature_map: dict[str, float] = {}
    ddx_candidates = [str(item).strip().lower() for item in perception.get("ddx_candidates", []) if str(item).strip()]
    for item in ddx_candidates:
        feature_map[f"ddx::{item}"] = 1.0
    feature_map["ddx_count"] = float(len(ddx_candidates))

    uncertainty_level = str(perception.get("uncertainty", {}).get("level", "unknown")).strip().lower() or "unknown"
    feature_map[f"uncertainty_initial::{uncertainty_level}"] = 1.0

    for field_name in METADATA_FIELDS:
        value = str(metadata.get(field_name, "")).strip().lower()
        if not value:
            continue
        if _is_number(value):
            feature_map[f"metadata_num::{field_name}"] = float(value)
        else:
            feature_map[f"metadata::{field_name}::{value}"] = 1.0

    confusion_pair = str(skill_retrieval_bundle.get("query_summary", {}).get("confusion_pair", "")).strip().lower()
    if confusion_pair:
        feature_map[f"confusion::{confusion_pair}"] = 1.0
    for pair_name in cognition.known_confusion_patterns.keys():
        normalized = str(pair_name).strip().lower()
        if normalized and normalized == confusion_pair:
            feature_map[f"confusion_known::{normalized}"] = 1.0

    feature_map["retrieval_raw_case_count"] = float(len(retrieved_experience_bundle.get("raw_case_results", []) or []))
    feature_map["retrieval_tactical_count"] = float(len(retrieved_experience_bundle.get("tactical_results", []) or []))
    feature_map["retrieval_abstract_count"] = float(len(retrieved_experience_bundle.get("abstract_results", []) or []))

    retrieval_scores = dict(skill_retrieval_bundle.get("retrieval_scores", {}))
    for skill_name in available_skill_names:
        normalized = str(skill_name).strip()
        if not normalized:
            continue
        feature_map[f"candidate::{normalized}"] = 1.0
        feature_map[f"retrieval_score::{normalized}"] = float(retrieval_scores.get(normalized, 0.0) or 0.0)
    return feature_map


def build_feature_vocab(feature_maps: list[dict[str, float]], *, min_feature_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature_map in feature_maps:
        for key, value in feature_map.items():
            if float(value) == 0.0:
                continue
            counts[key] = counts.get(key, 0) + 1
    selected = sorted(key for key, count in counts.items() if count >= max(1, int(min_feature_count)))
    return {key: idx for idx, key in enumerate(selected)}


def vectorize_feature_map(feature_map: dict[str, float], feature_vocab: dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(feature_vocab), dtype=torch.float32)
    for key, value in feature_map.items():
        index = feature_vocab.get(key)
        if index is None:
            continue
        vector[index] = float(value)
    return vector


def vectorize_feature_maps(feature_maps: list[dict[str, float]], feature_vocab: dict[str, int]) -> torch.Tensor:
    if not feature_maps:
        return torch.zeros((0, len(feature_vocab)), dtype=torch.float32)
    rows = [vectorize_feature_map(feature_map, feature_vocab) for feature_map in feature_maps]
    return torch.stack(rows, dim=0)


class ControllerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class ControllerPrediction:
    skill_probabilities: dict[str, float]
    ranked_skills: list[str]
    selected_skills: list[str]
    rejected_skills: list[str] = field(default_factory=list)
    selection_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_probabilities": dict(self.skill_probabilities),
            "ranked_skills": list(self.ranked_skills),
            "selected_skills": list(self.selected_skills),
            "rejected_skills": list(self.rejected_skills),
            "selection_info": dict(self.selection_info),
        }


@dataclass
class ControllerSelectionPolicy:
    threshold: float
    target_top_k: int
    min_select: int
    max_select: int
    top_k_buffer: int = 1
    relative_margin: float = 0.0
    floor_score: float = 0.0
    preserve_top1: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "target_top_k": int(self.target_top_k),
            "min_select": int(self.min_select),
            "max_select": int(self.max_select),
            "top_k_buffer": int(self.top_k_buffer),
            "relative_margin": float(self.relative_margin),
            "floor_score": float(self.floor_score),
            "preserve_top1": bool(self.preserve_top1),
        }


class LearnedControllerScorer:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.checkpoint_path = str(checkpoint_path)
        self.feature_vocab = dict(checkpoint.get("feature_vocab", {}))
        self.label_list = [str(item) for item in checkpoint.get("label_list", []) if str(item).strip()]
        if not self.feature_vocab or not self.label_list:
            raise ValueError(f"Invalid controller checkpoint: {checkpoint_path}")
        input_dim = int(checkpoint.get("input_dim", len(self.feature_vocab)))
        hidden_dim = int(checkpoint.get("hidden_dim", 128))
        output_dim = int(checkpoint.get("output_dim", len(self.label_list)))
        dropout = float(checkpoint.get("dropout", 0.1))
        self.model = ControllerMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        self.model.load_state_dict(checkpoint.get("model_state_dict", {}))
        self.model.eval()
        selection_policy_payload = dict(checkpoint.get("selection_policy", {}))
        saved_threshold = float(selection_policy_payload.get("threshold", checkpoint.get("threshold", 0.5)))
        saved_top_k = int(selection_policy_payload.get("target_top_k", checkpoint.get("top_k", 5)) or 5)
        saved_min_select = int(selection_policy_payload.get("min_select", min(saved_top_k, 4)) or 1)
        saved_max_select = int(selection_policy_payload.get("max_select", max(saved_top_k + 1, saved_min_select)) or max(saved_top_k + 1, saved_min_select))
        saved_buffer = int(selection_policy_payload.get("top_k_buffer", 1) or 1)
        saved_relative_margin = float(selection_policy_payload.get("relative_margin", 0.0) or 0.0)
        saved_floor_score = float(selection_policy_payload.get("floor_score", 0.0) or 0.0)
        saved_preserve_top1 = bool(selection_policy_payload.get("preserve_top1", True))
        resolved_top_k = int(top_k) if top_k is not None else saved_top_k
        self.selection_policy = ControllerSelectionPolicy(
            threshold=float(threshold) if threshold is not None else saved_threshold,
            target_top_k=max(1, resolved_top_k),
            min_select=max(1, saved_min_select),
            max_select=max(max(1, saved_min_select), saved_max_select),
            top_k_buffer=max(0, saved_buffer),
            relative_margin=max(0.0, saved_relative_margin),
            floor_score=max(0.0, saved_floor_score),
            preserve_top1=saved_preserve_top1,
        )
        self.threshold = self.selection_policy.threshold
        self.top_k = self.selection_policy.target_top_k
        self.feature_schema_version = str(checkpoint.get("feature_schema_version", "unknown"))

    def derive_selection_policy(
        self,
        *,
        top_k: int | None = None,
        min_select: int | None = None,
        max_select: int | None = None,
        threshold: float | None = None,
        top_k_buffer: int | None = None,
        preserve_top1: bool | None = None,
    ) -> ControllerSelectionPolicy:
        base = self.selection_policy
        resolved_top_k = max(1, int(top_k if top_k is not None else base.target_top_k))
        resolved_min = max(1, int(min_select if min_select is not None else min(base.min_select, resolved_top_k)))
        resolved_max = max(resolved_min, int(max_select if max_select is not None else max(base.max_select, resolved_top_k)))
        resolved_buffer = max(0, int(top_k_buffer if top_k_buffer is not None else base.top_k_buffer))
        return ControllerSelectionPolicy(
            threshold=float(threshold if threshold is not None else base.threshold),
            target_top_k=resolved_top_k,
            min_select=min(resolved_min, resolved_max),
            max_select=resolved_max,
            top_k_buffer=resolved_buffer,
            relative_margin=float(base.relative_margin),
            floor_score=float(base.floor_score),
            preserve_top1=bool(base.preserve_top1 if preserve_top1 is None else preserve_top1),
        )

    def predict_from_feature_map(
        self,
        feature_map: dict[str, float],
        *,
        available_skill_names: list[str],
        selection_policy: ControllerSelectionPolicy | None = None,
    ) -> ControllerPrediction:
        x = vectorize_feature_map(feature_map, self.feature_vocab).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x).squeeze(0)
            probs = torch.sigmoid(logits).cpu().tolist()
        score_map = {skill_name: float(prob) for skill_name, prob in zip(self.label_list, probs)}
        available = [str(item).strip() for item in available_skill_names if str(item).strip()]
        if available:
            for skill_name in available:
                score_map.setdefault(skill_name, 0.0)
            score_map = {key: value for key, value in score_map.items() if key in set(available)}
        ranked = [item[0] for item in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]
        resolved_policy = selection_policy or self.selection_policy
        selection_details = select_skills_with_policy_details(
            ranked_skills=ranked,
            score_map=score_map,
            policy=resolved_policy,
        )
        return ControllerPrediction(
            skill_probabilities={key: round(value, 6) for key, value in score_map.items()},
            ranked_skills=ranked,
            selected_skills=list(selection_details.get("selected_skills", [])),
            rejected_skills=list(selection_details.get("rejected_skills", [])),
            selection_info=dict(selection_details),
        )

    def score_from_planner_input(
        self,
        *,
        perception: dict[str, Any],
        metadata: dict[str, Any],
        skill_retrieval_bundle: dict[str, Any],
        retrieved_experience_bundle: dict[str, Any],
        available_skill_names: list[str],
        cognition: CognitionState,
        selection_policy: ControllerSelectionPolicy | None = None,
    ) -> ControllerPrediction:
        feature_map = build_runtime_controller_feature_map(
            perception=perception,
            metadata=metadata,
            skill_retrieval_bundle=skill_retrieval_bundle,
            retrieved_experience_bundle=retrieved_experience_bundle,
            available_skill_names=available_skill_names,
            cognition=cognition,
        )
        return self.predict_from_feature_map(
            feature_map,
            available_skill_names=available_skill_names,
            selection_policy=selection_policy,
        )


def multilabel_metrics(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    *,
    threshold: float,
    top_k: int,
    target_k: list[int] | None = None,
    y_harmful: torch.Tensor | None = None,
    min_select: int = 1,
    max_select: int = 5,
    top_k_buffer: int = 1,
) -> dict[str, float | None]:
    if y_true.numel() == 0:
        return {
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "exact_match_ratio": None,
            "topk_hit_rate": None,
            "avg_predicted_labels": None,
            "avg_true_labels": None,
            "harmful_skill_over_selection_rate": None,
        }
    y_true_i = (y_true > 0.5).int()
    y_pred_i = torch.zeros_like(y_true_i)
    for idx in range(y_prob.shape[0]):
        probs = y_prob[idx]
        ranked_indices = torch.argsort(probs, descending=True).tolist()
        score_map = {index: float(probs[index].item()) for index in ranked_indices}
        policy = ControllerSelectionPolicy(
            threshold=float(threshold),
            target_top_k=max(1, int(target_k[idx] if target_k and idx < len(target_k) else top_k)),
            min_select=max(1, int(min_select)),
            max_select=max(max(1, int(min_select)), int(max_select)),
            top_k_buffer=max(0, int(top_k_buffer)),
        )
        ranked_names = [str(index) for index in ranked_indices]
        selected_names = select_skills_with_policy(
            ranked_skills=ranked_names,
            score_map={str(index): value for index, value in score_map.items()},
            policy=policy,
        )
        selected_indices = {int(name) for name in selected_names}
        for column_index in selected_indices:
            y_pred_i[idx, column_index] = 1
    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum().item())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum().item())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum().item())
    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    exact_match = float((y_true_i == y_pred_i).all(dim=1).float().mean().item())

    k = max(1, int(top_k))
    hits: list[float] = []
    for idx in range(y_true_i.shape[0]):
        true_indices = set(torch.where(y_true_i[idx] > 0)[0].tolist())
        if not true_indices:
            continue
        probs = y_prob[idx]
        top_indices = set(torch.topk(probs, k=min(k, probs.numel())).indices.tolist())
        hits.append(1.0 if true_indices.intersection(top_indices) else 0.0)
    topk_hit_rate = sum(hits) / len(hits) if hits else None

    avg_predicted_labels = float(y_pred_i.sum(dim=1).float().mean().item())
    avg_true_labels = float(y_true_i.sum(dim=1).float().mean().item())
    harmful_rate = None
    if y_harmful is not None and y_harmful.numel() == y_pred_i.numel():
        harmful_mask = (y_harmful > 0.5).int()
        harmful_selected = ((harmful_mask == 1) & (y_pred_i == 1)).any(dim=1).float()
        harmful_rate = float(harmful_selected.mean().item())
    return {
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "exact_match_ratio": round(exact_match, 6),
        "topk_hit_rate": round(topk_hit_rate, 6) if topk_hit_rate is not None else None,
        "avg_predicted_labels": round(avg_predicted_labels, 6),
        "avg_true_labels": round(avg_true_labels, 6),
        "harmful_skill_over_selection_rate": round(harmful_rate, 6) if harmful_rate is not None else None,
    }


def compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    positives = y_train.sum(dim=0)
    total = torch.full_like(positives, float(y_train.shape[0]))
    negatives = torch.clamp(total - positives, min=1.0)
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=20.0)
    return pos_weight


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def select_skills_with_policy(
    *,
    ranked_skills: list[str],
    score_map: dict[str, float],
    policy: ControllerSelectionPolicy,
) -> list[str]:
    return list(select_skills_with_policy_details(ranked_skills=ranked_skills, score_map=score_map, policy=policy).get("selected_skills", []))


def select_skills_with_policy_details(
    *,
    ranked_skills: list[str],
    score_map: dict[str, float],
    policy: ControllerSelectionPolicy,
) -> dict[str, Any]:
    if not ranked_skills:
        return {
            "selected_skills": [],
            "rejected_skills": [],
            "threshold_candidates": [],
            "desired_k": 0,
            "top_score": 0.0,
            "selection_mode": "empty",
            "policy": policy.to_dict(),
        }
    ordered = list(dict.fromkeys(str(skill_name).strip() for skill_name in ranked_skills if str(skill_name).strip()))
    positive_pool = [skill_name for skill_name in ordered if float(score_map.get(skill_name, 0.0)) >= float(policy.floor_score)]
    if not positive_pool:
        positive_pool = list(ordered)
    threshold_selected = [
        skill_name
        for skill_name in positive_pool
        if float(score_map.get(skill_name, 0.0)) >= float(policy.threshold)
    ]
    desired_k = max(int(policy.min_select), int(policy.target_top_k) + int(policy.top_k_buffer))
    desired_k = min(max(1, desired_k), max(1, int(policy.max_select)))
    top_score = float(score_map.get(positive_pool[0], 0.0)) if positive_pool else 0.0
    if threshold_selected and float(policy.relative_margin) > 0.0:
        pruned_threshold = []
        for rank, skill_name in enumerate(threshold_selected):
            score = float(score_map.get(skill_name, 0.0))
            if rank < int(policy.target_top_k) or (top_score - score) <= float(policy.relative_margin):
                pruned_threshold.append(skill_name)
        if pruned_threshold:
            threshold_selected = pruned_threshold

    selected = list(threshold_selected[:desired_k])
    if len(selected) < int(policy.min_select):
        fallback_limit = min(len(positive_pool), max(desired_k, int(policy.min_select)))
        for skill_name in positive_pool[:fallback_limit]:
            if skill_name not in selected:
                selected.append(skill_name)
            if len(selected) >= int(policy.min_select):
                break
    if policy.preserve_top1 and positive_pool:
        top_skill = positive_pool[0]
        if top_skill not in selected:
            selected = [top_skill] + selected
    selected = list(dict.fromkeys(selected))[: max(1, int(policy.max_select))]
    rejected = [skill_name for skill_name in ordered if skill_name not in set(selected)]
    return {
        "selected_skills": selected,
        "rejected_skills": rejected,
        "threshold_candidates": list(threshold_selected),
        "desired_k": int(desired_k),
        "top_score": round(top_score, 6),
        "selection_mode": "threshold_margin_topk",
        "policy": policy.to_dict(),
    }
