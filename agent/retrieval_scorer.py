from __future__ import annotations

import math
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from agent.state import CaseState
from cognition.cognition_state import CognitionState


FEATURE_SCHEMA_VERSION = "retrieval_reranker_features_v1"
OBJECT_TYPES = ("tactical_experience", "abstract_experience", "skill_candidate")
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
REASON_KEYWORDS = (
    "foundational",
    "uncertainty",
    "confusion",
    "risk",
    "compare",
    "gap",
    "conflict",
    "metadata",
    "morphology",
    "specialist",
    "escalation",
)


@dataclass
class RetrievalScorerPrediction:
    object_type: str
    object_id: str
    probability: float
    blended_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "object_id": self.object_id,
            "probability": self.probability,
            "blended_score": self.blended_score,
        }


class RetrievalRerankerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def collect_retrieval_priors(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    object_counts: dict[str, dict[str, float]] = defaultdict(lambda: {"helpful": 0.0, "harmful": 0.0, "count": 0.0})
    skill_counts: dict[str, dict[str, float]] = defaultdict(lambda: {"helpful": 0.0, "harmful": 0.0, "count": 0.0})

    for record in records:
        impact_by_skill = _skill_impact_map(record)
        skill_outputs = dict(record.get("skill_outputs", {}))
        for skill_name, impact in impact_by_skill.items():
            skill_bucket = skill_counts[skill_name]
            skill_bucket["count"] += 1.0
            if impact == "helpful":
                skill_bucket["helpful"] += 1.0
            elif impact == "partially_helpful":
                skill_bucket["helpful"] += 0.5
            elif impact == "harmful":
                skill_bucket["harmful"] += 1.0
            output = dict(skill_outputs.get(skill_name, {}))
            for source_id in _normalize_refs(output.get("referenced_experiences", [])):
                obj_bucket = object_counts[source_id]
                obj_bucket["count"] += 1.0
                if impact == "helpful":
                    obj_bucket["helpful"] += 1.0
                elif impact == "partially_helpful":
                    obj_bucket["helpful"] += 0.5
                elif impact == "harmful":
                    obj_bucket["harmful"] += 1.0

    object_helpful_rate: dict[str, float] = {}
    object_harmful_rate: dict[str, float] = {}
    object_ref_count: dict[str, float] = {}
    for source_id, bucket in object_counts.items():
        count = max(1.0, float(bucket["count"]))
        object_helpful_rate[source_id] = round(float(bucket["helpful"]) / count, 6)
        object_harmful_rate[source_id] = round(float(bucket["harmful"]) / count, 6)
        object_ref_count[source_id] = float(bucket["count"])

    skill_helpful_rate: dict[str, float] = {}
    skill_failure_rate: dict[str, float] = {}
    for skill_name, bucket in skill_counts.items():
        count = max(1.0, float(bucket["count"]))
        skill_helpful_rate[skill_name] = round(float(bucket["helpful"]) / count, 6)
        skill_failure_rate[skill_name] = round(float(bucket["harmful"]) / count, 6)

    return {
        "object_helpful_rate": object_helpful_rate,
        "object_harmful_rate": object_harmful_rate,
        "object_ref_count": object_ref_count,
        "skill_helpful_rate": skill_helpful_rate,
        "skill_failure_rate": skill_failure_rate,
    }


def build_retrieval_scorer_samples(
    records: list[dict[str, Any]],
    *,
    object_types: tuple[str, ...] = OBJECT_TYPES,
    priors: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(
            build_retrieval_scorer_samples_from_record(
                record,
                object_types=object_types,
                priors=priors or {},
            )
        )
    return rows


def build_retrieval_scorer_samples_from_record(
    record: dict[str, Any],
    *,
    object_types: tuple[str, ...] = OBJECT_TYPES,
    priors: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    priors = priors or {}
    case_id = str(record.get("case_id", "")).strip()
    if not case_id:
        return []
    dataset_name = str(record.get("dataset_name", "unknown_dataset")).strip() or "unknown_dataset"
    qwen_initial = dict(record.get("qwen_initial", {}))
    clinical_metadata = dict(record.get("input_summary", {}).get("clinical_metadata", {}))
    uncertainty_level = str(qwen_initial.get("uncertainty", {}).get("level", "unknown")).strip().lower() or "unknown"
    case_context = {
        "initial_ddx": [str(item).strip() for item in qwen_initial.get("ddx_candidates", []) if str(item).strip()],
        "uncertainty_level": uncertainty_level,
        "metadata_summary": {
            field_name: str(clinical_metadata.get(field_name, "")).strip()
            for field_name in METADATA_FIELDS
            if str(clinical_metadata.get(field_name, "")).strip()
        },
        "confusion_tags": _extract_confusion_tags(record),
        "risk_flags": [
            str(item).strip()
            for item in record.get("reflection_summary", {}).get("case_outcome", {}).get("risk_flags", [])
            if str(item).strip()
        ],
    }
    skill_impacts = _skill_impact_map(record)
    object_reference_impacts = _object_reference_impacts(record, skill_impacts)
    cognition_skill_stats = dict(record.get("cognition_snapshot", {}).get("before", {}).get("skill_statistics", {}))
    evaluation = dict(record.get("evaluation", {}))
    delta = dict(evaluation.get("agent_vs_baseline_delta", {}))
    base_weight = 1.0
    if evaluation.get("correct") is True:
        base_weight += 0.2
    if float(delta.get("correct_delta", 0) or 0) > 0:
        base_weight += 0.2
    if float(delta.get("correct_delta", 0) or 0) < 0:
        base_weight += 0.3

    rows: list[dict[str, Any]] = []
    retrieval_before = dict(record.get("retrieval_bundle", {}).get("before_skills", {}))
    if "tactical_experience" in object_types:
        for packet in retrieval_before.get("tactical_results", []):
            rows.append(
                _build_experience_sample_row(
                    record=record,
                    packet=packet,
                    object_type="tactical_experience",
                    case_context=case_context,
                    object_reference_impacts=object_reference_impacts,
                    priors=priors,
                    base_weight=base_weight,
                )
            )
    if "abstract_experience" in object_types:
        for packet in retrieval_before.get("abstract_results", []):
            rows.append(
                _build_experience_sample_row(
                    record=record,
                    packet=packet,
                    object_type="abstract_experience",
                    case_context=case_context,
                    object_reference_impacts=object_reference_impacts,
                    priors=priors,
                    base_weight=base_weight,
                )
            )
    if "skill_candidate" in object_types:
        skill_retrieval = dict(record.get("skill_retrieval", {}))
        candidate_names = [
            str(item).strip()
            for item in (
                skill_retrieval.get("candidate_skill_names")
                or list(dict(skill_retrieval.get("retrieval_scores", {})).keys())
            )
            if str(item).strip()
        ]
        selected_skills = {
            str(item).strip()
            for item in (record.get("selected_skills") or record.get("planner_decision", {}).get("selected_skills", []))
            if str(item).strip()
        }
        retrieval_scores = dict(skill_retrieval.get("retrieval_scores", {}))
        trigger_hits = dict(skill_retrieval.get("trigger_hits", {}))
        match_reasons = dict(skill_retrieval.get("match_reasons", {}))
        for skill_name in candidate_names:
            impact = skill_impacts.get(skill_name, "neutral")
            label = _impact_to_soft_label(impact, selected=skill_name in selected_skills, is_skill_candidate=True)
            helpful_prior = float(priors.get("skill_helpful_rate", {}).get(skill_name, 0.0))
            failure_prior = float(priors.get("skill_failure_rate", {}).get(skill_name, 0.0))
            cognition_stats = dict(cognition_skill_stats.get(skill_name, {}))
            sample_weight = base_weight + (0.2 if impact in {"helpful", "harmful"} else 0.0)
            rows.append(
                {
                    "sample_id": f"{case_id}::skill_candidate::{skill_name}",
                    "case_id": case_id,
                    "dataset_name": dataset_name,
                    "object_type": "skill_candidate",
                    "object_id": skill_name,
                    "label": label,
                    "weight": round(sample_weight, 6),
                    "case_context": deepcopy(case_context),
                    "retrieval_object": {
                        "source_layer": "skill_retrieval",
                        "source_subtype": "candidate",
                        "experience_type": "skill_candidate",
                        "base_retrieval_score": float(retrieval_scores.get(skill_name, 0.0) or 0.0),
                        "trigger_hits": [
                            str(item).strip() for item in trigger_hits.get(skill_name, []) if str(item).strip()
                        ],
                        "match_reasons": [
                            str(item).strip() for item in match_reasons.get(skill_name, []) if str(item).strip()
                        ],
                    },
                    "prior_stats": {
                        "object_helpful_prior": helpful_prior,
                        "object_harmful_prior": failure_prior,
                        "skill_helpful_rate": float(cognition_stats.get("helpful_rate", helpful_prior) or 0.0),
                        "skill_failure_rate": float(cognition_stats.get("failure_rate", failure_prior) or 0.0),
                    },
                    "target_signal": {
                        "referenced_by_helpful_skill": False,
                        "referenced_by_harmful_skill": False,
                        "selected_by_planner": skill_name in selected_skills,
                        "impact": impact,
                        "final_correct": evaluation.get("correct"),
                        "delta_vs_baseline": float(delta.get("correct_delta", 0.0) or 0.0),
                    },
                }
            )
    return rows


def flatten_retrieval_training_sample_features(sample: dict[str, Any]) -> dict[str, float]:
    feature_map: dict[str, float] = {}
    object_type = str(sample.get("object_type", "unknown")).strip().lower() or "unknown"
    feature_map[f"object_type::{object_type}"] = 1.0

    case_context = dict(sample.get("case_context", {}))
    initial_ddx = [str(item).strip().lower() for item in case_context.get("initial_ddx", []) if str(item).strip()]
    for item in initial_ddx:
        feature_map[f"ddx::{item}"] = 1.0
    feature_map["ddx_count"] = float(len(initial_ddx))

    uncertainty_level = str(case_context.get("uncertainty_level", "unknown")).strip().lower() or "unknown"
    feature_map[f"uncertainty::{uncertainty_level}"] = 1.0

    metadata_summary = dict(case_context.get("metadata_summary", {}))
    for field_name in METADATA_FIELDS:
        value = str(metadata_summary.get(field_name, "")).strip().lower()
        if not value:
            continue
        if _is_number(value):
            feature_map[f"metadata_num::{field_name}"] = float(value)
        else:
            feature_map[f"metadata::{field_name}::{value}"] = 1.0

    for tag in case_context.get("confusion_tags", []):
        normalized = str(tag).strip().lower()
        if normalized:
            feature_map[f"confusion::{normalized}"] = 1.0
    for risk in case_context.get("risk_flags", []):
        normalized = str(risk).strip().lower()
        if normalized:
            feature_map[f"risk::{normalized}"] = 1.0

    retrieval_object = dict(sample.get("retrieval_object", {}))
    source_layer = str(retrieval_object.get("source_layer", "unknown")).strip().lower() or "unknown"
    source_subtype = str(retrieval_object.get("source_subtype", "unknown")).strip().lower() or "unknown"
    exp_type = str(retrieval_object.get("experience_type", "unknown")).strip().lower() or "unknown"
    feature_map[f"source_layer::{source_layer}"] = 1.0
    feature_map[f"source_subtype::{source_subtype}"] = 1.0
    feature_map[f"exp_type::{exp_type}"] = 1.0
    feature_map["base_retrieval_score"] = float(retrieval_object.get("base_retrieval_score", 0.0) or 0.0)

    for trigger in retrieval_object.get("trigger_hits", []):
        normalized = str(trigger).strip().lower()
        if normalized:
            feature_map[f"trigger::{normalized}"] = 1.0
    reasons_text = " ".join(str(item).strip().lower() for item in retrieval_object.get("match_reasons", []))
    for keyword in REASON_KEYWORDS:
        if keyword in reasons_text:
            feature_map[f"reason_kw::{keyword}"] = 1.0

    object_id = str(sample.get("object_id", "")).strip()
    if object_type == "skill_candidate" and object_id:
        feature_map[f"skill_name::{object_id}"] = 1.0

    prior = dict(sample.get("prior_stats", {}))
    feature_map["prior_object_helpful"] = float(prior.get("object_helpful_prior", 0.0) or 0.0)
    feature_map["prior_object_harmful"] = float(prior.get("object_harmful_prior", 0.0) or 0.0)
    feature_map["prior_skill_helpful"] = float(prior.get("skill_helpful_rate", 0.0) or 0.0)
    feature_map["prior_skill_failure"] = float(prior.get("skill_failure_rate", 0.0) or 0.0)
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


def pointwise_metrics(
    labels: torch.Tensor,
    probs: torch.Tensor,
    *,
    positive_label_threshold: float = 0.65,
    prediction_threshold: float = 0.5,
) -> dict[str, float | None]:
    if labels.numel() == 0 or probs.numel() == 0:
        return {
            "brier": None,
            "mae": None,
            "mse": None,
            "binary_precision": None,
            "binary_recall": None,
            "binary_f1": None,
            "avg_prob": None,
            "positive_rate": None,
        }
    labels = labels.float()
    probs = probs.float()
    brier = float(torch.mean((probs - labels) ** 2).item())
    mae = float(torch.mean(torch.abs(probs - labels)).item())
    mse = float(torch.mean((probs - labels) ** 2).item())
    y_true = (labels >= positive_label_threshold).int()
    y_pred = (probs >= prediction_threshold).int()
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "brier": round(brier, 6),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "binary_precision": round(precision, 6),
        "binary_recall": round(recall, 6),
        "binary_f1": round(f1, 6),
        "avg_prob": round(float(probs.mean().item()), 6),
        "positive_rate": round(float(y_true.float().mean().item()), 6),
    }


def grouped_top1_hit_rate(
    samples: list[dict[str, Any]],
    probs: torch.Tensor,
    *,
    positive_label_threshold: float = 0.65,
) -> dict[str, float | None]:
    if probs.numel() == 0 or not samples:
        return {"top1_hit_rate": None, "group_count": 0}
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        key = (str(sample.get("case_id", "")), str(sample.get("object_type", "")))
        grouped[key].append((idx, sample))
    hits = 0.0
    valid_groups = 0
    for rows in grouped.values():
        if not rows:
            continue
        best_idx = max(rows, key=lambda item: float(probs[item[0]].item()))[0]
        best_label = float(samples[best_idx].get("label", 0.0) or 0.0)
        valid_groups += 1
        if best_label >= positive_label_threshold:
            hits += 1.0
    if valid_groups == 0:
        return {"top1_hit_rate": None, "group_count": 0}
    return {
        "top1_hit_rate": round(hits / valid_groups, 6),
        "group_count": valid_groups,
    }


class LearnedRetrievalScorer:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        blend_weight: float | None = None,
    ) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.checkpoint_path = str(checkpoint_path)
        self.feature_schema_version = str(checkpoint.get("feature_schema_version", "unknown"))
        self.feature_vocab = dict(checkpoint.get("feature_vocab", {}))
        if not self.feature_vocab:
            raise ValueError(f"Invalid retrieval scorer checkpoint (empty feature vocab): {checkpoint_path}")
        input_dim = int(checkpoint.get("input_dim", len(self.feature_vocab)))
        hidden_dim = int(checkpoint.get("hidden_dim", 96))
        dropout = float(checkpoint.get("dropout", 0.1))
        self.model = RetrievalRerankerMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.model.load_state_dict(checkpoint.get("model_state_dict", {}))
        self.model.eval()
        self.object_priors = dict(checkpoint.get("object_priors", {}))
        self.skill_priors = dict(checkpoint.get("skill_priors", {}))
        self.blend_weight = float(blend_weight if blend_weight is not None else checkpoint.get("blend_weight", 2.5))

    def predict_probability(self, sample: dict[str, Any]) -> float:
        feature_map = flatten_retrieval_training_sample_features(sample)
        x = vectorize_feature_map(feature_map, self.feature_vocab).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(x).squeeze(0)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def rerank_experience_bundle(self, bundle: dict[str, Any], *, case_state: CaseState) -> dict[str, Any]:
        reranked = deepcopy(bundle)
        for field_name, object_type in (
            ("tactical_results", "tactical_experience"),
            ("abstract_results", "abstract_experience"),
        ):
            rows = []
            for packet in reranked.get(field_name, []) or []:
                sample = self._build_runtime_experience_sample(case_state=case_state, packet=packet, object_type=object_type)
                probability = self.predict_probability(sample)
                base_score = float(packet.get("retrieval_score", 0.0) or 0.0)
                prior = float(sample.get("prior_stats", {}).get("object_helpful_prior", 0.0) or 0.0)
                blended = base_score + self.blend_weight * probability + 0.5 * prior
                enriched = dict(packet)
                enriched.setdefault("base_retrieval_score", base_score)
                enriched["rerank_probability"] = round(probability, 6)
                enriched["rerank_score"] = round(blended, 6)
                rows.append(enriched)
            rows.sort(
                key=lambda item: (
                    float(item.get("rerank_score", 0.0)),
                    float(item.get("base_retrieval_score", item.get("retrieval_score", 0.0))),
                    str(item.get("source_id", "")),
                ),
                reverse=True,
            )
            reranked[field_name] = rows

        raw_rows = list(reranked.get("raw_case_results", []) or [])
        query = dict(reranked.get("query", {}))
        top_k_tactical = int(query.get("top_k_tactical", len(reranked.get("tactical_results", []) or [])) or 0)
        top_k_abstract = int(query.get("top_k_abstract", len(reranked.get("abstract_results", []) or [])) or 0)
        top_k_raw = int(query.get("top_k_raw", len(raw_rows)) or 0)
        top_k_merged = int(query.get("top_k_merged", max(top_k_tactical, top_k_abstract, 4)) or 0)
        reranked["tactical_results"] = list(reranked.get("tactical_results", []))[: max(0, top_k_tactical)]
        reranked["abstract_results"] = list(reranked.get("abstract_results", []))[: max(0, top_k_abstract)]
        reranked["raw_case_results"] = raw_rows[: max(0, top_k_raw)]
        reranked["planner_summary"] = _experience_summary_slice(
            primary=reranked["tactical_results"],
            secondary=reranked["abstract_results"],
            tertiary=reranked["raw_case_results"],
            top_k=max(0, top_k_tactical),
        )
        reranked["skill_summary"] = _experience_summary_slice(
            primary=reranked["tactical_results"],
            secondary=reranked["abstract_results"],
            tertiary=reranked["raw_case_results"],
            top_k=max(0, top_k_tactical),
        )
        reranked["aggregator_summary"] = _experience_merge_slice(
            tactical_results=reranked["tactical_results"],
            abstract_results=reranked["abstract_results"],
            raw_case_results=reranked["raw_case_results"],
            top_k=max(0, top_k_merged),
        )
        reranked["merged_results"] = list(reranked["aggregator_summary"])
        reranked["reranker"] = {
            "applied": True,
            "applied_to": ["tactical_experience", "abstract_experience"],
            "checkpoint_path": self.checkpoint_path,
            "feature_schema_version": self.feature_schema_version,
            "blend_weight": self.blend_weight,
        }
        return reranked

    def rerank_skill_bundle(
        self,
        bundle: dict[str, Any],
        *,
        case_state: CaseState,
        cognition: CognitionState,
    ) -> dict[str, Any]:
        reranked = deepcopy(bundle)
        candidate_names = [
            str(item).strip()
            for item in reranked.get("candidate_skill_names", [])
            if str(item).strip()
        ]
        retrieval_scores = dict(reranked.get("retrieval_scores", {}))
        skill_stats = dict(getattr(cognition, "skill_statistics", {}))
        scored_rows: list[tuple[str, float, float, float]] = []
        for skill_name in candidate_names:
            sample = self._build_runtime_skill_sample(
                case_state=case_state,
                bundle=reranked,
                skill_name=skill_name,
                skill_stats=skill_stats,
            )
            probability = self.predict_probability(sample)
            base_score = float(retrieval_scores.get(skill_name, 0.0) or 0.0)
            prior_help = float(sample.get("prior_stats", {}).get("skill_helpful_rate", 0.0) or 0.0)
            prior_fail = float(sample.get("prior_stats", {}).get("skill_failure_rate", 0.0) or 0.0)
            blended = base_score + self.blend_weight * probability + 0.8 * prior_help - 0.8 * prior_fail
            scored_rows.append((skill_name, blended, probability, base_score))

        scored_rows.sort(key=lambda item: (item[1], item[3], item[0]), reverse=True)
        reranked["candidate_skill_names"] = [item[0] for item in scored_rows]
        id_to_name = dict(reranked.get("skill_id_to_name", {}))
        name_to_id = {name: skill_id for skill_id, name in id_to_name.items()}
        reranked["candidate_skill_ids"] = [name_to_id[name] for name in reranked["candidate_skill_names"] if name in name_to_id]
        reranked["retrieval_scores"] = {
            skill_name: round(score, 6)
            for skill_name, score, _, _ in scored_rows
        }
        reranked.setdefault("reranker", {})
        reranked["reranker"] = {
            "applied": True,
            "applied_to": ["skill_candidate"],
            "checkpoint_path": self.checkpoint_path,
            "feature_schema_version": self.feature_schema_version,
            "blend_weight": self.blend_weight,
            "skill_rerank_probabilities": {name: round(prob, 6) for name, _, prob, _ in scored_rows},
        }
        decision_trace = list(reranked.get("decision_trace", []))
        score_map = {name: score for name, score, _, _ in scored_rows}
        prob_map = {name: prob for name, _, prob, _ in scored_rows}
        for item in decision_trace:
            skill_name = str(item.get("skill_name", "")).strip()
            if not skill_name or skill_name not in score_map:
                continue
            item["rerank_probability"] = round(float(prob_map.get(skill_name, 0.0)), 6)
            item["rerank_score"] = round(float(score_map.get(skill_name, 0.0)), 6)
        reranked["decision_trace"] = decision_trace
        return reranked

    def _build_runtime_experience_sample(
        self,
        *,
        case_state: CaseState,
        packet: dict[str, Any],
        object_type: str,
    ) -> dict[str, Any]:
        source_id = str(packet.get("source_id", "")).strip()
        object_helpful = float(self.object_priors.get("object_helpful_rate", {}).get(source_id, 0.0))
        object_harmful = float(self.object_priors.get("object_harmful_rate", {}).get(source_id, 0.0))
        return {
            "sample_id": f"{case_state.case_input.case_id}::{object_type}::{source_id}",
            "case_id": case_state.case_input.case_id,
            "dataset_name": str(case_state.case_input.dataset_name or "unknown_dataset"),
            "object_type": object_type,
            "object_id": source_id,
            "label": 0.0,
            "weight": 1.0,
            "case_context": _build_case_context_from_state(case_state),
            "retrieval_object": {
                "source_layer": str(packet.get("source_layer", "")).strip(),
                "source_subtype": str(packet.get("source_subtype", "")).strip(),
                "experience_type": str(packet.get("experience_type", "")).strip(),
                "base_retrieval_score": float(packet.get("retrieval_score", 0.0) or 0.0),
                "trigger_hits": [],
                "match_reasons": [],
            },
            "prior_stats": {
                "object_helpful_prior": object_helpful,
                "object_harmful_prior": object_harmful,
                "skill_helpful_rate": 0.0,
                "skill_failure_rate": 0.0,
            },
            "target_signal": {},
        }

    def _build_runtime_skill_sample(
        self,
        *,
        case_state: CaseState,
        bundle: dict[str, Any],
        skill_name: str,
        skill_stats: dict[str, Any],
    ) -> dict[str, Any]:
        cognition_stats = dict(skill_stats.get(skill_name, {}))
        helpful_prior = float(self.skill_priors.get("skill_helpful_rate", {}).get(skill_name, 0.0))
        failure_prior = float(self.skill_priors.get("skill_failure_rate", {}).get(skill_name, 0.0))
        return {
            "sample_id": f"{case_state.case_input.case_id}::skill_candidate::{skill_name}",
            "case_id": case_state.case_input.case_id,
            "dataset_name": str(case_state.case_input.dataset_name or "unknown_dataset"),
            "object_type": "skill_candidate",
            "object_id": skill_name,
            "label": 0.0,
            "weight": 1.0,
            "case_context": _build_case_context_from_state(case_state),
            "retrieval_object": {
                "source_layer": "skill_retrieval",
                "source_subtype": "candidate",
                "experience_type": "skill_candidate",
                "base_retrieval_score": float(dict(bundle.get("retrieval_scores", {})).get(skill_name, 0.0) or 0.0),
                "trigger_hits": [str(item).strip() for item in dict(bundle.get("trigger_hits", {})).get(skill_name, []) if str(item).strip()],
                "match_reasons": [
                    str(item).strip() for item in dict(bundle.get("match_reasons", {})).get(skill_name, []) if str(item).strip()
                ],
            },
            "prior_stats": {
                "object_helpful_prior": helpful_prior,
                "object_harmful_prior": failure_prior,
                "skill_helpful_rate": float(cognition_stats.get("helpful_rate", helpful_prior) or 0.0),
                "skill_failure_rate": float(cognition_stats.get("failure_rate", failure_prior) or 0.0),
            },
            "target_signal": {},
        }


def split_case_ids(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    case_ids = sorted(
        {
            str(record.get("case_id", "")).strip()
            for record in records
            if str(record.get("case_id", "")).strip()
        }
    )
    if not case_ids:
        return {"train": [], "val": [], "test": []}
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(case_ids), generator=generator).tolist()
    shuffled = [case_ids[idx] for idx in permutation]
    if len(shuffled) < 3:
        return {"train": shuffled, "val": [], "test": []}
    train_count = int(round(len(shuffled) * (train_ratio / total_ratio)))
    val_count = int(round(len(shuffled) * (val_ratio / total_ratio)))
    train_count = max(1, min(train_count, len(shuffled) - 2))
    val_count = max(1, min(val_count, len(shuffled) - train_count - 1))
    test_count = len(shuffled) - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def select_samples_by_case_ids(samples: list[dict[str, Any]], case_ids: list[str]) -> list[dict[str, Any]]:
    allowed = set(case_ids)
    if not allowed:
        return []
    return [sample for sample in samples if str(sample.get("case_id", "")) in allowed]


def train_step_with_weights(
    *,
    model: RetrievalRerankerMLP,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_w: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    logits = model(batch_x)
    loss_raw = F.binary_cross_entropy_with_logits(logits, batch_y, reduction="none")
    weights = torch.clamp(batch_w, min=0.05)
    loss = (loss_raw * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def infer_probabilities(model: RetrievalRerankerMLP, x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        return torch.sigmoid(logits).float().cpu()


def _build_experience_sample_row(
    *,
    record: dict[str, Any],
    packet: dict[str, Any],
    object_type: str,
    case_context: dict[str, Any],
    object_reference_impacts: dict[str, list[str]],
    priors: dict[str, dict[str, float]],
    base_weight: float,
) -> dict[str, Any]:
    case_id = str(record.get("case_id", "")).strip()
    dataset_name = str(record.get("dataset_name", "unknown_dataset")).strip() or "unknown_dataset"
    source_id = str(packet.get("source_id", "")).strip()
    impacts = object_reference_impacts.get(source_id, [])
    impact = _resolve_object_impact(impacts)
    label = _impact_to_soft_label(impact, selected=False, is_skill_candidate=False)
    sample_weight = base_weight + (0.25 if impact in {"helpful", "harmful"} else 0.0)
    evaluation = dict(record.get("evaluation", {}))
    delta = dict(evaluation.get("agent_vs_baseline_delta", {}))
    return {
        "sample_id": f"{case_id}::{object_type}::{source_id}",
        "case_id": case_id,
        "dataset_name": dataset_name,
        "object_type": object_type,
        "object_id": source_id,
        "label": label,
        "weight": round(sample_weight, 6),
        "case_context": deepcopy(case_context),
        "retrieval_object": {
            "source_layer": str(packet.get("source_layer", "")).strip(),
            "source_subtype": str(packet.get("source_subtype", "")).strip(),
            "experience_type": str(packet.get("experience_type", "")).strip(),
            "base_retrieval_score": float(packet.get("retrieval_score", 0.0) or 0.0),
            "trigger_hits": [],
            "match_reasons": [],
        },
        "prior_stats": {
            "object_helpful_prior": float(priors.get("object_helpful_rate", {}).get(source_id, 0.0)),
            "object_harmful_prior": float(priors.get("object_harmful_rate", {}).get(source_id, 0.0)),
            "skill_helpful_rate": 0.0,
            "skill_failure_rate": 0.0,
        },
        "target_signal": {
            "referenced_by_helpful_skill": "helpful" in impacts or "partially_helpful" in impacts,
            "referenced_by_harmful_skill": "harmful" in impacts,
            "selected_by_planner": False,
            "impact": impact,
            "final_correct": evaluation.get("correct"),
            "delta_vs_baseline": float(delta.get("correct_delta", 0.0) or 0.0),
        },
    }


def _skill_impact_map(record: dict[str, Any]) -> dict[str, str]:
    impact_map: dict[str, str] = {}
    for assessment in record.get("reflection_summary", {}).get("skill_assessments", []):
        if not isinstance(assessment, dict):
            continue
        skill_name = str(assessment.get("skill_name", "")).strip()
        if not skill_name:
            continue
        impact = str(assessment.get("impact", "")).strip().lower() or "neutral"
        impact_map[skill_name] = impact
    return impact_map


def _object_reference_impacts(record: dict[str, Any], skill_impacts: dict[str, str]) -> dict[str, list[str]]:
    output = dict(record.get("skill_outputs", {}))
    object_impacts: dict[str, list[str]] = defaultdict(list)
    for skill_name, skill_output in output.items():
        impact = skill_impacts.get(str(skill_name), "neutral")
        for source_id in _normalize_refs(dict(skill_output).get("referenced_experiences", [])):
            object_impacts[source_id].append(impact)
    return object_impacts


def _resolve_object_impact(impacts: list[str]) -> str:
    normalized = [str(item).strip().lower() for item in impacts if str(item).strip()]
    if any(item == "helpful" for item in normalized):
        return "helpful"
    if any(item == "partially_helpful" for item in normalized):
        return "partially_helpful"
    if any(item == "harmful" for item in normalized):
        return "harmful"
    return "neutral"


def _impact_to_soft_label(impact: str, *, selected: bool, is_skill_candidate: bool) -> float:
    normalized = str(impact).strip().lower()
    if normalized == "helpful":
        return 1.0
    if normalized == "partially_helpful":
        return 0.65
    if normalized == "harmful":
        return 0.0
    if is_skill_candidate and selected:
        return 0.35
    return 0.25 if not is_skill_candidate else 0.2


def _extract_confusion_tags(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    skill_retrieval = dict(record.get("skill_retrieval", {}))
    query_summary = dict(skill_retrieval.get("query_summary", {}))
    value = str(query_summary.get("confusion_pair", "")).strip()
    if value:
        tags.append(value)
    value = str(record.get("reflection_summary", {}).get("case_outcome", {}).get("confusion_pair", "")).strip()
    if value:
        tags.append(value)
    return list(dict.fromkeys(item for item in tags if item))


def _build_case_context_from_state(case_state: CaseState) -> dict[str, Any]:
    uncertainty_level = str(case_state.perception.get("uncertainty", {}).get("level", "unknown")).strip().lower() or "unknown"
    metadata = dict(case_state.clinical_metadata)
    skill_bundle = dict(case_state.skill_retrieval_bundle or {})
    confusion_pair = str(skill_bundle.get("query_summary", {}).get("confusion_pair", "")).strip()
    confusion_tags = [confusion_pair] if confusion_pair else []
    return {
        "initial_ddx": [str(item).strip() for item in case_state.perception.get("ddx_candidates", []) if str(item).strip()],
        "uncertainty_level": uncertainty_level,
        "metadata_summary": {
            field_name: str(metadata.get(field_name, "")).strip()
            for field_name in METADATA_FIELDS
            if str(metadata.get(field_name, "")).strip()
        },
        "confusion_tags": confusion_tags,
        "risk_flags": [str(item).strip() for item in case_state.risk_flags if str(item).strip()],
    }


def _normalize_refs(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _experience_summary_slice(
    *,
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    tertiary: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for packet in primary:
        rows.append((float(packet.get("rerank_score", packet.get("retrieval_score", 0.0))) + 0.2, packet))
    for packet in secondary:
        rows.append((float(packet.get("rerank_score", packet.get("retrieval_score", 0.0))) + 0.1, packet))
    for packet in tertiary:
        rows.append((float(packet.get("rerank_score", packet.get("retrieval_score", 0.0))), packet))
    rows.sort(key=lambda item: (item[0], str(item[1].get("source_id", ""))), reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, packet in rows:
        source_id = str(packet.get("source_id", "")).strip()
        if source_id and source_id in seen:
            continue
        if source_id:
            seen.add(source_id)
        deduped.append(packet)
        if len(deduped) >= max(0, int(top_k)):
            break
    return deduped


def _experience_merge_slice(
    *,
    tactical_results: list[dict[str, Any]],
    abstract_results: list[dict[str, Any]],
    raw_case_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = list(abstract_results) + list(tactical_results) + list(raw_case_results)
    rows.sort(
        key=lambda packet: (
            float(packet.get("rerank_score", packet.get("retrieval_score", 0.0))),
            str(packet.get("source_id", "")),
        ),
        reverse=True,
    )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for packet in rows:
        source_id = str(packet.get("source_id", "")).strip()
        if source_id and source_id in seen:
            continue
        if source_id:
            seen.add(source_id)
        deduped.append(packet)
        if len(deduped) >= max(0, int(top_k)):
            break
    return deduped

