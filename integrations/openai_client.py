from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

try:
    from openai import APIConnectionError
    from openai import APITimeoutError
    from openai import BadRequestError
    from openai import InternalServerError
    from openai import OpenAI
    from openai import RateLimitError
except ModuleNotFoundError:  # pragma: no cover - optional dependency for offline/unit-test environments
    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    OpenAI = None  # type: ignore[assignment]

from agent.evidence_package import EvidencePackage
from agent.state import CaseInput


LOGGER = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_RETRIES = 2
PROMPT_STACK_VERSION = "dermagent_prompt_stack_v1"
INITIAL_PERCEPTION_PROMPT_VERSION = "initial_perception_v1"
SKILL_PROMPT_VERSION = "skill_reasoning_v1"
FINAL_DIAGNOSIS_PROMPT_VERSION = "final_diagnosis_v1"
BASELINE_DIAGNOSIS_PROMPT_VERSION = "direct_baseline_v1"
INITIAL_PERCEPTION_MAX_TOKENS = 320
SKILL_MAX_TOKENS = 640
FINAL_DIAGNOSIS_MAX_TOKENS = 680
BASELINE_DIAGNOSIS_MAX_TOKENS = 420

FULL_CLINICAL_PROFILE_ID = "full_clinical"
COMPACT_PROFILE_PRESETS = (
    {"profile_id": "standard", "retrieval_top_k": 4, "max_skill_count": 16, "max_skill_fields": 7, "serialized_max_length": 3600},
    {"profile_id": "tight", "retrieval_top_k": 3, "max_skill_count": 12, "max_skill_fields": 5, "serialized_max_length": 2200},
    {"profile_id": "minimal", "retrieval_top_k": 2, "max_skill_count": 8, "max_skill_fields": 4, "serialized_max_length": 1200},
    {"profile_id": "emergency", "retrieval_top_k": 1, "max_skill_count": 5, "max_skill_fields": 3, "serialized_max_length": 600},
)

GENERIC_SKILL_FIELD_PRIORITY = (
    "primary_lesion_morphology",
    "lesion_type",
    "primary_color",
    "border_clarity",
    "border_irregularity",
    "surface_texture",
    "body_location",
    "risk_level",
    "candidate_pairs",
    "unlikely_candidates",
    "contradictions",
    "missing_information",
    "whether_escalation_needed",
    "critical_supporting_evidence",
    "supporting_evidence",
    "opposing_evidence",
    "conflicting_evidence",
    "alarm_signals",
    "risk_evidence",
    "exclusion_evidence",
    "required_missing_evidence",
    "differentiation_features",
    "reasoning_gaps",
    "missing_links",
    "caution_flags",
    "reasons",
    "why_it_matters",
    "impact_on_differential",
    "counterexample_watchouts",
    "further_observation_suggestions",
    "consistency_score",
    "conflicts",
    "suspicious_points",
    "uncertainty_level",
    "uncertainty_if_missing",
    "exclusion_confidence",
    "onset_type",
    "progression_speed",
    "stability",
    "recurrence",
    "size_range",
    "elevation",
    "count",
    "color_variation",
    "pigmentation_pattern",
    "asymmetry_color",
    "symmetry",
    "localized_vs_generalized",
    "clustering_pattern",
    "color",
    "border",
    "surface",
    "size_count",
    "distribution",
    "associated_context",
    "referenced_confusion_patterns",
    "evidence_strength",
    "recommendation_type",
)

SKILL_FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "lesion_description_structuring_skill": (
        "primary_lesion_morphology",
        "color",
        "border",
        "surface",
        "size_count",
        "distribution",
        "associated_context",
    ),
    "morphology_analysis_skill": ("lesion_type", "elevation", "count", "size_range"),
    "color_pattern_analysis_skill": ("primary_color", "color_variation", "pigmentation_pattern", "asymmetry_color"),
    "border_surface_analysis_skill": ("border_clarity", "border_irregularity", "surface_texture", "scaling_presence"),
    "distribution_analysis_skill": ("body_location", "localized_vs_generalized", "clustering_pattern", "symmetry"),
    "metadata_consistency_skill": ("consistency_score", "conflicts", "suspicious_points"),
    "temporal_evolution_skill": ("onset_type", "progression_speed", "stability", "recurrence"),
    "malignancy_risk_assessment_skill": ("risk_level", "alarm_signals", "risk_evidence"),
    "differential_compare_skill": ("candidate_pairs", "supporting_evidence", "conflicting_evidence"),
    "exclusion_reasoning_skill": ("unlikely_candidates", "exclusion_evidence", "required_missing_evidence", "exclusion_confidence"),
    "information_gap_detection_skill": (
        "missing_information",
        "why_it_matters",
        "impact_on_differential",
        "uncertainty_if_missing",
    ),
    "uncertainty_assessment_skill": ("uncertainty_level", "reasons", "missing_information"),
    "contradiction_check_skill": ("contradictions", "missing_links", "reasoning_gaps"),
    "escalation_recommendation_skill": (
        "whether_escalation_needed",
        "escalation_reason",
        "suggested_next_check_type",
        "caution_flags",
    ),
    "mel_nev_specialist_skill": (
        "differentiation_features",
        "critical_supporting_evidence",
        "supporting_evidence",
        "opposing_evidence",
        "counterexample_watchouts",
        "referenced_confusion_patterns",
        "further_observation_suggestions",
    ),
    "ack_scc_specialist_skill": (
        "differentiation_features",
        "critical_supporting_evidence",
        "supporting_evidence",
        "opposing_evidence",
        "counterexample_watchouts",
        "referenced_confusion_patterns",
        "further_observation_suggestions",
    ),
}


class DermOpenAIClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.getenv("OPENAI_MODEL", "Qwen2.5-VL-7B-Instruct")
        self.timeout = timeout if timeout is not None else _read_float_env("OPENAI_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        configured_retries = (
            max_retries if max_retries is not None else _read_int_env("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )
        self.max_retries = max(0, configured_retries)
        if OpenAI is None:
            raise ModuleNotFoundError(
                "openai package is not installed. Install `openai` to use DermOpenAIClient runtime calls."
            )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def prompt_manifest(self) -> dict[str, Any]:
        return {
            "prompt_stack_version": PROMPT_STACK_VERSION,
            "initial_perception_prompt_version": INITIAL_PERCEPTION_PROMPT_VERSION,
            "skill_prompt_version": SKILL_PROMPT_VERSION,
            "final_diagnosis_prompt_version": FINAL_DIAGNOSIS_PROMPT_VERSION,
            "baseline_diagnosis_prompt_version": BASELINE_DIAGNOSIS_PROMPT_VERSION,
        }

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "configured_model_name": self.model,
            "served_model_name": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "prompt_manifest": self.prompt_manifest(),
        }

    def initial_perception(self, case_input: CaseInput) -> dict[str, Any]:
        user_text = (
            "You are the initial perception stage in DermAgent.\n"
            "Return structured observation only. Do not produce a final diagnosis label.\n"
            "You must mimic the early dermatologist reasoning stage: observation, coarse description, "
            "tentative differential candidates, and uncertainty disclosure.\n"
            "Return JSON only with exactly these keys:\n"
            "- image_summary: short clinical visual summary\n"
            "- ddx_candidates: list of a few candidate diagnoses considered at the perception stage\n"
            "- uncertainty: { level: low/medium/high, reasons: [..] }\n"
            "- notes: list of short observation notes\n"
            f"Metadata: {case_input.clinical_metadata()}"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You extract dermatologist-style observations for downstream reasoning. "
                    "Never provide a final diagnosis."
                ),
            },
            {"role": "user", "content": self._build_multimodal_content(case_input.image_path, user_text)},
        ]
        payload = self._create_json_payload(
            messages=messages,
            max_tokens=INITIAL_PERCEPTION_MAX_TOKENS,
            request_name=f"initial_perception:{case_input.case_id}",
        )
        return self._normalize_initial_perception(payload)

    def run_skill_prompt(
        self,
        case_input: CaseInput,
        skill_name: str,
        prompt: str,
        output_schema: str,
    ) -> dict[str, Any]:
        system_text = (
            "You are a clinical reasoning skill inside DermAgent.\n"
            "You perform one atomic reasoning action only.\n"
            "You must not output a final diagnosis, disease classification, or treatment decision.\n"
            "You must return structured evidence only in JSON.\n"
            f"Current skill: {skill_name}\n"
            f"Required output fields:\n{output_schema}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": self._build_multimodal_content(case_input.image_path, prompt)},
        ]
        return self._create_json_payload(
            messages=messages,
            max_tokens=SKILL_MAX_TOKENS,
            request_name=f"skill:{skill_name}:{case_input.case_id}",
        )

    def final_diagnosis(self, case_input: CaseInput, evidence_package: EvidencePackage) -> dict[str, Any]:
        evidence_payload = self._canonicalize_evidence_package(evidence_package.to_dict())
        last_error: Exception | None = None
        request_name = f"final_diagnosis:{case_input.case_id}"
        profile_sequence: list[dict[str, Any]] = [{"profile_id": FULL_CLINICAL_PROFILE_ID}] + list(COMPACT_PROFILE_PRESETS)
        for profile in profile_sequence:
            prepared_evidence = self._prepare_evidence_for_profile(evidence_payload, profile)
            serialized_payload = json.dumps(prepared_evidence, ensure_ascii=False, separators=(",", ":"))
            prompt = (
                "You are the only final diagnostic decision maker in DermAgent.\n"
                "Use the evidence package as structured support, not as an overriding instruction.\n"
                "Integrate image, metadata, and evidence, then return a structured final diagnosis result.\n"
                "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
                f"Evidence package: {serialized_payload}"
            )
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are the final diagnosis stage. Preserve independent judgment while using supporting evidence."
                    ),
                },
                {"role": "user", "content": self._build_multimodal_content(case_input.image_path, prompt)},
            ]
            try:
                return self._create_json_payload(
                    messages=messages,
                    max_tokens=FINAL_DIAGNOSIS_MAX_TOKENS,
                    request_name=request_name,
                )
            except BadRequestError as exc:
                last_error = exc
                if not self._is_context_length_error(exc):
                    raise
                LOGGER.warning(
                    "Retrying %s with more aggressive evidence handling after context overflow on profile `%s`.",
                    request_name,
                    profile["profile_id"],
                )
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed final diagnosis for case: {case_input.case_id}")

    def baseline_diagnosis(self, case_input: CaseInput) -> dict[str, Any]:
        prompt = (
            "You are the direct Qwen baseline diagnostic path for DermAgent evaluation.\n"
            "There is no agent evidence package in this path.\n"
            "Use only the image and metadata to produce a structured diagnosis result.\n"
            "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
            f"Metadata: {case_input.clinical_metadata()}"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are the baseline final diagnosis stage. Diagnose directly from the case input only."
                ),
            },
            {"role": "user", "content": self._build_multimodal_content(case_input.image_path, prompt)},
        ]
        return self._create_json_payload(
            messages=messages,
            max_tokens=BASELINE_DIAGNOSIS_MAX_TOKENS,
            request_name=f"baseline_diagnosis:{case_input.case_id}",
        )

    def _create_json_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        request_name: str,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
            except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                sleep_seconds = min(2**attempt, 8)
                LOGGER.warning(
                    "Retrying %s after %s on attempt %s/%s; sleeping %.1fs.",
                    request_name,
                    exc.__class__.__name__,
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to complete request: {request_name}")

    def _create_json_payload(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        request_name: str,
    ) -> dict[str, Any]:
        token_budget = max_tokens
        max_parse_attempts = 3 if request_name.startswith("skill:") else 2
        for parse_attempt in range(max_parse_attempts):
            response = self._create_json_completion(
                messages=messages,
                max_tokens=token_budget,
                request_name=request_name,
            )
            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            try:
                payload = self._parse_json_response(content)
            except json.JSONDecodeError:
                if parse_attempt >= max_parse_attempts - 1:
                    raise
                token_budget += max(160, max_tokens // 2, token_budget // 3)
                LOGGER.warning(
                    "Retrying %s after malformed JSON response; increasing max_tokens to %s.",
                    request_name,
                    token_budget,
                )
                continue

            if finish_reason == "length" and parse_attempt < max_parse_attempts - 1:
                token_budget += max(160, max_tokens // 2, token_budget // 3)
                LOGGER.warning(
                    "Retrying %s because the model hit the max token budget; increasing max_tokens to %s.",
                    request_name,
                    token_budget,
                )
                continue
            return payload

        raise RuntimeError(f"Failed to parse JSON payload for request: {request_name}")

    @staticmethod
    def _build_multimodal_content(image_path: str, prompt_text: str) -> list[dict[str, Any]]:
        path = Path(image_path)
        if path.exists():
            mime_type, _ = mimetypes.guess_type(path.name)
            detected_mime_type = mime_type or "image/png"
            encoded_image = base64.b64encode(path.read_bytes()).decode("utf-8")
            image_url = f"data:{detected_mime_type};base64,{encoded_image}"
            return [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        return [{"type": "text", "text": prompt_text}]

    @staticmethod
    def _parse_json_response(content: str | None) -> dict[str, Any]:
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            candidate = content.strip()
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if len(lines) >= 3:
                    candidate = "\n".join(lines[1:-1]).strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    pass
            repaired = DermOpenAIClient._repair_truncated_json(candidate[start:] if start >= 0 else candidate)
            if repaired:
                return json.loads(repaired)
            raise

    @staticmethod
    def _repair_truncated_json(candidate: str) -> str | None:
        if not candidate:
            return None
        start = candidate.find("{")
        if start >= 0:
            candidate = candidate[start:]
        if not candidate.startswith("{"):
            return None

        result_chars: list[str] = []
        stack: list[str] = []
        in_string = False
        escape = False

        for char in candidate:
            result_chars.append(char)
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in {"}", "]"} and stack and stack[-1] == char:
                stack.pop()

        repaired = "".join(result_chars).rstrip()
        if in_string:
            repaired += '"'
        repaired = repaired.rstrip(", \n\t")
        while repaired.endswith(","):
            repaired = repaired[:-1].rstrip()
        while stack:
            closing = stack.pop()
            repaired = repaired.rstrip(", \n\t")
            repaired += closing
        repaired = repaired.replace(",}", "}").replace(",]", "]")
        return repaired

    @staticmethod
    def _normalize_initial_perception(payload: dict[str, Any]) -> dict[str, Any]:
        ddx_candidates = payload.get("ddx_candidates", [])
        if isinstance(ddx_candidates, str):
            ddx_candidates = [ddx_candidates]
        elif not isinstance(ddx_candidates, list):
            ddx_candidates = []
        ddx_candidates = [str(item) for item in ddx_candidates if str(item).strip()]

        uncertainty = payload.get("uncertainty", {})
        if isinstance(uncertainty, str):
            uncertainty = {"level": uncertainty, "reasons": []}
        elif not isinstance(uncertainty, dict):
            uncertainty = {}
        uncertainty = {
            "level": str(uncertainty.get("level", "unknown")).lower(),
            "reasons": [str(item) for item in uncertainty.get("reasons", [])]
            if isinstance(uncertainty.get("reasons", []), list)
            else [str(uncertainty.get("reasons"))] if uncertainty.get("reasons") else [],
        }

        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        elif not isinstance(notes, list):
            notes = []

        return {
            "image_summary": str(payload.get("image_summary", "")),
            "ddx_candidates": ddx_candidates,
            "uncertainty": uncertainty,
            "notes": [str(item) for item in notes if str(item).strip()],
        }

    @staticmethod
    def _compact_evidence_package(
        payload: dict[str, Any],
        *,
        retrieval_top_k: int = 4,
        max_skill_count: int = 16,
        max_skill_fields: int = 6,
        serialized_max_length: int = 3600,
        profile_id: str = "standard",
    ) -> dict[str, Any]:
        compact_raw = DermOpenAIClient._compact_retrieval_slice(payload.get("retrieved_raw_cases_summary", []), top_k=retrieval_top_k)
        compact_tactical = DermOpenAIClient._compact_retrieval_slice(
            payload.get("retrieved_tactical_experiences_summary", []),
            top_k=retrieval_top_k,
        )
        compact_abstract = DermOpenAIClient._compact_retrieval_slice(
            payload.get("retrieved_abstract_experiences_summary", []),
            top_k=retrieval_top_k,
        )
        if not compact_raw and not compact_tactical and not compact_abstract:
            compact_legacy = DermOpenAIClient._compact_retrieval_slice(payload.get("retrieved_experience", []), top_k=retrieval_top_k)
            compact_abstract = compact_legacy

        skill_outputs = payload.get("skill_outputs", {})
        compact_skills = {}
        preferred_skill_order = [
            "lesion_description_structuring_skill",
            "morphology_analysis_skill",
            "color_pattern_analysis_skill",
            "border_surface_analysis_skill",
            "distribution_analysis_skill",
            "metadata_consistency_skill",
            "temporal_evolution_skill",
            "differential_compare_skill",
            "exclusion_reasoning_skill",
            "information_gap_detection_skill",
            "mel_nev_specialist_skill",
            "ack_scc_specialist_skill",
            "malignancy_risk_assessment_skill",
            "uncertainty_assessment_skill",
            "contradiction_check_skill",
            "escalation_recommendation_skill",
        ]
        ordered_skill_names: list[str] = []
        for skill_name in preferred_skill_order:
            if skill_name in skill_outputs and skill_name not in ordered_skill_names:
                ordered_skill_names.append(skill_name)
        for skill_name in skill_outputs:
            if skill_name not in ordered_skill_names:
                ordered_skill_names.append(skill_name)
        for skill_name in ordered_skill_names[:max_skill_count]:
            output = skill_outputs.get(skill_name)
            if isinstance(output, dict):
                compact_skills[skill_name] = DermOpenAIClient._compact_skill_output(
                    skill_name,
                    output,
                    max_fields=max_skill_fields,
                )

        notes = payload.get("notes", [])
        if isinstance(notes, list):
            notes = [str(item).strip()[:160] for item in notes[:4] if str(item).strip()]
        else:
            notes = [str(notes).strip()[:160]] if notes else []

        planner_rationale = payload.get("planner_rationale", {})
        compact_planner_rationale = {
            "selected_skills": planner_rationale.get("selected_skills", [])[: min(max_skill_count, 10)],
            "selection_reasons": {
                skill_name: [str(item)[:180] for item in reasons[:2]]
                for skill_name, reasons in list(planner_rationale.get("selection_reasons", {}).items())[: min(max_skill_count, 6)]
            },
        }

        serialized_evidence_text = str(payload.get("serialized_evidence_text", "")).strip()
        if len(serialized_evidence_text) > serialized_max_length:
            serialized_evidence_text = DermOpenAIClient._compact_serialized_evidence_text(
                serialized_evidence_text,
                max_length=serialized_max_length,
            )

        return {
            "compression_profile": profile_id,
            "initial_perception_summary": payload.get("initial_perception_summary", payload.get("perception", {})),
            "retrieved_raw_cases_summary": compact_raw,
            "retrieved_tactical_experiences_summary": compact_tactical,
            "retrieved_abstract_experiences_summary": compact_abstract,
            "skill_outputs": compact_skills,
            "risk_flags": payload.get("risk_flags", []),
            "uncertainty_summary": payload.get("uncertainty_summary", payload.get("uncertainty", {})),
            "contradiction_summary": payload.get("contradiction_summary", {}),
            "information_gap_summary": payload.get("information_gap_summary", {}),
            "escalation_summary": payload.get("escalation_summary", {}),
            "planner_rationale": compact_planner_rationale,
            "notes": notes,
            "serialized_evidence_text": serialized_evidence_text,
        }

    @staticmethod
    def _compact_retrieval_slice(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        compact_records: list[dict[str, Any]] = []
        for record in records[:top_k]:
            compact_records.append(
                {
                    "source_id": record.get("source_id"),
                    "source_layer": record.get("source_layer"),
                    "experience_type": record.get("experience_type"),
                    "case_id": record.get("case_id"),
                    "perception_summary": str(record.get("perception_summary", ""))[:180],
                    "confusion_pair": str(record.get("confusion_pair", ""))[:80],
                    "learning_points": [str(item).strip()[:120] for item in record.get("learning_points", [])[:2] if str(item).strip()],
                }
            )
        return compact_records

    @staticmethod
    def _compact_skill_output(skill_name: str, output: dict[str, Any], *, max_fields: int = 6) -> dict[str, Any]:
        compact_output: dict[str, Any] = {}
        for field_name in DermOpenAIClient._ordered_skill_field_names(skill_name, output):
            value = output.get(field_name)
            if field_name == "referenced_experiences":
                continue
            if value in (None, "", [], {}, "unknown"):
                continue
            if isinstance(value, list):
                compact_values = [str(item).strip()[:140] for item in value[:2] if str(item).strip()]
                if compact_values:
                    compact_output[field_name] = compact_values
            else:
                compact_output[field_name] = str(value).strip()[:140]
            if len(compact_output) >= max_fields:
                break
        return compact_output

    @staticmethod
    def _ordered_skill_field_names(skill_name: str, output: dict[str, Any]) -> list[str]:
        preferred = list(SKILL_FIELD_PRIORITY.get(skill_name, ())) + list(GENERIC_SKILL_FIELD_PRIORITY)
        ordered: list[str] = []
        for field_name in preferred:
            if field_name in output and field_name not in ordered:
                ordered.append(field_name)
        for field_name in output:
            if field_name not in ordered:
                ordered.append(field_name)
        return ordered

    @staticmethod
    def _canonicalize_evidence_package(payload: dict[str, Any]) -> dict[str, Any]:
        skill_outputs = payload.get("skill_outputs", {})
        canonical_skills: dict[str, dict[str, Any]] = {}
        if isinstance(skill_outputs, dict):
            for skill_name, output in skill_outputs.items():
                if isinstance(output, dict):
                    canonical_skills[str(skill_name)] = DermOpenAIClient._full_skill_output(str(skill_name), output)
        notes = payload.get("notes", [])
        normalized_notes = []
        if isinstance(notes, list):
            normalized_notes = [str(item).strip()[:220] for item in notes[:6] if str(item).strip()]
        elif notes:
            normalized_notes = [str(notes).strip()[:220]]
        raw_cases = payload.get("retrieved_raw_cases_summary", [])
        tactical = payload.get("retrieved_tactical_experiences_summary", [])
        abstract = payload.get("retrieved_abstract_experiences_summary", [])
        risk_flags = payload.get("risk_flags", [])
        return {
            "compression_profile": FULL_CLINICAL_PROFILE_ID,
            "initial_perception_summary": payload.get("initial_perception_summary", payload.get("perception", {})),
            "retrieved_raw_cases_summary": list(raw_cases) if isinstance(raw_cases, list) else [],
            "retrieved_tactical_experiences_summary": list(tactical) if isinstance(tactical, list) else [],
            "retrieved_abstract_experiences_summary": list(abstract) if isinstance(abstract, list) else [],
            "skill_outputs": canonical_skills,
            "risk_flags": list(risk_flags) if isinstance(risk_flags, list) else [],
            "uncertainty_summary": payload.get("uncertainty_summary", payload.get("uncertainty", {})),
            "contradiction_summary": payload.get("contradiction_summary", {}),
            "information_gap_summary": payload.get("information_gap_summary", {}),
            "escalation_summary": payload.get("escalation_summary", {}),
            "planner_rationale": DermOpenAIClient._canonicalize_planner_rationale(payload.get("planner_rationale", {})),
            "notes": normalized_notes,
            "serialized_evidence_text": str(payload.get("serialized_evidence_text", "")).strip(),
        }

    @staticmethod
    def _canonicalize_planner_rationale(planner_rationale: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(planner_rationale, dict):
            return {}
        selection_reasons = planner_rationale.get("selection_reasons", {})
        selected_skills = planner_rationale.get("selected_skills", [])
        compact_reasons: dict[str, list[str]] = {}
        if isinstance(selection_reasons, dict):
            for skill_name, reasons in selection_reasons.items():
                if isinstance(reasons, list):
                    compact_reasons[str(skill_name)] = [str(item).strip()[:220] for item in reasons[:4] if str(item).strip()]
        return {
            "planner_type": str(planner_rationale.get("planner_type", "")).strip(),
            "planner_version": str(planner_rationale.get("planner_version", "")).strip(),
            "selected_skills": [str(item).strip() for item in selected_skills if str(item).strip()]
            if isinstance(selected_skills, list)
            else [],
            "selection_reasons": compact_reasons,
        }

    @staticmethod
    def _prepare_evidence_for_profile(payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(profile.get("profile_id", FULL_CLINICAL_PROFILE_ID))
        if profile_id == FULL_CLINICAL_PROFILE_ID:
            prepared = dict(payload)
            prepared["compression_profile"] = FULL_CLINICAL_PROFILE_ID
            return prepared
        return DermOpenAIClient._compact_evidence_package(
            payload,
            retrieval_top_k=int(profile["retrieval_top_k"]),
            max_skill_count=int(profile["max_skill_count"]),
            max_skill_fields=int(profile["max_skill_fields"]),
            serialized_max_length=int(profile["serialized_max_length"]),
            profile_id=profile_id,
        )

    @staticmethod
    def _full_skill_output(skill_name: str, output: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for field_name in DermOpenAIClient._ordered_skill_field_names(skill_name, output):
            if field_name == "referenced_experiences":
                continue
            value = output.get(field_name)
            if value in (None, "", [], {}, "unknown"):
                continue
            if isinstance(value, list):
                cleaned = [str(item).strip()[:220] for item in value[:6] if str(item).strip()]
                if cleaned:
                    normalized[field_name] = cleaned
            else:
                normalized[field_name] = str(value).strip()[:220]
        return normalized

    @staticmethod
    def _compact_serialized_evidence_text(text: str, *, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        sections = DermOpenAIClient._split_serialized_sections(text)
        if not sections:
            return text[: max(0, max_length - 3)] + "..."

        section_priority = [
            "[Observation Evidence]",
            "[Exclusion And Comparison Evidence]",
            "[Risk Evidence]",
            "[Conflict And Uncertainty]",
            "[Planner Rationale]",
        ]
        prioritized_sections = sorted(
            sections,
            key=lambda item: section_priority.index(item[0]) if item[0] in section_priority else len(section_priority),
        )
        rendered_sections = [{"header": header, "lines": [header], "body_lines": list(body_lines)} for header, body_lines in prioritized_sections]
        merged = "\n\n".join(section["header"] for section in rendered_sections)
        if len(merged) > max_length:
            return merged[: max(0, max_length - 3)] + "..."

        remaining_budget = max_length - len(merged)
        line_index = 0
        while remaining_budget > 0:
            added_any = False
            for section in rendered_sections:
                body_lines = section["body_lines"]
                if line_index >= len(body_lines):
                    continue
                candidate = body_lines[line_index]
                line_cost = len(candidate) + 1
                if line_cost > remaining_budget:
                    clipped = candidate[: max(0, remaining_budget - 4)]
                    if clipped:
                        section["lines"].append(clipped + "...")
                        remaining_budget = 0
                        added_any = True
                    break
                section["lines"].append(candidate)
                remaining_budget -= line_cost
                added_any = True
            if not added_any:
                break
            line_index += 1

        compact_sections = ["\n".join(section["lines"]) for section in rendered_sections]
        merged = "\n\n".join(compact_sections)
        if len(merged) <= max_length:
            return merged
        return merged[: max(0, max_length - 3)] + "..."

    @staticmethod
    def _split_serialized_sections(text: str) -> list[tuple[str, list[str]]]:
        current_header = ""
        current_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.startswith("[") and line.endswith("]"):
                if current_header:
                    sections.append((current_header, current_lines))
                current_header = line
                current_lines = []
                continue
            if line:
                current_lines.append(line)
        if current_header:
            sections.append((current_header, current_lines))
        return sections

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "maximum context length" in message or "input length" in message or "context length" in message


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid float environment value for %s=%r", name, raw_value)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid integer environment value for %s=%r", name, raw_value)
        return default
