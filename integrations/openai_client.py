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
    from openai import InternalServerError
    from openai import OpenAI
    from openai import RateLimitError
except ModuleNotFoundError:  # pragma: no cover - optional dependency for offline/unit-test environments
    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
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
        compact_evidence = self._compact_evidence_package(evidence_package.to_dict())
        prompt = (
            "You are the only final diagnostic decision maker in DermAgent.\n"
            "Use the evidence package as structured support, not as an overriding instruction.\n"
            "Integrate image, metadata, and evidence, then return a structured final diagnosis result.\n"
            "Include: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
            f"Evidence package: {compact_evidence}"
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
        return self._create_json_payload(
            messages=messages,
            max_tokens=FINAL_DIAGNOSIS_MAX_TOKENS,
            request_name=f"final_diagnosis:{case_input.case_id}",
        )

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
        for parse_attempt in range(2):
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
                if parse_attempt >= 1:
                    raise
                token_budget += max(160, max_tokens // 2)
                LOGGER.warning(
                    "Retrying %s after malformed JSON response; increasing max_tokens to %s.",
                    request_name,
                    token_budget,
                )
                continue

            if finish_reason == "length" and parse_attempt < 1:
                token_budget += max(160, max_tokens // 2)
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
    def _compact_evidence_package(payload: dict[str, Any]) -> dict[str, Any]:
        compact_raw = DermOpenAIClient._compact_retrieval_slice(payload.get("retrieved_raw_cases_summary", []), top_k=4)
        compact_tactical = DermOpenAIClient._compact_retrieval_slice(
            payload.get("retrieved_tactical_experiences_summary", []),
            top_k=4,
        )
        compact_abstract = DermOpenAIClient._compact_retrieval_slice(
            payload.get("retrieved_abstract_experiences_summary", []),
            top_k=4,
        )
        if not compact_raw and not compact_tactical and not compact_abstract:
            compact_legacy = DermOpenAIClient._compact_retrieval_slice(payload.get("retrieved_experience", []), top_k=4)
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
        for skill_name in ordered_skill_names:
            output = skill_outputs.get(skill_name)
            if isinstance(output, dict):
                compact_skills[skill_name] = DermOpenAIClient._compact_skill_output(output)

        notes = payload.get("notes", [])
        if isinstance(notes, list):
            notes = notes[:6]
        else:
            notes = [str(notes)] if notes else []

        planner_rationale = payload.get("planner_rationale", {})
        compact_planner_rationale = {
            "selected_skills": planner_rationale.get("selected_skills", [])[:12],
            "selection_reasons": {
                skill_name: [str(item) for item in reasons[:2]]
                for skill_name, reasons in list(planner_rationale.get("selection_reasons", {}).items())[:8]
            },
        }

        serialized_evidence_text = str(payload.get("serialized_evidence_text", "")).strip()
        if len(serialized_evidence_text) > 3600:
            serialized_evidence_text = serialized_evidence_text[:3597] + "..."

        return {
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
                    "perception_summary": record.get("perception_summary"),
                    "confusion_pair": record.get("confusion_pair"),
                    "learning_points": record.get("learning_points", [])[:3],
                }
            )
        return compact_records

    @staticmethod
    def _compact_skill_output(output: dict[str, Any]) -> dict[str, Any]:
        compact_output: dict[str, Any] = {}
        for field_name, value in output.items():
            if field_name == "referenced_experiences":
                continue
            if value in (None, "", [], {}, "unknown"):
                continue
            if isinstance(value, list):
                compact_values = [str(item).strip() for item in value[:3] if str(item).strip()]
                if compact_values:
                    compact_output[field_name] = compact_values
            else:
                compact_output[field_name] = str(value).strip()[:180]
            if len(compact_output) >= 6:
                break
        return compact_output


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
