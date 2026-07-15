from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from integrations.openai_client import DermOpenAIClient
from project_paths import outputs_root

QA_SESSION_VERSION = "qa_session_v1"


def qa_sessions_root() -> Path:
    root = outputs_root() / "qa_sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_path(session_id: str) -> Path:
    return qa_sessions_root() / f"{session_id}.json"


def _clamp_text(value: Any, *, max_chars: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _prune_for_prompt(value: Any, *, depth: int = 0, max_depth: int = 3, max_items: int = 6) -> Any:
    if depth >= max_depth:
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return _clamp_text(value, max_chars=240)
    if isinstance(value, dict):
        return {
            str(k): _prune_for_prompt(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for k, v in list(value.items())[:max_items]
        }
    if isinstance(value, list):
        return [
            _prune_for_prompt(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for v in value[:max_items]
        ]
    if isinstance(value, str):
        return _clamp_text(value, max_chars=400)
    return value


def _summarize_skill_outputs(skill_outputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for skill_name, payload in list(dict(skill_outputs or {}).items())[:8]:
        summary[str(skill_name)] = _prune_for_prompt(payload, max_depth=2, max_items=5)
    return summary


def _summarize_physician_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = [
        "status",
        "summary_version",
        "detail_level",
        "clinical_impression",
        "doctor_summary",
        "doctor_facing_summary",
        "findings_summary",
        "recommended_follow_up",
        "recommended_tests",
        "triage_flags",
    ]
    summary = {k: payload.get(k) for k in keys if k in payload}
    return summary or _prune_for_prompt(payload, max_depth=2, max_items=6)


def _summarize_evidence_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "risk_flags": list(payload.get("risk_flags", []) or [])[:8],
        "uncertainty_summary": _prune_for_prompt(payload.get("uncertainty_summary", {}), max_depth=2, max_items=6),
        "contradiction_summary": _prune_for_prompt(payload.get("contradiction_summary", {}), max_depth=2, max_items=6),
        "information_gap_summary": _prune_for_prompt(payload.get("information_gap_summary", {}), max_depth=2, max_items=6),
        "planner_rationale": _prune_for_prompt(payload.get("planner_rationale", {}), max_depth=2, max_items=6),
        "selected_evidence": _prune_for_prompt(payload.get("selected_evidence", []), max_depth=2, max_items=6),
        "notes": [str(item) for item in list(payload.get("notes", []) or [])[:6]],
    }


def _build_context_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    input_summary = dict(record.get("input_summary", {}) or {})
    return {
        "case_id": str(record.get("case_id", "")).strip(),
        "dataset_name": str(record.get("dataset_name", "")).strip(),
        "image_path": str(input_summary.get("image_path", "")).strip(),
        "clinical_metadata": deepcopy(input_summary.get("clinical_metadata", {}) or {}),
        "workflow_context": deepcopy(input_summary.get("workflow_context", {}) or {}),
        "initial_perception": _prune_for_prompt(record.get("qwen_initial", {}), max_depth=2, max_items=6),
        "baseline_diagnosis": _prune_for_prompt(record.get("baseline_qwen", {}), max_depth=2, max_items=6),
        "agent_final_diagnosis": _prune_for_prompt(record.get("qwen_final", {}), max_depth=2, max_items=6),
        "selected_skills": list(record.get("selected_skills", []) or []),
        "skill_outputs_summary": _summarize_skill_outputs(record.get("skill_outputs", {}) or {}),
        "evidence_summary": _summarize_evidence_bundle(record.get("evidence_bundle", {}) or {}),
        "physician_evidence_summary": _summarize_physician_evidence(record.get("physician_evidence_summary", {}) or {}),
    }


def _default_suggested_questions(audience_mode: str) -> list[str]:
    mode = str(audience_mode or "patient").strip().lower()
    if mode == "doctor":
        return [
            "What are the strongest image findings supporting the current diagnosis?",
            "What are the main differential diagnoses and how are they separated?",
            "Which DermAgent skills contributed the most evidence?",
            "What uncertainties remain and what extra history or tests would reduce them?",
            "What follow-up or triage actions would you recommend?",
        ]
    return [
        "What is wrong with my skin?",
        "Why do you think that?",
        "What causes this kind of problem?",
        "What should I do next?",
        "What other possibilities are you still considering?",
    ]


def _normalize_qa_answer(payload: dict[str, Any], *, audience_mode: str) -> dict[str, Any]:
    candidate_sources: list[str] = []
    if isinstance(payload.get("answer"), str):
        candidate_sources.append(str(payload.get("answer", "")).strip())
    if isinstance(payload.get("raw_text"), str):
        candidate_sources.append(str(payload.get("raw_text", "")).strip())

    for raw_answer in candidate_sources:
        if not raw_answer or not raw_answer.startswith("{"):
            continue
        try:
            nested = json.loads(raw_answer)
            if isinstance(nested, dict):
                payload = {**payload, **nested}
                break
        except Exception:
            marker = '"answer"'
            marker_index = raw_answer.find(marker)
            if marker_index < 0:
                continue
            colon_index = raw_answer.find(":", marker_index + len(marker))
            quote_index = raw_answer.find('"', colon_index + 1)
            if colon_index < 0 or quote_index < 0:
                continue
            chars: list[str] = []
            escaped = False
            for ch in raw_answer[quote_index + 1 :]:
                if escaped:
                    chars.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    break
                chars.append(ch)
            extracted = "".join(chars).strip()
            if extracted:
                payload["answer"] = extracted
                break

    answer = _clamp_text(payload.get("answer") or payload.get("raw_text") or "", max_chars=4000)
    evidence_refs = [str(item).strip() for item in list(payload.get("evidence_refs", []) or []) if str(item).strip()][:8]
    follow_up_questions = [str(item).strip() for item in list(payload.get("follow_up_questions", []) or []) if str(item).strip()][:5]
    confidence = str(payload.get("confidence", "unknown")).strip().lower() or "unknown"
    if confidence not in {"low", "medium", "high", "unknown"}:
        confidence = "unknown"
    return {
        "answer": answer,
        "evidence_refs": evidence_refs,
        "follow_up_questions": follow_up_questions or _default_suggested_questions(audience_mode)[:3],
        "confidence": confidence,
        "audience_mode": str(audience_mode or "patient").strip().lower() or "patient",
    }

def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if isinstance(row, dict):
                records.append(row)
    return records


def load_execution_record(path: str | Path, *, case_id: str | None = None) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Execution record source not found: {target}")
    if target.suffix.lower() == ".jsonl":
        records = _load_jsonl_records(target)
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and str(payload.get("record_version", "")).strip() and str(payload.get("case_id", "")).strip():
            records = [payload]
        elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            records = [item for item in payload.get("cases", []) if isinstance(item, dict)]
        else:
            raise ValueError(f"Unsupported execution record payload shape: {target}")
    if not records:
        raise ValueError(f"No execution records found in {target}")
    if case_id:
        for record in records:
            if str(record.get("case_id", "")).strip() == str(case_id).strip():
                return record
        raise ValueError(f"Case id {case_id} not found in {target}")
    if len(records) == 1:
        return records[0]
    raise ValueError(f"Multiple execution records found in {target}; provide case_id explicitly.")


@dataclass
class QATurn:
    turn_index: int
    question: str
    answer: str
    audience_mode: str
    include_image: bool
    confidence: str = "unknown"
    evidence_refs: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QASession:
    session_id: str
    session_version: str = QA_SESSION_VERSION
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    session_label: str = ""
    source_mode: str = "execution_record"
    source_record_path: str = ""
    source_case_id: str = ""
    base_url: str = "http://127.0.0.1:8013/v1"
    api_key: str = "EMPTY"
    model: str = "Hulu-Med-4B"
    audience_mode_default: str = "patient"
    include_image_default: bool = False
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    suggested_questions: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def image_path(self) -> str:
        return str(self.context_snapshot.get("image_path", "")).strip()


def save_qa_session(session: QASession) -> Path:
    session.updated_at = _utc_now()
    path = _session_path(session.session_id)
    path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_qa_session(session_id_or_path: str | Path) -> QASession:
    raw = Path(str(session_id_or_path))
    path = raw if raw.exists() else _session_path(raw.stem or str(session_id_or_path))
    if not path.exists():
        raise FileNotFoundError(f"QA session not found: {session_id_or_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return QASession(**payload)


def create_qa_session(*, execution_record_path: str | Path, case_id: str | None = None, session_label: str = "", base_url: str, api_key: str, model: str, audience_mode: str = "patient", include_image: bool = False) -> tuple[QASession, Path]:
    record = load_execution_record(execution_record_path, case_id=case_id)
    case_key = str(record.get("case_id", "case")).strip() or "case"
    mode = str(audience_mode or "patient").strip().lower() or "patient"
    session = QASession(
        session_id=f"qa_{case_key}_{uuid.uuid4().hex[:10]}",
        session_label=session_label,
        source_mode="execution_record",
        source_record_path=str(Path(execution_record_path).resolve()),
        source_case_id=case_key,
        base_url=str(base_url).strip(),
        api_key=str(api_key).strip(),
        model=str(model).strip(),
        audience_mode_default=mode,
        include_image_default=bool(include_image),
        context_snapshot=_build_context_snapshot(record),
        suggested_questions=_default_suggested_questions(mode),
    )
    path = save_qa_session(session)
    return session, path


def create_direct_qa_session(
    *,
    image_path: str | Path,
    session_label: str = "",
    case_id: str | None = None,
    dataset_name: str = "direct_hulumed",
    clinical_metadata: dict[str, Any] | None = None,
    workflow_context: dict[str, Any] | None = None,
    context_note: str = "",
    base_url: str,
    api_key: str,
    model: str,
    audience_mode: str = "patient",
    include_image: bool = True,
) -> tuple[QASession, Path]:
    image_path_obj = Path(image_path).resolve()
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path_obj}")
    case_key = str(case_id or image_path_obj.stem or "direct_case").strip() or "direct_case"
    mode = str(audience_mode or "patient").strip().lower() or "patient"
    context_snapshot = {
        "case_id": case_key,
        "dataset_name": str(dataset_name or "direct_hulumed").strip() or "direct_hulumed",
        "image_path": str(image_path_obj),
        "clinical_metadata": deepcopy(clinical_metadata or {}),
        "workflow_context": deepcopy(workflow_context or {}),
        "context_note": str(context_note or "").strip(),
        "initial_perception": {},
        "baseline_diagnosis": {},
        "agent_final_diagnosis": {},
        "selected_skills": [],
        "skill_outputs_summary": {},
        "evidence_summary": {},
        "physician_evidence_summary": {},
    }
    session = QASession(
        session_id=f"qa_direct_{case_key}_{uuid.uuid4().hex[:10]}",
        session_label=session_label,
        source_mode="direct_image",
        source_record_path="",
        source_case_id=case_key,
        base_url=str(base_url).strip(),
        api_key=str(api_key).strip(),
        model=str(model).strip(),
        audience_mode_default=mode,
        include_image_default=bool(include_image),
        context_snapshot=context_snapshot,
        suggested_questions=_default_suggested_questions(mode),
    )
    path = save_qa_session(session)
    return session, path


def ask_qa_session(*, session_id_or_path: str | Path, question: str, audience_mode: str | None = None, include_image: bool | None = None, max_tokens: int = 512) -> tuple[QASession, dict[str, Any], Path]:
    session = load_qa_session(session_id_or_path)
    mode = str(audience_mode or session.audience_mode_default or "patient").strip().lower() or "patient"
    include_image_flag = session.include_image_default if include_image is None else bool(include_image)
    client = DermOpenAIClient(base_url=session.base_url, api_key=session.api_key, model=session.model, timeout=300, max_retries=0)
    payload = client.case_qa_answer(
        question=question,
        context=session.context_snapshot,
        image_path=session.image_path,
        audience_mode=mode,
        history=session.turns,
        include_image=include_image_flag,
        max_tokens=max_tokens,
    )
    normalized = _normalize_qa_answer(payload, audience_mode=mode)
    turn = QATurn(
        turn_index=len(session.turns) + 1,
        question=str(question).strip(),
        answer=normalized["answer"],
        audience_mode=normalized["audience_mode"],
        include_image=include_image_flag,
        confidence=normalized["confidence"],
        evidence_refs=list(normalized["evidence_refs"]),
        follow_up_questions=list(normalized["follow_up_questions"]),
    )
    session.turns.append(turn.to_dict())
    session.suggested_questions = list(normalized["follow_up_questions"])
    path = save_qa_session(session)
    return session, turn.to_dict(), path


def _session_export_text(session: QASession) -> str:
    lines: list[str] = []
    lines.append(f"session_id: {session.session_id}")
    lines.append(f"source_mode: {session.source_mode}")
    lines.append(f"source_case_id: {session.source_case_id}")
    lines.append(f"source_record_path: {session.source_record_path}")
    lines.append(f"model: {session.model}")
    lines.append(f"audience_mode_default: {session.audience_mode_default}")
    lines.append("")
    ctx = dict(session.context_snapshot or {})
    lines.append("[context]")
    lines.append(f"case_id: {ctx.get('case_id', '')}")
    lines.append(f"dataset_name: {ctx.get('dataset_name', '')}")
    lines.append(f"image_path: {ctx.get('image_path', '')}")
    baseline = dict(ctx.get("baseline_diagnosis", {}) or {})
    agent_final = dict(ctx.get("agent_final_diagnosis", {}) or {})
    lines.append(f"baseline_final_diagnosis: {baseline.get('final_diagnosis', '')}")
    lines.append(f"agent_final_diagnosis: {agent_final.get('final_diagnosis', '')}")
    lines.append("")
    lines.append("[turns]")
    for turn in session.turns:
        lines.append(f"turn {turn.get('turn_index', '')}")
        lines.append(f"user: {str(turn.get('question', '')).strip()}")
        lines.append(f"assistant: {str(turn.get('answer', '')).strip()}")
        refs = list(turn.get("evidence_refs", []) or [])
        if refs:
            lines.append(f"evidence_refs: {', '.join(str(item) for item in refs)}")
        lines.append(f"confidence: {str(turn.get('confidence', 'unknown')).strip()}")
        lines.append(f"created_at: {str(turn.get('created_at', '')).strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _session_export_markdown(session: QASession) -> str:
    ctx = dict(session.context_snapshot or {})
    baseline = dict(ctx.get("baseline_diagnosis", {}) or {})
    agent_final = dict(ctx.get("agent_final_diagnosis", {}) or {})
    lines: list[str] = []
    lines.append(f"# DermAgent QA Session `{session.session_id}`")
    lines.append("")
    lines.append(f"- source_case_id: `{session.source_case_id}`")
    lines.append(f"- source_mode: `{session.source_mode}`")
    lines.append(f"- source_record_path: `{session.source_record_path}`")
    lines.append(f"- model: `{session.model}`")
    lines.append(f"- audience_mode_default: `{session.audience_mode_default}`")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- case_id: `{ctx.get('case_id', '')}`")
    lines.append(f"- dataset_name: `{ctx.get('dataset_name', '')}`")
    lines.append(f"- image_path: `{ctx.get('image_path', '')}`")
    lines.append(f"- baseline_final_diagnosis: `{baseline.get('final_diagnosis', '')}`")
    lines.append(f"- agent_final_diagnosis: `{agent_final.get('final_diagnosis', '')}`")
    lines.append("")
    lines.append("## Dialogue")
    lines.append("")
    for turn in session.turns:
        turn_index = turn.get("turn_index", "")
        lines.append(f"### Turn {turn_index}")
        lines.append("")
        lines.append(f"**User**: {str(turn.get('question', '')).strip()}")
        lines.append("")
        lines.append(f"**Assistant**: {str(turn.get('answer', '')).strip()}")
        refs = list(turn.get("evidence_refs", []) or [])
        if refs:
            lines.append("")
            lines.append(f"- evidence_refs: {', '.join(str(item) for item in refs)}")
        lines.append(f"- confidence: `{str(turn.get('confidence', 'unknown')).strip()}`")
        lines.append(f"- created_at: `{str(turn.get('created_at', '')).strip()}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_qa_session(
    session_id_or_path: str | Path,
    output_path: str | Path,
    *,
    export_format: str | None = None,
) -> Path:
    session = load_qa_session(session_id_or_path)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(export_format or target.suffix.lstrip(".") or "md").strip().lower()
    if fmt == "json":
        target.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "jsonl":
        with target.open("w", encoding="utf-8") as handle:
            for turn in session.turns:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
    elif fmt == "txt":
        target.write_text(_session_export_text(session), encoding="utf-8")
    elif fmt in {"md", "markdown"}:
        target.write_text(_session_export_markdown(session), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return target


