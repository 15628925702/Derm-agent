from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent.case_qa import ask_qa_session, create_qa_session, load_qa_session
from agent.case_qa import create_direct_qa_session
from project_paths import outputs_root


class CreateSessionRequest(BaseModel):
    execution_record_path: str | None = None
    image_path: str | None = None
    case_id: str | None = None
    session_label: str | None = None
    dataset_name: str = "direct_hulumed"
    clinical_metadata: dict[str, Any] | None = None
    workflow_context: dict[str, Any] | None = None
    context_note: str = ""
    model: str = "Hulu-Med-4B"
    audience_mode: str = "patient"
    include_image: bool = False
    base_url: str = "http://127.0.0.1:8013/v1"
    api_key: str = "EMPTY"


class AskSessionRequest(BaseModel):
    question: str
    audience_mode: str | None = None
    include_image: bool | None = None
    max_tokens: int = 512


DEMO_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DermAgent QA Demo</title>
  <style>
    :root {
      --bg: #f5f2ee;
      --panel: #fbfaf8;
      --line: #e5ddd2;
      --user: #c8d9ee;
      --assistant: #f6efcf;
      --text: #1f2a37;
      --muted: #6b7280;
      --chip: #eef2f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f7f4f1 0%, #f2ede8 100%);
      color: var(--text);
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }
    .panel {
      background: rgba(255,255,255,0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(31, 42, 55, 0.06);
    }
    .sidebar {
      padding: 18px;
      position: sticky;
      top: 20px;
      align-self: start;
    }
    .main {
      padding: 18px;
      min-height: 85vh;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 16px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 22px;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 13px;
      font-weight: 600;
    }
    select, input[type="text"], input[type="number"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
    }
    textarea {
      min-height: 86px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .checks {
      display: flex;
      gap: 14px;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .checks label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin: 0;
      font-weight: 500;
    }
    button {
      border: 0;
      border-radius: 12px;
      background: #2f5d8a;
      color: white;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }
    button.secondary {
      background: #e8eef5;
      color: #24415f;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .controls {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .case-card {
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 16px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.75);
    }
    .case-card img {
      width: 220px;
      height: 220px;
      object-fit: cover;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #eaeaea;
    }
    .case-meta h2 {
      margin: 0 0 8px;
      font-size: 18px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .chip {
      padding: 6px 10px;
      background: var(--chip);
      border-radius: 999px;
      font-size: 12px;
      color: #334155;
      border: 1px solid #dbe4ee;
    }
    .chat {
      padding: 10px 4px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow: auto;
      min-height: 360px;
    }
    .msg {
      display: flex;
      gap: 10px;
      align-items: flex-start;
    }
    .msg.user { justify-content: flex-start; }
    .msg.assistant { justify-content: flex-start; }
    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      background: #d9e7f5;
      border: 1px solid #c7d8ea;
      flex: 0 0 auto;
    }
    .bubble {
      max-width: 78%;
      padding: 12px 14px;
      border-radius: 16px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
    }
    .user .bubble {
      background: var(--user);
    }
    .assistant .bubble {
      background: var(--assistant);
    }
    .meta {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .composer {
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .composer .row {
      grid-template-columns: 1fr 140px 120px;
      align-items: end;
    }
    .status {
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 24px;
      text-align: center;
      background: rgba(255,255,255,0.45);
    }
    @media (max-width: 960px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .case-card { grid-template-columns: 1fr; }
      .case-card img { width: 100%; height: auto; max-height: 320px; }
      .composer .row { grid-template-columns: 1fr; }
      .bubble { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="layout">
      <aside class="panel sidebar">
        <h1>DermAgent QA</h1>
        <div class="sub">基于现有 execution record 创建会话，然后围绕同一个 case 做逐步问答。</div>

        <label for="recordSelect">最近记录</label>
        <select id="recordSelect"></select>

        <label for="recordPath">或手动输入记录路径</label>
        <input id="recordPath" type="text" placeholder="G:\0-newResearch\temp\Derm-agent\outputs\...\compare_agent_vs_qwen_*.json" />

        <div class="row">
          <div>
            <label for="audienceMode">回答模式</label>
            <select id="audienceMode">
              <option value="patient">patient</option>
              <option value="doctor">doctor</option>
            </select>
          </div>
          <div>
            <label for="maxTokens">max tokens</label>
            <input id="maxTokens" type="number" value="256" min="64" max="2048" step="32" />
          </div>
        </div>

        <div class="checks">
          <label><input id="includeImage" type="checkbox" checked /> 带图问答</label>
        </div>

        <div class="controls">
          <button id="createBtn">创建会话</button>
          <button id="reloadBtn" class="secondary">刷新记录</button>
        </div>

        <label for="sessionPath">当前 session</label>
        <input id="sessionPath" type="text" readonly />
      </aside>

      <main class="panel main">
        <div id="status" class="status">等待创建会话。</div>

        <section id="caseCard" class="case-card" style="display:none;">
          <img id="caseImage" alt="case image" />
          <div class="case-meta">
            <h2 id="caseTitle">Case</h2>
            <div id="caseMeta" class="meta-grid"></div>
            <div id="suggested" class="chips"></div>
          </div>
        </section>

        <section id="chat" class="chat">
          <div class="empty">先创建一个 QA session，再开始追问。</div>
        </section>

        <section class="composer">
          <div class="row">
            <div>
              <label for="questionInput">问题</label>
              <textarea id="questionInput" placeholder="例如：What is wrong with my skin?"></textarea>
            </div>
            <div>
              <label for="askAudienceMode">本轮模式</label>
              <select id="askAudienceMode">
                <option value="">使用会话默认</option>
                <option value="patient">patient</option>
                <option value="doctor">doctor</option>
              </select>
            </div>
            <div style="display:flex;align-items:end;">
              <button id="askBtn" style="width:100%;">发送问题</button>
            </div>
          </div>
        </section>
      </main>
    </div>
  </div>

  <script>
    const state = {
      auth: 'Bearer EMPTY',
      sessionId: '',
      session: null,
      imageUrl: '',
    };

    const el = {
      recordSelect: document.getElementById('recordSelect'),
      recordPath: document.getElementById('recordPath'),
      audienceMode: document.getElementById('audienceMode'),
      askAudienceMode: document.getElementById('askAudienceMode'),
      includeImage: document.getElementById('includeImage'),
      maxTokens: document.getElementById('maxTokens'),
      createBtn: document.getElementById('createBtn'),
      reloadBtn: document.getElementById('reloadBtn'),
      askBtn: document.getElementById('askBtn'),
      questionInput: document.getElementById('questionInput'),
      status: document.getElementById('status'),
      chat: document.getElementById('chat'),
      sessionPath: document.getElementById('sessionPath'),
      caseCard: document.getElementById('caseCard'),
      caseImage: document.getElementById('caseImage'),
      caseTitle: document.getElementById('caseTitle'),
      caseMeta: document.getElementById('caseMeta'),
      suggested: document.getElementById('suggested'),
    };

    async function api(path, options = {}) {
      const headers = new Headers(options.headers || {});
      headers.set('Authorization', state.auth);
      if (options.body && !headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
      }
      const res = await fetch(path, { ...options, headers });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return res.json();
      }
      return res.blob();
    }

    function setStatus(text) {
      el.status.textContent = text;
    }

    function clearChat() {
      el.chat.innerHTML = '';
    }

    function addMessage(role, text, meta = '') {
      const row = document.createElement('div');
      row.className = `msg ${role}`;
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = role === 'user' ? '🧑' : '🧠';
      const bubbleWrap = document.createElement('div');
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      bubbleWrap.appendChild(bubble);
      if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta';
        metaDiv.textContent = meta;
        bubbleWrap.appendChild(metaDiv);
      }
      row.appendChild(avatar);
      row.appendChild(bubbleWrap);
      el.chat.appendChild(row);
      el.chat.scrollTop = el.chat.scrollHeight;
    }

    function renderSuggested(questions) {
      el.suggested.innerHTML = '';
      for (const question of questions || []) {
        const btn = document.createElement('button');
        btn.className = 'chip';
        btn.textContent = question;
        btn.onclick = () => {
          el.questionInput.value = question;
          sendQuestion();
        };
        el.suggested.appendChild(btn);
      }
    }

    function renderCaseCard(session) {
      const ctx = session.context_snapshot || {};
      el.caseCard.style.display = 'grid';
      el.caseTitle.textContent = `${ctx.case_id || 'case'} · ${ctx.dataset_name || ''}`;
      const meta = [
        ['baseline', ctx.baseline_diagnosis?.final_diagnosis || ''],
        ['agent_final', ctx.agent_final_diagnosis?.final_diagnosis || ''],
        ['skills', (ctx.selected_skills || []).join(', ') || 'none'],
        ['label_space', ctx.clinical_metadata?.label_space_id || ''],
      ];
      el.caseMeta.innerHTML = '';
      for (const [k, v] of meta) {
        const item = document.createElement('div');
        item.innerHTML = `<strong>${k}</strong><br>${String(v || '').replace(/</g, '&lt;')}`;
        el.caseMeta.appendChild(item);
      }
      renderSuggested(session.suggested_questions || []);
    }

    async function loadCaseImage() {
      if (!state.sessionId) return;
      try {
        const blob = await api(`/v1/dermagent/qa/session/${state.sessionId}/image`);
        if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
        state.imageUrl = URL.createObjectURL(blob);
        el.caseImage.src = state.imageUrl;
      } catch {
        el.caseImage.removeAttribute('src');
      }
    }

    async function loadRecentRecords() {
      setStatus('加载最近记录...');
      const payload = await api('/v1/dermagent/qa/records/recent');
      el.recordSelect.innerHTML = '';
      for (const item of payload.records || []) {
        const opt = document.createElement('option');
        opt.value = item.path;
        opt.textContent = `${item.case_id || 'multi-case'} | ${item.dataset_name || ''} | ${item.path}`;
        el.recordSelect.appendChild(opt);
      }
      if (payload.records?.length) {
        el.recordPath.value = payload.records[0].path;
      }
      setStatus(`已加载 ${payload.records?.length || 0} 条记录。`);
    }

    async function createSession() {
      clearChat();
      setStatus('创建 QA session...');
      const executionRecordPath = (el.recordPath.value || el.recordSelect.value || '').trim();
      const payload = {
        execution_record_path: executionRecordPath,
        model: 'Hulu-Med-4B',
        audience_mode: el.audienceMode.value,
        include_image: el.includeImage.checked,
        base_url: 'http://127.0.0.1:8013/v1',
        api_key: 'EMPTY',
      };
      const resp = await api('/v1/dermagent/qa/session/create', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      state.sessionId = resp.session_id;
      state.session = resp.session;
      el.sessionPath.value = resp.session_path;
      renderCaseCard(resp.session);
      await loadCaseImage();
      clearChat();
      addMessage('assistant', 'QA session is ready. Ask your first question.', `session_id=${resp.session_id}`);
      setStatus('会话已创建。');
    }

    async function sendQuestion() {
      const question = el.questionInput.value.trim();
      if (!state.sessionId || !question) return;
      addMessage('user', question);
      el.questionInput.value = '';
      el.askBtn.disabled = true;
      setStatus('正在生成回答...');
      try {
        const payload = {
          question,
          audience_mode: el.askAudienceMode.value || null,
          include_image: el.includeImage.checked,
          max_tokens: Number(el.maxTokens.value || 256),
        };
        const resp = await api(`/v1/dermagent/qa/session/${state.sessionId}/ask`, {
          method: 'POST',
          body: JSON.stringify(payload),
        });
        const turn = resp.turn || {};
        const meta = [
          turn.confidence ? `confidence=${turn.confidence}` : '',
          Array.isArray(turn.evidence_refs) && turn.evidence_refs.length ? `refs=${turn.evidence_refs.join(', ')}` : '',
        ].filter(Boolean).join(' | ');
        addMessage('assistant', turn.answer || '', meta);
        renderSuggested(resp.suggested_questions || []);
        setStatus('回答完成。');
      } catch (err) {
        addMessage('assistant', `Request failed: ${err.message || err}`, 'error');
        setStatus('本轮回答失败。');
      } finally {
        el.askBtn.disabled = false;
      }
    }

    el.reloadBtn.onclick = () => loadRecentRecords().catch(err => setStatus(`加载失败: ${err.message || err}`));
    el.createBtn.onclick = () => createSession().catch(err => setStatus(`创建失败: ${err.message || err}`));
    el.askBtn.onclick = () => sendQuestion();

    loadRecentRecords().catch(err => setStatus(`初始化失败: ${err.message || err}`));
  </script>
</body>
</html>
"""


def _resolve_runtime_path(path_text: str) -> Path:
    candidate = Path(str(path_text or "").strip())
    if candidate.exists():
        return candidate
    text = str(path_text or "").strip()
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5:6]
        rest = text[6:].lstrip("/").replace("/", "\\")
        host_path = Path(f"{drive.upper()}:\\{rest}")
        if host_path.exists():
            return host_path
    return candidate


def _iter_recent_execution_records(limit: int = 30) -> list[dict[str, Any]]:
    root = outputs_root()
    candidates = list(root.rglob("compare_agent_vs_qwen_*.json")) + list(root.rglob("case_execution_record.json"))
    candidates = [path for path in candidates if "qa_sessions" not in str(path)]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    results: list[dict[str, Any]] = []
    for path in candidates[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_id = ""
        dataset_name = ""
        if isinstance(payload, dict):
            if str(payload.get("case_id", "")).strip():
                case_id = str(payload.get("case_id", "")).strip()
                dataset_name = str(payload.get("dataset_name", "")).strip()
            elif isinstance(payload.get("cases"), list) and payload["cases"]:
                first_case = payload["cases"][0]
                if isinstance(first_case, dict):
                    case_id = str(first_case.get("case_id", "")).strip()
                    dataset_name = str(first_case.get("dataset_name", "")).strip()
        results.append(
            {
                "path": str(path.resolve()),
                "case_id": case_id,
                "dataset_name": dataset_name,
                "updated_at": path.stat().st_mtime,
                "name": path.name,
            }
        )
    return results


def create_app(api_key: str) -> FastAPI:
    app = FastAPI(title="DermAgent QA Server")

    def authorize(authorization: str | None) -> None:
        if not api_key:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.post("/v1/dermagent/qa/session/create")
    def create_session(request: CreateSessionRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        if request.execution_record_path:
            session, path = create_qa_session(
                execution_record_path=request.execution_record_path,
                case_id=request.case_id,
                session_label=str(request.session_label or "").strip(),
                base_url=request.base_url,
                api_key=request.api_key,
                model=request.model,
                audience_mode=request.audience_mode,
                include_image=request.include_image,
            )
        elif request.image_path:
            session, path = create_direct_qa_session(
                image_path=request.image_path,
                session_label=str(request.session_label or "").strip(),
                case_id=request.case_id,
                dataset_name=request.dataset_name,
                clinical_metadata=dict(request.clinical_metadata or {}),
                workflow_context=dict(request.workflow_context or {}),
                context_note=str(request.context_note or "").strip(),
                base_url=request.base_url,
                api_key=request.api_key,
                model=request.model,
                audience_mode=request.audience_mode,
                include_image=True if request.include_image is None else bool(request.include_image),
            )
        else:
            raise HTTPException(status_code=400, detail="Provide execution_record_path or image_path.")
        return JSONResponse({
            "session_id": session.session_id,
            "session_path": str(path),
            "session": session.to_dict(),
        })

    @app.get("/v1/dermagent/qa/records/recent")
    def recent_records(authorization: str | None = Header(default=None), limit: int = 30) -> JSONResponse:
        authorize(authorization)
        return JSONResponse({"records": _iter_recent_execution_records(limit=max(1, min(limit, 100)))})

    @app.get("/v1/dermagent/qa/session/{session_id}")
    def get_session(session_id: str, authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        session = load_qa_session(session_id)
        return JSONResponse(session.to_dict())

    @app.get("/v1/dermagent/qa/session/{session_id}/image")
    def get_session_image(session_id: str, authorization: str | None = Header(default=None)) -> FileResponse:
        authorize(authorization)
        session = load_qa_session(session_id)
        image_path = _resolve_runtime_path(session.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Session image not found.")
        media_type, _ = mimetypes.guess_type(image_path.name)
        return FileResponse(str(image_path), media_type=media_type or "image/jpeg")

    @app.post("/v1/dermagent/qa/session/{session_id}/ask")
    def ask_session(session_id: str, request: AskSessionRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        session, turn, path = ask_qa_session(
            session_id_or_path=session_id,
            question=request.question,
            audience_mode=request.audience_mode,
            include_image=request.include_image,
            max_tokens=request.max_tokens,
        )
        return JSONResponse({
            "session_id": session.session_id,
            "session_path": str(path),
            "turn": turn,
            "suggested_questions": session.suggested_questions,
        })

    @app.get("/qa-demo")
    def qa_demo() -> HTMLResponse:
        return HTMLResponse(DEMO_HTML)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve DermAgent QA session endpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--api-key", default="EMPTY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
