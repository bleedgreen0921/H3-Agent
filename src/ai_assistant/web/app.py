"""本地单用户 Web 入口：对话、附件、知识库和规划状态。"""
from __future__ import annotations

import asyncio
from html import escape
import json
import logging
from pathlib import Path
import re
import secrets
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ai_assistant.attachments.service import AttachmentError
from ai_assistant.web.runtime import WebRuntime, failure_message

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent


def _token_path(config):
    return config.web.token_path()


def _load_token(config):
    path = _token_path(config)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = secrets.token_urlsafe(32)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return value


def _authorized(request: Request, token: str) -> bool:
    header = request.headers.get("x-assistant-token")
    cookie = request.cookies.get("assistant_token")
    query = request.query_params.get("token")
    return token in {header, cookie, query}


def _safe_html(text: str) -> str:
    """Markdown 渲染前转义原始 HTML，并限制链接协议。"""
    import markdown
    rendered = markdown.markdown(escape(text or ""), extensions=[
        "fenced_code", "tables", "nl2br", "sane_lists"])
    rendered = re.sub(r'<(img|script|style|iframe)[^>]*>', "", rendered,
                      flags=re.IGNORECASE)
    rendered = re.sub(r'href="(?!https?://|/|#|mailto:)[^"]*"',
                      'href="#"', rendered)
    return rendered


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)
    planning_action: str | None = None
    planning_goal: str | None = None


class ReferenceRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=8192)
    identity: str = "user"


def create_app(config) -> FastAPI:
    token = _load_token(config)
    runtime = WebRuntime(config)

    app = FastAPI(title="AI Assistant", docs_url=None, redoc_url=None)
    app.state.runtime = runtime
    app.state.token = token
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=ROOT / "templates")

    @app.middleware("http")
    async def local_auth(request: Request, call_next):
        path = request.url.path
        if path in {"/health", "/login"} or path.startswith("/static"):
            return await call_next(request)
        if _authorized(request, token):
            response = await call_next(request)
            if request.query_params.get("token") == token:
                response.set_cookie("assistant_token", token, httponly=True,
                                    samesite="strict", max_age=60 * 60 * 24 * 30)
            return response
        if path.startswith("/api/"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return templates.TemplateResponse(
            request, "login.html", {"request": request}, status_code=401)

    @app.on_event("startup")
    async def startup():
        await runtime.start()

    @app.on_event("shutdown")
    async def shutdown():
        await runtime.close()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html",
                                          {"request": request, "error": None})

    @app.post("/login")
    async def login(request: Request, token_value: Annotated[str, Form()] = ""):
        if secrets.compare_digest(token_value.strip(), token):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("assistant_token", token, httponly=True,
                                samesite="strict", max_age=60 * 60 * 24 * 30)
            return response
        return templates.TemplateResponse(
            request, "login.html", {"request": request, "error": "令牌不正确。"},
            status_code=401)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {"request": request})

    @app.get("/api/sessions")
    async def list_sessions():
        return await runtime.list_sessions()

    @app.post("/api/sessions")
    async def create_session():
        return {"id": await runtime.create_session()}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            await runtime.delete_session(session_id)
        except Exception as error:
            raise HTTPException(status_code=400, detail=failure_message(error)) from None
        return {"deleted": session_id}

    @app.get("/api/sessions/{session_id}/messages")
    async def messages(session_id: str):
        rows = await runtime.session_messages(session_id)
        return [
            {**row, "html": _safe_html(row["text"]) if row["role"] == "assistant"
             else escape(row["text"]).replace("\n", "<br>")}
            for row in rows
        ]

    @app.post("/api/sessions/{session_id}/turns")
    async def start_turn(session_id: str, body: TurnRequest):
        text = body.text.strip()
        planning_action = body.planning_action
        planning_goal = body.planning_goal
        # 兼容 /plan 文本指令；Web 按钮和文本走同一规划入口。
        if planning_action is None and text.startswith("/plan"):
            head, _, rest = text.partition(" ")
            rest = rest.strip()
            if rest in {"run", "exit", "status"}:
                planning_action = rest
                text = {"run": "执行计划", "exit": "退出规划模式",
                        "status": "查看计划状态"}[rest]
            else:
                planning_action = "create"
                planning_goal = rest or None
                text = rest or "规划任务"
            if not planning_goal and planning_action == "create":
                raise HTTPException(status_code=400, detail="用法：/plan <目标>。")
        try:
            input_id = await runtime.broker.start(
                session_id, text, body.attachment_ids,
                planning_action=planning_action,
                planning_goal=planning_goal)
        except Exception as error:
            raise HTTPException(status_code=400, detail=failure_message(error)) from None
        return {"input_id": input_id, "session_id": session_id}

    @app.get("/api/turns/{input_id}/events")
    async def turn_events(input_id: str):
        try:
            turn, queue = runtime.broker.subscribe(input_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="turn not found") from None

        async def stream():
            try:
                if turn.done:
                    if turn.result:
                        yield _sse("answer.final", turn.result)
                    elif turn.error:
                        yield _sse("turn.failed", {"input_id": input_id,
                                                   "message": turn.error})
                    yield _sse("done", {"input_id": input_id})
                    return
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    event = item.get("event", "message")
                    yield _sse(event, item)
                    if event in {"done", "answer.final", "turn.failed", "turn.cancelled"}:
                        if event != "answer.final":
                            break
            finally:
                runtime.broker.unsubscribe(turn, queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/turns/{input_id}")
    async def turn_result(input_id: str):
        turn = runtime.broker.turns.get(input_id)
        if turn is None:
            raise HTTPException(status_code=404, detail="turn not found")
        return {"input_id": input_id, "done": turn.done,
                "result": turn.result, "error": turn.error}

    @app.post("/api/turns/{input_id}/cancel")
    async def cancel_turn(input_id: str):
        return {"cancelled": await runtime.broker.cancel(input_id)}

    @app.post("/api/attachments")
    async def upload_attachment(
        session_id: Annotated[str, Form()],
        file: Annotated[UploadFile, File()],
    ):
        try:
            data = await file.read()
            record = await runtime.attachments.create(
                session_id, file.filename or "attachment", data)
        except AttachmentError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        except Exception:
            raise HTTPException(status_code=400, detail="附件上传失败。") from None
        return {
            "id": record.id, "name": record.original_name,
            "media_type": record.media_type, "bytes": record.bytes,
        }

    @app.post("/api/attachments/{attachment_id}/ingest")
    async def ingest_attachment(attachment_id: str, session_id: str):
        try:
            return await runtime.attachments.submit_to_knowledge(
                attachment_id, session_id)
        except AttachmentError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    @app.get("/api/plan/{session_id}")
    async def plan(session_id: str):
        return {"planning": await runtime.plan_state(session_id)}

    async def _knowledge_store():
        if config.knowledge is None:
            raise HTTPException(status_code=400, detail="知识库未启用。")
        from ai_assistant.knowledge.config import resolve_knowledge_paths
        from ai_assistant.storage.rag import RagStore
        base = config.config_path.parent if config.config_path else Path.cwd()
        return RagStore(resolve_knowledge_paths(config.knowledge, base))

    @app.get("/api/knowledge/sources")
    async def knowledge_sources():
        store = await _knowledge_store()
        try:
            from ai_assistant.knowledge.management import sources
            return await sources(config, store)
        finally:
            await store.close()

    @app.get("/api/knowledge/documents")
    async def knowledge_documents():
        store = await _knowledge_store()
        try:
            return await store.document_list()
        finally:
            await store.close()

    @app.post("/api/knowledge/sync")
    async def knowledge_sync():
        store = await _knowledge_store()
        try:
            from ai_assistant.knowledge.sync import configure
            await configure(config, store)
            return {"sync_id": await store.request_sync({})}
        finally:
            await store.close()

    @app.post("/api/knowledge/submit")
    async def knowledge_submit(body: ReferenceRequest):
        store = await _knowledge_store()
        try:
            from ai_assistant.knowledge.sync import register
            return await register(config, body.reference, identity=body.identity)
        finally:
            await store.close()

    @app.get("/api/knowledge/status/{job_id}")
    async def knowledge_status(job_id: str):
        store = await _knowledge_store()
        try:
            from ai_assistant.knowledge.management import status
            result = await status(store, job_id)
            if result is None:
                raise HTTPException(status_code=404, detail="任务不存在。")
            return result
        finally:
            await store.close()

    @app.get("/api/settings")
    async def settings():
        return {
            "default_model": config.default_model,
            "models": list(config.models),
            "workspace": str(runtime.workspace_root()),
            "knowledge": config.knowledge is not None,
            "feishu_docs": config.integrations.feishu is not None,
            "attachments": {
                "enabled": config.attachments.enabled,
                "max_bytes": config.attachments.max_bytes,
                "extensions": config.attachments.allowed_extensions,
            },
            "planning": config.planning.enabled,
        }

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
