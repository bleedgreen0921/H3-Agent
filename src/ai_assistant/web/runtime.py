"""本地 Web 会话服务：复用 AssistantSession，管理后台轮次和事件。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from ai_assistant.config import ConfigError
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import assistant_session
from ai_assistant.runtime.types import final_text
from ai_assistant.storage.attachments import AttachmentService
from ai_assistant.storage.sessions import session_storage
from ai_assistant.storage.web import WebMessageStore
from ai_assistant.web.render import safe_html

logger = logging.getLogger(__name__)


def _task_results_view(value):
    """只向 UI 暴露状态和短摘要，避免每次轮询重复传输完整产物。"""
    if not isinstance(value, dict):
        return None
    items = {}
    for task_id, result in (value.get("items") or {}).items():
        if not isinstance(result, dict):
            continue
        items[task_id] = {
            key: result[key] for key in
            ("id", "status", "summary", "error", "questions", "attempts")
            if key in result
        }
    return {"plan_id": value.get("plan_id"), "items": items}


def failure_message(error):
    """把运行异常转换为可展示的安全摘要。"""
    from ai_assistant.agents.context import ContextBudgetError
    from ai_assistant.runtime import TurnIncompleteError
    from ai_assistant.storage.sessions import SessionError, SessionStorageError
    if getattr(error, "turn_message", None):
        return error.turn_message
    if isinstance(error, TurnIncompleteError):
        return str(error)
    if isinstance(error, ContextBudgetError):
        return "上下文超出预算：请缩小请求、附件或工具结果范围。"
    if isinstance(error, SessionStorageError):
        return "会话存储暂不可用；本轮未执行或结果待核实。"
    if isinstance(error, SessionError) and getattr(error, "safe_message", None):
        return error.safe_message
    if isinstance(error, TimeoutError):
        return "本轮执行超时。"
    return f"本轮执行失败（{type(error).__name__}）；未自动重试。"


@dataclass
class Turn:
    input_id: str
    session_id: str
    queues: list[asyncio.Queue] = field(default_factory=list)
    task: asyncio.Task | None = None
    done: bool = False
    result: dict | None = None
    error: str | None = None


class TurnBroker:
    """每个输入一个后台任务；SSE 只订阅，不重复执行。"""

    def __init__(self, runtime):
        self.runtime = runtime
        self.turns: dict[str, Turn] = {}

    def _publish(self, turn: Turn, event: dict):
        for queue in list(turn.queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def start(self, session_id: str, text: str, attachment_ids=(),
                    planning_action: str | None = None,
                    planning_goal: str | None = None):
        input_id = str(uuid4())
        turn = Turn(input_id=input_id, session_id=session_id)
        self.turns[input_id] = turn
        attachment_ids = list(attachment_ids or ())
        await self.runtime.messages.append(
            session_id, "user", text, input_id=input_id,
            attachment_ids=attachment_ids)
        turn.task = asyncio.create_task(
            self._run(turn, text, attachment_ids, planning_action, planning_goal))
        return input_id

    async def _run(self, turn: Turn, text: str, attachment_ids,
                   planning_action=None, planning_goal=None):
        self._publish(turn, {"event": "turn.started", "input_id": turn.input_id,
                             "session_id": turn.session_id})
        try:
            caller = Caller("web", True, scope="local")
            await self.runtime.record_attachment_inputs(turn.session_id, attachment_ids)
            display_text = text
            if attachment_ids:
                lines = [text, "", "附件："]
                for item in await self.runtime.attachment_metadata(turn.session_id, attachment_ids):
                    lines.append(
                        f"- attachment_id={item['id']}, name={item['original_name']}, "
                        f"type={item['media_type']}, bytes={item['bytes']}")
                display_text = "\n".join(lines)

            async with assistant_session(
                    self.runtime.config, session_id=turn.session_id,
                    caller=caller, resources=self.runtime.resources) as session:
                result = await session.run_turn(
                    display_text, input_id=turn.input_id,
                    user_source=text, attachments=attachment_ids,
                    planning_action=planning_action, planning_goal=planning_goal,
                    on_event=lambda item: self._on_event(turn, item))
                body = final_text(result.messages)
                status = (result.result or {}).get("status", "complete")
                await self.runtime.messages.append(
                    turn.session_id, "assistant", body or "",
                    input_id=turn.input_id, status=status,
                    attachment_ids=attachment_ids)
                turn.result = {
                    "input_id": turn.input_id,
                    "session_id": turn.session_id,
                    "status": status,
                    "text": body,
                    "html": safe_html(body or ""),
                    "sources": self.runtime.serialize_sources(result.sources),
                    "usage": self.runtime.serialize_usage(result.usage),
                }
                await session.after_reply()
            self._publish(turn, {"event": "answer.final", **turn.result})
        except asyncio.CancelledError:
            turn.error = "已取消当前执行；已发生的操作不会自动回滚。"
            await self.runtime.messages.append(
                turn.session_id, "assistant", turn.error, input_id=turn.input_id,
                status="cancelled")
            self._publish(turn, {"event": "turn.cancelled", "input_id": turn.input_id,
                                 "message": turn.error})
        except Exception as error:
            turn.error = failure_message(error)
            await self.runtime.messages.append(
                turn.session_id, "assistant", turn.error, input_id=turn.input_id,
                status="failed")
            self._publish(turn, {"event": "turn.failed", "input_id": turn.input_id,
                                 "message": turn.error})
            logger.warning("Web turn failed: input_id=%s category=%s",
                           turn.input_id, type(error).__name__)
        finally:
            turn.done = True
            self._publish(turn, {"event": "done", "input_id": turn.input_id})

    def _on_event(self, turn: Turn, item):
        event = item.get("event") if isinstance(item, dict) else None
        if event == "on_text":
            self._publish(turn, {"event": "model.delta", "text": item.get("text", ""),
                                 "input_id": turn.input_id})
        elif event == "on_model_start":
            self._publish(turn, {"event": "model.start", "input_id": turn.input_id})
        elif event == "on_tool_start":
            self._publish(turn, {"event": "tool.started", "name": item.get("name"),
                                 "label": item.get("label"), "input_id": turn.input_id})
        elif event == "on_tool_end":
            self._publish(turn, {"event": "tool.finished", "name": item.get("name"),
                                 "input_id": turn.input_id})
        elif event == "on_tool_error":
            self._publish(turn, {"event": "tool.failed", "name": item.get("name"),
                                 "input_id": turn.input_id})

    def subscribe(self, input_id: str):
        turn = self.turns.get(input_id)
        if turn is None:
            raise KeyError(input_id)
        queue = asyncio.Queue(maxsize=1000)
        turn.queues.append(queue)
        return turn, queue

    def unsubscribe(self, turn: Turn, queue: asyncio.Queue):
        if queue in turn.queues:
            turn.queues.remove(queue)

    async def cancel(self, input_id: str):
        turn = self.turns.get(input_id)
        if turn is None or turn.done or turn.task is None:
            return False
        turn.task.cancel()
        try:
            await turn.task
        except asyncio.CancelledError:
            pass
        return True


class WebRuntime:
    """Web 进程内共享的配置、Skills 和消息服务。"""

    def __init__(self, config):
        self.config = config
        self.messages = WebMessageStore(config.sessions)
        self.attachments = AttachmentService(config)
        self.skills = None
        self.resources = None
        self.broker = TurnBroker(self)
        self._closed = False

    async def start(self):
        if self.config.sessions is None:
            raise ConfigError("Web 入口需要 [sessions]。")
        # 启动时只检查已显式初始化的结构，不自动迁移。
        from ai_assistant.storage.attachments import attachment_store
        from ai_assistant.storage.web import web_store
        async with session_storage(self.config.sessions):
            pass
        async with web_store(self.config.sessions):
            pass
        if self.config.attachments.enabled:
            async with attachment_store(self.config.sessions):
                pass
        if self.config.skills is not None:
            from ai_assistant.skills.catalog import SkillCatalog, configured_roots
            self.skills = SkillCatalog(
                configured_roots(self.config), disabled=self.config.skills.disabled,
                config_files=self.config.config_files)
        self.resources = SimpleNamespace(skills=self.skills, capability_cache={})

    async def close(self):
        if self._closed:
            return
        self._closed = True
        for turn in list(self.broker.turns.values()):
            if turn.task and not turn.task.done():
                turn.task.cancel()
                try:
                    await turn.task
                except asyncio.CancelledError:
                    pass
        if self.skills is not None:
            self.skills.close()

    def workspace_root(self) -> Path:
        base = self.config.config_path.parent if self.config.config_path else Path.cwd()
        root = Path(self.config.workspace.root)
        return (root if root.is_absolute() else base / root).resolve()

    async def list_sessions(self, limit=200):
        async with session_storage(self.config.sessions) as repo:
            rows = await (await repo.execute(
                '''SELECT s.id,s.title,s.updated_at,s.status FROM assistant_sessions s
                WHERE s.binding IS NULL OR s.binding LIKE %s
                ORDER BY s.updated_at DESC,s.id LIMIT %s''', ("web:%", limit))).fetchall()
        return [dict(row) for row in rows]

    async def create_session(self):
        async with session_storage(self.config.sessions) as repo:
            row = await repo.open(self.workspace_root(), binding=None)
        return str(row["id"])

    async def delete_session(self, session_id):
        # 先记录附件目录，再删除会话；数据库级联删除元数据。
        attachments = await self.attachments.list(session_id)
        async with session_storage(self.config.sessions) as repo:
            await repo.delete(session_id)
        import shutil
        from pathlib import Path
        for record in attachments:
            shutil.rmtree(Path(record.storage_path).parent, ignore_errors=True)
        return True

    async def session_messages(self, session_id):
        return await self.messages.list(session_id)

    async def plan_state(self, session_id):
        """返回规划状态和任务结果；两者都来自 checkpoint。"""
        empty = {"planning": None, "task_results": None}
        async with session_storage(self.config.sessions) as repo:
            row = await (await repo.execute(
                "SELECT thread_id FROM assistant_sessions WHERE id=%s",
                (str(session_id),))).fetchone()
            if not row:
                return empty
            try:
                snapshot = await repo.saver.aget_tuple(
                    {"configurable": {"thread_id": row["thread_id"]}})
            except Exception:
                return empty
            values = (snapshot.checkpoint.get("channel_values", {})
                      if snapshot is not None else {})
            return {"planning": values.get("planning"),
                    "task_results": _task_results_view(values.get("task_results"))}

    async def attachment_metadata(self, session_id, ids):
        result = []
        for ident in ids:
            record = await self.attachments.get(ident, session_id)
            result.append({
                "id": record.id, "original_name": record.original_name,
                "media_type": record.media_type, "bytes": record.bytes,
            })
        return result

    async def record_attachment_inputs(self, session_id, ids):
        # 附件记录在创建时已经关联会话；这里只做访问校验。
        for ident in ids:
            await self.attachments.get(ident, session_id)

    @staticmethod
    def serialize_sources(sources):
        return [
            {"metadata": dict(getattr(doc, "metadata", {}) or {}),
             "preview": (getattr(doc, "page_content", "") or "")[:500]}
            for doc in (sources or [])
        ]

    @staticmethod
    def serialize_usage(usage):
        from dataclasses import asdict
        try:
            return asdict(usage)
        except Exception:
            return {}
