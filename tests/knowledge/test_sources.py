"""Offline source extraction and durable sync integration; never calls real accounts."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from ai_assistant.config import AppConfig
from ai_assistant.knowledge.sources import (
    FeishuReader,
    SourceFailure,
    local_inventory,
    save_documents,
)
from ai_assistant.knowledge.sync import configure, settings, sync_once
from ai_assistant.storage.rag import RagStore

URL = os.environ.get("RAG_TEST_DATABASE_URL")
SESSION_URL = os.environ.get("SESSION_TEST_DATABASE_URL")


def config(tmp_path, **kwargs):
    return AppConfig(
        default_model="fake",
        workspace={"root": str(tmp_path)},
        models={
            "fake": {"provider": "ollama", "model": "fake", "max_input_tokens": 24000}
        },
        knowledge={
            "database_url": URL or "postgresql+asyncpg://unused/db",
            "knowledge_base_id": str(uuid4()),
            "materials_dir": tmp_path / ".rag/materials",
            "diagnostics_dir": tmp_path / ".rag/diagnostics",
            **kwargs,
        },
    )


def test_directory_scope_and_immutable_cloud_snapshot(tmp_path):
    (tmp_path / "good.md").write_text("正文")
    (tmp_path / "hidden.secret").write_text("secret")
    (tmp_path / "link.md").symlink_to(tmp_path / "good.md")
    (tmp_path / ".env").write_text("secret")
    c = config(tmp_path)
    entries = local_inventory(
        c, {"path": str(tmp_path), "include": ["*.md"], "exclude": []}
    )
    assert [d["name"] for _, d in entries] == ["good.md"]
    docs = [Document(page_content="正文", metadata={"block_start": 1})]
    digest, path = save_documents(tmp_path / "snapshots", docs)
    assert save_documents(tmp_path / "snapshots", docs) == (digest, path)
    assert json.loads(path.read_text())[0]["page_content"] == "正文"
    assert path.stat().st_mode & 0o777 == 0o400
    path.chmod(0o600)
    path.write_text("corrupt")
    with pytest.raises(Exception, match="校验"):
        save_documents(tmp_path / "snapshots", docs)


def reader(client):
    r = FeishuReader.__new__(FeishuReader)
    r.client = client
    r.tokens = None
    r.api = SimpleNamespace(access=AsyncMock(return_value="fixture"))
    return r


def test_cloud_docx_blocks_and_bad_pagination():
    async def run():
        def handler(req):
            if req.url.params.get("page_token"):
                return httpx.Response(
                    200,
                    json={
                        "data": {"items": [], "has_more": True, "page_token": "same"}
                    },
                )
            return httpx.Response(
                200,
                json={
                    "data": {
                        "items": [
                            {
                                "block_id": "b1",
                                "block_type": 3,
                                "heading1": {
                                    "elements": [{"text_run": {"content": "标题"}}]
                                },
                            }
                        ],
                        "has_more": False,
                    }
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r = reader(client)
            docs = await r.read({"kind": "docx", "token": "fixture"})
            assert docs[0].page_content == "标题"
            assert docs[0].metadata["heading_path"] == ["标题"]
            with pytest.raises(SourceFailure, match="pagination"):
                await r.pages("/fixture", {"page_token": "same"})

    asyncio.run(run())


def test_cloud_tables_preserve_locations():
    async def run():
        def handler(req):
            path = req.url.path
            if path.endswith("/sheets/query"):
                data = {
                    "sheets": [
                        {
                            "sheet_id": "s1",
                            "title": "费用",
                            "grid_properties": {"row_count": 2, "column_count": 2},
                        }
                    ]
                }
            elif "/values/" in path:
                data = {"valueRange": {"values": [["项目", "金额"], ["住宿", 500]]}}
            elif path.endswith("/tables"):
                data = {"items": [{"table_id": "t1", "name": "任务"}]}
            elif path.endswith("/fields"):
                data = {"items": [{"field_name": "状态"}]}
            elif path.endswith("/records/search"):
                data = {"items": [{"record_id": "r1", "fields": {"状态": "完成"}}]}
            else:
                raise AssertionError(path)
            return httpx.Response(200, json={"data": data})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r = reader(client)
            sheets = await r.read({"kind": "sheet", "token": "fixture"})
            assert sheets[-1].metadata["cell_range"] == "A2:B2"
            assert "金额: 500" in sheets[-1].page_content
            records = await r.read({"kind": "bitable", "token": "fixture"})
            assert records[0].metadata["record_id"] == "r1"
            assert "完成" in records[0].page_content

    asyncio.run(run())


@pytest.mark.skipif(not URL, reason="Set a disposable RAG_TEST_DATABASE_URL")
def test_sync_refresh_frozen_builds_and_removed_file(tmp_path):
    from tests.knowledge.test_database import add_chunk

    async def finish(store):
        while job := await store.claim():
            await add_chunk(store, job)
            await store.publish(job)
        await store.finish_syncs()

    async def run():
        path = tmp_path / "a.md"
        path.write_text("第一版")
        c = config(
            tmp_path,
            sources={
                "local": {
                    "kind": "local_directory",
                    "path": str(tmp_path),
                    "include": ["*.md"],
                }
            },
        )
        store = RagStore(settings(c))
        try:
            await store.init()
            await configure(c, store)
            first = await store.request_sync({}, "request1")
            assert await store.request_sync({}, "request1") == first
            assert await store.request_sync({}, "request2") == first
            assert await sync_once(c, store)
            await finish(store)
            one = await store.sync_status(first)
            assert one["state"] == "ready" and len(one["builds"]) == 1
            second = await store.request_sync({})
            await sync_once(c, store)
            await finish(store)
            assert (await store.sync_status(second))["builds"] == one["builds"]
            path.write_text("第二版")
            third = await store.request_sync({})
            await sync_once(c, store)
            await finish(store)
            newer = await store.sync_status(third)
            assert newer["builds"] != one["builds"]
            path.unlink()
            fourth = await store.request_sync({})
            await sync_once(c, store)
            await finish(store)
            assert (await store.sync_status(fourth))["builds"] == []
            assert await store.eligible_builds(one["builds"]) == []
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.skipif(not URL or not SESSION_URL, reason="Set disposable database URLs")
def test_native_pause_reopen_resume_without_duplicate_model_or_effect(
    tmp_path, monkeypatch
):
    from ai_assistant.runtime import assistant_session, WaitingTurn, final_text
    from ai_assistant.storage.sessions import session_storage
    from tests.runtime.test_session import ScriptedModel

    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "内部规则"},
                        "id": "query",
                    }
                ],
            ),
            AIMessage(content="没有有效证据"),
        ]
    )

    @asynccontextmanager
    async def fake(*args, **kwargs):
        yield model

    monkeypatch.setattr("ai_assistant.runtime.model_session", fake)

    async def run():
        c = config(tmp_path).model_copy(
            update={
                "sessions": __import__(
                    "ai_assistant.config", fromlist=["SessionsConfig"]
                ).SessionsConfig(database_url=SESSION_URL)
            }
        )
        store = RagStore(settings(c))
        await store.init()
        async with session_storage(c.sessions, initialize=True):
            pass
        try:
            async with assistant_session(c) as session:
                ident = session.session_id
                waiting = await session.run_turn(
                    "查内部规则", input_id="original-input"
                )
                assert isinstance(waiting, WaitingTurn)
                sync_id = waiting.sync_id
            await sync_once(c, store)
            await store.finish_syncs()
            assert (await store.sync_status(sync_id))["state"] == "ready"
            async with assistant_session(c, session_id=ident) as session:
                assert session._pending_input == "original-input"
                result = await session.run_turn("", resume=True)
                assert final_text(result.messages) == "没有有效证据"
                assert result.input_id == "original-input"
                assert len([m for m in result.messages if m.type == "human"]) == 1
                await session.repository.delete(ident)
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.skipif(not URL, reason="Set a disposable RAG_TEST_DATABASE_URL")
def test_partial_failure_revocation_restore_and_disabled_source(tmp_path):
    from tests.knowledge.test_database import add_chunk

    async def run():
        (tmp_path / "valid.md").write_text("有效内容")
        (tmp_path / "unsupported.bin").write_bytes(b"unsupported")
        c = config(
            tmp_path,
            sources={"local": {"kind": "local_directory", "path": str(tmp_path)}},
        )
        store = RagStore(settings(c))
        try:
            await store.init()
            await configure(c, store)
            ident = await store.request_sync({})
            await sync_once(c, store)
            job = await store.claim()
            assert job
            await add_chunk(store, job)
            await store.publish(job)
            await store.finish_syncs()
            row = await store.sync_status(ident)
            assert row["builds"] and row["errors"]
            member = next(
                m for m in await store.members() if m["uri"].endswith("valid.md")
            )
            await store.observe(
                "local",
                member["uri"],
                member["descriptor"],
                "unavailable",
                "access_denied",
            )
            assert not await store.eligible_builds(row["builds"])
            await store.observe(
                "local", member["uri"], member["descriptor"], "available"
            )
            assert await store.eligible_builds(row["builds"]) == row["builds"]
            await configure(
                c.model_copy(
                    update={"knowledge": c.knowledge.model_copy(update={"sources": {}})}
                ),
                store,
            )
            assert not await store.eligible_builds(row["builds"])
            assert not any(s["enabled"] for s in await store.source_list())
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.skipif(not URL, reason="Set a disposable RAG_TEST_DATABASE_URL")
def test_expired_sync_cannot_submit_or_publish(tmp_path):
    from sqlalchemy import text
    from ai_assistant.knowledge import LeaseLost

    async def run():
        store = RagStore(settings(config(tmp_path)))
        try:
            await store.init()
            ident = await store.request_sync({})
            run = await store.claim_sync()
            async with store.engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE rag_sync_runs SET lease_until=now()-interval '1 second',deadline=now()-interval '1 second' WHERE id=:id"
                    ),
                    {"id": ident},
                )
            with pytest.raises(LeaseLost):
                await store.submit(
                    source_uri="file:///expired",
                    digest="a",
                    snapshot="/unused",
                    metadata={},
                    sync_run=run,
                )
            await store.settle_scan(run, [], [])
            await store.finish_syncs()
            row = await store.sync_status(ident)
            assert row["state"] == "ready" and not row["builds"] and row["errors"]
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.skipif(not URL or not SESSION_URL, reason="Set disposable database URLs")
def test_cancel_waiting_branches_from_successful_history(tmp_path, monkeypatch):
    from ai_assistant.runtime import assistant_session, WaitingTurn, final_text
    from ai_assistant.config import SessionsConfig
    from ai_assistant.storage.sessions import session_storage
    from tests.runtime.test_session import ScriptedModel

    model = ScriptedModel(
        responses=[
            AIMessage(content="baseline"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "内部规则"},
                        "id": "cancel-query",
                    }
                ],
            ),
            AIMessage(content="新问题回答"),
        ]
    )

    @asynccontextmanager
    async def fake(*args, **kwargs):
        yield model

    monkeypatch.setattr("ai_assistant.runtime.model_session", fake)

    async def run():
        c = config(tmp_path).model_copy(
            update={"sessions": SessionsConfig(database_url=SESSION_URL)}
        )
        store = RagStore(settings(c))
        await store.init()
        async with session_storage(c.sessions, initialize=True):
            pass
        try:
            async with assistant_session(c) as session:
                await session.run_turn("基线")
                waiting = await session.run_turn("旧的知识请求")
                assert isinstance(waiting, WaitingTurn)
                await session.cancel_waiting()
                result = await session.run_turn("新的普通问题")
                assert final_text(result.messages) == "新问题回答"
                assert all(m.content != "旧的知识请求" for m in result.messages)
                await session.repository.delete(session.session_id)
        finally:
            await store.close()

    asyncio.run(run())


def test_admission_failure_is_safe_observation_not_turn_failure(tmp_path, monkeypatch):
    from ai_assistant.knowledge.admission import admit_message
    from ai_assistant.knowledge import KnowledgeError

    async def fail(*args, **kwargs):
        raise KnowledgeError("private credential")

    monkeypatch.setattr("ai_assistant.knowledge.admission.register", fail)
    result = asyncio.run(
        admit_message(config(tmp_path), "请分析 https://example.feishu.cn/docx/fixture")
    )
    assert result == [{"status": "failed", "reason": "KnowledgeError"}]
