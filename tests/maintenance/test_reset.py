"""破坏性维护只针对本测试临时创建的独立数据库和目录。"""
import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
from ai_assistant.config import ConfigError
from ai_assistant.maintenance.reset import data_files, reset
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.storage.sessions import session_storage
from tests.integrations.test_mcp import config

URL = os.environ.get('SESSION_TEST_DATABASE_URL')


def test_generated_paths_exclude_sources_and_links(tmp_path):
    generated = tmp_path / 'generated'
    generated.mkdir()
    source = tmp_path / 'original.txt'
    source.write_text('source')
    snapshot = generated / ('a' * 64 + '.txt')
    snapshot.write_text('copy')
    (generated / ('b' * 64 + '.txt')).symlink_to(source)
    c = config(tmp_path, knowledge={'database_url': 'postgresql+asyncpg://user@localhost/test', 'materials_dir': generated})
    assert data_files(c) == [snapshot]
    with pytest.raises(ConfigError):
        reset(c, execute=True)
    unsafe = c.model_copy(update={'knowledge': c.knowledge.model_copy(update={'materials_dir': tmp_path})})
    with pytest.raises(ConfigError):
        data_files(unsafe)


@pytest.mark.skipif(not URL, reason='Set disposable SESSION_TEST_DATABASE_URL')
def test_preview_execute_repeat_preserves_unrelated_data(tmp_path):
    import psycopg
    from psycopg import sql
    parts = urlsplit(URL)
    name = 'reset_fixture_' + uuid4().hex
    target = urlunsplit(parts._replace(path='/' + name))
    with psycopg.connect(URL, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    try:
        c = config(tmp_path, sessions={'database_url': target}, knowledge={
            'database_url': target.replace('postgresql://', 'postgresql+asyncpg://'),
            'materials_dir': tmp_path / 'generated', 'diagnostics_dir': tmp_path / 'diagnostics'})

        async def setup():
            async with session_storage(c.sessions, initialize=True) as repo:
                row = await repo.open(tmp_path)
                from langchain_core.messages import HumanMessage
                from ai_assistant.agents import create_assistant_agent
                from ai_assistant.config import AgentSettings
                from tests.runtime.test_session import ScriptedModel
                from langchain_core.messages import AIMessage
                agent = create_assistant_agent(ScriptedModel(responses=[AIMessage(content='x')]), [], settings=AgentSettings(), checkpointer=repo.saver)
                await agent.aupdate_state({'configurable': {'thread_id': row['thread_id']}}, {'messages': [HumanMessage('owned')]})
                await agent.aupdate_state({'configurable': {'thread_id': 'unrelated-thread'}}, {'messages': [HumanMessage('preserve')]})
            async with operation_storage(c.sessions, initialize=True):
                pass
            from ai_assistant.storage.rag import RagStore
            store = RagStore(c.knowledge)
            try:
                await store.init()
                await store.submit(source_uri='file:///fixture', digest='test', snapshot='/unused', metadata={})
            finally:
                await store.close()
        asyncio.run(setup())
        generated = Path(c.knowledge.materials_dir)
        generated.mkdir()
        artifact = generated / ('a' * 64 + '.txt')
        artifact.write_text('snapshot')
        original = tmp_path / 'original.txt'
        original.write_text('keep')
        with psycopg.connect(target, autocommit=True) as conn:
            conn.execute('CREATE TABLE unrelated(id integer)')
            conn.execute('INSERT INTO unrelated VALUES(1)')
            conn.execute('CREATE TABLE store(prefix text,key text, PRIMARY KEY(prefix,key))')
            conn.execute("INSERT INTO store VALUES('personal_memory.app.owner','old'),('personalXmemory.app.owner','keep')")
            conn.execute("INSERT INTO upsertion_record(uuid,namespace) VALUES('old','rag:old'),('keep','other:keep')")
        preview = reset(c)
        assert not preview['executed'] and artifact.exists()
        with psycopg.connect(target, autocommit=True) as conn:
            assert conn.execute('SELECT count(*) FROM assistant_sessions').fetchone()[0] == 1
            conn.execute('SELECT pg_advisory_lock(714160004)')
            with pytest.raises(ConfigError):
                reset(c, execute=True, services_stopped=True)
            conn.execute('SELECT pg_advisory_unlock(714160004)')
        done = reset(c, execute=True, services_stopped=True)
        assert done['executed'] and not done['file_failures']
        assert not artifact.exists() and original.read_text() == 'keep'
        reset(c, execute=True, services_stopped=True)
        with psycopg.connect(target, autocommit=True) as conn:
            assert conn.execute('SELECT count(*) FROM assistant_sessions').fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM rag_documents').fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM unrelated').fetchone()[0] == 1
            assert conn.execute('SELECT prefix FROM store').fetchall() == [('personalXmemory.app.owner',)]
            assert conn.execute('SELECT namespace FROM upsertion_record').fetchall() == [('other:keep',)]
            assert conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id='unrelated-thread'").fetchone()[0] == 1
    finally:
        with psycopg.connect(URL, autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
