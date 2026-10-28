"""Opt-in native PostgreSQL integration; only deletes sessions created here."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from ai_assistant.config import AppConfig, AgentSettings, ModelConfig, SessionsConfig, WorkspaceConfig
from ai_assistant.storage.sessions import SessionError, SessionStorageError, session_storage
from ai_assistant.runtime import assistant_session, final_text
from langchain_core.messages import AIMessage, ToolMessage
from tests.runtime.test_session import ScriptedModel, call

URL = os.environ.get('SESSION_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')


def config(tmp_path):
    return AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')},
        workspace=WorkspaceConfig(root=str(tmp_path)), sessions=SessionsConfig(database_url=URL))


def fake_models(monkeypatch, responses):
    @asynccontextmanager
    async def fake(*args, **kwargs):
        yield ScriptedModel(responses=responses)
    monkeypatch.setattr('ai_assistant.runtime.model_session', fake)


async def initialize(c):
    async with session_storage(c.sessions, initialize=True):
        pass


async def delete(c, sid):
    async with session_storage(c.sessions) as repo:
        await repo.delete(sid)


def test_lock_reopen_workspace_new_and_delete(tmp_path, monkeypatch):
    fake_models(monkeypatch, [AIMessage(content='answer')])
    async def run():
        c = config(tmp_path)
        await initialize(c)
        ids = []
        try:
            async with assistant_session(c) as first:
                ids.append(first.session_id)
                turn = await first.run_turn('baseline')
                assert final_text(turn.messages) == 'answer'
                old_thread = first._thread
                async with session_storage(c.sessions) as other:
                    with pytest.raises(SessionError, match='already in use'):
                        await other.delete(first.session_id)
                with pytest.raises(SessionError, match='already in use'):
                    async with assistant_session(c, session_id=first.session_id):
                        pass
            # Closing releases resources but retains native history.
            async with assistant_session(c, session_id=ids[0]) as second:
                turn = await second.run_turn('follow up')
                assert any(m.content == 'baseline' for m in turn.messages)
                assert second._thread == old_thread
                await second.new_conversation()
                ids.append(second.session_id)
                assert not any(m.content == 'baseline' for m in (await second.run_turn('new')).messages)
            other = tmp_path / 'other'
            other.mkdir()
            wrong = c.model_copy(update={'workspace': WorkspaceConfig(root=str(other))})
            with pytest.raises(SessionError, match='different workspace'):
                async with assistant_session(wrong, session_id=ids[0]):
                    pass
            async with session_storage(c.sessions) as repo:
                assert set(ids) <= {str(row['id']) for row in await repo.list()}
                await repo.delete(ids[0])
                assert (await repo.saver.aget_tuple({'configurable': {'thread_id': old_thread}})) is None
                ids.pop(0)
        finally:
            for sid in ids:
                await delete(c, sid)
    asyncio.run(run())


def test_failed_turns_keep_successful_head_and_do_not_repeat_tools(tmp_path, monkeypatch):
    async def run():
        c = config(tmp_path)
        await initialize(c)
        for baseline in (False, True):
            responses = ([AIMessage(content='baseline answer')] if baseline else []) + [
                call('list_directory', 'a'), AIMessage(content='FAIL'), AIMessage(content='recovered')]
            # list_directory accepts no explicit args, so it is a real native tool invocation.
            fake_models(monkeypatch, responses)
            sid = None
            try:
                async with assistant_session(c) as s:
                    sid = s.session_id
                    if baseline:
                        await s.run_turn('baseline')
                    head, old_thread = s._complete, s._thread
                    with pytest.raises(RuntimeError, match='private diagnostics'):
                        await s.run_turn('failed request')
                    assert s._complete == head
                    out = await s.run_turn('fresh request')
                    assert all(m.content != 'failed request' for m in out.messages)
                    assert not any(isinstance(m, ToolMessage) for m in out.messages)
                    assert (s._thread == old_thread) == baseline
            finally:
                if sid:
                    await delete(c, sid)
    asyncio.run(run())


def test_checkpoint_complete_but_index_commit_fails(tmp_path, monkeypatch):
    async def run():
        c = config(tmp_path)
        await initialize(c)
        fake_models(monkeypatch, [AIMessage(content='baseline answer'), AIMessage(content='unconfirmed answer')])
        sid = None
        try:
            async with assistant_session(c) as s:
                sid = s.session_id
                await s.run_turn('baseline')
                head = s._complete
                async def fail(config, **kwargs):
                    raise SessionStorageError('injected commit failure')
                monkeypatch.setattr(s.repository, 'complete', fail)
                with pytest.raises(SessionStorageError):
                    await s.run_turn('unconfirmed request')
                assert s._complete == head
            fake_models(monkeypatch, [AIMessage(content='recovered')])
            async with assistant_session(c, session_id=sid) as s:
                out = await s.run_turn('fresh')
                assert any(m.content == 'baseline' for m in out.messages)
                assert not any(m.content == 'unconfirmed request' for m in out.messages)
        finally:
            if sid:
                await delete(c, sid)
    asyncio.run(run())


def test_interrupted_delete_is_retryable(tmp_path, monkeypatch):
    async def run():
        c = config(tmp_path)
        await initialize(c)
        fake_models(monkeypatch, [AIMessage(content='answer')])
        async with assistant_session(c) as s:
            sid = s.session_id
            await s.run_turn('hello')
        async with session_storage(c.sessions) as repo:
            async def fail(thread):
                raise RuntimeError('internal diagnostics')
            monkeypatch.setattr(repo.saver, 'adelete_thread', fail)
            with pytest.raises(SessionStorageError, match='retry'):
                await repo.delete(sid)
        with pytest.raises(SessionError, match='deletion is incomplete'):
            async with assistant_session(c, session_id=sid):
                pass
        await delete(c, sid)
    asyncio.run(run())


@pytest.mark.parametrize('crash_mode', ['crash', 'crash-effect'])
def test_actual_process_crash_and_restart(tmp_path, crash_mode):
    c = config(tmp_path)
    asyncio.run(initialize(c))
    worker = Path(__file__).with_name('session_worker.py')
    def invoke(mode, sid=''):
        return subprocess.run([sys.executable, '-m', 'tests.storage.session_worker', mode, str(tmp_path), sid],
                              cwd=worker.parents[2], capture_output=True, text=True, timeout=30)
    seed = invoke('seed')
    assert seed.returncode == 0, seed.stderr
    sid = seed.stdout.strip()
    try:
        crash = invoke(crash_mode, sid)
        assert crash.returncode == (17 if crash_mode == 'crash' else 18), crash.stderr
        assert (tmp_path / 'effect.txt').read_text() == 'observed effect'
        resumed = invoke('resume', sid)
        assert resumed.returncode == 0, resumed.stderr
        assert resumed.stdout.strip() == 'recovered'
        assert (tmp_path / 'effect.txt').read_text() == 'observed effect'
    finally:
        asyncio.run(delete(c, sid))


def test_lost_lock_connection_cancels_active_turn_and_cleans_up(tmp_path, monkeypatch):
    from langchain_core.tools import tool
    from psycopg import AsyncConnection
    async def run():
        c = config(tmp_path)
        await initialize(c)
        fake_models(monkeypatch, [call('slow', 'slow-id'), AIMessage(content='should not complete')])
        entered, cleaned = asyncio.Event(), asyncio.Event()
        @tool
        async def slow():
            """Wait until cancellation, then clean up."""
            entered.set()
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(.01)
                cleaned.set()
            return 'done'
        monkeypatch.setattr('ai_assistant.runtime.create_builtin_tools', lambda **kwargs: [slow])
        sid = None
        try:
            async with assistant_session(c) as s:
                sid = s.session_id
                task = asyncio.create_task(s.run_turn('wait'))
                await asyncio.wait_for(entered.wait(), 5)
                async with await AsyncConnection.connect(URL, autocommit=True) as conn:
                    await conn.execute('SELECT pg_terminate_backend(%s)', (s.repository.conn.info.backend_pid,))
                with pytest.raises(SessionStorageError, match='connection lost'):
                    await asyncio.wait_for(task, 5)
                assert cleaned.is_set()
                with pytest.raises(SessionStorageError):
                    await s.run_turn('cannot continue with lost lock')
        finally:
            if sid:
                await delete(c, sid)
    asyncio.run(run())


def test_missing_schema_is_not_implicitly_initialized(tmp_path):
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    from psycopg import AsyncConnection, sql
    async def run():
        schema = 'session_test_' + uuid4().hex
        async with await AsyncConnection.connect(URL, autocommit=True) as conn:
            await conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            try:
                url = urlsplit(URL)
                query = dict(parse_qsl(url.query))
                query['options'] = '-csearch_path=' + schema
                isolated = urlunsplit(url._replace(query=urlencode(query)))
                settings = SessionsConfig(database_url=isolated)
                with pytest.raises(SessionStorageError, match='init'):
                    async with session_storage(settings):
                        pass
                row = await (await conn.execute('SELECT to_regclass(%s)', (schema + '.assistant_sessions',))).fetchone()
                assert row[0] is None
            finally:
                await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    asyncio.run(run())


def test_unlock_cleanup_does_not_mask_unknown_caller_exception(tmp_path, monkeypatch):
    async def run():
        c = config(tmp_path)
        await initialize(c)
        sid = None
        try:
            with pytest.raises(RuntimeError, match='caller bug'):
                async with session_storage(c.sessions) as repo:
                    row = await repo.open(tmp_path)
                    sid = str(row['id'])
                    async def failed_unlock():
                        raise SessionStorageError('unlock failed')
                    monkeypatch.setattr(repo, 'release', failed_unlock)
                    raise RuntimeError('caller bug')
        finally:
            if sid:
                await delete(c, sid)
    asyncio.run(run())


def test_title_and_success_pointer_commit_together(tmp_path, monkeypatch):
    async def run():
        c = config(tmp_path)
        await initialize(c)
        async with session_storage(c.sessions) as repo:
            row = await repo.open(tmp_path)
            sid = row['id']
            checkpoint = {'configurable': {'thread_id': row['thread_id'], 'checkpoint_ns': '', 'checkpoint_id': str(uuid4())}}
            original = repo.update_current

            async def fail_pointer(sql, params):
                if 'SET complete=' in sql:
                    raise SessionStorageError('fixture failure after title update')
                return await original(sql, params)

            monkeypatch.setattr(repo, 'update_current', fail_pointer)
            with pytest.raises(SessionStorageError):
                await repo.complete(checkpoint, title='未提交主题')
            stored = await (await repo.conn.execute('SELECT title,complete FROM assistant_sessions WHERE id=%s', (sid,))).fetchone()
            assert stored == {'title': '', 'complete': None}
            monkeypatch.setattr(repo, 'update_current', original)
            await repo.complete(checkpoint, title='已提交主题')
            stored = await (await repo.conn.execute('SELECT title,complete FROM assistant_sessions WHERE id=%s', (sid,))).fetchone()
            assert stored['title'] == '已提交主题' and stored['complete'] == checkpoint
            await repo.delete(sid)
    asyncio.run(run())
