"""Worker supervision, cancellation and unknown-commit reconciliation."""
import pytest
pytest.importorskip("langchain_postgres")

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from ai_assistant.knowledge import KnowledgeError, LeaseLost
from ai_assistant.knowledge.ingestion import run_once


class Store:
    config = SimpleNamespace(heartbeat_seconds=0.01)
    def __init__(self):
        self.state = 'running'
        self.failures = []
    async def claim(self):
        return dict(id='j', token='t', deadline=datetime.now(timezone.utc) + timedelta(seconds=10))
    async def heartbeat(self, job):
        raise LeaseLost('expired')
    async def status(self, job):
        return dict(state=self.state)
    async def fail(self, job, **kwargs):
        self.failures.append(kwargs)


def test_expired_worker_stops_before_publication():
    async def scenario():
        store = Store()
        stopped = asyncio.Event()
        async def slow(store, job):
            try:
                await asyncio.sleep(5)
                raise AssertionError('obsolete worker published')
            finally:
                stopped.set()
        with patch('ai_assistant.knowledge.ingestion.build', slow):
            assert await run_once(store)
        assert stopped.is_set()
        assert store.failures[0]['error'] == 'expired'
    asyncio.run(scenario())


def test_committed_publication_is_not_retried_after_unknown_result():
    async def scenario():
        store = Store()
        async def committed(store, job):
            store.state = 'succeeded'
            raise ConnectionError('commit response lost')
        with patch('ai_assistant.knowledge.ingestion.build', committed):
            assert await run_once(store)
        assert store.failures == []
    asyncio.run(scenario())


def test_known_permanent_failure_and_external_cancellation():
    async def scenario():
        store = Store()
        async def broken(store, job):
            raise KnowledgeError('missing local resource')
        with patch('ai_assistant.knowledge.ingestion.build', broken):
            assert await run_once(store)
        assert store.failures == [dict(transient=False, error='missing local resource', error_code='ingestion_failed', diagnostic_path=None)]
        started, stopped = asyncio.Event(), asyncio.Event()
        async def waiting(store, job):
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()
        with patch('ai_assistant.knowledge.ingestion.build', waiting):
            task = asyncio.create_task(run_once(Store()))
            await started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert stopped.is_set()
    asyncio.run(scenario())


def test_ingestion_diagnostic_save_failure_keeps_original_error(tmp_path):
    from ai_assistant.knowledge import IngestionError
    from uuid import uuid4
    async def scenario():
        store = Store()
        store.config = SimpleNamespace(heartbeat_seconds=0.01, knowledge_base_id='kb', diagnostics_dir=tmp_path)
        async def broken(store, job):
            raise IngestionError('parser_resource_missing', '缺少离线资源。', 'private traceback')
        with patch('ai_assistant.knowledge.ingestion.build', broken), \
             patch('ai_assistant.knowledge.ingestion.write_diagnostic', side_effect=OSError('disk full')):
            await run_once(store)
        failure = store.failures[0]
        assert failure['error_code'] == 'parser_resource_missing'
        assert failure['error'].startswith('缺少离线资源。') and '未能保存' in failure['error']
        assert failure['diagnostic_path'] is None and not failure['transient']
    asyncio.run(scenario())


def test_ingestion_diagnostic_written_and_only_reference_saved(tmp_path):
    from ai_assistant.knowledge import IngestionError
    from uuid import uuid4
    async def scenario():
        store = Store()
        store.config = SimpleNamespace(heartbeat_seconds=1, knowledge_base_id='kb', diagnostics_dir=tmp_path)
        async def claim():
            return dict(id=str(uuid4()), build_id=str(uuid4()), attempts=1,
                        deadline=datetime.now(timezone.utc) + timedelta(seconds=10))
        store.claim = claim
        async def broken(store, job):
            raise IngestionError('empty_document', '没有正文。', 'private diagnostic')
        with patch('ai_assistant.knowledge.ingestion.build', broken):
            await run_once(store)
        from pathlib import Path
        failure = store.failures[0]
        assert Path(failure['diagnostic_path']).read_text().endswith('private diagnostic')
        assert failure['error_code'] == 'empty_document' and failure['error'] == '没有正文。'
        assert 'private diagnostic' not in str(failure)
    asyncio.run(scenario())
