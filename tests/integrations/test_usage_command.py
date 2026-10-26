"""/usage 在任务运行中读取本绑定，不打开模型或取得 Agent 会话锁。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from ai_assistant.interfaces.feishu import BotWorker
from ai_assistant.config import ConfigError


def test_usage_uses_trusted_binding_without_runtime(monkeypatch):
    async def run():
        config = SimpleNamespace(integrations=SimpleNamespace(feishu=SimpleNamespace(owner_open_id='owner')))
        worker = BotWorker(config, SimpleNamespace(conn='independent-connection'), None, None)
        worker.active = object()
        worker.active_row = {'binding': 'private-owner'}
        worker.binding_session = AsyncMock(return_value='trusted-session')
        worker.enqueue_reply = AsyncMock()
        report = AsyncMock(return_value='服务已报告的累计用量')
        monkeypatch.setattr('ai_assistant.storage.usage.UsageStore.report', report)
        def forbidden(*args, **kwargs):
            raise AssertionError('不能打开 Agent 会话')
        monkeypatch.setattr('ai_assistant.interfaces.feishu.assistant_session', forbidden)
        row = {'binding': 'private-owner', 'body': {'owner_open_id': 'owner', 'group': False, 'text': '/usage'}}
        await worker.show_usage(row)
        report.assert_awaited_once_with('trusted-session')
        assert worker.active is not None
        worker.binding_session.assert_awaited_once_with(row)
        worker.enqueue_reply.assert_awaited_once_with(row, '服务已报告的累计用量')
        row['body']['owner_open_id'] = 'untrusted'
        with pytest.raises(ConfigError):
            await worker.show_usage(row)
        assert report.await_count == 1
    asyncio.run(run())
