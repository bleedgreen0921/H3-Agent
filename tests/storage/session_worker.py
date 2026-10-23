"""Subprocess fixture for real crash recovery, never calls a provider."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys

from langchain_core.messages import AIMessage
from ai_assistant.config import AppConfig, ModelConfig, SessionsConfig, ToolsConfig, FilesConfig, WorkspaceConfig
from ai_assistant.runtime import assistant_session
from tests.runtime.test_session import ScriptedModel
import ai_assistant.runtime as runtime

mode, workspace, sid = sys.argv[1:]


class CrashModel(ScriptedModel):
    async def _agenerate(self, messages, **kwargs):
        if self.responses[self.i].content == 'CRASH':
            os._exit(17)
        return await super()._agenerate(messages, **kwargs)


@asynccontextmanager
async def fake_model(*args, **kwargs):
    scripted_mode = 'crash' if mode == 'crash-effect' else mode
    responses = {'seed': [AIMessage(content='baseline answer')],
        'crash': [AIMessage(content='', tool_calls=[dict(name='create_file',
            args={'path': 'effect.txt', 'content': 'observed effect'}, id='write-effect')]), AIMessage(content='CRASH')],
        'resume': [AIMessage(content='recovered')]}[scripted_mode]
    yield CrashModel(responses=responses)


async def run():
    runtime.model_session = fake_model
    config = AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')},
        sessions=SessionsConfig(database_url=os.environ['SESSION_TEST_DATABASE_URL']),
        workspace=WorkspaceConfig(root=workspace), tools=ToolsConfig(files=FilesConfig(enabled=True)))
    async with assistant_session(config, session_id=sid or None) as session:
        def event_callback(event):
            if mode == 'crash-effect' and event['event'] == 'on_tool_end' and event['name'] == 'create_file':
                os._exit(18)
        prompt_mode = 'crash' if mode == 'crash-effect' else mode
        out = await session.run_turn({'seed':'baseline request', 'crash':'failed request', 'resume':'fresh request'}[prompt_mode], on_event=event_callback)
        if mode == 'seed':
            print(session.session_id)
        elif mode == 'resume':
            assert any(m.content == 'baseline request' for m in out.messages)
            assert not any(m.content == 'failed request' for m in out.messages)
            print(out.messages[-1].text)


asyncio.run(run())
