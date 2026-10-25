"""Personal style loading without real model or database calls."""
import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import AIMessage
from pydantic import Field

from ai_assistant.agents import SYSTEM_PROMPT, load_persona
from ai_assistant.config import AppConfig, ConfigError, load_config
from ai_assistant.runtime import assistant_session
from tests.runtime.test_session import ScriptedModel


def configured(tmp_path):
    file = tmp_path / 'assistant.toml'
    file.write_text('default_model = "fake"\n[models.fake]\nprovider = "ollama"\nmodel = "fake"\n[agent]\npersona_file = "PERSONA.md"\n[workspace]\nroot = "."\n')
    return load_config(file)


def test_persona_path_is_relative_to_config_and_opt_in(tmp_path, monkeypatch):
    config = configured(tmp_path)
    (tmp_path / 'PERSONA.md').write_text('直接自然地交流。', encoding='utf-8')
    monkeypatch.chdir(tmp_path.parent)
    assert load_persona(config) == '直接自然地交流。'
    default = AppConfig(default_model='fake', models={'fake': {'provider':'ollama','model':'fake'}})
    assert load_persona(default) == ''


def test_persona_errors_do_not_silently_fall_back(tmp_path):
    config = configured(tmp_path)
    with pytest.raises(ConfigError, match='无法读取'):
        load_persona(config)
    path = tmp_path / 'PERSONA.md'
    path.write_bytes(b'\xff')
    with pytest.raises(ConfigError, match='UTF-8'):
        load_persona(config)
    path.write_bytes(b'x' * 16385)
    with pytest.raises(ConfigError, match='16 KiB'):
        load_persona(config)


def test_runtime_injects_style_as_system_prompt_on_every_turn(tmp_path, monkeypatch):
    config = configured(tmp_path)
    (tmp_path / 'PERSONA.md').write_text('用自然直接的中文回答。', encoding='utf-8')
    class CapturingModel(ScriptedModel):
        prompts: list = Field(default_factory=list)
        async def _agenerate(self, messages, **kwargs):
            self.prompts.append(list(messages))
            return await super()._agenerate(messages, **kwargs)
    model = CapturingModel(responses=[AIMessage(content='第一轮'), AIMessage(content='第二轮')])
    @asynccontextmanager
    async def fake(*args, **kwargs):
        yield model
    monkeypatch.setattr('ai_assistant.runtime.model_session', fake)
    async def run():
        async with assistant_session(config) as session:
            await session.run_turn('你好')
            await session.run_turn('继续')
    asyncio.run(run())
    assert len(model.prompts) == 2
    for messages in model.prompts:
        assert messages[0].type == 'system'
        assert SYSTEM_PROMPT in messages[0].content
        assert '用自然直接的中文回答。' in messages[0].content
        assert '不改变上述业务规则' in messages[0].content
        assert all('用自然直接的中文回答。' not in str(m.content) for m in messages[1:])
