"""Only main answer text reaches the independent output callback."""
import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import AgentSettings
from ai_assistant.runtime import AssistantSession
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace


class StreamingModel(FakeListChatModel):
    ended: bool = False

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content=[{'type': 'reasoning', 'reasoning': 'PRIVATE'}]))
        for text in ('你', '好'):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            await asyncio.sleep(0)
        self.ended = True


def test_body_before_model_end_and_completion_after_checkpoint(tmp_path):
    async def run():
        model = StreamingModel(responses=['unused'])
        settings = AgentSettings()
        agent = create_assistant_agent(model, [], checkpointer=InMemorySaver(), settings=settings)
        session = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path)), settings)
        events = []
        def output(event):
            if event['kind'] == 'delta':
                assert not model.ended
            if event['kind'] == 'complete':
                assert session._complete is not None
            events.append(event)
        turn = await session.run_turn('你好', on_output=output)
        assert ''.join(e['text'] for e in events if e['kind'] == 'delta') == '你好'
        assert 'PRIVATE' not in str(events)
        assert events[-1] == {'kind': 'complete', 'text': '你好'}
        assert turn.usage.first_body_seconds is not None
    asyncio.run(run())


def test_cancel_marks_preview_unfinished_and_does_not_commit(tmp_path):
    async def run():
        ready = asyncio.Event()
        class BlockingModel(StreamingModel):
            async def _astream(self, messages, **kwargs):
                yield ChatGenerationChunk(message=AIMessageChunk(content='预览'))
                await asyncio.Event().wait()
        settings = AgentSettings()
        agent = create_assistant_agent(BlockingModel(responses=['unused']), [],
                                       checkpointer=InMemorySaver(), settings=settings)
        session = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path)), settings)
        events = []
        def output(event):
            events.append(event)
            if event['kind'] == 'delta':
                ready.set()
        task = asyncio.create_task(session.run_turn('开始', on_output=output))
        await ready.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert session._complete is None
        assert events[-1]['kind'] == 'cancelled'
        assert not any(e['kind'] == 'complete' for e in events)
    asyncio.run(run())
