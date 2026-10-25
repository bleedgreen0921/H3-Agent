import asyncio
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from ai_assistant.config import AppConfig
from ai_assistant.tools import ToolContext, create_builtin_tools, tool_context


def configuration(root, *, enabled=True, terminal=False):
    tools = {'files': {'enabled': enabled}}
    if terminal:
        tools['terminal'] = {'timeout': 5}
    return AppConfig(default_model='local', models={'local': {'provider': 'ollama', 'model': 'unused'}},
                     workspace={'root': str(root)}, tools=tools)


def graph():
    tools = create_builtin_tools(include_write=True, include_terminal=True)
    app = StateGraph(MessagesState, context_schema=ToolContext)
    app.add_node('tools', ToolNode(tools))
    app.add_edge(START, 'tools')
    app.add_edge('tools', END)
    return app.compile()


async def invoke(app, context, name, args):
    result = await app.ainvoke({'messages': [AIMessage(content='', tool_calls=[{
        'name': name, 'args': args, 'id': 'mutation-check', 'type': 'tool_call',
    }])]}, context=context)
    return result['messages'][-1]


class WriteToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_tools_schema_and_context_gates(self):
        tools = create_builtin_tools(include_web=True, include_write=True, include_terminal=True)
        self.assertEqual(len(tools), 8)
        for tool in tools:
            fields = tool.tool_call_schema.model_json_schema()['properties']
            self.assertNotIn('runtime', fields)
            self.assertNotIn('profile', fields)
        with TemporaryDirectory() as directory:
            app = graph()
            async with tool_context(configuration(directory)) as context:
                created = await invoke(app, context, 'create_file', {'path': 'config.toml', 'content': 'number=1\n'})
                self.assertEqual(created.status, 'success')
                read = await invoke(app, context, 'read_text', {'path': 'config.toml'})
                self.assertIn(created.artifact['sha256'], read.content)
                edited = await invoke(app, context, 'edit_file', {
                    'path': 'config.toml', 'expected_sha256': read.artifact['sha256'],
                    'edits': [{'old_text': 'number=1', 'new_text': 'number=2'}],
                })
                self.assertEqual(edited.status, 'success')
                self.assertEqual(edited.tool_call_id, 'mutation-check')
                error = await invoke(app, context, 'edit_file', {
                    'path': 'config.toml', 'expected_sha256': read.artifact['sha256'],
                    'edits': [{'old_text': 'number=2', 'new_text': 'number=3'}],
                })
                self.assertEqual(error.status, 'error')
            async with tool_context(configuration(directory, enabled=False)) as context:
                for name, args in [('create_file', {'path': 'no', 'content': 'x'}),
                                   ('run_command', {'command': 'true'})]:
                    self.assertEqual((await invoke(app, context, name, args)).status, 'error')
            self.assertFalse((Path(directory) / 'no').exists())

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS Seatbelt required')
    async def test_terminal_messages_and_shared_lock(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            app = graph()
            async with tool_context(configuration(root, terminal=True)) as context:
                self.assertIs(context.terminal.lock, context.operation_lock)
                original = context.writes.create_file
                started = asyncio.Event()
                loop = asyncio.get_running_loop()
                def slow_create(*args):
                    loop.call_soon_threadsafe(started.set)
                    time.sleep(.15)
                    return original(*args)
                with patch.object(context.writes, 'create_file', side_effect=slow_create):
                    create = asyncio.create_task(invoke(app, context, 'create_file', {'path': 'created', 'content': 'ok'}))
                    await started.wait()
                    command = asyncio.create_task(invoke(app, context, 'run_command', {'command': 'test -f created'}))
                    self.assertEqual((await create).status, 'success')
                    self.assertEqual((await command).status, 'success')
                error = await invoke(app, context, 'run_command', {'command': 'exit 7'})
                self.assertEqual(error.status, 'error')
                self.assertEqual(error.artifact['exit_code'], 7)
                error = await invoke(app, context, 'run_command', {'command': 'sleep 10', 'timeout': .2})
                self.assertEqual(error.status, 'error')
                self.assertTrue(error.artifact['timed_out'])

    async def test_cancel_write_joins_thread_before_releasing_lock(self):
        with TemporaryDirectory() as directory:
            async with tool_context(configuration(directory)) as context:
                original = context.writes.create_file
                def slow(*args):
                    time.sleep(.15)
                    return original(*args)
                with patch.object(context.writes, 'create_file', side_effect=slow):
                    task = asyncio.create_task(invoke(graph(), context, 'create_file', {'path': 'created', 'content': 'ok'}))
                    await asyncio.sleep(.05)
                    task.cancel()
                    await asyncio.sleep(.02)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertFalse(context.operation_lock.locked())
                self.assertEqual((Path(directory) / 'created').read_text(), 'ok')

    def test_configuration_limits(self):
        for settings in ({'profile': 'host'}, {'timeout': 0}, {'timeout': 1801},
                         {'timeout': float('inf')}, {'network': True}):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                AppConfig(default_model='x', models={'x': {'provider': 'ollama', 'model': 'unused'}},
                          tools={'terminal': settings})
