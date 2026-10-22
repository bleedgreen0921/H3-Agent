from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from ai_assistant.config import AppConfig, ConfigError, load_config
from ai_assistant.tools import ToolContext, create_builtin_tools, tool_context
from ai_assistant.tools.web import DeepSeekWeb
from ai_assistant.workspace import ReadOnlyWorkspace


class BuiltinTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_tools_native_graph(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.txt').write_text('hello')
            tools = create_builtin_tools(include_web=True)
            self.assertEqual(len(tools), 5)
            self.assertEqual(len(create_builtin_tools()), 3)
            for tool in tools:
                self.assertNotIn('runtime', tool.tool_call_schema.model_json_schema()['properties'])
            graph = StateGraph(MessagesState, context_schema=ToolContext)
            graph.add_node('tools', ToolNode(tools))
            graph.add_edge(START, 'tools')
            graph.add_edge('tools', END)
            app = graph.compile()
            def handler(request):
                data = {'results': []} if request.url.path.endswith('search') else {'title': 'Page', 'content': 'body'}
                return httpx.Response(200, json=data)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                context = ToolContext(ReadOnlyWorkspace(root), DeepSeekWeb(client, 'key'))
                args = [{}, {'path': 'a.txt'}, {'query': 'hello'}, {'query': 'test'}, {'url': 'https://example.com'}]
                calls = [{'name': tool.name, 'args': arg, 'id': str(i), 'type': 'tool_call'}
                         for i, (tool, arg) in enumerate(zip(tools, args))]
                result = await app.ainvoke({'messages': [AIMessage(content='', tool_calls=calls)]}, context=context)
                messages = result['messages'][1:]
                self.assertEqual(len(messages), 5)
                for i, message in enumerate(messages):
                    self.assertIsInstance(message, ToolMessage)
                    self.assertEqual(message.tool_call_id, str(i))
                    self.assertEqual(message.status, 'success')
                    self.assertIsInstance(message.artifact, dict)
                for name, arguments in [('read_text', {'path': '../secret'}), ('web_search', {'query': ' '}),
                                        ('read_text', {'path': 'a.txt', 'max_lines': 0})]:
                    result = await app.ainvoke({'messages': [AIMessage(content='', tool_calls=[{
                        'name': name, 'args': arguments, 'id': 'error', 'type': 'tool_call'}])]}, context=context)
                    self.assertEqual(result['messages'][-1].status, 'error')
                with patch.object(context.workspace, 'read_text', side_effect=RuntimeError('bug')):
                    with self.assertRaises(RuntimeError):
                        await app.ainvoke({'messages': [AIMessage(content='', tool_calls=[{
                            'name': 'read_text', 'args': {'path': 'a.txt'}, 'id': 'bug', 'type': 'tool_call'}])]}, context=context)

    async def test_configuration_resolution_and_client_ownership(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.toml'
            path.write_text('default_model="d"\n[models.d]\nprovider="deepseek"\nmodel="deepseek-chat"\napi_key="SECRET"\n[tools.web]\ncredential_profile="d"\n')
            config = load_config(path)
            async with tool_context(config) as context:
                self.assertEqual(context.workspace.root, path.parent)
                client = context.web._client
                self.assertFalse(client.is_closed)
                with self.assertRaises(ValueError):
                    context.workspace.read_text('custom.toml')
            self.assertTrue(client.is_closed)
            path.write_text(path.read_text().replace('credential_profile="d"', 'credential_profile="missing"'))
            with self.assertRaises(ConfigError):
                load_config(path)
            config = AppConfig(default_model='x', models={'x': {'provider': 'ollama', 'model': 'x'}})
            async with tool_context(config) as context:
                self.assertIsNone(context.web)

    async def test_context_closes_on_failure_and_cancellation(self):
        import asyncio
        for error in (RuntimeError('failure'), asyncio.CancelledError()):
            with TemporaryDirectory() as directory:
                config = AppConfig(default_model='d', models={'d': {
                    'provider': 'deepseek', 'model': 'deepseek-chat', 'api_key': 'key'}},
                    workspace={'root': directory}, tools={'web': {'credential_profile': 'd'}})
                with self.assertRaises(type(error)):
                    async with tool_context(config) as context:
                        client = context.web._client
                        raise error
                self.assertTrue(client.is_closed)

    def test_web_configuration_constraints(self):
        from pydantic import ValidationError
        base = {'default_model': 'd', 'models': {'d': {
            'provider': 'deepseek', 'model': 'deepseek-chat', 'api_key': 'SECRET'}}}
        for settings in ({'credential_profile': 'missing'}, {'credential_profile': 'd', 'timeout': 0},
                         {'credential_profile': 'd', 'timeout': float('inf')},
                         {'credential_profile': 'd', 'timeout': float('nan')},
                         {'credential_profile': 'd', 'base_url': 'https://other.example'}):
            with self.subTest(settings=settings), self.assertRaises(ValidationError) as caught:
                AppConfig.model_validate(dict(base, tools={'web': settings}))
            self.assertNotIn('SECRET', str(caught.exception))
        base['models']['d']['provider'] = 'openai'
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(dict(base, tools={'web': {'credential_profile': 'd'}}))
