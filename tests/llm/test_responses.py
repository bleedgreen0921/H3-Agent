"""通过真实原生 SDK 和 mock HTTP 验证 Responses 请求。"""
import json
import unittest

import httpx
from pydantic import BaseModel

from ai_assistant.config import AppConfig, ModelConfig
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.structured import create_structured_runnable


class Answer(BaseModel):
    ok: bool


class ResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_schema_tools_usage_and_history(self):
        requests = []

        def respond(request):
            self.assertEqual(request.url.path, '/responses')
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body['reasoning'], {'effort': 'none'})
            self.assertFalse(body['store'])
            self.assertNotIn('thinking', body)
            self.assertNotIn('previous_response_id', body)
            return httpx.Response(200, json={
                'id': 'resp_test', 'object': 'response', 'created_at': 0,
                'model': 'deepseek-flash', 'status': 'completed',
                'output': [{'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                            'status': 'completed', 'content': [{'type': 'output_text',
                            'text': '{"ok":true}', 'annotations': []}]}],
                'usage': {'input_tokens': 5, 'output_tokens': 3, 'total_tokens': 8},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = AppConfig(default_model='test', models={'test': ModelConfig(
                provider='deepseek', model='deepseek-flash', api_key='offline-key',
                parameters={'use_responses_api': True, 'max_retries': 0,
                            'http_async_client': client, 'extra_body': {'thinking': {'type': 'disabled'}}})})
            model = create_model(config)
            result = await model.ainvoke('Hi')
            self.assertEqual(result.usage_metadata['total_tokens'], 8)
            parsed = await create_structured_runnable(model, Answer).ainvoke('JSON')
            self.assertEqual(parsed, Answer(ok=True))
            await model.bind_tools([{'type': 'function', 'function': {'name': 'lookup',
                'description': 'Lookup', 'parameters': {'type': 'object', 'properties': {}}}}]).ainvoke('Lookup')
            self.assertEqual(requests[1]['text']['format']['type'], 'json_schema')
            self.assertTrue(requests[1]['text']['format']['strict'])
            self.assertEqual(requests[2]['tools'][0]['name'], 'lookup')
        self.assertTrue(client.is_closed)

    async def test_stream_completion(self):
        from ai_assistant.llm.checks import check_model

        complete = True

        def respond(request):
            self.assertEqual(request.url.path, '/responses')
            response = {'id': 'resp_test', 'object': 'response', 'created_at': 0,
                'model': 'deepseek-flash', 'status': 'completed',
                'output': [{'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                    'status': 'completed', 'content': [{'type': 'output_text', 'text': 'OK', 'annotations': []}]}],
                'usage': {'input_tokens': 5, 'output_tokens': 3, 'total_tokens': 8}}
            events = [
                {'type': 'response.created', 'sequence_number': 0, 'response': {**response, 'status': 'in_progress', 'output': []}},
                {'type': 'response.output_text.delta', 'sequence_number': 1, 'item_id': 'msg_test', 'output_index': 0, 'content_index': 0, 'delta': 'OK'},
                {'type': 'response.completed', 'sequence_number': 2, 'response': response},
            ]
            if not complete:
                events.pop()
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                text=''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in events))

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = AppConfig(default_model='test', models={'test': ModelConfig(
                provider='deepseek', model='deepseek-flash', api_key='offline-key',
                parameters={'use_responses_api': True, 'max_retries': 0, 'http_async_client': client})})
            results = await check_model(config, capability='stream')
            self.assertEqual(results[1].status, 'passed')
            self.assertEqual(results[1].usage['total_tokens'], 8)
            complete = False
            results = await check_model(config, capability='stream')
            self.assertEqual(results[1].error, 'incomplete_response')

    async def test_cancel_and_timeout_cleanup(self):
        import asyncio
        from ai_assistant.llm.checks import check_model

        for cancel in (False, True):
            started, closed = asyncio.Event(), asyncio.Event()

            class HangingStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    started.set()
                    await asyncio.Event().wait()
                    yield b''

                async def aclose(self):
                    closed.set()

            def respond(request):
                self.assertEqual(request.url.path, '/responses')
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=HangingStream())

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                config = AppConfig(default_model='test', models={'test': ModelConfig(
                    provider='deepseek', model='deepseek-flash', api_key='offline-key',
                    parameters={'use_responses_api': True, 'max_retries': 0, 'http_async_client': client})})
                task = asyncio.create_task(check_model(config, capability='stream', timeout=5 if cancel else 1))
                await asyncio.wait_for(started.wait(), 3)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    self.assertEqual((await task)[1].error, 'timeout')
                self.assertTrue(closed.is_set())
                self.assertFalse(client.is_closed)

    async def test_tool_result_replays_full_history(self):
        from langchain_core.messages import HumanMessage, ToolMessage

        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            output = ([{'id': 'fc_test', 'type': 'function_call', 'call_id': 'call_test',
                        'name': 'lookup', 'arguments': '{}', 'status': 'completed'}]
                      if len(requests) == 1 else [{'id': 'msg_test', 'type': 'message',
                        'role': 'assistant', 'status': 'completed', 'content': [
                        {'type': 'output_text', 'text': 'found', 'annotations': []}]}])
            return httpx.Response(200, json={'id': 'resp_test', 'object': 'response',
                'created_at': 0, 'model': 'deepseek-flash', 'status': 'completed', 'output': output})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = AppConfig(default_model='test', models={'test': ModelConfig(
                provider='deepseek', model='deepseek-flash', api_key='offline-key',
                parameters={'use_responses_api': True, 'max_retries': 0, 'http_async_client': client})})
            model = create_model(config).bind_tools([{'type': 'function', 'function': {
                'name': 'lookup', 'description': 'Lookup',
                'parameters': {'type': 'object', 'properties': {}}}}])
            messages = [HumanMessage('Lookup')]
            call = await model.ainvoke(messages)
            messages.extend([call, ToolMessage('found', tool_call_id=call.tool_calls[0]['id'])])
            result = await model.ainvoke(messages)
            self.assertEqual(result.text, 'found')
            self.assertNotIn('previous_response_id', requests[1])
            self.assertEqual([item.get('type', item.get('role')) for item in requests[1]['input']],
                             ['message', 'function_call', 'function_call_output'])
            self.assertEqual(requests[1]['input'][0]['role'], 'user')

    async def test_response_failure_categories(self):
        from langchain_core.messages import AIMessage
        from ai_assistant.llm.checks import _validate_response, _error, ProbeFailure
        from openai.types.responses import ResponseError

        variants = [
            (AIMessage('', response_metadata={'status': 'incomplete'}), 'truncated'),
            (AIMessage('', response_metadata={'status': 'failed'}), 'request_failed'),
            (AIMessage([{'type': 'refusal', 'refusal': 'blocked'}]), 'refused'),
            (AIMessage([{'type': 'reasoning', 'content': []}]), 'thinking_mismatch'),
        ]
        for message, expected in variants:
            with self.assertRaises(ProbeFailure) as error:
                _validate_response(message, deepseek=True)
            self.assertEqual(str(error.exception), expected)
        self.assertEqual(_error(ValueError(ResponseError(code='server_error', message='private'))),
                         ('request_failed', None))

    async def test_production_agent_blocks_incomplete_tool_and_accepts_old_history(self):
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver
        from types import SimpleNamespace
        from ai_assistant.agents import create_assistant_agent
        from ai_assistant.config import AgentSettings
        from ai_assistant.llm.completion import IncompleteResponseError

        executed, requests = [], []
        status = 'completed'

        @tool
        def lookup() -> str:
            """Read a verification code."""
            executed.append(True)
            return 'verified'

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            has_result = any(item.get('type') == 'function_call_output'
                             and item.get('call_id') == 'new_call' for item in body['input'])
            output = ([{'id': 'msg_done', 'type': 'message', 'role': 'assistant',
                       'status': 'completed', 'content': [{'type': 'output_text',
                       'text': 'verified', 'annotations': []}]}] if has_result else
                      [{'id': 'fc_new', 'type': 'function_call', 'call_id': 'new_call',
                        'name': 'lookup', 'arguments': '{}', 'status': 'completed'}])
            return httpx.Response(200, json={'id': 'resp_new', 'object': 'response',
                'created_at': 0, 'model': 'deepseek-flash', 'status': status, 'output': output,
                'usage': {'input_tokens': 5, 'output_tokens': 3, 'total_tokens': 8}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = AppConfig(default_model='test', models={'test': ModelConfig(
                provider='deepseek', model='deepseek-flash', api_key='offline-key',
                parameters={'use_responses_api': True, 'max_retries': 0, 'http_async_client': client})})
            agent = create_assistant_agent(create_model(config), [lookup],
                checkpointer=InMemorySaver(), settings=AgentSettings())
            history = [HumanMessage('old'), AIMessage('', tool_calls=[
                {'id': 'old_call', 'name': 'lookup', 'args': {}}]),
                ToolMessage('old result', tool_call_id='old_call', artifact={'preserved': True}),
                HumanMessage('read again')]
            context = SimpleNamespace(project_rules='')
            result = await agent.ainvoke({'messages': history},
                {'configurable': {'thread_id': 'complete'}}, context=context)
            self.assertEqual(result['messages'][-1].text, 'verified')
            self.assertEqual(len(executed), 1)
            self.assertEqual(result['messages'][2].artifact, {'preserved': True})
            self.assertNotIn('previous_response_id', requests[-1])
            status = 'incomplete'
            with self.assertRaises(IncompleteResponseError):
                await agent.ainvoke({'messages': [HumanMessage('lookup')]},
                    {'configurable': {'thread_id': 'incomplete'}}, context=context)
            self.assertEqual(len(executed), 1)
