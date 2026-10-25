import asyncio
import unittest

import httpx

from ai_assistant.tools.web import DeepSeekWeb, WebError, validate_url, SEARCH_TEXT_LIMIT, FETCH_TEXT_LIMIT


ITEM = {'title': 'Title', 'url': 'https://example.com', 'snippet': 'short',
        'summary': 'long', 'content': '', 'site_name': None, 'date': None}


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_and_responses(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith('search'):
                return httpx.Response(200, json={'results': [ITEM]})
            return httpx.Response(200, json={'title': 'Page', 'content': '# Body'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            web = DeepSeekWeb(client, 'SECRET')
            content, artifact = await web.search('site:example.com topic')
            self.assertIn('long', content)
            self.assertEqual(artifact['response']['results'], [ITEM])
            content, artifact = await web.fetch('https://example.com/page')
            self.assertIn('# Body', content)
            self.assertFalse(artifact['truncated'])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].headers['Authorization'], 'Bearer SECRET')
        self.assertEqual(requests[0].url, 'https://api.deepseek.com/v1/tools/search')
        import json
        self.assertEqual(json.loads(requests[0].content), {
            'query': 'site:example.com topic', 'count': 5, 'need_content': False})

    async def test_errors_no_retry_no_redirect(self):
        for status in (301, 400, 401, 402, 403, 404, 429, 500, 504):
            requests = []
            def handler(request):
                requests.append(request)
                return httpx.Response(status, headers={'location': 'https://other.example'}, text='SECRET')
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
                with self.assertRaises(WebError) as caught:
                    await DeepSeekWeb(client, 'SECRET').search('test')
                self.assertNotIn('SECRET', str(caught.exception))
            self.assertEqual(len(requests), 1)

    async def test_bad_response_empty_results_and_truncation(self):
        for data in ({'results': []}, {'results': [dict(ITEM, summary='x' * 21000)]}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data))) as client:
                content, artifact = await DeepSeekWeb(client, 'key').search('x')
                self.assertLessEqual(len(content), SEARCH_TEXT_LIMIT)
                self.assertEqual(artifact['truncated'], bool(data['results']))
        for body in (b'not json', b'[]', b'{}', b'x' * (2 * 1024 * 1024 + 1)):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))) as client:
                with self.assertRaises(WebError):
                    await DeepSeekWeb(client, 'key').search('x')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'title': '', 'content': 'x' * 31000}))) as client:
            content, artifact = await DeepSeekWeb(client, 'key').fetch('https://example.com')
            self.assertEqual(len(content), FETCH_TEXT_LIMIT)
            self.assertIn('内容已截断', content)
            self.assertTrue(artifact['truncated'])

    async def test_transport_and_cancellation(self):
        for error in (httpx.ReadTimeout('SECRET'), httpx.ConnectError('SECRET'), asyncio.CancelledError()):
            calls = []
            def handler(request):
                calls.append(request)
                raise error
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(asyncio.CancelledError if isinstance(error, asyncio.CancelledError) else WebError):
                    await DeepSeekWeb(client, 'key').search('test')
            self.assertEqual(len(calls), 1)

    def test_url_policy(self):
        for url in ('file:///tmp/a', 'http://localhost', 'http://127.0.0.1', 'https://[::1]',
                    'https://10.0.0.1', 'https://user:pass@example.com', 'https://intranet',
                    'https://x.local', 'https://example.com:bad', 'http://2130706433',
                    'http://0x7f000001', 'https://example.com/\n'):
            with self.subTest(url=url), self.assertRaises(WebError):
                validate_url(url)
        self.assertEqual(validate_url('https://example.com/page'), 'https://example.com/page')

    async def test_invalid_input_sends_nothing(self):
        requests = []
        def handler(request):
            requests.append(request)
            raise AssertionError('invalid input must not be sent')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            web = DeepSeekWeb(client, 'key')
            for query, count in [(' ', 5), ('x' * 129, 5), ('x', 11)]:
                with self.assertRaises(WebError):
                    await web.search(query, count)
            with self.assertRaises(WebError):
                await web.fetch('http://localhost')
        self.assertEqual(requests, [])


def test_combined_web_results_fit_default_context_budget():
    from ai_assistant.agents.context import count_tokens, fit_messages, usable_budget
    from ai_assistant.agents import SYSTEM_PROMPT
    from ai_assistant.tools.builtin import create_builtin_tools
    from ai_assistant.tools.web import bounded_text
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    tools = create_builtin_tools(include_web=True)
    messages = [HumanMessage(content='x' * 857), AIMessage(content='', tool_calls=[
        {'id':'search','name':'web_search','args':{'query':'fixture'}},
        {'id':'fetch','name':'web_fetch','args':{'url':'https://example.com'}}]),
        ToolMessage(content=bounded_text('x'*30000,SEARCH_TEXT_LIMIT),tool_call_id='search'),
        ToolMessage(content=bounded_text('x'*30000,FETCH_TEXT_LIMIT),tool_call_id='fetch')]
    counter=lambda items:count_tokens([SystemMessage(content=SYSTEM_PROMPT)]+list(items),tools=tools)
    assert fit_messages(messages,usable_budget(24000),counter=counter) == messages
