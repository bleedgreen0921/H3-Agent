"""Wire-level offline probes: profile retry policy and native diagnostics."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from ai_assistant.config import AppConfig, ConfigError, ModelConfig, load_config
from ai_assistant.llm.checks import CAPABILITIES, CheckResult, check_model
from ai_assistant.llm.factory import create_model

KEY = "secret-must-never-be-printed"


def config(client=None):
    parameters = {"max_retries": 0, "max_tokens": 9999}
    if client is not None:
        parameters["http_async_client"] = client
    return AppConfig(default_model="deepseek", models={"deepseek": ModelConfig(
        provider="deepseek", model="deepseek-flash", api_key=KEY, parameters=parameters,
    )})


def reply(body, *, content=None, reasoning=None, finish=None):
    message = {"role": "assistant", "content": content or "OK"}
    if "tools" in body:
        message.update(content=None, tool_calls=[{"id": "check-1", "type": "function", "function": {
            "name": "connection_check", "arguments": '{"ok":true}'}}])
    elif "response_format" in body:
        message["content"] = '{"ok":true}' if content is None else content
    if reasoning:
        message["reasoning_content"] = reasoning
    return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
        "model": "deepseek-flash", "choices": [{"index": 0, "message": message,
        "finish_reason": finish or ("tool_calls" if "tools" in body else "stop")}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})


class ThinkingTests(unittest.TestCase):
    def test_defaults_and_no_mutation(self):
        loaded = config()
        loaded.models["deepseek"].parameters["extra_body"] = {"other": {"flag": True}}
        with patch("ai_assistant.llm.factory.init_chat_model") as factory:
            create_model(loaded)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"other": {"flag": True}, "thinking": {"type": "disabled"}})
        kwargs["extra_body"]["other"]["flag"] = False
        self.assertEqual(loaded.models["deepseek"].parameters["extra_body"], {"other": {"flag": True}})

    def test_invalid_controls_and_compatible_alias(self):
        for provider in ("deepseek", "openai"):
            for params in (
                {"extra_body": {"thinking": {"type": "enabled"}}},
                {"extra_body": {"thinking": "disabled"}},
                {"extra_body": {"thinking": None}},
                {"thinking": {"type": "disabled"}},
                {"model_kwargs": {"thinking": {"type": "disabled"}}},
                {"use_responses_api": True, "reasoning": {"effort": "high"}},
                {"use_responses_api": True, "previous_response_id": "resp_old"},
                {"use_responses_api": True, "extra_body": {"conversation": "conv_old"}},
                {"use_responses_api": True, "store": True},
            ):
                with self.subTest(provider=provider, params=params), self.assertRaises(ValidationError):
                    ModelConfig(provider=provider, model="deepseek-flash", api_key=KEY, parameters=params)
        loaded = config()
        loaded.models["deepseek"].parameters["extra_body"] = {"thinking": {"type": "enabled"}}
        with self.assertRaises(ConfigError):
            create_model(loaded)
        with patch("ai_assistant.llm.factory.init_chat_model") as factory:
            compatible = AppConfig(default_model="test", models={"test": ModelConfig(
                provider="openai", model="deepseek-flash", api_key=KEY)})
            create_model(compatible)
        self.assertEqual(factory.call_args.kwargs["extra_body"]["thinking"], {"type": "disabled"})


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_five_wire_requests_and_original_configuration(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(request.url.path, "/chat/completions")
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertIn(256, (body.get("max_tokens"), body.get("max_completion_tokens")))
            if body.get("stream"):
                chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0,
                    "model": "deepseek-flash", "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                    text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
            return reply(body)

        observed = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            loaded = config(client)
            results = await check_model(loaded, on_result=observed.append)
            self.assertFalse(client.is_closed)
        self.assertEqual([r.capability for r in results], list(CAPABILITIES))
        self.assertTrue(all(r.status == "passed" for r in results), results)
        self.assertEqual(observed, results)
        self.assertEqual(len(requests), 5)
        self.assertEqual(requests[-2]["response_format"]["type"], "json_object")
        self.assertEqual(requests[-1]["response_format"]["type"], "json_schema")
        self.assertNotIn("tools", requests[-1])
        self.assertEqual(results[0].usage["total_tokens"], 7)
        self.assertIsNone(results[1].usage)
        self.assertEqual(loaded.models["deepseek"].parameters["max_retries"], 0)
        self.assertNotIn("extra_body", loaded.models["deepseek"].parameters)

    async def test_http_errors_continue_without_retry(self):
        for status in (401, 402, 403, 429, 500, 503):
            requests, printed = [], []

            def respond(request):
                # The previous failure callback must run before the next request.
                self.assertEqual(len(printed), len(requests))
                requests.append(request)
                return httpx.Response(status, json={"error": {"message": KEY}})

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                results = await check_model(config(client), on_result=printed.append)
            self.assertEqual(len(requests), 5)
            self.assertTrue(all(r.status == "failed" and r.http_status == status for r in results))
            self.assertNotIn(KEY, repr(results))

    async def test_connection_timeout_and_no_retry(self):
        for exception in (httpx.ConnectError(KEY), httpx.ReadTimeout(KEY)):
            requests = []

            def respond(request):
                requests.append(request)
                raise exception

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                results = await check_model(config(client))
            self.assertEqual(len(requests), 5)
            self.assertTrue(all(r.status == "failed" for r in results))
            expected = "timeout" if isinstance(exception, httpx.ReadTimeout) else "connection"
            self.assertTrue(all(r.error == expected for r in results))
            self.assertNotIn(KEY, repr(results))

    async def test_profile_sdk_retry_is_not_overridden(self):
        requests = []
        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(429, json={"error": {"message": KEY}})
            return reply(json.loads(request.content))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            loaded = config(client)
            loaded.models["deepseek"].parameters["max_retries"] = 1
            with patch("openai._base_client.BaseClient._calculate_retry_timeout", return_value=0):
                results = await check_model(loaded, capability="chat")
        self.assertEqual(len(requests), 2)
        self.assertEqual(results[0].status, "passed")
        self.assertEqual(loaded.models["deepseek"].parameters["max_retries"], 1)

    async def test_unexpected_program_error_propagates(self):
        with patch("ai_assistant.llm.checks._probe", side_effect=RuntimeError("probe bug")):
            with self.assertRaisesRegex(RuntimeError, "probe bug"):
                await check_model(config(), capability="chat")

    async def test_output_failure_and_single_selection(self):
        variants = [
            ("chat", {"reasoning": KEY}, "thinking_mismatch"),
            ("chat", {"finish": "length"}, "truncated"),
            ("json_mode", {"content": "not JSON " + KEY}, "invalid_json"),
            ("json_schema", {"content": '{"ok":true,"extra":1}'}, "invalid_fields"),
            ("json_schema", {"content": '{"ok":false}'}, "unexpected_value"),
            ("json_schema", {"content": '{"ok":1}'}, "invalid_fields"),
            ("json_schema", {"content": '{"ok":"true"}'}, "invalid_fields"),
        ]
        for capability, options, error in variants:
            requests = []

            def respond(request):
                requests.append(request)
                return reply(json.loads(request.content), **options)

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                results = await check_model(config(client), capability=capability)
            self.assertEqual(len(requests), 1)
            result = next(r for r in results if r.capability == capability)
            self.assertEqual(result.error, error)
            self.assertEqual(sum(r.status == "skipped" for r in results), 4)
            self.assertNotIn(KEY, repr(results))

    async def test_invalid_tool_and_empty_text(self):
        def respond(request):
            body = json.loads(request.content)
            response = reply(body)
            data = response.json()
            message = data["choices"][0]["message"]
            if "tools" in body:
                message["tool_calls"][0]["function"]["name"] = "wrong_tool"
            else:
                message["content"] = ""
            return httpx.Response(200, json=data)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for capability, error in (("tools", "invalid_tool_call"), ("chat", "empty_output")):
                results = await check_model(config(client), capability=capability)
                self.assertEqual(next(r for r in results if r.capability == capability).error, error)

    async def test_per_probe_deadline(self):
        calls = []

        async def respond(request):
            calls.append(request)
            await asyncio.sleep(10)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            results = await check_model(config(client), timeout=0.01)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(r.error == "timeout" for r in results))

    async def test_stream_cancel_closes_response_and_stops_next_probes(self):
        started, closed = asyncio.Event(), asyncio.Event()
        calls = []

        class HangingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.Event().wait()
                yield b""

            async def aclose(self):
                closed.set()

        def respond(request):
            calls.append(request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=HangingStream())

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            task = asyncio.create_task(check_model(config(client), capability="stream"))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(closed.is_set())
            self.assertFalse(client.is_closed)
        self.assertEqual(len(calls), 1)

    async def test_setup_errors_without_requests(self):
        for capability, timeout in (("bad", 60), ("all", 0), ("all", float("nan"))):
            with self.assertRaises(ConfigError):
                await check_model(config(), capability=capability, timeout=timeout)
        with self.assertRaises(ConfigError):
            await check_model(config(), name="bad")
        with patch("ai_assistant.llm.checks.create_model", side_effect=ConfigError("Integration unavailable")):
            with self.assertRaises(ConfigError):
                await check_model(config())


