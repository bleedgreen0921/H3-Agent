"""Characterize native structured Runnables, not a second parsing protocol."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError, OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ai_assistant.config import AppConfig, ModelConfig
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.structured import create_structured_runnable


class Answer(BaseModel):
    count: int


class StrictAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    count: int


class ValidatedAnswer(BaseModel):
    count: int

    @field_validator("count")
    @classmethod
    def positive(cls, value):
        if value < 1:
            raise ValueError("must be positive")
        return value


def response(content='{"count":3}', *, tools=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tools is not None:
        message["tool_calls"] = tools
    return httpx.Response(200, json={
        "id": "offline", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
    })


def call(name="Answer", arguments='{"count":3}', identity="call-1"):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


class NativeStructuredTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        guard = patch("socket.socket.connect", side_effect=AssertionError("No real network"))
        guard.start()
        self.addCleanup(guard.stop)
        self.requests = []
        self.reply = response()

        async def respond(request):
            self.requests.append(request)
            if callable(self.reply):
                return await self.reply(request)
            return self.reply

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.sync_client = httpx.Client(transport=httpx.MockTransport(lambda request: response()))
        self.addAsyncCleanup(self.client.aclose)
        self.addCleanup(self.sync_client.close)

    def model(self, *, provider="deepseek", name="deepseek-flash", retries=0):
        config = AppConfig(default_model="test", models={"test": ModelConfig(
            provider=provider, model=name, api_key="offline-placeholder",
            parameters={"max_retries": retries, "http_async_client": self.client,
                        "http_client": self.sync_client},
        )})
        return create_model(config)

    async def test_native_value_and_raw_envelope(self):
        model = self.model()
        runnable = create_structured_runnable(model, Answer)
        self.assertIsInstance(runnable, Runnable)
        self.assertEqual(await runnable.ainvoke('Return JSON {"count":3}'), Answer(count=3))
        result = await create_structured_runnable(model, Answer, include_raw=True).ainvoke("JSON")
        self.assertEqual(set(result), {"raw", "parsed", "parsing_error"})
        self.assertEqual(result["parsed"], Answer(count=3))
        self.assertIsNone(result["parsing_error"])
        self.assertEqual(result["raw"].usage_metadata["total_tokens"], 20)
        for request in self.requests:
            body = json.loads(request.content)
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertFalse(self.client.is_closed)

    async def test_native_tolerant_json_without_another_request(self):
        runnable = create_structured_runnable(self.model(), Answer)
        for text in ('{"count":3', '```json\n{"count":3}\n```'):
            self.reply = response(text)
            before = len(self.requests)
            self.assertEqual(await runnable.ainvoke("JSON"), Answer(count=3))
            self.assertEqual(len(self.requests), before + 1)

    async def test_native_validation_and_raw_errors(self):
        model = self.model()
        for text, schema in (('{}', Answer), ('{"count":"bad"}', Answer),
                             ('{"count":0}', ValidatedAnswer), ('not JSON', Answer)):
            self.reply = response(text)
            runnable = create_structured_runnable(model, schema, include_raw=True)
            result = await runnable.ainvoke("JSON")
            self.assertIsNone(result["parsed"])
            self.assertIsInstance(result["parsing_error"], OutputParserException)
            self.assertEqual(result["raw"].usage_metadata["total_tokens"], 20)
            with self.assertRaises(OutputParserException):
                await create_structured_runnable(model, schema).ainvoke("JSON")

    async def test_schema_configuration_is_native(self):
        self.reply = response('{"count":"3","extra":"ignored"}')
        self.assertEqual(await create_structured_runnable(self.model(), Answer).ainvoke("JSON"), Answer(count=3))
        with self.assertRaises(OutputParserException):
            await create_structured_runnable(self.model(), StrictAnswer).ainvoke("JSON")

    async def test_native_tool_binding_and_first_match(self):
        model = self.model()
        runnable = create_structured_runnable(model, Answer, method="function_calling", include_raw=True)
        for tools in ([call()], [call(), call(arguments='{"count":7}', identity="call-2")]):
            self.reply = response(None, tools=tools, finish="tool_calls")
            result = await runnable.ainvoke("Return the count using the Answer tool")
            self.assertEqual(result["parsed"], Answer(count=3))
            self.assertIsNone(result["parsing_error"])
            body = json.loads(self.requests[-1].content)
            self.assertEqual(body["tools"][0]["function"]["name"], "Answer")
            self.assertNotIn("response_format", body)
            self.assertNotIn("beta", self.requests[-1].url.path)
            self.assertEqual(body["thinking"], {"type": "disabled"})
        self.reply = response(None, tools=[])
        empty = await runnable.ainvoke("Use the tool")
        self.assertIsNone(empty["parsed"])
        self.assertIsNone(empty["parsing_error"])

    async def test_native_bad_tool_arguments(self):
        runnable = create_structured_runnable(self.model(), Answer, method="function_calling", include_raw=True)
        self.reply = response(None, tools=[call(arguments='{"count":"bad"}')], finish="tool_calls")
        result = await runnable.ainvoke("Use tool")
        self.assertIsNone(result["parsed"])
        self.assertIsInstance(result["parsing_error"], ValidationError)
        self.assertEqual(len(self.requests), 1)

    async def test_early_method_and_provider_rejection(self):
        deepseek = self.model()
        compatible = self.model(provider="openai")
        for model in (deepseek, compatible):
            with self.assertRaisesRegex(ValueError, "JSON Schema"):
                create_structured_runnable(model, Answer, method="json_schema")
        with self.assertRaises(ValueError):
            create_structured_runnable(deepseek, Answer, method="unknown")
        with self.assertRaisesRegex(ValueError, "DeepSeek only"):
            create_structured_runnable(self.model(provider="openai", name="other-model"), Answer)
        self.assertEqual(len(self.requests), 0)
        result = await create_structured_runnable(compatible, Answer).ainvoke("JSON")
        self.assertEqual(result.count, 3)

    async def test_sdk_finish_exceptions_propagate_even_with_raw(self):
        for reason, error in (("length", LengthFinishReasonError), ("content_filter", ContentFilterFinishReasonError)):
            self.reply = response('{"count":', finish=reason)
            before = len(self.requests)
            with self.assertRaises(error):
                await create_structured_runnable(self.model(), Answer, include_raw=True).ainvoke("JSON")
            self.assertEqual(len(self.requests), before + 1)

    async def test_native_prompt_config_and_usage_callback(self):
        handler = UsageMetadataCallbackHandler()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Return JSON matching this schema: {schema}"), ("user", "{text}"),
        ]).partial(schema=json.dumps(Answer.model_json_schema()))
        chain = prompt | create_structured_runnable(self.model(), Answer)
        result = await chain.with_config(tags=["offline"], metadata={"purpose": "test"}, callbacks=[handler]).ainvoke({"text": "Count 3"})
        self.assertEqual(result.count, 3)
        self.assertEqual(sum(value["total_tokens"] for value in handler.usage_metadata.values()), 20)
        self.assertEqual(len(self.requests), 1)

    async def test_connection_and_timeout_exceptions_are_native(self):
        for error, expected in ((httpx.ConnectError, ModelConnectionError), (httpx.ReadTimeout, ModelTimeoutError)):
            async def fail(request):
                raise error("offline")
            self.reply = fail
            before = len(self.requests)
            with self.assertRaises(expected):
                await create_structured_runnable(self.model(), Answer, include_raw=True).ainvoke("JSON")
            self.assertEqual(len(self.requests), before + 1)

    async def test_sdk_retry_ownership(self):
        attempts = 0
        async def transient(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(429, json={"error": {"message": "offline"}}) if attempts == 1 else response()
        self.reply = transient
        with patch("openai._base_client.BaseClient._calculate_retry_timeout", return_value=0):
            parsed = await create_structured_runnable(self.model(retries=1), Answer).ainvoke("JSON")
        self.assertEqual(parsed.count, 3)
        self.assertEqual(len(self.requests), 2)

    async def test_cancellation_and_programmer_errors_propagate(self):
        started = asyncio.Event()
        async def wait(request):
            started.set()
            await asyncio.Event().wait()
        self.reply = wait
        task = asyncio.create_task(create_structured_runnable(self.model(), Answer).ainvoke("JSON"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.client.is_closed)
        self.reply = response()
        class Broken(BaseModel):
            count: int
            @field_validator("count")
            @classmethod
            def broken(cls, value):
                raise RuntimeError("validator bug")
        with self.assertRaisesRegex(RuntimeError, "validator bug"):
            await create_structured_runnable(self.model(), Broken).ainvoke("JSON")
