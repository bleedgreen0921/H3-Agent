"""Offline configuration and native provider integration checks."""

import json
import unittest
from unittest.mock import patch

import httpx
from ai_assistant.config import AppConfig, ConfigError, ModelConfig
from ai_assistant.llm.factory import create_model


def catalog():
    return AppConfig(default_model="local", models={
        "local": ModelConfig(
            provider="openai", model="test-model",
            base_url="https://model.invalid/v1", api_key="offline-key",
            parameters={"max_retries": 0},
        ),
        "other": ModelConfig(provider="anthropic", model="test-other", api_key="other-key"),
    })


class ModelTests(unittest.TestCase):
    @patch("ai_assistant.llm.factory.init_chat_model")
    def test_provider_initialization_errors_do_not_expose_credentials(self, factory):
        for exception in (ImportError("offline-key"), ValueError("offline-key"), TypeError("offline-key")):
            factory.side_effect = exception
            with self.subTest(exception=type(exception)), self.assertRaises(ConfigError) as error:
                create_model(catalog())
            self.assertNotIn("offline-key", str(error.exception))

    @patch("ai_assistant.llm.factory.init_chat_model")
    def test_selected_profile_and_provider_options(self, factory):
        loaded = catalog()
        create_model(loaded)
        factory.assert_called_once_with(
            "test-model", model_provider="openai", base_url="https://model.invalid/v1",
            api_key="offline-key", max_retries=0,
        )
        factory.reset_mock()
        create_model(loaded, "other")
        factory.assert_called_once_with("test-other", model_provider="anthropic", api_key="other-key")
        self.assertNotIn("api_key", loaded.models["local"].parameters)

    @patch("ai_assistant.llm.factory.init_chat_model")
    def test_unknown_name_fails_before_factory(self, factory):
        for name in ("unknown", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_model(catalog(), name)
        factory.assert_not_called()


class NativeOpenAIIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_messages_usage_and_tool_binding(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(request.headers["authorization"], "Bearer offline-key")
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "test-model", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            loaded = catalog()
            loaded.models["local"].parameters["http_async_client"] = client
            with patch.dict("os.environ", {"OPENAI_API_KEY": "wrong-environment-key"}):
                model = create_model(loaded)
            response = await model.ainvoke([{"role": "user", "content": "Hi"}])
            self.assertEqual(response.content, "hello")
            self.assertEqual(response.usage_metadata["total_tokens"], 7)
            tool = {"type": "function", "function": {"name": "lookup", "description": "Lookup",
                "parameters": {"type": "object", "properties": {}}}}
            await model.bind_tools([tool]).ainvoke("Hi")
        self.assertEqual(requests[0]["model"], "test-model")
        self.assertEqual(requests[1]["tools"][0]["function"]["name"], "lookup")

    async def test_native_streaming(self):
        def respond(request):
            self.assertTrue(json.loads(request.content)["stream"])
            chunk = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0,
                "model": "test-model", "choices": [{"index": 0,
                "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            loaded = catalog()
            loaded.models["local"].parameters["http_async_client"] = client
            model = create_model(loaded)
            text = "".join([chunk.content async for chunk in model.astream("Hi")])
            self.assertEqual(text, "hello")

    async def test_native_structured_output(self):
        schema = {"title": "Answer", "type": "object", "properties": {
            "answer": {"type": "string"}}, "required": ["answer"]}

        def respond(request):
            body = json.loads(request.content)
            self.assertEqual(body["response_format"]["type"], "json_schema")
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "test-model", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": '{"answer":"hello"}'},
                    "finish_reason": "stop"}],
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            loaded = catalog()
            loaded.models["local"].parameters["http_async_client"] = client
            model = create_model(loaded)
            result = await model.with_structured_output(
                schema, method="json_schema"
            ).ainvoke("Hi")
            self.assertEqual(result, {"answer": "hello"})


if __name__ == "__main__":
    unittest.main()
