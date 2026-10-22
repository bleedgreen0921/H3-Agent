"""Exercise domestic provider request paths using real SDKs and mock HTTP."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from ai_assistant.config import AppConfig, ModelConfig, load_config
from ai_assistant.llm.factory import create_model


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PATHS = {
    "deepseek": ("api.deepseek.com", "/chat/completions"),
    "glm": ("open.bigmodel.cn", "/api/paas/v4/chat/completions"),
    "qwen": ("dashscope.aliyuncs.com", "/compatible-mode/v1/chat/completions"),
    "kimi": ("api.moonshot.ai", "/v1/chat/completions"),
}


class DomesticConfigTests(unittest.TestCase):
    def test_all_domestic_providers_require_explicit_key(self):
        for provider in EXPECTED_PATHS:
            with self.subTest(provider=provider), self.assertRaises(ValidationError):
                ModelConfig(provider=provider, model="test")

    @patch("ai_assistant.llm.factory.init_chat_model")
    def test_endpoint_and_compatible_parameter_overrides(self, factory):
        for provider in EXPECTED_PATHS:
            config = AppConfig(default_model="test", models={"test": ModelConfig(
                provider=provider, model="custom-model", api_key="custom-key",
                base_url="https://custom.invalid/region/v1",
                parameters={"stream_usage": True, "max_retries": 0},
            )})
            create_model(config)
            self.assertEqual(factory.call_args.kwargs["base_url"], "https://custom.invalid/region/v1")
            self.assertTrue(factory.call_args.kwargs["stream_usage"])
            self.assertEqual(factory.call_args.kwargs["model_provider"], "deepseek" if provider == "deepseek" else "openai")


class DomesticSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_keys_tool_calls_usage_and_native_reasoning(self):
        for name, (host, path) in EXPECTED_PATHS.items():
            with self.subTest(provider=name):
                config = load_config(ROOT / "assistant.example.toml")
                profile = config.models[name]

                def respond(request):
                    self.assertEqual((request.url.host, request.url.path), (host, path))
                    self.assertEqual(request.headers["authorization"], "Bearer " + profile.api_key.get_secret_value())
                    body = json.loads(request.content)
                    self.assertEqual(body["model"], profile.model)
                    self.assertEqual(body["tools"][0]["function"]["name"], "lookup")
                    self.assertEqual(body["custom_option"], "preserved")
                    return httpx.Response(200, json={
                        "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                        "model": profile.model,
                        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                            "role": "assistant", "content": None, "reasoning_content": "native reasoning",
                            "tool_calls": [{"id": "call-1", "type": "function", "function": {
                                "name": "lookup", "arguments": '{"query":"test"}'}}],
                        }}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                    })

                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    profile.parameters.update(http_async_client=client, extra_body={"custom_option": "preserved"})
                    model = create_model(config, name)
                    tool = {"type": "function", "function": {"name": "lookup", "description": "Lookup",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
                    result = await model.bind_tools([tool]).ainvoke("Test")
                self.assertEqual(result.tool_calls[0]["args"], {"query": "test"})
                self.assertEqual(result.usage_metadata["total_tokens"], 8)
                if name == "deepseek":
                    self.assertEqual(type(model).__name__, "ChatDeepSeek")
                    self.assertEqual(result.additional_kwargs["reasoning_content"], "native reasoning")

    async def test_streaming_chat_completions_for_each_provider(self):
        for name, (host, path) in EXPECTED_PATHS.items():
            with self.subTest(provider=name):
                config = load_config(ROOT / "assistant.example.toml")

                def respond(request):
                    self.assertEqual((request.url.host, request.url.path), (host, path))
                    body = json.loads(request.content)
                    self.assertTrue(body["stream"])
                    if name != "deepseek":
                        self.assertNotIn("stream_options", body)
                    chunk = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0,
                        "model": config.models[name].model, "choices": [{"index": 0,
                        "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]}
                    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                        text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    config.models[name].parameters["http_async_client"] = client
                    model = create_model(config, name)
                    self.assertEqual("".join([c.content async for c in model.astream("Hi")]), "hello")
