"""Diagnostic limits and HTTP client ownership remain application concerns."""

import unittest

from ai_assistant.config import AppConfig, ConfigError, ModelConfig
from ai_assistant.llm.requests import model_session, prepare_request


def config(retries=None):
    params = {"max_tokens": 9999}
    if retries is not None:
        params["max_retries"] = retries
    return AppConfig(default_model="test", models={"test": ModelConfig(
        provider="deepseek", model="deepseek-flash", api_key="offline-placeholder", parameters=params,
    )})


class RequestTests(unittest.IsolatedAsyncioTestCase):
    def test_prepare_preserves_profile_retries_without_mutation(self):
        for retries in (None, 0, 1):
            original = config(retries)
            prepared = prepare_request(original, "test", 30, 256)
            self.assertEqual(prepared.models["test"].parameters.get("max_retries"), retries)
            self.assertEqual(original.models["test"].parameters["max_tokens"], 9999)
            self.assertEqual(prepared.models["test"].parameters["max_tokens"], 256)
            self.assertFalse(prepared.models["test"].parameters["cache"])

    async def test_owned_clients_closed_on_success_and_error(self):
        for raises in (False, True):
            clients = []
            def factory(config, selected):
                params = config.models[selected].parameters
                clients.extend([params["http_client"], params["http_async_client"]])
                self.assertTrue(all(not client.is_closed for client in clients))
                if raises:
                    raise ConfigError("factory error")
                return object()
            prepared = prepare_request(config(0), "test", 60, 256)
            if raises:
                with self.assertRaises(ConfigError):
                    async with model_session(prepared, "test", 60, factory=factory):
                        pass
            else:
                async with model_session(prepared, "test", 60, factory=factory):
                    pass
            self.assertTrue(all(client.is_closed for client in clients))
            self.assertNotIn("http_client", prepared.models["test"].parameters)
