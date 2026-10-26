"""Configuration and CLI checks with no network or optional SDK requirements."""

from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ai_assistant.interfaces.cli import main
from ai_assistant.config import AppConfig, ConfigError, ModelConfig, load_config


VALID = '''default_model = "test"
[models.test]
provider = "openai"
model = "test-model"
api_key = "SENTINEL_PRIVATE_KEY"
'''


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "assistant.toml"
        self.path.write_text(VALID)

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["config", "check", "--config", str(self.path), *args])
        return code, out.getvalue(), err.getvalue()

    def test_example_loads_without_creating_models(self):
        path = Path(__file__).resolve().parents[2] / "assistant.example.toml"
        with patch("ai_assistant.llm.factory.init_chat_model", side_effect=AssertionError("No model creation")):
            config = load_config(path)
        self.assertEqual(config.default_model, "deepseek")
        self.assertEqual(set(config.models), {"deepseek"})

    def test_default_path_is_cwd_without_parent_search(self):
        old = Path.cwd()
        try:
            os.chdir(self.temp.name)
            self.assertEqual(load_config().default_model, "test")
            child = Path("child")
            child.mkdir()
            os.chdir(child)
            with self.assertRaises(ConfigError):
                load_config()
        finally:
            os.chdir(old)

    def test_missing_file_permission_directory_and_encoding(self):
        with self.assertRaisesRegex(ConfigError, "not found"):
            load_config(self.path.with_name("missing.toml"))
        with patch.object(Path, "open", side_effect=PermissionError("SENTINEL_PRIVATE_KEY")):
            with self.assertRaisesRegex(ConfigError, "Permission denied"):
                load_config(self.path)
        with self.assertRaisesRegex(ConfigError, "Unable to read"):
            load_config(self.path.parent)
        self.path.write_bytes(b'\xff')
        with self.assertRaisesRegex(ConfigError, "UTF-8"):
            load_config(self.path)

    def test_invalid_inputs_are_safe_in_cli_and_exception(self):
        variants = [
            VALID + '\nsecret = "SENTINEL_PRIVATE_KEY"\n',
            VALID.replace('default_model = "test"', 'default_model = "missing"'),
            VALID.replace('api_key = "SENTINEL_PRIVATE_KEY"', 'api_key_env = "OLD_VARIABLE"'),
            VALID.replace('api_key = "SENTINEL_PRIVATE_KEY"', ''),
            VALID.replace('api_key = "SENTINEL_PRIVATE_KEY"', 'api_key = "  "'),
            VALID.replace('provider = "openai"', 'provider = " "'),
            VALID + '\n[models.test.parameters]\napi_key = "SENTINEL_PRIVATE_KEY"\n',
            VALID + '\nSENTINEL_PRIVATE_KEY ???',
        ]
        for content in variants:
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(ConfigError) as error:
                    load_config(self.path)
                self.assertNotIn("SENTINEL_PRIVATE_KEY", str(error.exception))
                code, out, err = self.cli()
                self.assertEqual(code, 2)
                self.assertNotIn("SENTINEL_PRIVATE_KEY", out + err)
                self.assertNotIn("Traceback", err)

    def test_secrets_hidden_in_config_display_and_validation(self):
        config = load_config(self.path)
        for text in (str(config), repr(config), config.model_dump_json(), str(config.model_dump())):
            self.assertNotIn("SENTINEL_PRIVATE_KEY", text)
        self.assertEqual(config.models["test"].api_key.get_secret_value(), "SENTINEL_PRIVATE_KEY")
        with self.assertRaises(ValidationError) as error:
            AppConfig.model_validate({"default_model": "bad", "models": {"test": {
                "provider": "openai", "model": "test", "api_key": "SENTINEL_PRIVATE_KEY"}}})
        self.assertNotIn("SENTINEL_PRIVATE_KEY", str(error.exception))

    def test_key_required_for_known_cloud_providers_and_optional_for_ollama(self):
        for provider in ("openai", "anthropic", "google_genai"):
            with self.subTest(provider=provider), self.assertRaises(ValidationError):
                ModelConfig(provider=provider, model="test")
        self.assertIsNone(ModelConfig(provider="ollama", model="test").api_key)

    def test_reserved_parameter_aliases(self):
        for name in ("model", "api_key", "api_key_env", "openai_api_key", "google_api_key",
                     "base_url", "configurable_fields", "model_provider"):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                ModelConfig(provider="ollama", model="test", parameters={name: "x"})

    def test_load_and_cli_do_not_mutate_environment_or_create_models(self):
        before = dict(os.environ)
        with patch("ai_assistant.llm.factory.init_chat_model", side_effect=AssertionError("No model creation")), \
             patch("socket.socket.connect", side_effect=AssertionError("No network")):
            config = load_config(self.path)
            code, out, err = self.cli()
            self.assertEqual(code, 0)
            self.assertIn("未检查连通性", out)
            self.assertIn("test", out)
            self.assertEqual(err, "")
        self.assertEqual(dict(os.environ), before)
        self.path.write_text(VALID.replace("SENTINEL_PRIVATE_KEY", "${UNCHANGED_LITERAL}"))
        self.assertEqual(load_config(self.path).models["test"].api_key.get_secret_value(), "${UNCHANGED_LITERAL}")

    def test_selected_profile_and_unknown_profile(self):
        self.assertEqual(self.cli("--model", "test")[0], 0)
        self.assertEqual(self.cli("--model", "missing")[0], 2)

    def test_cli_bad_arguments_exit_two(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["config", "check", "--unknown"])
        self.assertEqual(error.exception.code, 2)

    def test_module_entrypoint(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "ai_assistant", "config", "check", "--config", str(self.path)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("未检查连通性", result.stdout)
        self.assertNotIn("SENTINEL_PRIVATE_KEY", result.stdout + result.stderr)


def test_agent_timezone_and_task_default_validation():
    import pytest
    from ai_assistant.config import AgentSettings
    from ai_assistant.integrations.config import FeishuConfig
    from pydantic import ValidationError
    assert AgentSettings().timezone == 'Asia/Shanghai'
    with pytest.raises(ValidationError):
        AgentSettings(timezone='Invalid/Zone')
    assert FeishuConfig(app_id='cli_fixture', app_secret='fixture', owner_open_id='fixture').default_task_assignee == 'none'
    with pytest.raises(ValidationError):
        FeishuConfig(app_id='cli_fixture', app_secret='fixture', owner_open_id='fixture', default_task_assignee='anyone')
