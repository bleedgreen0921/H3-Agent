"""CLI results, cancellation and argument validation."""

from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from ai_assistant.config import AppConfig, ConfigError, ModelConfig
from ai_assistant.interfaces.cli import main
from ai_assistant.llm.checks import CheckResult

KEY = "secret-must-never-be-printed"


class ProbeCLITests(unittest.TestCase):
    def test_success_failure_and_safe_terminal_output(self):
        root = Path(__file__).resolve().parents[2]
        for status, expected in (("passed", 0), ("failed", 1)):
            async def fake(*args, on_result, **kwargs):
                result = CheckResult("chat", status, error="authentication" if status == "failed" else None)
                on_result(result)
                return [result]

            out, err = io.StringIO(), io.StringIO()
            with patch("ai_assistant.llm.checks.check_model", side_effect=fake), redirect_stdout(out), redirect_stderr(err):
                code = main(["model", "check", "--config", str(root / "assistant.example.toml")])
            self.assertEqual(code, expected)
            self.assertIn("max_retries=0（SDK）", out.getvalue())
            self.assertNotIn(KEY, out.getvalue() + err.getvalue())

    def test_setup_and_interrupt_exit_codes(self):
        path = str(Path(__file__).resolve().parents[2] / "assistant.example.toml")
        for error, code in ((ConfigError("setup error"), 2), (KeyboardInterrupt(), 130)):
            with patch("ai_assistant.llm.checks.check_model", side_effect=error), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["model", "check", "--config", path]), code)

    def test_reports_explicit_or_default_sdk_retries(self):
        for parameters, text in (({"max_retries": 1}, "max_retries=1（SDK）"), ({}, "沿用 SDK 默认")):
            config = AppConfig(default_model="test", models={"test": ModelConfig(
                provider="deepseek", model="deepseek-flash", api_key=KEY, parameters=parameters,
            )})
            async def fake(*args, **kwargs):
                return []
            out = io.StringIO()
            with patch("ai_assistant.interfaces.cli.load_config", return_value=config), \
                 patch("ai_assistant.llm.checks.check_model", side_effect=fake), redirect_stdout(out):
                self.assertEqual(main(["model", "check"]), 0)
            self.assertIn(text, out.getvalue())
            self.assertNotIn("禁止自动重试", out.getvalue())
            self.assertNotIn(KEY, out.getvalue())

    def test_invalid_timeout_arguments(self):
        for timeout in ("0", "-1", "nan", "inf", "bad"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["model", "check", "--timeout", timeout])
            self.assertEqual(error.exception.code, 2)
