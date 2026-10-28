"""共享响应检查必须阻止可解析但未完成的输出。"""
import unittest

from langchain_core.messages import AIMessage
from ai_assistant.llm.completion import IncompleteResponseError, validate_completion


class CompletionTests(unittest.TestCase):
    def test_completed_and_legacy_chat(self):
        for metadata, responses in (({}, False), ({'status': 'completed'}, True)):
            message = AIMessage('ok', response_metadata=metadata)
            self.assertIs(validate_completion(message, responses=responses), message)

    def test_incomplete_failed_missing_and_refusal(self):
        for metadata in ({}, {'status': 'incomplete'}, {'status': 'failed'},
                         {'status': 'in_progress'}, {'finish_reason': 'length'},
                         {'finish_reason': 'content_filter'}):
            with self.subTest(metadata=metadata), self.assertRaises(IncompleteResponseError):
                validate_completion(AIMessage('{"ok":true}', response_metadata=metadata), responses=True)
        for message in (AIMessage('', additional_kwargs={'refusal': 'private'}),
                        AIMessage([{'type': 'refusal', 'refusal': 'private'}])):
            with self.assertRaises(IncompleteResponseError) as error:
                validate_completion(message)
            self.assertNotIn('private', str(error.exception))

    def test_deepseek_nested_controls_and_factory_copy(self):
        from copy import deepcopy
        from unittest.mock import patch
        from ai_assistant.config import AppConfig, ModelConfig, ConfigError
        from ai_assistant.llm.factory import create_model

        for container in ('extra_body', 'model_kwargs'):
            for controls in ({'use_responses_api': False}, {'reasoning': {'effort': 'high'}},
                             {'store': True}, {'conversation': 'remote'},
                             {'previous_response_id': 'remote'}, {'use_previous_response_id': True}):
                parameters = {'use_responses_api': True, container: controls}
                config = AppConfig.model_construct(default_model='test', models={'test': ModelConfig.model_construct(
                    provider='deepseek', model='deepseek-flash', api_key='offline-key', parameters=parameters)})
                before = deepcopy(config.models['test'].parameters)
                with self.subTest(container=container, controls=controls), patch(
                        'ai_assistant.llm.factory.init_chat_model') as factory:
                    with self.assertRaises(ConfigError):
                        create_model(config)
                    factory.assert_not_called()
                    self.assertEqual(config.models['test'].parameters, before)
