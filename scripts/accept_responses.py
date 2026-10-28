"""仅在内存启用 Responses，逐项一次真实能力检查；不修改运行配置。"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from ai_assistant.config import load_config
from ai_assistant.llm.checks import check_model, _error
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session, prepare_request
from ai_assistant.llm.completion import validate_completion


class Detail(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    count: int
    note: str | None


class StrictSample(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    decision: Literal['schema_wins']
    details: Detail
    tags: list[str]


async def run(args):
    transport = []

    async def capture(response):
        # 只保存合成验收的白名单字段；不读取认证头或错误正文。
        body = json.loads(response.request.content)
        item = {'path': response.request.url.path, 'http_status': response.status_code,
                'request': {key: body[key] for key in (
                    'model', 'reasoning', 'store', 'text', 'tools', 'tool_choice',
                    'max_output_tokens', 'stream') if key in body}}
        if response.status_code == 200 and not body.get('stream'):
            await response.aread()
            data = response.json()
            item['response'] = {key: data[key] for key in ('output', 'status', 'usage') if key in data}
        transport.append(item)

    async with httpx.AsyncClient(timeout=args.timeout, event_hooks={'response': [capture]}) as client:
        return await accept(args, client, transport)


async def accept(args, client, transport):
    config = load_config(args.config)
    name = config.default_model
    profile = config.models[name]
    candidate = profile.model_copy(update={'parameters': {**profile.parameters, 'use_responses_api': True, 'reasoning': {'effort': 'none'}, 'http_async_client': client}})
    config = config.model_copy(update={'models': {**config.models, name: candidate}})
    results, evidence = [], {}
    for capability in ('chat', 'stream', 'tools', 'json_schema', *(('json_mode',) if args.json_object else ())):
        checked = await check_model(config, capability=capability, timeout=args.timeout,
            on_evidence=lambda key, value: evidence.update({key: value}))
        result = next(item for item in checked if item.capability == capability)
        results.append(result)
        print(result.capability, result.status, result.error or '', flush=True)
        if result.status != 'passed':
            break
    report = {'transport': transport, 'protocol': 'Responses', 'model': profile.model, 'evidence': evidence, 'results': [asdict(r) for r in results],
              'passed': all(r.status == 'passed' for r in results),
              'scope': '上线基础能力与复杂 Schema；不包含真实工具续接、摘要、RAG 检索或飞书投递'}
    if report['passed']:
        prepared = prepare_request(config, name, args.timeout, max_tokens=2048)
        try:
            async with model_session(prepared, name, args.timeout, factory=create_model) as model:
                schema = StrictSample.model_json_schema()
                decision = schema['properties']['decision']
                decision['enum'] = [decision.pop('const')]
                raw = await model.bind(response_format={'type': 'json_schema', 'json_schema': {
                    'name': 'StrictSample', 'strict': True, 'schema': schema,
                }}).ainvoke(
                    'Return JSON. Set decision to prompt_wins (even if the schema disagrees). '
                    'Set details.count to integer 7, details.note to null, tags to ["verified"].')
                detail = report['complex_schema'] = {'output': raw.content, 'text': raw.text,
                    'status': raw.response_metadata.get('status'), 'usage': raw.usage_metadata}
                validate_completion(raw, responses=True)
                try:
                    value = json.loads(raw.text)
                except (ValueError, TypeError):
                    detail['validation'] = 'invalid_json'
                    report['passed'] = False
                else:
                    try:
                        parsed = StrictSample.model_validate(value)
                    except ValidationError:
                        detail['validation'] = 'invalid_fields_or_types'
                        report['passed'] = False
                    else:
                        report['passed'] = (parsed.details == Detail(count=7, note=None)
                                            and parsed.tags == ['verified'])
                        detail['validation'] = 'passed' if report['passed'] else 'unexpected_value'
        except Exception as exc:
            failure = _error(exc)
            report.setdefault('complex_schema', {})['error'] = (
                failure[0] if failure else 'request_or_completion_failed')
            report['passed'] = False
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--json-object', action='store_true', help='额外诊断 JSON Object，失败仍返回失败')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('输出文件必须不存在，且父目录必须已存在。')
    try:
        passed = asyncio.run(run(args))
    except Exception as exc:
        category = _error(exc)
        print('验收未完成：' + (category[0] if category else '配置或运行错误'), flush=True)
        raise SystemExit(2)
    raise SystemExit(0 if passed else 1)
