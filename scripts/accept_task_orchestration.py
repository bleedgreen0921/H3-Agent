"""显式真实模型决策验收；真实模型计费，飞书和终端全部模拟。

用法：项目 Python scripts/accept_task_orchestration.py --config assistant.toml
自动创建并删除独立 PostgreSQL 数据库；不连接真实飞书、不重试用例。
"""
import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import SecretStr
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from ai_assistant.config import load_config
from ai_assistant.integrations.config import MCPServer
from ai_assistant.integrations.mcp import Catalog, Entry, stable_id, schema_tokens
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import assistant_session, final_text
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.storage.sessions import session_storage
from ai_assistant.tools.builtin import create_builtin_tools, run_command

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/feishu_task_schemas.json'


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls.fromisoformat('2026-10-26T12:00:00+08:00')
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)


@contextmanager
def disposable_database(settings):
    """仅清理本次创建的随机库，初始化绝不触及业务库。"""
    options = conninfo_to_dict(os.environ.get('SESSION_TEST_DATABASE_URL') or settings.database_url.get_secret_value())
    options['dbname'] = 'postgres'
    admin = make_conninfo(**options)
    name = 'assistant_accept_' + uuid4().hex
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    try:
        options['dbname'] = name
        yield settings.model_copy(update={'database_url': SecretStr(make_conninfo(**options))})
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def simulated_directory(wrong=False):
    fixture = json.loads(FIXTURE.read_text())
    names = list(fixture['tools'])
    settings = MCPServer(transport='feishu', identity='user', tools=names,
        write_tools=[n for n in names if n.endswith(('_patch', '_create', '_addMembers'))], allow_write=True)
    actual = {'guid': 'fixture-guid', 'summary': '模拟面试',
        'due': {'timestamp': '1793002800000', 'is_all_day': False}, 'members': [],
        'reminders': [{'relative_fire_minute': 30}]}
    calls = []
    entries = []
    for name, schema in fixture['tools'].items():
        def bind(name):
            async def invoke(**kwargs):
                calls.append(name)
                if name.endswith('_patch'):
                    if wrong:
                        return json.dumps({'code': 1254000, 'msg': '模拟写入拒绝'})
                    actual.update(deepcopy(kwargs.get('data', {}).get('task', {})))
                elif name.endswith('_addMembers'):
                    actual['members'].extend(deepcopy(kwargs.get('data', {}).get('members', [])))
                return json.dumps({'code': 0, 'data': {'task': deepcopy(actual)}})
            return invoke
        ident = stable_id('fixture', 'user', name)
        tool = StructuredTool.from_function(name=ident, description=fixture['descriptions'][name],
            args_schema=schema, coroutine=bind(name))
        entries.append(Entry(ident, 'fixture', name, 'schema', tool, settings))
    return Catalog(entries), settings, actual, calls


async def run_case(config, index, wrong, request):
    directory, settings, actual, calls = simulated_directory(wrong)
    terminal_calls = []
    events = []
    async def terminal(**kwargs):
        terminal_calls.append(True)
        return '模拟终端：命令未执行。日期换算请使用 resolve_datetime。'
    fake_terminal = StructuredTool.from_function(name=run_command.name, description=run_command.description,
        args_schema=run_command.tool_call_schema, coroutine=terminal)
    def builtins(**kwargs):
        return create_builtin_tools(**{**kwargs, 'include_terminal': False}) + [fake_terminal]
    async def event(e):
        if e['event'] == 'on_tool_start':
            name = e.get('name')
            data = e.get('data', {}).get('input', {})
            # 只记录虚构输入的发现/加载信息，不记录参数和身份。
            events.append({'tool': name, 'loaded': data.get('tool_ids', []) if isinstance(data, dict) else []})
    report = {'case': index, 'wrong_receipt': wrong}
    with TemporaryDirectory(prefix='assistant-accept-') as root:
        c = config.model_copy(update={
            'workspace': config.workspace.model_copy(update={'root': root}),
            'knowledge': None, 'skills': None, 'memory': config.memory.model_copy(update={'enabled': False}),
            'agent': config.agent.model_copy(update={'timezone': 'Asia/Shanghai', 'persona_file': None}),
            'tools': config.tools.model_copy(update={'terminal': None, 'web': None}),
            'integrations': config.integrations.model_copy(update={'feishu': config.integrations.feishu.model_copy(
                update={'owner_open_id': 'owner-fixture', 'default_task_assignee': 'owner'})}),
            'mcp': config.mcp.model_copy(update={'servers': {'fixture': settings}})})
        with patch('ai_assistant.runtime.create_builtin_tools', builtins), \
             patch('ai_assistant.agents.mcp.datetime', FixedDatetime), \
             patch('ai_assistant.tools.datetime.datetime', FixedDatetime), \
             patch('ai_assistant.integrations.feishu.OfficialConnection.discover', side_effect=AssertionError('禁止真实飞书连接')):
            async with assistant_session(c, caller=Caller('feishu_private'), resources=directory) as session:
                seeded = await session.agent.aupdate_state({'configurable': {'thread_id': session._thread}},
                    {'messages': [HumanMessage('创建模拟面试任务，今天晚上七点，负责人是我，提前半小时提醒。'),
                                  AIMessage('已完成，任务 fixture-guid，今天晚上七点，负责人是你。')]})
                session._complete = seeded
                try:
                    turn = await session.run_turn(request, on_event=event)
                    report.update(usage=asdict(turn.usage), final=final_text(turn.messages),
                        operation_statuses=[{'status': o['status'], 'business_outcome': bool((o.get('result') or {}).get('business_outcome'))}
                                            for o in turn.operations])
                    # 轨迹全部来自虚构身份及模拟服务，不包含真实配置或凭据。
                    report['trace'] = [{'type': m.type, 'content': m.content,
                        'tool_calls': getattr(m, 'tool_calls', [])} for m in turn.messages]
                except Exception as exc:
                    report.update(error_type=type(exc).__name__, final='未完成：验收轮次异常')
    report.update(actual=deepcopy(actual), calls=calls, events=events, terminal_calls=len(terminal_calls))
    loaded = [directory.entries[k].original_name for e in events for k in e['loaded'] if k in directory.entries]
    searches = sum(e['tool'] == 'search_mcp_tools' for e in events)
    usage = report.get('usage', {})
    common = (not report.get('error_type') and usage.get('main_model_calls', 99) <= 10 and searches <= 2
        and not terminal_calls and 'task_v2_task_create' not in calls + loaded)
    statuses = report.get('operation_statuses', [])
    if wrong:
        passed = common and report['final'].startswith('未完成') and calls.count('task_v2_task_patch') == 1 \
            and calls.count('task_v2_task_addMembers') == 0 and actual['due']['timestamp'] == '1793002800000'
    else:
        passed = common and actual['due']['timestamp'] == '1793012400000' \
            and any(m.get('id') == 'owner-fixture' and m.get('role') == 'assignee' for m in actual['members']) \
            and actual['reminders'] == [{'relative_fire_minute': 30}] \
            and len(statuses) == 2 and all(o['status'] == 'succeeded' and not o['business_outcome'] for o in statuses) \
            and '未完成' not in report['final'] and '19:00' in report['final'] \
            and '负责人' in report['final'] and 'owner-fixture' in report['final']
    report.update(passed=bool(passed), target_eight_calls=usage.get('main_model_calls', 99) <= 8)
    return report


async def main(args):
    config = load_config(args.config)
    if not config.sessions or not config.integrations.feishu or not config.context or not config.mcp:
        raise RuntimeError('验收需要现有 sessions、feishu、context、mcp 配置。')
    if config.context.mcp_schema_tokens != 6000:
        raise RuntimeError('任务编排验收要求现有 6000 schema 预算。')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with disposable_database(config.sessions) as sessions:
        c = config.model_copy(update={'sessions': sessions})
        async with session_storage(sessions, initialize=True):
            pass
        async with operation_storage(sessions, initialize=True):
            pass
        original = '刚才说已完成，但任务时间不对，负责人也是空的。请把任务 fixture-guid 改为今天晚上七点，负责人设为我，保留提前半小时提醒。'
        cases = [(False, original)] * 3 + [(False, '请核对 fixture-guid：把截止时间修正到今晚19:00，添加我为负责人，保持原提醒。'), (True, original)]
        for index, (wrong, request) in enumerate(cases, 1):
            print(f'用例 {index}/5 开始（不自动重跑）', flush=True)
            report = await run_case(c, index, wrong, request)
            (args.output / f'case-{index}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            results.append({k: report[k] for k in ('case', 'passed', 'target_eight_calls')})
            print(json.dumps({**results[-1], 'calls': report.get('usage', {}).get('main_model_calls')}, ensure_ascii=False), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return all(r['passed'] for r in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--output', type=Path, default=ROOT / '.data/task-orchestration-acceptance')
    try:
        success = asyncio.run(main(parser.parse_args()))
    except Exception as error:
        print(f'验收未完成：{type(error).__name__}；请检查本机配置、模型及数据库连接。')
        raise SystemExit(1)
    raise SystemExit(0 if success else 1)
