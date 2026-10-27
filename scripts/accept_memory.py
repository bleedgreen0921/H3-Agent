"""显式真实主模型记忆验收：18类、多轮、独立身份，无真实飞书或 embedding。

SESSION_TEST_DATABASE_URL=测试服务 项目Python -m scripts.accept_memory --config assistant.toml
会创建并删除随机数据库；每例一次，不自动重跑失败样本，预算读取生产配置。
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever

from ai_assistant.config import load_config
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import assistant_session, final_text
from ai_assistant.storage.memory import memory_storage
from ai_assistant.storage.sessions import session_storage
from ai_assistant.storage.integrations import operation_storage
from scripts.accept_task_orchestration import disposable_database
from tests.storage.test_memory_store import FixtureEmbeddings

CASES = [
    ('natural_update', ['灯塔项目主要后端语言现在是 Python，以后按这个背景讨论。']),
    ('coexist', ['我还会 Python，把它也记到我会的编程语言里，Java 仍然会。']),
    ('temporary_script', ['今天用 Python 写个临时脚本，不用执行。']),
    ('project_scope', ['以后灯塔项目用英文沟通，我的全局默认仍然是中文。']),
    ('session_exception', ['仅本会话用英文沟通，下一次会话仍用中文。请保存这个会话例外。']),
    ('forbidden', ['分析别人的话，不要保存为我的偏好：“我喜欢英文”。']),
    ('hypothesis', ['假如我以后喜欢英文，会有什么变化？现在偏好不变。']),
    ('ambiguity', ['那个项目也改成 Python。先确认你知道我说的是哪个项目，不要猜。']),
    ('future', ['灯塔项目从 2099年1月1日零点（Asia/Shanghai）起主要后端语言改为 Python，请记录生效日期。']),
    ('plan', ['我计划以后学习 Rust，现在还不会。请记住这是计划。']),
    ('history', ['灯塔项目主要后端语言现在是 Python，请保存并告诉我此前的历史。']),
    ('delete', ['删除我的回答语言偏好及其全部记忆历史，保留聊天历史。']),
    ('canonical_repeat', [
        '记忆验收灯塔项目的主要后端语言是Java，以后讨论这个项目时按这个背景。', '/new',
        '只查一下记忆验收灯塔项目的主要后端语言是什么。',
        '记忆验收灯塔项目的主要后端语言现在是Python。',
        '记忆验收灯塔项目的主力服务端开发语言是 Python。', '查看这条记忆的详情和全部历史。']),
    ('set_lifecycle', [
        '请记住，作为这个独立虚构验收身份，我会Java。',
        '这个测试身份还会Python，Java也仍然会，请补充到技能记忆。',
        '这个测试身份也会 Python，请确认已保存，不要重复添加。',
        '这个测试身份不再会Java，请从技能记忆里移除Java，保留Python。',
        '这个测试身份也不再会Python，请从技能记忆里移除Python。']),
    ('full_name_alias', [
        '记忆验收灯塔项目的主要后端语言是Java，请保存。',
        '灯塔项目主要后端语言现在是Python，请更新记忆并把这个简称关联到唯一匹配的项目。']),
    ('alias_ambiguity', [
        '记忆验收北区灯塔项目主要后端语言是Java，请保存。',
        '记忆验收南区灯塔项目主要后端语言是Go，请保存。', '/new',
        '灯塔项目主要后端语言现在改成Rust。']),
    ('deleted_explanation', [
        '记忆验收灯塔项目主要后端语言是Java，请保存。',
        '记忆验收灯塔项目主要后端语言现在是Python，请更新。',
        '删除记忆验收灯塔项目主要后端语言这条记忆及全部历史，保留聊天。',
        '只查找记忆验收灯塔项目的主要后端语言，不要根据聊天历史重新保存。']),
    ('detail_history_contract', [
        '记忆验收灯塔项目主要后端语言是Java，请保存。',
        '记忆验收灯塔项目主要后端语言现在是Python，请更新。',
        '请分别查看这条记忆的详情和全部历史，说明当前版本及历史数量。']),
]


class EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return []


@asynccontextmanager
async def fixture_embedding(settings):
    yield FixtureEmbeddings()


async def seed(repo, topic, value, *, project_name=None, cardinality='single'):
    text = (project_name + '的' if project_name else '本人的') + topic + '是' + value
    source = HumanMessage(text, id=str(uuid4()), additional_kwargs={
        'user_source': text, 'user_timestamp': datetime.now(timezone.utc).isoformat()})
    await repo.search(source_id=source.id)
    raw = dict(category='project' if project_name else 'personal_fact', topic=topic, values=[value],
               cardinality=cardinality, quote=text, source_id=source.id,
               scope='project' if project_name else 'global', project_name=project_name)
    response = json.loads(await repo.remember(raw, [source]))
    if response['status'] != 'saved':
        raise RuntimeError('fixture seed failed: ' + response['reason'])
    return response['memory_id']


async def snapshot(repo):
    return {i.key: i.value for i in await repo.all(repo.ns)}


def assess(name, before, after, history, seeds, steps, projects, attributes, barriers):
    if name in {'temporary_script', 'forbidden', 'hypothesis', 'ambiguity'}:
        return after == before
    if name == 'delete':
        return seeds['language'] not in after and not any(v['id'] == seeds['language'] for v in history)
    if name == 'coexist':
        value = after.get(seeds['skill'], {})
        return value.get('values') == ['Java', 'Python'] and value.get('cardinality') == 'multiple' and any(
            call['args'].get('change', {}).get('action') == 'append' for step in steps for call in step['calls'])
    if name in {'project_scope', 'session_exception'}:
        scope = 'project' if name == 'project_scope' else 'session'
        return after.get(seeds['language']) == before[seeds['language']] and any(
            v['scope'] == scope and v['values'] == ['英文'] for v in after.values())
    if name == 'plan':
        return any(v['kind'] == 'plan' and 'Rust' in v['values'] for v in after.values())
    if name in {'natural_update', 'future', 'history'}:
        value = after.get(seeds['project'], {})
        passed = value.get('values') == ['Python'] and value.get('revision') == 2
        if name == 'future':
            passed = passed and bool(value.get('valid_from')) and value['valid_from'].startswith('2099-01-01')
        if name == 'history':
            passed = passed and any(v['id'] == seeds['project'] and v['values'] == ['Java'] for v in history)
        return passed
    if name == 'alias_ambiguity':
        return len(projects) == 2 and len(after) == 2 and steps[-1]['before'] == steps[-1]['after'] and not any(
            c['name'] == 'remember_memory' for c in steps[-1]['calls']) and any(
                word in steps[-1]['final'] for word in ('哪个', '哪一个', '北区', '南区'))
    if name == 'set_lifecycle':
        if len(after) != 1:
            return False
        value = next(iter(after.values()))
        operations = [c['args'].get('change', {}).get('action', 'add') for step in steps for c in step['calls']
                      if c['name'] == 'remember_memory']
        return (value['cardinality'] == 'multiple' and value['values'] == [] and value['status'] == 'inactive'
                and value['revision'] == 4 and len(history) == 3 and 'append' in operations and operations.count('remove') >= 2
                and any(a['cardinality'] == 'multiple' for a in attributes))
    if name == 'deleted_explanation':
        return not after and not history and len(barriers) == 1 and '已删除' in steps[-1]['final'] and not re.search(
            r'(?:可能|推测|大概|或许).{0,20}索引', steps[-1]['final']) and not any(
                c['name'] == 'remember_memory' for c in steps[-1]['calls'])
    if len(after) != 1 or len(projects) != 1:
        return False
    value = next(iter(after.values()))
    passed = (projects[0]['name'] == '记忆验收灯塔项目' and value['values'] == ['Python'] and value['revision'] == 2
              and len(history) == 1 and history[0]['values'] == ['Java'] and history[0]['id'] == value['id'])
    if name == 'full_name_alias':
        passed = passed and '灯塔项目' in projects[0]['aliases'] and projects[0]['revision'] == 2
    if name == 'detail_history_contract':
        receipts = steps[-1]['read_receipts']
        passed = passed and any(r.get('history_loaded') is False and 'history' not in r for r in receipts)
        passed = passed and any(r.get('history_loaded') is True and r.get('history_count') == 1 and len(r.get('history', [])) == 1 for r in receipts)
    return passed


async def run_case(config, index, name, requests):
    report = {'case': index, 'name': name, 'passed': False, 'turns': []}
    with TemporaryDirectory(prefix='memory-accept-') as root:
        c = config.model_copy(update={
            'workspace': config.workspace.model_copy(update={'root': root}), 'skills': None, 'mcp': None,
            'memory': config.memory.model_copy(update={'enabled': True}),
            'agent': config.agent.model_copy(update={'persona_file': None}),
            'tools': config.tools.model_copy(update={'terminal': None, 'web': None}),
            'integrations': config.integrations.model_copy(update={'feishu': config.integrations.feishu.model_copy(
                update={'owner_open_id': 'fictional-memory-' + str(uuid4())})})})
        report['budgets'] = c.agent.model_dump(include={'model_call_limit', 'tool_call_limit', 'summary_call_limit', 'turn_timeout_seconds', 'knowledge_call_limit'})
        async with session_storage(c.sessions, initialize=True), operation_storage(c.sessions, initialize=True):
            pass
        seeds = {}
        async with memory_storage(c, initialize=True) as repo:
            repo.session_id = 'fixture-seed'
            if index <= 12:
                seeds['project'] = await seed(repo, '主要后端语言', 'Java', project_name='灯塔项目')
                seeds['language'] = await seed(repo, '回答语言', '中文')
                seeds['skill'] = await seed(repo, '掌握的编程语言', 'Java', cardinality='multiple')
                if name == 'ambiguity':
                    await seed(repo, '主要后端语言', 'Go', project_name='海鸥项目')
        async with assistant_session(c, caller=Caller('feishu_private'), retriever=EmptyRetriever()) as session:
            repo = session.context.memory
            before = await snapshot(repo)
            session_ids = [session.session_id]
            try:
                for request in requests:
                    if request == '/new':
                        await session.new_conversation()
                        session_ids.append(session.session_id)
                        continue
                    step = {'request': request, 'before': await snapshot(repo), 'calls': [], 'read_receipts': []}
                    report['turns'].append(step)
                    async def event(item):
                        if item['event'] == 'on_tool_start':
                            step['calls'].append({'name': item['name'], 'args': item.get('data', {}).get('input', {})})
                    turn = await session.run_turn(request, on_event=event)
                    step.update(usage=asdict(turn.usage), final=final_text(turn.messages), after=await snapshot(repo))
                    current = next((m.id for m in reversed(turn.messages) if isinstance(m, HumanMessage) and 'user_source' in m.additional_kwargs), None)
                    start = next((i for i, m in enumerate(turn.messages) if m.id == current), 0)
                    for message in turn.messages[start:]:
                        if isinstance(message, ToolMessage) and message.name == 'manage_memory':
                            data = json.loads(message.content)
                            if 'history_loaded' in data:
                                step['read_receipts'].append(data)
                after = await snapshot(repo)
                history = [i.value for i in await repo.all(repo.area('history'))]
                projects = [i.value for i in await repo.all(repo.area('projects'))]
                attributes = [i.value for i in await repo.all(repo.area('attributes'))]
                barriers = [i.value for i in await repo.all(repo.area('barriers'))]
                report['passed'] = bool(assess(name, before, after, history, seeds, report['turns'], projects, attributes, barriers))
                if name == 'session_exception':
                    await session.new_conversation()
                    session_ids.append(session.session_id)
                    repo.session_id, repo.project_id = session.session_id, None
                    await repo.recall('回答语言')
                    report['passed'] = report['passed'] and '：中文。' in repo.reference and '：英文。' not in repo.reference
                if name == 'deleted_explanation':
                    ids = list(report['turns'][1]['after'])
                    count = await (await repo.conn.execute('SELECT count(*) AS n FROM store_vectors WHERE key=ANY(%s)', (ids,))).fetchone()
                    report['passed'] = report['passed'] and count['n'] == 0
                report.update(records=list(after.values()), history=history, projects=projects, attributes=attributes)
            except Exception as error:
                report['error_type'] = type(error).__name__
            # 跨 /new 汇总真实服务报告，异常调用同样计入；不重复执行样本。
            usage = await (await repo.conn.execute('''SELECT profile,role,count(*) AS calls,
                sum(input_tokens) AS input_tokens,sum(output_tokens) AS output_tokens,sum(total_tokens) AS total_tokens,
                count(*) FILTER(WHERE total_tokens IS NULL) AS unreported
                FROM assistant_model_usage WHERE session_id=ANY(%s::uuid[]) GROUP BY profile,role''', (session_ids,))).fetchall()
            report['service_usage'] = usage
    print(json.dumps({'case': index, 'name': name, 'passed': report['passed']}, ensure_ascii=False), flush=True)
    return report


async def run(args):
    if args.output and Path(args.output).exists():
        raise FileExistsError('验收报告路径已存在，请指定新路径；尚未调用模型。')
    config = load_config(args.config)
    if not config.knowledge or not config.sessions or not config.integrations.feishu:
        raise RuntimeError('需要 sessions、knowledge、飞书身份配置；仅使用虚构主人，不连接飞书。')
    reports = []
    with disposable_database(config.sessions) as settings, \
         patch('ai_assistant.knowledge.models.embedding', fixture_embedding), \
         patch('ai_assistant.runtime.create_builtin_tools', return_value=[]), \
         patch('ai_assistant.knowledge.tools.knowledge_tools', return_value=[]), \
         patch('ai_assistant.integrations.feishu.OfficialConnection.discover', side_effect=AssertionError('禁止真实飞书连接')):
        config = config.model_copy(update={'sessions': settings})
        for index, (name, requests) in enumerate(CASES, 1):
            try:
                reports.append(await run_case(config, index, name, requests))
            except Exception as error:
                reports.append({'case': index, 'name': name, 'passed': False, 'error_type': type(error).__name__})
    output = Path(args.output) if args.output else Path('.data') / ('memory-v3-acceptance-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    # 不覆盖历史运行，失败样本保留本次原始轨迹。
    with output.open('x') as file:
        json.dump({'scenarios': reports, 'passed': sum(r['passed'] for r in reports), 'total': len(CASES),
                   'real_feishu_verified': False}, file, ensure_ascii=False, indent=2, default=str)
    print(str(output), flush=True)
    return 0 if all(r['passed'] for r in reports) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--output', help='新报告路径；必须尚不存在')
    raise SystemExit(asyncio.run(run(parser.parse_args())))
