"""显式 Responses 业务验收：独立会话库、临时文件、摘要恢复及现有文档只读检索。

仅使用已安装依赖。需设置 SESSION_TEST_DATABASE_URL 指向独立临时库；可能产生模型费用。
每个场景只执行一次，失败停止。不会恢复文档、连接 MCP 或发送飞书消息。
"""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_assistant.agents.context import is_summary
from ai_assistant.config import load_config, SessionsConfig
from ai_assistant.knowledge.retrieval import create_knowledge_retriever
from ai_assistant.knowledge.sync import settings
from ai_assistant.runtime import assistant_session, final_text
from ai_assistant.storage.sessions import session_storage

DOCUMENT = '75ad0ab4-db15-467f-a473-4c6fd28963fe'
EXPECTED = ('2030-01-15', '评测负责人甲', 'VERIFY-a1e7945f625c96ac')


class AcceptanceFailure(RuntimeError):
    """仅携带验收入口预定义的错误码。"""


def require(condition, code):
    if not condition:
        raise AcceptanceFailure(code)


async def file_roundtrip(config, report):
    token = 'CHECK-' + uuid4().hex
    (Path(config.workspace.root) / 'source.txt').write_text(token, encoding='utf-8')
    async with assistant_session(config) as session:
        turn = await session.run_turn('调用 read_text 读取 source.txt，原样回答文件中的校验码。')
        tools = [m for m in turn.messages if isinstance(m, ToolMessage) and m.name == 'read_text']
        require(tools and all(m.status == 'success' for m in tools), 'read_tool_not_executed')
        require(any(token in m.text for m in tools), 'tool_result_mismatch')
        require(token in final_text(turn.messages), 'final_code_mismatch')
        report.update(passed=True, session_id=session.session_id, tool='read_text',
                      code_matched=True, usage=asdict(turn.usage))


async def summary_reopen(config, report):
    token = 'SUMMARY-' + uuid4().hex
    goal = '星河项目交付清单'
    selected = config.model_copy(update={'context': config.context.model_copy(
        update={'summary_every_turns': 1, 'summary_tokens': 1000})})
    async with assistant_session(selected) as session:
        sid = session.session_id
        # 合成历史直接写入独立 checkpoint；通过生产轮次触发原生定期摘要。
        history = [HumanMessage(f'用户目标：{goal}。核验码：{token}。请持续保留目标和核验码。'),
                   AIMessage('已记录用户目标和核验码。')]
        for index in range(4):
            history.extend([HumanMessage(f'合成记录 {index}：暂不改变目标和核验码。'),
                            AIMessage('收到，维持原目标和核验码。')])
        checkpoint = await session.agent.aupdate_state(
            {'configurable': {'thread_id': session._thread}}, {'messages': history})
        await session.repository.complete(checkpoint, title='Responses 合成摘要验收')
        session._complete = checkpoint
        turn = await session.run_turn('只回答当前用户目标和核验码，不调用工具。')
        state = await session.agent.aget_state(session._complete)
        summaries = [m.text for m in state.values['messages'] if is_summary(m)]
        require(turn.usage.summary_model_calls == 1, 'summary_not_called_once')
        require(summaries and token in '\n'.join(summaries) and goal in '\n'.join(summaries),
                'summary_lost_goal_or_code')
        require(token in final_text(turn.messages), 'pre_reopen_code_mismatch')
        report.update(session_id=sid, summary_preserved_goal=True, summary_preserved_code=True,
                      first_usage=asdict(turn.usage))
    # 第二次打开同一持久会话，不再主动触发定期整理，核对上次摘要恢复。
    reopened = selected.model_copy(update={'context': selected.context.model_copy(
        update={'summary_every_turns': 0})})
    async with assistant_session(reopened, session_id=sid) as session:
        state = await session.agent.aget_state(session._complete)
        require(any(is_summary(m) and token in m.text for m in state.values['messages']),
                'summary_checkpoint_not_restored')
        turn = await session.run_turn('继续之前的话题，只回答用户目标和核验码，不调用工具。')
        require(token in final_text(turn.messages) and goal in final_text(turn.messages),
                'reopened_answer_mismatch')
        rows = await (await session.context.usage_store.conn.execute('''
            SELECT role,count(*) AS calls,sum(total_tokens) AS total_tokens
            FROM assistant_model_usage WHERE session_id=%s GROUP BY role ORDER BY role
        ''', (sid,))).fetchall()
        counters = {r['role']: int(r['calls']) for r in rows}
        require(counters == {'main': 2, 'summary': 1}, 'usage_roles_mismatch')
        require(all(r['total_tokens'] is not None and r['total_tokens'] > 0 for r in rows),
                'usage_not_reported')
        report.update(passed=True, reopened=True, continued_goal_and_code=True,
                      second_usage=asdict(turn.usage), cumulative_usage=[{'role': r['role'], 'calls': int(r['calls']),
                          'total_tokens': int(r['total_tokens'])} for r in rows])


async def existing_document(config, original, report):
    # 只检索现有可见构建，不启动云端同步、恢复文档或更改知识库资料。
    retriever = create_knowledge_retriever(settings(original), filters={'document_id': DOCUMENT})
    async with assistant_session(config, retriever=retriever) as session:
        turn = await session.run_turn(
            '只调用 search_knowledge 检索项目 EVAL-20261027-225526 的交付日期、负责人和核验码，'
            '回答这三项并引用片段 ID，日期使用 YYYY-MM-DD。不调用文件工具、不猜测。检索已限定指定测试文档。')
        require(turn.sources and all(d.metadata.get('document_id') == DOCUMENT for d in turn.sources),
                'missing_or_out_of_scope_sources')
        evidence = '\n'.join(d.page_content for d in turn.sources)
        answer = final_text(turn.messages)
        require(all(value in evidence for value in EXPECTED), 'evidence_values_mismatch')
        require(all(value in answer for value in EXPECTED), 'answer_values_mismatch')
        ids = [d.id for d in turn.sources if d.id]
        require(any(identity in answer for identity in ids), 'missing_chunk_citation')
        report.update(passed=True, document_id=DOCUMENT, expected_fields_matched=True,
                      chunk_citation_matched=True, chunk_ids=ids, usage=asdict(turn.usage),
                      scope='现有可见快照检索与生产 Agent 回答；未刷新云端来源')


async def run(args):
    original = load_config(args.config)
    url = os.environ.get('SESSION_TEST_DATABASE_URL')
    require(url and original.sessions and url != original.sessions.database_url.get_secret_value(),
            'independent_test_database_required')
    report = {'protocol': 'Responses', 'passed': False, 'stages': {}, 'requests': []}

    async def capture(request):
        body = json.loads(request.content)
        report['requests'].append({'path': request.url.path,
            **{key: body[key] for key in ('model', 'reasoning', 'store', 'stream') if key in body},
            'has_previous_response_id': 'previous_response_id' in body})

    with args.output.open('x', encoding='utf-8') as output:
        args.output.chmod(0o600)
        def save():
            # 先序列化再写入，避免类型错误留下半份 JSON；每个场景结束即保存。
            content = json.dumps(report, ensure_ascii=False, indent=2)
            output.seek(0)
            output.write(content)
            output.truncate()
            output.flush()

        try:
            with TemporaryDirectory(prefix='responses-business-') as root:
                async with httpx.AsyncClient(timeout=90, event_hooks={'request': [capture]}) as client:
                    name = original.default_model
                    profile = original.models[name]
                    candidate = profile.model_copy(update={'parameters': {**profile.parameters,
                        'use_responses_api': True, 'reasoning': {'effort': 'none'},
                        'max_retries': 0, 'timeout': 90, 'http_async_client': client}})
                    config = original.model_copy(update={'models': {**original.models, name: candidate},
                        'sessions': SessionsConfig(database_url=url), 'knowledge': None, 'mcp': None,
                        'skills': None, 'workspace': original.workspace.model_copy(update={'root': root}),
                        'tools': original.tools.model_copy(update={'terminal': None, 'web': None,
                            'files': original.tools.files.model_copy(update={'enabled': False})})})
                    require(config.context is not None and not config.context.summary_model,
                            'acceptance_requires_shared_main_summary_model')
                    async with session_storage(config.sessions, initialize=True):
                        pass
                    for name, operation in (('file_roundtrip', file_roundtrip),
                                            ('summary_reopen', summary_reopen),
                                            ('existing_document', existing_document)):
                        stage = report['stages'][name] = {'passed': False}
                        print(name, 'started', flush=True)
                        if name == 'existing_document':
                            await operation(config, original, stage)
                        else:
                            await operation(config, stage)
                        save()
                        print(name, 'passed', flush=True)
                    require(report['requests'] and all(r['path'].endswith('/responses')
                        and r.get('store') is False and not r['has_previous_response_id']
                        for r in report['requests']), 'protocol_or_history_mismatch')
                    report['passed'] = True
        except Exception as exc:
            report['error_type'] = type(exc).__name__
            if isinstance(exc, AcceptanceFailure):
                report['error'] = str(exc)
            print('acceptance failed:', report['error_type'], report.get('error', ''), flush=True)
        finally:
            save()
    print('report:', args.output, 'passed:', report['passed'], flush=True)
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('结果文件必须不存在且父目录必须存在。')
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
