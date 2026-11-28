"""任务执行器：调用任务 Agent、重试、解析结构化结果和效果回执。"""
from __future__ import annotations

import asyncio
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from ai_assistant.agents.planning.models import TaskReport
from ai_assistant.agents.planning.prompts import _task_prompt
from ai_assistant.agents.planning.results import (
    _brief_text, _dedupe_effects, _task_result,
)
from ai_assistant.capabilities.model import CapabilityTurnState, ExposurePlan
from ai_assistant.llm.structured import StructuredOutputError, parse_structured_result
from ai_assistant.observability import event
from ai_assistant.runtime.execution import raise_if_fatal
from ai_assistant.runtime.types import final_text
from ai_assistant.tools.context import ToolContext


def _task_report(messages):
    """解析任务 agent 的 JSON 最终答复；失败返回 None，由调用方重试或失败关闭。"""
    last = next((message for message in reversed(messages or ())
                 if isinstance(message, AIMessage)), None)
    if last is None:
        return None
    try:
        return parse_structured_result(TaskReport, {'raw': last})
    except StructuredOutputError:
        return None


def _task_config(config, plan_id, task_id):
    base = dict(config or {})
    base['metadata'] = {
        **(base.get('metadata') or {}),
        'assistant_phase': 'task',
        'assistant_plan_id': plan_id,
        'assistant_plan_task': task_id,
    }
    return base


def _task_effects(runtime, before):
    facts = runtime.context.effect_facts[before:]
    return [dict(item) for item in facts if isinstance(item, dict)]


def _published_effects(messages):
    """从任务私有 ToolMessage 中提取本地文件发布事实，避免规划模式丢失回执。"""
    from langchain_core.messages import ToolMessage
    result = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact if isinstance(message.artifact, dict) else {}
        fact = artifact.get('execution_fact') if isinstance(
            artifact.get('execution_fact'), dict) else {}
        if not (fact.get('published') or artifact.get('published')):
            continue
        result.append({
            'kind': 'local_file',
            'tool': message.name,
            'status': 'saved',
            'path': artifact.get('path'),
            'sha256': artifact.get('sha256'),
            'effect_id': artifact.get('effect_id'),
            'summary': '本地文件已发布',
        })
    return result


async def _operation_ids(context):
    if context.operations is None or not context.input_id:
        return set()
    try:
        rows = await context.operations.list(context.input_id)
    except Exception as error:
        raise_if_fatal(error)
        return set()
    return {row.get('id') for row in rows if isinstance(row, dict)}


def _task_failure(task, error, *, write, attempts, effects=None, uncertain=False):
    status = 'unknown' if write and uncertain else 'failed'
    category = type(error).__name__ if error is not None else 'task_failed'
    if status == 'unknown':
        summary = '写操作效果未确认，未自动重试。'
        message = '执行异常；未确认效果。'
    else:
        summary = '任务执行失败；未观察到写入效果。' if write else '任务执行失败。'
        message = '任务未完成；没有确认的写入效果。' if write else '任务未完成。'
    if status != 'unknown':
        from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
        from ai_assistant.agents.context import ContextBudgetError
        if isinstance(error, ModelCallLimitExceededError):
            message = '任务模型调用次数已达上限，尚未产出可用结果。'
        elif isinstance(error, ToolCallLimitExceededError):
            message = '任务工具调用次数已达上限，尚未完成。'
        elif isinstance(error, ContextBudgetError):
            message = '任务输入超出模型上下文预算，需要缩小输入或分段读取。'
        elif isinstance(error, TimeoutError):
            message = '任务执行超时，尚未完成。'
    event('plan.task_failed', task_id=task.get('id'), status=status, category=category, attempts=attempts)
    return _task_result(
        task.get('id'), status, summary,
        error={'category': category, 'message': message},
        effects=effects, attempts=attempts)


async def _execute_task(payload, config, runtime: Runtime[ToolContext], *, write,
                        task_agent, registry, settings):
    task = payload.get('task') or {}
    plan_id = payload.get('plan_id')
    task_id = task.get('id')
    context = runtime.context
    execution = payload.get('execution') or {}
    tool_ids = list(execution.get('tool_ids') or [])
    capability_plan = ExposurePlan(
        tool_ids=tool_ids, server=execution.get('server'),
        identity=execution.get('identity'), effect=execution.get('effect'),
        forbid_create=bool(execution.get('forbid_create')),
        tool_schema_tokens=execution.get('schema_tokens') or registry.schema_tokens(tool_ids),
        max_schema_tokens=getattr(registry.settings, 'max_schema_tokens', 0) or 0)
    task_context = replace(
        context,
        capabilities=CapabilityTurnState(registry=registry, plan=capability_plan,
                                         read_only=not write),
        executed_tool_calls=set(),
        task_id=task_id,
    )
    metadata = _task_config(config, plan_id, task_id)
    message = HumanMessage(_task_prompt(
        task, payload.get('dependencies'), payload.get('goal'), write=write,
        request=payload.get('request'), resources=execution.get('resources') or []))
    before_effects = len(context.effect_facts)
    before_operations = await _operation_ids(context) if write else set()
    attempts_limit = settings.task_retry_attempts if not write else 1
    last_error = None
    result = None
    used = 1
    for used in range(1, attempts_limit + 1):
        try:
            async with asyncio.timeout(settings.task_timeout_seconds):
                result = await task_agent.ainvoke(
                    {'messages': [message]}, config=metadata, context=task_context)
            last_error = None
            execution = result.get('execution') or {}
            execution_status = execution.get('status')
            text = final_text(result.get('messages') or [])
            report = _task_report(result.get('messages') or ()) if execution_status not in {'unknown', 'failed'} else None
            # 结构化结果无效时只读任务重试一次；绝不能把原始文本当成成功。
            if (report is None and text and not write
                    and execution_status not in {'unknown', 'failed'}
                    and used < attempts_limit):
                last_error = StructuredOutputError('任务输出不符合结构化契约。')
                message = HumanMessage(message.text + (
                    '\n\n上一次响应不是合法的结构化结果。请严格只输出规定 JSON，'
                    '字符串中的换行必须转义，不要添加 Markdown 代码块或解释。'))
                await asyncio.sleep(0.2)
                continue
            break
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise_if_fatal(error)
            last_error = error
            from ai_assistant.agents.context import ContextBudgetError
            from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
            from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
            non_retryable = isinstance(error, (
                ContextBudgetError, ModelCallLimitExceededError, ToolCallLimitExceededError))
            if write or non_retryable or used >= attempts_limit:
                break
            await asyncio.sleep(0.2)
    effects = _task_effects(runtime, before_effects)
    if result is not None:
        effects = _dedupe_effects(effects + _published_effects(result.get('messages')))
    if result is None:
        new_operations = (await _operation_ids(context)) - before_operations if write else set()
        return {'task_results': {'plan_id': plan_id, 'items': {
            task_id: _task_failure(
                task, last_error, write=write, attempts=used, effects=effects,
                uncertain=bool(write and (effects or new_operations)))}}}
    execution = result.get('execution') or {}
    execution_status = execution.get('status')
    text = final_text(result.get('messages') or ())
    report = _task_report(result.get('messages') or ()) if execution_status not in {'unknown', 'failed'} else None
    if execution_status == 'unknown':
        summary = _brief_text(text, 600) or '任务效果未确认。'
        return {'task_results': {'plan_id': plan_id, 'items': {
            task_id: _task_result(task_id, 'unknown', summary,
                                  error={'category': execution.get('phase') or 'unknown',
                                         'message': '执行结果未确认，未自动重试。'},
                                  effects=effects, attempts=used)}}}
    if execution_status == 'failed':
        summary = _brief_text(text, 600) or '任务未完成。'
        return {'task_results': {'plan_id': plan_id, 'items': {
            task_id: _task_result(task_id, 'failed', summary,
                                  error={'category': execution.get('phase') or 'failed',
                                         'message': '任务未完成。'},
                                  effects=effects, attempts=used)}}}
    if not text and report is None:
        return {'task_results': {'plan_id': plan_id, 'items': {
            task_id: _task_result(task_id, 'failed', '任务未返回结果。',
                                  error={'category': 'empty_task_result',
                                         'message': '任务没有最终回答。'},
                                  effects=effects, attempts=used)}}}
    if report is None:
        # 解析失败且重试耗尽：停止任务，不把原始文本传给下游。
        return {'task_results': {'plan_id': plan_id, 'items': {
            task_id: _task_result(task_id, 'failed', '任务输出不是有效的结构化结果。',
                                  error={'category': 'invalid_task_output',
                                         'message': '任务输出不符合结构化契约；未采用原始文本。'},
                                  effects=effects, attempts=used)}}}
    if report.questions:
        summary = report.summary.strip()
        output = (report.output or '').strip()
        return {'task_results': {'plan_id': plan_id, 'items': {
            task_id: _task_result(
                task_id, 'needs_input', summary, output=output,
                references=list(report.references),
                questions=[str(item).strip() for item in report.questions
                           if str(item).strip()],
                effects=effects, attempts=used)}}}
    summary = report.summary.strip()
    output = (report.output or '').strip() or summary
    references = list(report.references)
    for effect in effects:
        url = effect.get('url') or effect.get('link') if isinstance(effect, dict) else None
        if isinstance(url, str) and url and url not in references:
            references.append(url)
    if write:
        after_operations = await _operation_ids(context)
        if not effects and not (after_operations - before_operations):
            return {'task_results': {'plan_id': plan_id, 'items': {
                task_id: _task_result(task_id, 'failed', '未观察到写入回执，任务未确认完成。',
                                      output=output, references=references,
                                      error={'category': 'missing_write_receipt',
                                             'message': '没有确认的写入效果。'},
                                      effects=effects, attempts=used)}}}
    return {'task_results': {'plan_id': plan_id, 'items': {
        task_id: _task_result(task_id, 'done', summary, output=output,
                              references=references, effects=effects, attempts=used)}}}
