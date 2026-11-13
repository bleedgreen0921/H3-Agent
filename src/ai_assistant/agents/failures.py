"""当前工具批次失败后，通过原生图跳转停止执行。"""
import asyncio

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp


from ai_assistant.runtime.execution import ToolCleanupError, tool_scope, raise_if_fatal
from ai_assistant.capabilities.policy import CapabilityPolicy


FAILURE_REPLY = '未完成：本轮已停止，不再重试或执行后续步骤。已确认的回执保留。'


def tool_label(context, name):
    """把内部 stable id 映射成业务名称；不把 mcp_* 暴露给用户。"""
    catalog = getattr(context, 'services', {}).get('mcp')
    entry = catalog.entries.get(name) if catalog else None
    original = getattr(entry, 'original_name', None) if entry else None
    if not original:
        return '外部工具'
    from ai_assistant.integrations.adapters import service_adapter
    adapter = service_adapter(entry)
    label = getattr(adapter, 'display_name', None) if adapter else None
    if label is None:
        return original
    try:
        return label(entry)
    except Exception:
        return original


def readonly_call(request, policy=None):
    policy = policy or CapabilityPolicy.from_context(request.runtime.context)
    return policy.effect(request.tool_call['name']) == 'read'


def execution_fact(result, request, *, unexpected=False, policy=None):
    """程序分类保存在 artifact；模型文本不能授予重试权限。"""
    from langgraph.types import Command
    from dataclasses import replace
    if isinstance(result, Command) and isinstance(result.update, dict):
        return replace(result, update={**result.update, 'messages': [
            execution_fact(m, request, unexpected=unexpected) for m in result.update.get('messages', [])]})
    if not isinstance(result, ToolMessage):
        return result
    artifact = result.artifact if isinstance(result.artifact, dict) else {'original_artifact': result.artifact}
    reading = readonly_call(request, policy)
    existing = artifact.get('execution_fact')
    existing = existing if isinstance(existing, dict) else {}
    error = result.status == 'error' or existing.get('control') == 'stop'
    recoverable = error and reading and not unexpected
    # 本地 preflight 已明确“无副作用、可修正”，即使工具是写能力也保留该事实。
    if existing.get('control') == 'continue' and existing.get('nature') == 'read':
        fact = existing
    else:
        fact = {'version': 1, 'nature': 'read' if reading else 'effect',
                'error_category': ('read_error' if recoverable else 'unclassified' if unexpected else 'effect_error') if error else None,
                'control': 'continue' if not error or recoverable else 'stop'}
    if existing.get('control') == 'stop':
        fact = existing
    if not error and request.tool_call['name'] in {'create_file', 'edit_file'}:
        fact['published'] = True
    return result.model_copy(update={'status': 'error' if error else result.status,
        'artifact': {**artifact, 'execution_fact': fact}})


class ToolFailureMiddleware(AgentMiddleware):
    def __init__(self, *, policy=None, notices=()):
        self.policy = policy
        self.notices = tuple(notices)

    async def abefore_agent(self, state, runtime):
        runtime.context.executed_tool_calls.clear()
        return {'execution': None, 'diagnostics': []}

    async def awrap_tool_call(self, request, handler):
        from ai_assistant.agents.context import ContextBudgetError, SummaryCallLimitError
        from ai_assistant.storage.sessions import SessionError
        from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
        async def execute():
            token = tool_scope.set(object())
            try:
                return await handler(request)
            finally:
                tool_scope.reset(token)
        task = asyncio.create_task(execute())
        calls = request.runtime.context.inflight_tools
        calls.add(task)
        try:
            result = await asyncio.shield(task)
            request.runtime.context.executed_tool_calls.add(request.tool_call['id'])
            return execution_fact(result, request, policy=self.policy)
        except asyncio.CancelledError:
            from ai_assistant.cancellation import _drain_worker
            if not task.done() and not task.cancelling():
                task.cancel()
            await _drain_worker(task)
            if not task.cancelled() and task.exception() is not None:
                raise ToolCleanupError('工具取消清理失败；本会话停止执行，请核实已发生效果。') from None
            raise
        except (GraphBubbleUp, SessionError, ContextBudgetError, SummaryCallLimitError,
                ModelCallLimitExceededError, ToolCallLimitExceededError):
            raise
        except Exception as error:
            raise_if_fatal(error)
            # 不将原始异常、参数或凭据写入回复和 checkpoint。
            request.runtime.context.executed_tool_calls.add(request.tool_call['id'])
            from langchain_core.tools import ToolException
            from pydantic import ValidationError
            expected = isinstance(error, (ToolException, ValidationError, OSError, TimeoutError)) and readonly_call(request)
            content = '读取失败，请检查路径、参数、权限或网络后选择其他读取方式。' if expected else '工具执行异常；实际效果未确认。'
            return execution_fact(ToolMessage(content=content, status='error',
                               artifact={'full_tool_content': content},
                               tool_call_id=request.tool_call['id'], name=request.tool_call['name']), request, unexpected=not expected, policy=self.policy)
        finally:
            calls.discard(task)

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self, state, runtime):
        # 仅消费本次实际执行的原生结果；Command 中的其他消息不会遮蔽失败。
        batch = [message for message in state['messages'] if isinstance(message, ToolMessage)
                 and message.tool_call_id in runtime.context.executed_tool_calls]
        errors = [message for message in batch if message.status == 'error'
                  and (message.artifact or {}).get('execution_fact', {}).get('control') != 'continue']
        if batch and errors:
            details = []
            unknown = False
            for message in errors:
                fact = (message.artifact or {}).get('execution_fact', {})
                label = tool_label(runtime.context, message.name)
                category = fact.get('error_category')
                if category == 'invalid_arguments':
                    details.append(f'{label}：参数未通过本地校验，未发送到远端；没有写入效果。')
                elif category == 'effect_denied':
                    details.append(f'{label}：当前任务为只读，未执行写入。')
                elif category == 'identity_mismatch':
                    details.append(f'{label}：目标文档已绑定其他飞书身份，未执行写入。')
                elif category == 'permission':
                    details.append(f'{label}：权限或授权不足，未执行写入。')
                elif category == 'remote_rejected':
                    details.append(f'{label}：远端明确拒绝，未执行写入。')
                else:
                    details.append(f'{label}：工具执行失败；远端效果未确认，不重试。')
                    unknown = unknown or category in {'unknown', 'unclassified'}
            reply = FAILURE_REPLY + ('\n' + '\n'.join(details) if details else '')
            for notice in self.notices:
                try:
                    addition = await notice(runtime.context)
                    if addition:
                        reply += addition
                except Exception:
                    pass
            return {'messages': [AIMessage(reply)], 'execution': {'status': 'unknown' if unknown else 'failed',
                    'phase': 'tool', 'reason': 'tool_error' if errors else 'business_check_failed'}, 'jump_to': 'end'}


