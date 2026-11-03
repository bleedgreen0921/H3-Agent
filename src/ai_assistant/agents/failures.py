"""当前工具批次失败后，通过原生图跳转停止执行。"""
import asyncio

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp


from ai_assistant.runtime.execution import ToolCleanupError, tool_scope, raise_if_fatal


FAILURE_REPLY = '未完成：工具执行失败，本轮已停止，不再重试或执行后续步骤。已确认的结果和回执保留；未确认的效果仍需核实，不能视为已回滚。'


READ_TOOLS = frozenset({'read_text', 'read_file', 'search_text', 'search_files',
    'read_pdf_attachment', 'read_document_attachment', 'list_directory', 'web_search', 'web_fetch', 'search_knowledge', 'knowledge_status',
    'search_skills', 'load_skill', 'unload_skill', 'read_skill_resource',
    'get_current_identity', 'resolve_datetime', 'search_email', 'read_email', 'prepare_email_task'})


def readonly_call(request):
    name = request.tool_call['name']
    catalog = getattr(request.runtime.context, 'mcp', None)
    entry = catalog.entries.get(name) if catalog else None
    return entry.original_name not in entry.settings.write_tools if entry else name in READ_TOOLS


def execution_fact(result, request, *, unexpected=False):
    """程序分类保存在 artifact；模型文本不能授予重试权限。"""
    from langgraph.types import Command
    from dataclasses import replace
    if isinstance(result, Command) and isinstance(result.update, dict):
        return replace(result, update={**result.update, 'messages': [
            execution_fact(m, request, unexpected=unexpected) for m in result.update.get('messages', [])]})
    if not isinstance(result, ToolMessage):
        return result
    artifact = result.artifact if isinstance(result.artifact, dict) else {'original_artifact': result.artifact}
    reading = readonly_call(request)
    error = result.status == 'error' or artifact.get('execution_fact', {}).get('control') == 'stop'
    recoverable = error and reading and not unexpected
    fact = {'version': 1, 'nature': 'read' if reading else 'effect',
            'error_category': ('read_error' if recoverable else 'unclassified' if unexpected else 'effect_error') if error else None,
            'control': 'continue' if not error or recoverable else 'stop'}
    if artifact.get('execution_fact', {}).get('control') == 'stop':
        fact = artifact['execution_fact']
    if not error and request.tool_call['name'] in {'create_file', 'edit_file'}:
        fact['published'] = True
    return result.model_copy(update={'status': 'error' if error else result.status,
        'artifact': {**artifact, 'execution_fact': fact}})


class ToolFailureMiddleware(AgentMiddleware):
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
            return execution_fact(result, request)
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
                               tool_call_id=request.tool_call['id'], name=request.tool_call['name']), request, unexpected=not expected)
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
            names = '、'.join(dict.fromkeys(message.name or '工具' for message in errors))
            reply = FAILURE_REPLY + ('\n失败事项：' + names if names else '\n远端业务核对未通过。')
            from ai_assistant.integrations.feishu_adapter import unfinished_document_notice
            reply += await unfinished_document_notice(runtime.context)
            unknown = any((m.artifact or {}).get('execution_fact', {}).get('error_category') in {'unknown', 'unclassified'} for m in errors)
            return {'messages': [AIMessage(reply)], 'execution': {'status': 'unknown' if unknown else 'failed',
                    'phase': 'tool', 'reason': 'tool_error' if errors else 'business_check_failed'}, 'jump_to': 'end'}


