"""通用 MCP 授权及调用；不包含产品资源身份绑定。"""
import asyncio
from copy import deepcopy

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException

from ai_assistant.capabilities.policy import CapabilityPolicy
from ai_assistant.integrations.mcp import validate_arguments
from ai_assistant.integrations.policy import authorize, operation_principal
from ai_assistant.integrations.tool_policy import PreflightIssue, ToolPreflightError
from ai_assistant.observability import event


class MCPMiddleware(AgentMiddleware):
    """通用 MCP 工具执行边界：授权、预检、写回执和超时。"""

    def prepare(self, request):
        entries = list(request.runtime.context.services['mcp'].entries.values())
        unavailable = request.runtime.context.services['mcp'].unavailable
        services = ', '.join(sorted({f'{entry.server} ({entry.settings.identity})' for entry in entries}
            | {f"{name} ({status['settings'].identity})" for name, status in unavailable.items()}))
        original = request.system_message
        guidance = '\n已配置 MCP 服务：' + (services or '当前入口无授权服务') + '；直接使用已提供的工具，缺失能力如实说明。'
        if unavailable:
            guidance += '\n当前不可用的 MCP 服务：' + ', '.join(sorted(unavailable)) + '；如实说明当前不可用。'
        return request.override(system_message=SystemMessage(
            (original.text if original else '') + guidance,
            additional_kwargs=original.additional_kwargs if original else {}))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    async def awrap_tool_call(self, request, handler):
        return await self.execute_tool(request, handler)

    @classmethod
    def _preflight_failure(cls, entry, error, request):
        """尚未产生外部副作用；模型可以修正参数后继续。"""
        event('boundary.preflight', tool=entry.id, code=error.issue.code,
              status='error', recoverable=error.issue.recoverable)
        state = getattr(request.runtime.context, 'capabilities', None)
        stop = False
        if state is not None:
            state.preflight_failures = getattr(state, 'preflight_failures', 0) + 1
            stop = state.preflight_failures >= 3
        message = error.issue.message + (
            ' 本轮已连续多次预检失败，停止继续尝试。' if stop else '')
        return ToolMessage(
            message, status='error', tool_call_id=request.tool_call['id'],
            name=entry.id,
            artifact={'execution_fact': {
                'version': 1, 'nature': 'read',
                'control': 'stop' if stop else 'continue',
                'error_code': error.issue.code,
                'error_category': error.issue.code,
            }})

    async def execute_tool(self, request, handler):
        context = request.runtime.context
        entry = context.services['mcp'].entries.get(request.tool_call['name'])
        if entry is None:
            return await handler(request)
        try:
            authorize(context.caller, entry.settings, entry.original_name)
            arguments = deepcopy(request.tool_call['args'])
            writing = (CapabilityPolicy.from_context(context).effect(entry.id) == 'write'
                       or entry.original_name in entry.settings.write_tools)
            state = getattr(context, 'capabilities', None)
            plan = getattr(state, 'plan', None)
            if plan is not None and plan.effect == 'read' and writing:
                raise ToolPreflightError(PreflightIssue(
                    'effect_denied', '当前任务为只读，不能执行写入。', recoverable=False))
            schema = getattr(getattr(entry, 'tool', None), 'args_schema', None)
            if schema is not None and not isinstance(schema, dict):
                schema = schema.model_json_schema()
            try:
                validate_arguments(schema, arguments)
            except ValueError as error:
                # 参数校验发生在 operation 创建和远端调用之前；这是无副作用的
                # 预检失败，必须允许模型按同一 schema 修正一次，而不是当成写失败。
                raise ToolPreflightError(PreflightIssue(
                    'invalid_arguments', str(error), recoverable=True)) from None

            operation = None
            if writing:
                if context.operations is None or context.input_id is None:
                    raise ToolException('External writes require durable operations and an input ID.')
                fresh, operation = await context.operations.begin(
                    context.input_id, operation_principal(),
                    entry.id, arguments)
                if not fresh:
                    result = operation.get('result') or {}
                    status = 'success' if (
                        operation['status'] == 'succeeded'
                        and result.get('verification') != 'mismatch'
                        and not result.get('business_outcome')) else 'error'
                    return ToolMessage(
                        str({'operation_id': operation['id'],
                             'status': operation['status'],
                             'result': operation['result']}),
                        tool_call_id=request.tool_call['id'], name=entry.id,
                        status=status)
            request = request.override(tool_call={**request.tool_call, 'args': arguments})
            try:
                async with asyncio.timeout(context.services['mcp'].timeout):
                    result = await handler(request)
            except BaseException:
                if operation:
                    await asyncio.shield(context.operations.finish(operation['id'], 'unknown'))
                raise

            status = 'unknown' if getattr(result, 'status', '') == 'error' else 'succeeded'
            if operation:
                operation['status'] = status
                receipt = {'version': 1, 'accepted': status == 'succeeded',
                           'verification': 'unverified'}
                if isinstance(result, ToolMessage):
                    receipt.update(content=result.content, artifact=result.artifact)
                else:
                    receipt['receipt'] = str(result)
                try:
                    await context.operations.finish(operation['id'], status, receipt)
                except Exception:
                    return ToolMessage(
                        '远端调用已返回，但本地回执保存未确认；任务效果待核实，不得自动重发。',
                        tool_call_id=request.tool_call['id'], name=entry.id,
                        status='error',
                        artifact={'protected_receipt': True,
                                  'remote_result': receipt,
                                  'execution_fact': {
                                      'version': 1, 'nature': 'effect',
                                      'error_category': 'unknown', 'control': 'stop'}})
            return result
        except ToolPreflightError as error:
            # 预检阶段的本地校验失败：operation 尚未创建。
            return self._preflight_failure(entry, error, request)
        except ToolException as error:
            return ToolMessage(str(error), status='error',
                               tool_call_id=request.tool_call['id'], name=entry.id)
