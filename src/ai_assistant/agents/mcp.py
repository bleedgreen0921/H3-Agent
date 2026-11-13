"""通用 MCP 授权及调用；服务事实由 transport 适配解释。"""
import asyncio
import inspect
from copy import deepcopy
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from ai_assistant.integrations.mcp import validate_arguments
from ai_assistant.integrations.policy import authorize, operation_principal
from ai_assistant.integrations.adapters import service_adapter
from ai_assistant.integrations.tool_policy import ToolPreflightError
from ai_assistant.capabilities.policy import CapabilityPolicy
from ai_assistant.observability import event


class MCPMiddleware(AgentMiddleware):
    def prepare(self, request):
        entries = list(request.runtime.context.services['mcp'].entries.values())
        unavailable = request.runtime.context.services['mcp'].unavailable
        services = ', '.join(sorted({f'{entry.server} ({entry.settings.identity})' for entry in entries}
            | {f"{name} ({status['settings'].identity})" for name, status in unavailable.items()}))
        original = request.system_message
        guidance = '\n已配置 MCP 服务：' + (services or '当前入口无授权服务') + '；直接使用已提供的工具，缺失能力如实说明。'
        if unavailable:
            guidance += '\n当前不可用的 MCP 服务：' + ', '.join(sorted(unavailable)) + '；如实说明当前不可用。'
        adapters = {service_adapter(entry) for entry in entries} - {None}
        for adapter in sorted(adapters, key=lambda item: item.__name__):
            guidance += '\n' + adapter.guidance(entries, request.runtime.context)
        return request.override(system_message=SystemMessage(
            (original.text if original else '') + guidance,
            additional_kwargs=original.additional_kwargs if original else {}))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    async def awrap_tool_call(self, request, handler):
        return await self.execute_tool(request, handler)

    @staticmethod
    def _project(entry, result, context):
        """统一投影模型可见结果；原始回执已在调用前保存。"""
        adapter = service_adapter(entry)
        project = getattr(adapter, 'project', None) if adapter is not None else None
        return project(entry, result, context, mode='context') if project else result

    @classmethod
    def _preflight_failure(cls, entry, error, request, context):
        """尚未产生外部副作用；模型可以修正参数后继续。"""
        event('boundary.preflight', tool=entry.id, code=error.issue.code,
              status='error', recoverable=error.issue.recoverable)
        state = getattr(context, 'capabilities', None)
        if state is not None:
            state.preflight_failures = getattr(state, 'preflight_failures', 0) + 1
            if state.preflight_failures >= 3:
                return cls._project(entry, ToolMessage(
                    error.issue.message + ' 本轮已连续多次预检失败，停止继续尝试。',
                    status='error',
                    tool_call_id=request.tool_call['id'],
                    name=entry.id,
                    artifact={'execution_fact': {
                        'version': 1,
                        'nature': 'read',
                        'control': 'stop',
                        'error_code': error.issue.code,
                    }},
                ), context)
        return cls._project(entry, ToolMessage(
            error.issue.message,
            status='error',
            tool_call_id=request.tool_call['id'],
            name=entry.id,
            artifact={'execution_fact': {
                'version': 1,
                'nature': 'read',
                'control': 'continue',
                'error_code': error.issue.code,
            }},
        ), context)

    async def execute_tool(self, request, handler):
        context = request.runtime.context
        entry = context.services['mcp'].entries.get(request.tool_call['name'])
        if entry is None:
            return await handler(request)
        adapter = service_adapter(entry)
        try:
            authorize(context.caller, entry.settings, entry.original_name)
            arguments = deepcopy(request.tool_call['args'])
            schema = getattr(getattr(entry, 'tool', None), 'args_schema', None)
            if schema is not None and not isinstance(schema, dict):
                schema = schema.model_json_schema()
            try:
                validate_arguments(schema, arguments)
            except ValueError as error:
                raise ToolException(f'{entry.original_name} 参数不符合 schema：{error}') from None
            preflight = getattr(adapter, 'preflight', None) if adapter is not None else None
            if preflight is not None:
                checked = preflight(entry, arguments, context)
                if inspect.isawaitable(checked):
                    await checked
            expected, early = await adapter.prepare(entry, arguments, context, request.state, request.runtime.config) if adapter else ({}, None)
            expected = deepcopy(expected)
            expected.pop('client_token', None)
            if early is not None:
                return self._project(entry, ToolMessage(
                    early, tool_call_id=request.tool_call['id'], name=entry.id), context)
            path = adapter.idempotency_path(entry) if adapter else None
            if path and isinstance(arguments.get(path[0]), dict):
                arguments[path[0]].pop(path[1], None)
            writing = (CapabilityPolicy.from_context(context).effect(entry.id) == 'write'
                       or entry.original_name in entry.settings.write_tools)
            operation = None
            if writing:
                if context.operations is None or context.input_id is None:
                    raise ToolException('External writes require durable operations and an input ID.')
                source_key = None
                if adapter and hasattr(adapter, 'email_source_key'):
                    source_key = adapter.email_source_key(entry, arguments, context)
                options = {'source_key': source_key} if source_key else {}
                fresh, operation = await context.operations.begin(context.input_id, operation_principal(context.caller), entry.id, arguments, **options)
                if not fresh:
                    return self._project(entry, ToolMessage(
                        str({'operation_id': operation['id'], 'status': operation['status'], 'result': operation['result']}),
                        tool_call_id=request.tool_call['id'], name=entry.id, status='success' if operation['status'] == 'succeeded' and not ((operation['result'] or {}).get('verification') == 'mismatch' or (operation['result'] or {}).get('business_outcome')) else 'error'), context)
                if path:
                    arguments.setdefault(path[0], {})[path[1]] = operation['id']
            request = request.override(tool_call={**request.tool_call, 'args': arguments})
            try:
                async with asyncio.timeout(context.services['mcp'].timeout):
                    result = await handler(request)
            except BaseException:
                if operation:
                    await asyncio.shield(context.operations.finish(operation['id'], 'unknown'))
                raise
            result, status = adapter.interpret(entry, result) if adapter else (result,
                'unknown' if getattr(result, 'status', '') == 'error' else 'succeeded')
            if operation:
                operation['status'] = status
                receipt = {'version': 1, 'accepted': status == 'succeeded', 'verification': 'unverified',
                    **({'content': result.content, 'artifact': result.artifact} if isinstance(result, ToolMessage) else {'receipt': str(result)})}
                try:
                    await context.operations.finish(operation['id'], status, receipt)
                except Exception:
                    return self._project(entry, ToolMessage(
                        '远端调用已返回，但本地回执保存未确认；任务效果待核实，不得自动重发。',
                        tool_call_id=request.tool_call['id'], name=entry.id, status='error',
                        artifact={'protected_receipt': True, 'remote_result': receipt,
                                  'execution_fact': {'version': 1, 'nature': 'effect',
                                                     'error_category': 'unknown', 'control': 'stop'}}), context)
                if adapter:
                    result, detail = await adapter.verify(entry, operation, arguments, expected, result, context, request.runtime.config)
                    if detail:
                        receipt.update(detail)
                        await context.operations.finish(operation['id'], status, receipt)

            result = await adapter.after_read(entry, arguments, result, context) if adapter else result
            return self._project(entry, result, context)
        except ToolPreflightError as error:
            # 预检或 prepare 阶段的本地校验失败：operation 尚未创建。
            return self._preflight_failure(entry, error, request, context)
        except ToolException as error:
            return self._project(entry, ToolMessage(
                str(error), status='error', tool_call_id=request.tool_call['id'], name=entry.id), context)
