"""通用 MCP 授权及调用；服务事实由 transport 适配解释。"""
import asyncio
from copy import deepcopy
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from ai_assistant.integrations.policy import authorize, operation_principal
from ai_assistant.integrations.adapters import service_adapter


class MCPMiddleware(AgentMiddleware):
    def prepare(self, request):
        entries = list(request.runtime.context.mcp.entries.values())
        unavailable = request.runtime.context.mcp.unavailable
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

    async def execute_tool(self, request, handler):
        context = request.runtime.context
        entry = context.mcp.entries.get(request.tool_call['name'])
        if entry is None:
            return await handler(request)
        adapter = service_adapter(entry)
        try:
            authorize(context.caller, entry.server, entry.settings, entry.original_name, request.tool_call['args'])
            arguments = deepcopy(request.tool_call['args'])
            expected, early = await adapter.prepare(entry, arguments, context, request.state, request.runtime.config) if adapter else ({}, None)
            expected = deepcopy(expected)
            expected.pop('client_token', None)
            if early is not None:
                return ToolMessage(early, tool_call_id=request.tool_call['id'], name=entry.id)
            path = adapter.idempotency_path(entry) if adapter else None
            if path and isinstance(arguments.get(path[0]), dict):
                arguments[path[0]].pop(path[1], None)
            writing = entry.original_name in entry.settings.write_tools
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
                    return ToolMessage(str({'operation_id': operation['id'], 'status': operation['status'], 'result': operation['result']}),
                        tool_call_id=request.tool_call['id'], name=entry.id, status='success' if operation['status'] == 'succeeded' and not ((operation['result'] or {}).get('verification') == 'mismatch' or (operation['result'] or {}).get('business_outcome')) else 'error')
                if path:
                    arguments.setdefault(path[0], {})[path[1]] = operation['id']
            request = request.override(tool_call={**request.tool_call, 'args': arguments})
            try:
                async with asyncio.timeout(context.mcp.timeout):
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
                    return ToolMessage('远端调用已返回，但本地回执保存未确认；任务效果待核实，不得自动重发。',
                        tool_call_id=request.tool_call['id'], name=entry.id, status='error',
                        artifact={'protected_receipt': True, 'remote_result': receipt,
                                  'execution_fact': {'version': 1, 'nature': 'effect',
                                                     'error_category': 'unknown', 'control': 'stop'}})
                if adapter:
                    result, detail = await adapter.verify(entry, operation, arguments, expected, result, context, request.runtime.config)
                    if detail:
                        receipt.update(detail)
                        await context.operations.finish(operation['id'], status, receipt)

            return await adapter.after_read(entry, arguments, result, context) if adapter else result
        except ToolException as error:
            return ToolMessage(str(error), status='error', tool_call_id=request.tool_call['id'], name=entry.id)
