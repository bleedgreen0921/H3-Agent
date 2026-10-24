"""Native request selection and execution-time office authorization."""
import asyncio
from copy import deepcopy
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from ai_assistant.integrations.policy import authorize


class MCPMiddleware(AgentMiddleware):
    def prepare(self, request):
        catalog = request.runtime.context.mcp
        active = {e.id for e in catalog.active(request.state.get('active_mcp_tools', {}))}
        return request.override(tools=[t for t in request.tools
            if t.name not in catalog.entries or t.name in active])

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    async def awrap_tool_call(self, request, handler):
        context = request.runtime.context
        entry = context.mcp.entries.get(request.tool_call['name'])
        if entry is None:
            return await handler(request)
        try:
            if request.state.get('active_mcp_tools', {}).get(entry.id) != entry.digest:
                raise ToolException('Load this MCP tool with the current schema before calling it.')
            authorize(context.caller, entry.server, entry.settings, entry.original_name,
                      request.tool_call['args'])
            writing = entry.original_name in entry.settings.write_tools
            idempotency_paths = {
                'im_v1_message_create': ('data', 'uuid'),
                'im_v1_message_reply': ('data', 'uuid'),
                'calendar_v4_calendarEvent_create': ('params', 'idempotency_key'),
                'task_v2_task_create': ('data', 'client_token'),
                'bitable_v1_appTableRecord_create': ('params', 'client_token'),
            }
            path = idempotency_paths.get(entry.original_name) if entry.settings.transport == 'feishu' else None
            arguments = deepcopy(request.tool_call['args'])
            mail_send = entry.settings.transport == 'netease' and entry.original_name == 'send'
            if mail_send:
                arguments.pop('operation_id', None)
            if path and isinstance(arguments.get(path[0]), dict):
                arguments[path[0]].pop(path[1], None)
            operation = None
            if writing:
                if context.operations is None or context.input_id is None:
                    raise ToolException('External writes require durable operations and an input ID.')
                fresh, operation = await context.operations.begin(context.input_id,
                    context.caller.binding, entry.id, arguments)
                if not fresh:
                    return ToolMessage(content=str({'operation_id': operation['id'],
                        'status': operation['status'], 'result': operation['result']}),
                        tool_call_id=request.tool_call['id'], name=entry.id)
                if mail_send:
                    arguments['operation_id'] = operation['id']
                    await context.operations.finish(operation['id'], 'pending',
                        {'correlation': {'operation_id': operation['id'], 'server': entry.server,
                                         'account': entry.settings.account}})
                    request = request.override(tool_call={**request.tool_call, 'args': arguments})
                if path:
                    arguments.setdefault(path[0], {})[path[1]] = operation['id']
                    request = request.override(tool_call={**request.tool_call, 'args': arguments})
            try:
                async with asyncio.timeout(context.mcp.timeout):
                    result = await handler(request)
            except BaseException:
                if operation:
                    # A transport failure cannot prove that a remote write did not happen.
                    await asyncio.shield(context.operations.finish(operation['id'], 'unknown'))
                raise
            if operation:
                status = 'unknown' if getattr(result, 'status', '') == 'error' else 'succeeded'
                artifact = getattr(result, 'artifact', None)
                structured = artifact.get('structured_content') if isinstance(artifact, dict) else None
                if isinstance(structured, dict) and structured.get('status') in {'unknown', 'rejected', 'partially_accepted', 'replacement_saved_old_retained'}:
                    status = 'unknown' if structured['status'] == 'unknown' else 'failed' if structured['status'] == 'rejected' else 'succeeded'
                await context.operations.finish(operation['id'], status,
                    {'content': result.content} if isinstance(result, ToolMessage) else {'receipt': str(result)})
            return result
        except ToolException as exc:
            return ToolMessage(content=str(exc), status='error', tool_call_id=request.tool_call['id'], name=entry.id)
