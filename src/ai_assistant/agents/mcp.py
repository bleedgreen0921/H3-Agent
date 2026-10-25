"""Native request selection and execution-time office authorization."""
import asyncio
from datetime import datetime
from copy import deepcopy
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from ai_assistant.integrations.policy import authorize


class MCPMiddleware(AgentMiddleware):
    def prepare(self, request):
        catalog = request.runtime.context.mcp
        active = {e.id for e in catalog.active(request.state.get('active_mcp_tools', {}))}
        services = ', '.join(sorted({f'{e.server} ({e.settings.identity})' for e in catalog.entries.values()}))
        original = request.system_message.text if request.system_message else ''
        guidance = ('\n当前本机时间（含 UTC 偏移）：' + datetime.now().astimezone().isoformat(timespec='seconds') + '。'
            '日期相关问题按当前时间检索；跨时区换算使用 IANA 时区及对应日期的夏令时规则，不猜测偏移。'
            '赛程等时效事实引用实际取得的来源链接；来源冲突时明确未确认，不能声称已核实。'
            '\n已配置 MCP 服务：' + (services or '当前入口无授权服务') + '。'
            '飞书任务、文档、Wiki 等请求先 search_mcp_tools 再 load_mcp_tools 后调用；'
            '目录按原始名称和描述匹配，中文无结果时改用 task、wiki、docx 等关键词或空查询并分页。'
            '飞书链接优先使用已认证 MCP；Wiki 链接先 getNode 解析 obj_token/obj_type，再读取文档正文。'
            '不要因匿名网页出现登录页就断言没有飞书凭据，也不要要求用户重复配置已有集成。'
            '只激活当前任务需要的工具，任务结束后可卸载。')
        return request.override(system_message=SystemMessage(content=original + guidance), tools=[t for t in request.tools
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
