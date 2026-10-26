"""Native request selection and execution-time office authorization."""
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from copy import deepcopy
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from ai_assistant.integrations.policy import authorize


class MCPMiddleware(AgentMiddleware):
    def __init__(self, model_call_limit=10):
        self.model_call_limit = model_call_limit

    def prepare(self, request):
        catalog = request.runtime.context.mcp
        active = {e.id for e in catalog.active(request.state.get('active_mcp_tools', {}))}
        services = ', '.join(sorted({f'{e.server} ({e.settings.identity})' for e in catalog.entries.values()}))
        original = request.system_message.text if request.system_message else ''
        used = request.state.get('run_model_call_count', 0)
        remaining = max(0, self.model_call_limit - used)
        import logging
        logging.getLogger(__name__).info('Model allowance: used=%s remaining=%s', used, remaining)
        guidance = ('\n当前本机时间（含 UTC 偏移）：' + datetime.now(ZoneInfo(request.runtime.context.timezone)).isoformat(timespec='seconds') + '。'
            '日期相关问题按当前时间检索；跨时区换算使用 IANA 时区及对应日期的夏令时规则，不猜测偏移。'
            '赛程等时效事实引用实际取得的来源链接；来源冲突时明确未确认，不能声称已核实。'
            '\n已配置 MCP 服务：' + (services or '当前入口无授权服务') + '。'
            '飞书任务、文档、Wiki 等请求先 search_mcp_tools 再 load_mcp_tools 后调用；'
            '目录支持中英文关键词及 server/domain 筛选，找到目标即停止翻页；缺失能力直接报告，不能遍历目录猜测身份。'
            '飞书链接优先使用已认证 MCP；Wiki 链接先 getNode 解析 obj_token/obj_type，再读取文档正文。'
            '不要因匿名网页出现登录页就断言没有飞书凭据，也不要要求用户重复配置已有集成。'
            '修改已有任务只加载 get、patch、addMembers，不加载 create。已有时间直接读取 get 的规范化结果；日期换算禁止使用终端。目标时间先结合当前日期规范化为 ISO，再调用 resolve_datetime；不明确的日期先澄清。主人私聊为自己创建任务可用 get_current_identity，未指定负责人时遵循配置默认规则，群聊须明确日期、时区与身份，不继承主人默认时区或负责人。修改任务先按 GUID get，同时可获取可信主人身份并转换目标时间，独立只读调用可在同一响应发出。然后串行 patch 时间、addMembers 负责人；已有负责人不重复添加，保留提醒，分别核对回执后生成最终说明。只有核实回执才能宣布成功。')
        guidance += f'\n本轮主模型调用已用 {used} 次，剩余 {remaining} 次（包含当前请求），上限 {self.model_call_limit} 次；摘要另计。'
        if remaining <= 2:
            guidance += '停止无关发现，优先必要操作，保留最终确认；未完成必须如实说明。'
        return request.override(system_message=SystemMessage(content=original + guidance), tools=[t for t in request.tools
            if t.name not in catalog.entries or t.name in active])

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    async def awrap_tool_call(self, request, handler):
        # 原生工具节点可能并行调度；任务修改在可信上下文锁内串行化。
        context = request.runtime.context
        entry = context.mcp.entries.get(request.tool_call['name'])
        if entry and entry.original_name.startswith('task_v2_task_'):
            async with context.operation_lock:
                return await self.execute_tool(request, handler)
        return await self.execute_tool(request, handler)

    async def execute_tool(self, request, handler):
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
                'task_v2_task_addMembers': ('data', 'client_token'),
                'bitable_v1_appTableRecord_create': ('params', 'client_token'),
            }
            path = idempotency_paths.get(entry.original_name) if entry.settings.transport == 'feishu' else None
            arguments = deepcopy(request.tool_call['args'])
            from ai_assistant.agents.tasks import TASK_WRITES, prepare_task, read_task, task_data, discrepancies, outcome
            task_write = entry.settings.transport == 'feishu' and entry.original_name in TASK_WRITES
            if task_write and (context.task_outcomes or request.state.get('turn_outcome')):
                raise ToolException('本轮已有未核实或不符的任务回执；禁止自动补写或重放，请先报告未完成。')
            expected = prepare_task(context, entry, arguments) if task_write else {}
            if task_write and entry.original_name != 'task_v2_task_create':
                guid = arguments.get('path', {}).get('task_guid')
                actual = await read_task(context, entry, guid, params=arguments.get('params'))
                if entry.original_name == 'task_v2_task_addMembers':
                    existing = {(m.get('id'), m.get('role')) for m in actual.get('members', [])}
                    expected['members'] = [m for m in expected.get('members', []) if (m.get('id'), m.get('role')) not in existing]
                    if not expected['members']:
                        return ToolMessage(content=str({'status': 'already_present', 'task': actual}),
                            tool_call_id=request.tool_call['id'], name=entry.id)
            request = request.override(tool_call={**request.tool_call, 'args': arguments})
            if path and isinstance(arguments.get(path[0]), dict):
                arguments[path[0]].pop(path[1], None)
            operation = None
            if writing:
                if context.operations is None or context.input_id is None:
                    raise ToolException('External writes require durable operations and an input ID.')
                fresh, operation = await context.operations.begin(context.input_id,
                    context.caller.binding, entry.id, arguments)
                if not fresh and task_write:
                    from langgraph.types import Command
                    stored = operation['result'] or {}
                    detail = stored.get('business_outcome')
                    actual = task_data(ToolMessage(content=stored.get('content', ''), tool_call_id=request.tool_call['id']))
                    issues = discrepancies(expected, actual, entry.original_name, arguments.get('path', {}).get('task_guid'))
                    if detail is None and (issues or operation['status'] != 'succeeded'):
                        detail = outcome(context, operation, expected, actual, issues + ['已存在操作；未自动重放，需核实回执'])
                    if detail:
                        context.task_outcomes[operation['id']] = detail
                        return Command(update={'turn_outcome': {operation['id']: detail}, 'messages': [ToolMessage(
                            content=str(stored), tool_call_id=request.tool_call['id'], name=entry.id)]})
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
                from ai_assistant.agents.tasks import payload
                remote = payload(result)
                if isinstance(remote, dict) and remote.get('code', 0) != 0:
                    status = 'failed'
                import logging
                logging.getLogger(__name__).info('MCP write: tool=%s status=%s', entry.original_name, status)
                operation['status'] = status
                await context.operations.finish(operation['id'], status,
                    {'content': result.content} if isinstance(result, ToolMessage) else {'receipt': str(result)})
            if operation and task_write:
                actual = task_data(result)
                issues = discrepancies(expected, actual, entry.original_name, arguments.get('path', {}).get('task_guid'))
                guid = arguments.get('path', {}).get('task_guid') or actual.get('guid')
                if issues and guid and operation['status'] == 'succeeded':
                    try:
                        actual = await read_task(context, entry, guid, params=arguments.get('params'))
                        issues = discrepancies(expected, actual, entry.original_name, arguments.get('path', {}).get('task_guid'))
                    except Exception:
                        issues.append('写后回读失败；实际字段未确认')
                if operation['status'] != 'succeeded':
                    issues.append('远端状态未确认成功')
                if isinstance(result, ToolMessage):
                    result = result.model_copy(update={'artifact': {**(result.artifact if isinstance(result.artifact, dict) else {}),
                        'protected_receipt': True}})
                if issues:
                    from langgraph.types import Command
                    detail = outcome(context, operation, expected, actual, issues)
                    context.task_outcomes[operation['id']] = detail
                    await context.operations.finish(operation['id'], operation['status'],
                        {'content': result.content, 'business_outcome': detail})
                    return Command(update={'turn_outcome': {operation['id']: detail}, 'messages': [result]})
            if not writing and context.knowledge_config is not None and getattr(result, 'status', '') != 'error':
                paths = {'docx_v1_document_get': ('docx','document_id'),
                         'docx_v1_document_rawContent': ('docx','document_id'),
                         'wiki_v2_space_getNode': ('wiki','token')}
                selected = paths.get(entry.original_name)
                if selected:
                    kind, key = selected
                    args = request.tool_call['args']
                    token = (args.get('path') or {}).get(key) or (args.get('params') or {}).get(key)
                    if isinstance(token, str) and token.isalnum():
                        from ai_assistant.knowledge.sync import register
                        try:
                            await register(context.knowledge_config, f'https://docs.feishu.cn/{kind}/{token}',
                                           identity=entry.settings.identity)
                        except Exception:
                            # 读取结果仍有效；入库失败不能悄悄描述成成功。
                            if isinstance(result, ToolMessage):
                                result = result.model_copy(update={'content': str(result.content) + '\n应用状态：本次读取成功，但长期入库登记失败，请检查 knowledge_status。'})
            if entry.original_name == 'task_v2_task_get' and isinstance(result, ToolMessage):
                from ai_assistant.agents.tasks import task_model_result
                result = task_model_result(result, context)
            return result
        except ToolException as exc:
            import logging
            logging.getLogger(__name__).warning('MCP rejected: tool=%s category=authorization_or_precondition', entry.original_name)
            return ToolMessage(content=str(exc), status='error', tool_call_id=request.tool_call['id'], name=entry.id)
