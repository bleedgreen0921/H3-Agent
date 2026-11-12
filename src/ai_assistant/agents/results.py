"""限制模型可见结果，完整原始内容和必要写入回执保留在 artifact。"""
import json
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from ai_assistant.agents.context import ContextBudgetError, count_tokens

PAGINATION_KEYS = ('has_more', 'total', 'page_token', 'next_page_token', 'page_size',
                   'pageToken', 'pageSize')


def pagination_metadata(value, found=None, depth=0):
    """从嵌套 JSON 文本中提取分页元数据；跳过 items，不展开整表记录。"""
    found = {} if found is None else found
    if depth > 6:
        return found
    if isinstance(value, str):
        text = value.strip()
        if len(text) < 2 or text[0] not in '[{':
            return found
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            return found
    if isinstance(value, dict):
        # 只从分页响应对象取元数据，避免把业务字段里的同名值当成分页信息。
        if any(key in value for key in ('items', 'has_more', 'page_token', 'next_page_token')):
            for key in PAGINATION_KEYS:
                item = value.get(key)
                if isinstance(item, (str, int, float, bool)) and key not in found:
                    found[key] = item
            if isinstance(value.get('items'), list):
                found.setdefault('page_items', len(value['items']))
        for key, item in value.items():
            if key != 'items' and isinstance(item, (dict, list, str)):
                pagination_metadata(item, found, depth + 1)
    elif isinstance(value, list):
        for item in value:
            pagination_metadata(item, found, depth + 1)
    return found


class ToolResultMiddleware(AgentMiddleware):
    def __init__(self, budget=12000, policy=None):
        self.budget = budget
        self.policy = policy

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        from langgraph.types import Command
        if isinstance(result, Command) and isinstance(result.update, dict):
            from dataclasses import replace
            messages = []
            for message in result.update.get('messages', []):
                if isinstance(message, ToolMessage):
                    async def saved_message(request, message=message):
                        return message
                    message = await self.awrap_tool_call(request, saved_message)
                messages.append(message)
            return replace(result, update={**result.update, 'messages': messages})
        if not isinstance(result, ToolMessage):
            return result
        from ai_assistant.capabilities.policy import CapabilityPolicy
        context = request.runtime.context
        policy = CapabilityPolicy.from_context(context)
        if policy.effect(request.tool_call['name']) == 'unknown' and self.policy is not None:
            policy = self.policy
        catalog = getattr(context, 'mcp', None)
        entry = catalog.entries.get(request.tool_call['name']) if catalog else None
        writing = policy.effect(request.tool_call['name']) == 'write' or (
            entry and entry.original_name in entry.settings.write_tools)
        # 写回执完整保留；最终预算检查会明确拒绝无法容纳的必需回执。
        if writing:
            return result.model_copy(update={'artifact': {
                **(result.artifact if isinstance(result.artifact, dict) else {'original_artifact': result.artifact}),
                'full_tool_content': (result.artifact.get('full_tool_content', result.content)
                    if isinstance(result.artifact, dict) else result.content), 'protected_receipt': True}})
        original = result.artifact if isinstance(result.artifact, dict) else {}
        artifact = {**original, 'full_tool_content': original.get('full_tool_content', result.content),
                    'original_artifact': result.artifact}
        if result.status == 'error' or count_tokens([result]) <= self.budget:
            return result.model_copy(update={'artifact': artifact})
        content = result.content if isinstance(result.content, str) else json.dumps(result.content, ensure_ascii=False)
        receipts = {'content_truncated': True}
        artifact['content_truncated'] = True
        pagination = pagination_metadata(result.artifact)
        pagination.update(pagination_metadata(content))
        receipts.update({f'pagination.{key}': value for key, value in pagination.items()})
        def collect(value, path=''):
            if isinstance(value, dict):
                for key, item in value.items():
                    location = path + '.' + str(key)
                    if (key in {'id', 'status', 'success', 'exit_code', 'timed_out', 'path', 'sha256'}
                            or str(key).endswith('_id')) and not isinstance(item, (dict, list)):
                        receipts[location] = item
                    elif isinstance(item, (dict, list)):
                        collect(item, location)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    collect(item, f'{path}[{index}]')
        collect(result.artifact)
        try:
            collect(json.loads(content))
        except (ValueError, TypeError):
            pass
        receipt = '工具状态：' + result.status + '\n必要定位与回执：' + json.dumps(receipts, ensure_ascii=False) + '\n'
        if count_tokens([ToolMessage(receipt, tool_call_id=result.tool_call_id)]) > self.budget - 500:
            raise ContextBudgetError('单个工具结果的必要资源 ID 或回执超出预算，请缩小查询范围或分页。')
        notice = ('\n结果已截断；完整原始内容保存在 artifact。'
                  '分页字段见回执；需要更多内容时请使用业务工具的分页或缩小查询范围。')
        preview_size = self.budget - len(receipt) - 300
        return result.model_copy(update={'content': receipt + content[:preview_size] + notice, 'artifact': artifact})


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == 'true'
    return value == 1


class CoverageMiddleware(AgentMiddleware):
    """未完成分页时提醒模型：total 不是已读条数，不能声称全量。"""

    def _latest_pagination(self, messages):
        last_human = -1
        for index, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                last_human = index
        for message in reversed(messages[last_human + 1:]):
            if not isinstance(message, ToolMessage):
                continue
            metadata = pagination_metadata(message.artifact)
            metadata.update(pagination_metadata(message.content))
            if isinstance(message.artifact, dict) and _truthy(message.artifact.get('content_truncated')):
                metadata['content_truncated'] = True
            elif isinstance(message.content, str) and '结果已截断' in message.content:
                metadata['content_truncated'] = True
            if metadata:
                return metadata
        return None

    def prepare(self, request):
        metadata = self._latest_pagination(request.messages)
        if not metadata:
            return request
        has_more = _truthy(metadata.get('has_more'))
        truncated = _truthy(metadata.get('content_truncated'))
        if not has_more and not truncated:
            return request
        flags = [name for name, active in (('has_more=true', has_more),
                                           ('content_truncated=true', truncated)) if active]
        details = [f'{key}={metadata[key]}' for key in ('total', 'page_items', 'page_token')
                   if key in metadata]
        note = ('\n分页覆盖提醒：最近一次工具结果未完整覆盖当前查询（' + '，'.join(flags) + '）'
                + (f'，{"，".join(details)}' if details else '')
                + '。total 是匹配总数，不是已读条数；只有 has_more=false 且未发生本地截断时才可能覆盖全量。'
                '如果用户需要全量明细，必须继续分页或缩小页大小；'
                '如果只需要数量且已有 total，可直接用 total 回答，不必继续翻页；'
                '不得声称已读取全表、全部记录或逐页读完。')
        original = request.system_message
        return request.override(system_message=SystemMessage(
            (original.text if original else '') + note,
            additional_kwargs=original.additional_kwargs if original else {}))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))
