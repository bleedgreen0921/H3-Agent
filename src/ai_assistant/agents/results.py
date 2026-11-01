"""限制模型可见结果，完整原始内容和必要写入回执保留在 artifact。"""
import json
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from ai_assistant.agents.context import ContextBudgetError, count_tokens


class ToolResultMiddleware(AgentMiddleware):
    def __init__(self, budget=12000):
        self.budget = budget

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
        context = request.runtime.context
        catalog = getattr(context, 'mcp', None)
        entry = catalog.entries.get(request.tool_call['name']) if catalog else None
        # 写回执完整保留；最终预算检查会明确拒绝无法容纳的必需回执。
        if entry and entry.original_name in entry.settings.write_tools:
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
        receipts = {}
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
        preview_size = self.budget - len(receipt) - 300
        notice = '\n结果已截断；完整原始内容保存在 artifact。需要更多内容时请使用业务工具的分页或缩小查询范围。'
        return result.model_copy(update={'content': receipt + content[:preview_size] + notice, 'artifact': artifact})
