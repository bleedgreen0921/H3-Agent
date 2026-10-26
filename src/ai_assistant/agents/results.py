"""Bound ordinary results while retaining the full content in checkpoint artifacts."""
import json
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from ai_assistant.tools.context import ToolContext
from ai_assistant.agents.context import ContextBudgetError, count_tokens


class ToolResultMiddleware(AgentMiddleware):
    def __init__(self, budget=4000):
        self.budget = budget

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        context = request.runtime.context
        entry = context.mcp.entries.get(request.tool_call['name']) if context.mcp else None
        if entry and entry.original_name == 'task_v2_task_get':
            from ai_assistant.agents.tasks import task_model_result
            result = task_model_result(result, context)
        # 写回执完整保留；最终预算检查会明确拒绝无法容纳的必需回执。
        if entry and entry.original_name in entry.settings.write_tools:
            return result.model_copy(update={'artifact': {
                **(result.artifact if isinstance(result.artifact, dict) else {'original_artifact': result.artifact}),
                'protected_receipt': True}})
        original = result.artifact if isinstance(result.artifact, dict) else {}
        artifact = {**original, 'full_tool_content': original.get('full_tool_content', result.content),
                    'original_artifact': result.artifact}
        if count_tokens([result]) <= self.budget:
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
        notice = f'\n结果已分页；使用 read_tool_result(result_id="{result.tool_call_id}", offset={preview_size}) 继续读取。'
        return result.model_copy(update={'content': receipt + content[:preview_size] + notice, 'artifact': artifact})


@tool
async def read_tool_result(result_id: str, runtime: ToolRuntime[ToolContext], offset: int = 0,
                           history_cursor: str | None = None) -> str:
    """Read a bounded page of a full result artifact, including after history compression.

    A history_cursor searches another page of checkpoints in this conversation only.
    """
    if offset < 0:
        return 'offset 必须大于等于 0。'

    def find(messages):
        for message in reversed(messages):
            if isinstance(message, ToolMessage) and message.tool_call_id == result_id:
                artifact = message.artifact
                if isinstance(artifact, dict) and 'full_tool_content' in artifact:
                    text = artifact['full_tool_content']
                    if not isinstance(text, str):
                        text = json.dumps(text, ensure_ascii=False)
                    end = min(len(text), offset + 3000)
                    return text[offset:end] + f'\nnext_offset={end if end < len(text) else None}'
        return None

    found = find(runtime.state['messages'])
    if found is not None:
        return found
    history = getattr(runtime.context, 'checkpoint_history', None)
    if history is not None:
        # thread_id 只来自可信运行配置；模型只能指定本会话内的分页位置。
        config = {'configurable': {'thread_id': runtime.config['configurable']['thread_id']}}
        before = {'configurable': {**config['configurable'], 'checkpoint_id': history_cursor}} if history_cursor else None
        last = None
        try:
            async for snapshot in history(config, limit=20, before=before):
                found = find(snapshot.values.get('messages', []))
                if found is not None:
                    return found
                last = snapshot.config['configurable']['checkpoint_id']
        except Exception:
            return '历史 artifact 读取失败，请检查会话存储；未重新执行原工具。'
        if last:
            return f'当前历史页未找到，使用 history_cursor="{last}" 继续搜索；不会重新执行原工具。'
    return '未找到完整结果，请重新分页查询原工具。'
