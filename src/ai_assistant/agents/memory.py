"""Reference memory and authoritative per-turn receipts in native Agent state."""
import re
from typing import Annotated

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage

from ai_assistant.tools.mcp import OfficeState, merge_outcome


class MemoryState(OfficeState):
    pending_memories: list[dict]
    memory_project_id: str | None
    memory_epoch: int | None
    memory_results: Annotated[dict | None, merge_outcome]
    memory_reads: Annotated[dict | None, merge_outcome]


class MemoryMiddleware(AgentMiddleware):
    def prepare(self, request):
        memory = getattr(request.runtime.context, 'memory', None)
        if memory is None:
            return request
        original = request.system_message.text if request.system_message else ''
        source = next((m for m in reversed(request.messages) if 'user_source' in m.additional_kwargs), None)
        identity = f'\n当前用户消息 ID：{source.id}' if source else ''
        instruction = ('\n工具回执是记忆状态唯一依据。最终答复将全部记忆相关说明放入 '
                       '<memory_result>...</memory_result>，其他业务答复放在标签外；系统会用真实回执校验和补充。'
                       '不要在标签外声称已记住、更新或猜测索引延迟；不要自行要求再次授权已允许的只读核查。')
        return request.override(system_message=SystemMessage(original + '\n' + memory.reference + identity + instruction))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        memory = getattr(request.runtime.context, 'memory', None)
        if memory is not None:
            try:
                policy = await memory.get_policy()
                # 清空或格式升级后不沿用旧 checkpoint 的项目选择。
                memory.project_id = (request.state.get('memory_project_id')
                                     if request.state.get('memory_epoch') == policy['version'] else None)
                await memory.recall(memory.query)
            except Exception:
                memory.reference = '记忆存储暂不可用，不得使用旧缓存或声称已保存。'
                memory.policy = {'paused': True}
        return await handler(self.prepare(request))

    async def aafter_agent(self, state, runtime):
        writes = state.get('memory_results') or {}
        reads = state.get('memory_reads') or {}
        last = state['messages'][-1]
        labels = {'saved': '已保存', 'unchanged': '未变化', 'needs_clarification': '需要澄清',
                  'rejected': '未保存', 'conflict': '版本冲突', 'failed': '操作失败', 'unknown': '状态未知',
                  'found': '已找到', 'deleted': '已删除', 'not_found': '未找到', 'unavailable': '暂不可用'}
        lines = []
        for receipt in writes.values():
            locator = ('（' + receipt['memory_id'] + ('，revision ' + str(receipt['revision']) if receipt.get('revision') is not None else '') + '）'
                       if receipt.get('memory_id') else '')
            lines.append(labels[receipt['status']] + locator + '：' + receipt.get('summary', receipt.get('reason', '')))
        # 只读时展示最后一个查询的状态，避免将写入前的空查询当成写入后状态。
        latest = next(reversed(reads.values()), None) if reads else None
        unsafe = any(r['status'] not in {'saved', 'unchanged'} for r in writes.values())
        if latest and not writes:
            lines.append(labels[latest['status']] + '：' + latest.get('summary', ''))
            unsafe = latest['status'] in {'deleted', 'not_found', 'unavailable', 'rejected', 'failed'} or bool(latest.get('deleted_ids'))
            if latest.get('deleted_ids'):
                lines.append('已确认删除：' + '、'.join(latest['deleted_ids']) + '；不是索引延迟。')
        text = last.text
        if '<memory_result>' in text:
            if unsafe:
                text = re.sub(r'<memory_result>.*?(?:</memory_result>|$)', '', text, flags=re.S).strip()
            else:
                text = text.replace('<memory_result>', '').replace('</memory_result>', '')
        elif unsafe:
            # 模型未遵守分段协议时，不采用无法确认的完成声明。
            text = '记忆结果以以下回执为准；其他操作未自动回滚或重放。'
        if lines:
            text = text.rstrip() + '\n记忆回执：\n' + '\n'.join(lines)
        if text != last.text:
            return {'messages': [AIMessage(content=text.strip(), id=last.id)]}
        return None


class MemoryPreview:
    """流式隐藏记忆草稿，等待提交后的真实回执；标签可跨任意 chunk。"""
    def __init__(self):
        self.pending = ''
        self.inside = False

    def feed(self, text):
        self.pending += text
        output = []
        while self.pending:
            tag = '</memory_result>' if self.inside else '<memory_result>'
            index = self.pending.find(tag)
            if index >= 0:
                if not self.inside:
                    output.append(self.pending[:index])
                self.pending = self.pending[index + len(tag):]
                self.inside = not self.inside
                continue
            keep = next((n for n in range(min(len(tag) - 1, len(self.pending)), 0, -1)
                         if self.pending.endswith(tag[:n])), 0)
            if not self.inside:
                output.append(self.pending[:-keep] if keep else self.pending)
            self.pending = self.pending[-keep:] if keep else ''
            break
        return ''.join(output)

    def finish(self):
        result = self.pending if not self.inside else ''
        self.pending = ''
        return result
