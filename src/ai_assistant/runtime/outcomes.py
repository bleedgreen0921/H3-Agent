"""由真实回执生成统一轮次终态，入口不使用模型正文判断完成。"""
from dataclasses import asdict, dataclass, field

TERMINAL_STATES = frozenset({'complete', 'partial', 'failed', 'unknown', 'cancelled', 'needs_clarification'})
SUCCESS = frozenset({'saved', 'unchanged', 'found', 'deleted', 'not_found', 'succeeded'})


@dataclass(frozen=True)
class TurnResult:
    status: str
    items: list[dict] = field(default_factory=list)

    execution: dict | None = None
    diagnostics: list[dict] = field(default_factory=list)
    body_locator: dict | None = None
    version: int = 1

    def to_dict(self):
        result = asdict(self)
        if self.execution is None:
            result.pop('execution')
        if not self.diagnostics:
            result.pop('diagnostics')
        for key in ('body_locator',):
            if result[key] is None:
                result.pop(key)
        return result


def turn_result(state, operations=(), *, interrupted=None, executed_tool_calls=None):
    items = list(state.get('effect_facts') or [])
    confirmed = any(item.get('kind') == 'local_file' and item.get('status') == 'saved' for item in items)
    from ai_assistant.runtime.effects import operation_receipt
    for operation in operations:
        receipt = operation_receipt(operation)
        confirmed |= receipt.accepted
        items.append(receipt.to_item())
    from langchain_core.messages import HumanMessage, ToolMessage
    messages = []
    for message in reversed(state.get('messages', [])):
        if isinstance(message, HumanMessage):
            break
        messages.append(message)
    for message in reversed(messages):
        if not isinstance(message, ToolMessage) or (executed_tool_calls is not None and message.tool_call_id not in executed_tool_calls):
            continue
        artifact = message.artifact if isinstance(message.artifact, dict) else {}
        fact = artifact.get('execution_fact', {})
        if fact.get('published'):
            confirmed = True
            items.append({'kind': 'local_file', 'tool': message.name, 'status': 'saved',
                'path': artifact.get('path'), 'sha256': artifact.get('sha256'),
                'effect_id': artifact.get('effect_id'),
                'summary': '本地文件已发布'})
        if fact.get('registered'):
            confirmed = True
            registration = artifact.get('registration', {})
            items.append({'kind': 'registration', 'status': 'succeeded',
                'source_id': registration.get('source_id'), 'sync_id': registration.get('sync_id'),
                'searchable_confirmed': False, 'summary': '来源已登记，后台任务已提交；尚未确认可检索'})
        if message.status == 'error':
            recoverable = fact.get('control') == 'continue'
            items.append({'kind': 'read_error' if recoverable else 'tool_failure', 'tool': message.name,
                'tool_call_id': message.tool_call_id,
                'status': 'read_error' if recoverable else 'unknown' if fact.get('error_category') in {'unknown', 'unclassified'} else 'failed',
                'summary': '可恢复读取错误，已交回模型处理' if recoverable else '工具执行停止；未确认效果需核实'})
    bad = [item['status'] for item in items if item['status'] not in SUCCESS and item['status'] != 'read_error']
    execution = state.get('execution')
    unknown = any(value in {'unknown', 'pending', 'sending'} for value in bad)
    execution_status = (execution or {}).get('status')
    if interrupted == 'cancelled' or execution_status == 'cancelled':
        status = 'partial' if confirmed else 'cancelled'
    elif unknown or interrupted == 'unknown' or execution_status == 'unknown':
        status = 'partial' if confirmed else 'unknown'
    elif interrupted or bad or (execution and execution.get('status') in {'failed', 'unknown'}):
        status = 'partial' if confirmed else 'failed'
    else:
        status = 'complete'
    return TurnResult(status, items, execution, state.get('diagnostics') or [])
