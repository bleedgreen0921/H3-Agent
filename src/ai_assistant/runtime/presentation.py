"""已保存轮次的纯呈现投影；不执行模型、工具或业务状态推断。"""
from dataclasses import asdict, dataclass

from ai_assistant.integrations.tool_policy import project_output_text


SAFE_MESSAGE = '保存未确认、状态待核实；请读取历史记录及操作回执核实，未自动重跑。'
LABELS = {'partial': '部分完成', 'failed': '未完成', 'unknown': '状态待核实',
          'cancelled': '已取消', 'needs_clarification': '需要补充信息'}


@dataclass(frozen=True)
class Presentation:
    state: str
    label: str
    body: str
    source: dict
    diagnostics: list

    @property
    def text(self):
        text = (self.label + '\n\n' if self.label else '') + self.body
        if self.diagnostics:
            text += '\n\n必要诊断：\n' + '\n'.join(
                str(item.get('summary') or '会话维护未完成，业务正文已保留。') for item in self.diagnostics)
        return text

    def snapshot(self):
        return {'version': 1, **asdict(self), 'text': self.text}


def project(result, body):
    """调用方仅传入持久结果和对应原文；不转换缺失当前契约的结果。"""
    state = result['status']
    label = '' if state == 'complete' else LABELS.get(state, '状态待核实')
    body = project_output_text(body)
    return Presentation(state, label, body, {'input_id': result.get('input_id'),
                        'body_locator': result.get('body_locator')}, result.get('diagnostics') or [])


def saved_body(result, messages):
    locator = result.get('body_locator') or {}
    if not locator.get('message_id'):
        return None
    return next((message.text for message in messages if message.id == locator['message_id']), None)
