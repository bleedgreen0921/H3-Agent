"""运行时返回值、异常和消息展示辅助函数。"""
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage

from ai_assistant.storage.sessions import SessionError
from ai_assistant.runtime.observation import TurnUsage

RECOVERY_NOTICE = ('应用状态：上一轮未完成，已发生的文件或命令效果不会回滚。'
                   '不要自动重放上一轮动作；直接处理本轮请求，仅在本轮需要操作相关资源时核实状态。')


class TurnIncompleteError(SessionError):
    """仅包含运行时生成的安全状态说明，可直接交付各入口。"""



@dataclass(frozen=True)
class AgentTurn:
    messages: list[BaseMessage]
    sources: list[Document]
    usage: TurnUsage = field(default_factory=TurnUsage)
    skill_ids: list[str] = field(default_factory=list)
    input_id: str = ""
    operations: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)


def request_preview(text):
    """只保留用户原文的单行预览，附件优先显示文件名。"""
    import json
    if text.startswith('File reference: '):
        try:
            text = json.loads(text.removeprefix('File reference: ')).get('file_name') or '文件请求'
        except (ValueError, TypeError):
            text = '文件请求'
    return ' '.join(text.split())[:40]


def final_text(messages):
    if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
        return ''
    return messages[-1].text.strip()

