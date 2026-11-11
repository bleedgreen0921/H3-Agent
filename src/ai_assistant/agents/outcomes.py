"""原生轮次状态；保留业务回执与 Skills 状态。"""
from typing import Annotated
from langchain.agents import AgentState
from ai_assistant.tools.state import merge_records


class TurnState(AgentState):
    active_skills: Annotated[dict[str, str | None], merge_records]
    conversation_title: str
    execution: dict | None
    diagnostics: list[dict]
    turn_result: dict | None
    project_rules_snapshot: str
    completed_turns: int
    session_state: dict
    last_counted_input_id: str
