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
    # 规划入口的输入字段；普通 Agent 路径不会读取，但共享父图需要稳定 schema。
    input_id: str
    planning_action: str | None
    planning_goal: str | None
    planning_note: str | None
    dispatch: list[str]
    effect_facts: list[dict]
