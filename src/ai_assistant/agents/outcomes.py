"""业务回执独立于个人记忆，保留远端写入核对保护。"""
import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from ai_assistant.tools.mcp import OfficeState


class TurnState(OfficeState):
    conversation_title: str
    turn_result: dict | None
    project_rules_snapshot: str
    completed_turns: int
    summarized_turns: int
    last_counted_input_id: str


class OutcomeMiddleware(AgentMiddleware):
    async def aafter_agent(self, state, runtime):
        from ai_assistant.runtime.outcomes import turn_result
        update = {'turn_result': turn_result(state).to_dict()}
        if state.get('turn_outcome'):
            text = ('未完成：远端任务操作已发生，但业务核对未通过；未自动重放。\n' +
                    state['messages'][-1].text + '\n' + '\n'.join(
                        json.dumps(value, ensure_ascii=False) for value in state['turn_outcome'].values()))
            update['messages'] = [AIMessage(content=text)]
        return update
