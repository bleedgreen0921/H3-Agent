"""Owner-only memory tools with verified user provenance."""
from langchain.tools import ToolRuntime, tool
from ai_assistant.tools.context import ToolContext


@tool
async def remember_memory(category: str, topic: str, content: str, quote: str,
                          source_id: str, runtime: ToolRuntime[ToolContext]) -> str:
    """Persist an explicitly requested personal memory. Cite exact user text and message ID."""
    memory = runtime.context.memory
    if memory is None:
        return '当前入口未启用个人记忆。'
    try:
        return await memory.remember(dict(category=category, topic=topic, content=content,
                                          quote=quote, source_id=source_id), runtime.state['messages'])
    except Exception:
        return '记忆持久化失败，尚未确认保存；请检查数据库与本地向量服务。'


@tool
async def manage_memory(runtime: ToolRuntime[ToolContext], action: str = 'list',
                        memory_id: str | None = None, offset: int = 0) -> str:
    """View or manage personal memory: list, delete, clear, pause, resume. Preserve chat history."""
    if runtime.context.memory is None:
        return '当前入口未启用个人记忆。'
    try:
        return await runtime.context.memory.manage(action, memory_id, offset)
    except Exception:
        return '记忆管理失败，请检查数据库；未确认完成。'
