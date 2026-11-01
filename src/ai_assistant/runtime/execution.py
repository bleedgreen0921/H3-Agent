"""原生执行共用的预算及清理支撑，不调度模型或工具。"""
import asyncio
from contextvars import ContextVar

from ai_assistant.storage.sessions import SessionStorageError

tool_scope = ContextVar('assistant_tool_scope', default=None)


class ToolCleanupError(SessionStorageError):
    """清理失败后，必须阻止会话继续派发业务。"""


async def drain_tools(context):
    """异常离开工具节点时，等待所有已派发调用完成取消清理。"""
    from ai_assistant.cancellation import _drain_worker
    tasks = [task for task in context.inflight_tools if task is not asyncio.current_task()]
    for task in tasks:
        if not task.done() and not task.cancelling():
            task.cancel()
    if tasks:
        worker = asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True))
        await _drain_worker(worker)
        for result in worker.result():
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise ToolCleanupError('工具清理失败；本会话停止执行，请核实已发生效果。') from None


def raise_if_fatal(error):
    """辅助整理也不能吞掉存储故障、图中断或清理失败。"""
    from langgraph.errors import GraphBubbleUp
    if isinstance(error, (SessionStorageError, GraphBubbleUp)):
        raise error
    try:
        from psycopg import Error as DatabaseError
    except ImportError:
        DatabaseError = ()
    try:
        from sqlalchemy.exc import SQLAlchemyError
    except ImportError:
        SQLAlchemyError = ()
    if isinstance(error, DatabaseError) or isinstance(error, SQLAlchemyError):
        raise error
