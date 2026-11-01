"""仅在知识操作边界转换预期故障；会话存储与控制异常保持原样。"""
import asyncio
from functools import wraps

from ai_assistant.knowledge import KnowledgeError


def knowledge_operation(function):
    @wraps(function)
    async def invoke(*args, **kwargs):
        try:
            return await function(*args, **kwargs)
        except Exception as error:
            # 只识别知识操作产生的数据库异常，不按异常文本判断或吞掉其他错误。
            from ai_assistant.runtime.execution import ToolCleanupError
            from ai_assistant.storage.sessions import SessionError
            from langgraph.errors import GraphBubbleUp
            if isinstance(error, (SessionError, ToolCleanupError, GraphBubbleUp)):
                raise
            original = error.__context__
            while original is not None:
                if isinstance(original, (asyncio.CancelledError, SessionError, GraphBubbleUp)):
                    raise ToolCleanupError('知识工具清理失败，本会话停止执行。') from None
                original = original.__context__
            if isinstance(error, ImportError):
                raise KnowledgeError('知识能力依赖不可用，请检查已安装的 RAG 可选依赖。') from None
            if isinstance(error, (ConnectionError, TimeoutError)):
                raise KnowledgeError('知识能力连接失败，请检查知识库及服务状态。') from None
            try:
                from sqlalchemy.exc import SQLAlchemyError
            except ImportError:
                SQLAlchemyError = ()
            try:
                from psycopg import Error as DatabaseError
            except ImportError:
                DatabaseError = ()
            if isinstance(error, SQLAlchemyError) or isinstance(error, DatabaseError):
                raise KnowledgeError('知识库数据库操作失败，请检查知识服务状态。') from None
            raise
    return invoke
