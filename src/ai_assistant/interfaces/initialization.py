"""显式初始化已配置存储；各数据库独立提交，允许失败后幂等重跑。"""
import sys

from ai_assistant.config import ConfigError


async def execute(config):
    """显式初始化已配置存储，分别汇总成功和失败项。"""
    completed, failed = [], []
    if config.sessions is None and config.knowledge is None:
        raise ConfigError('没有可初始化的存储，请配置 [sessions] 或 [knowledge]。')

    # 渠道操作存储依赖会话存储，只有会话初始化成功后才尝试。
    if config.sessions is not None:
        try:
            from ai_assistant.storage.sessions import session_storage
            async with session_storage(config.sessions, initialize=True):
                pass
            completed.append('会话')
        except Exception:
            failed.append('会话')
        if '会话' in completed:
            try:
                from ai_assistant.storage.integrations import operation_storage
                async with operation_storage(config.sessions, initialize=True):
                    pass
                completed.append('渠道操作')
            except Exception:
                failed.append('渠道操作')
        else:
            failed.append('渠道操作（依赖会话）')

    # 知识库独立初始化；无论成功失败都关闭已创建的存储对象。
    if config.knowledge is not None:
        store = None
        try:
            from ai_assistant.storage.rag import RagStore
            from ai_assistant.knowledge.sync import settings
            store = RagStore(settings(config))
            await store.init()
            completed.append('知识库')
        except Exception:
            failed.append('知识库')
        finally:
            if store is not None:
                await store.close()

    # 各存储独立提交；部分失败时报告完成范围，允许后续重跑。
    print('已完成：' + ('、'.join(completed) or '无'))
    if failed:
        print('未完成：' + '、'.join(failed) + '；请检查依赖和连接后重跑 init。', file=sys.stderr)
        return 2
    return 0
