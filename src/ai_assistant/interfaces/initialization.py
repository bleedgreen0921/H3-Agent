"""显式初始化已配置存储；各数据库独立提交，允许失败后幂等重跑。"""
import sys

from ai_assistant.config import ConfigError


async def execute(config):
    completed, failed = [], []
    if config.sessions is None and config.knowledge is None:
        raise ConfigError('没有可初始化的存储，请配置 [sessions] 或 [knowledge]。')
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
    print('已完成：' + ('、'.join(completed) or '无'))
    if failed:
        print('未完成：' + '、'.join(failed) + '；请检查依赖和连接后重跑 init。', file=sys.stderr)
        return 2
    return 0
